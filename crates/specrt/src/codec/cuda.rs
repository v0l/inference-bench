use super::compile::{Kind, LaunchPlan, Part, plan};
use super::graph::{CodecOp, Stack};
use super::host::{HostCodec, HostStack, Param};
use crate::codegen::Module;
use crate::jit::Jit;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use kir::conv::{ConvKernel, LINE_ROW_KERNELS, RING_THREADS, THREADS};
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn module(all: &[ConvKernel]) -> Result<Module> {
    let source = kir::cuda::emit_line(all).map_err(anyhow::Error::msg)?;
    let mut names: Vec<&'static str> = LINE_ROW_KERNELS.to_vec();
    for k in all {
        names.push(Box::leak(k.name().into_boxed_str()));
    }
    names.sort_unstable();
    names.dedup();
    Ok(Module { source, kernels: names, smem: BTreeMap::new(), arch_features: false, kir_hash: kir::fold_hashes(all.iter().map(ConvKernel::hash)) })
}

struct Dev {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

pub struct GpuStack {
    stack: Stack,
    weights: BTreeMap<(usize, Part), Dev>,
}

pub struct Sessions {
    pub batch: usize,
    state: BTreeMap<usize, [CudaSlice<f32>; 2]>,
    flip: bool,
}

impl Sessions {
    pub fn floats_per_session(&self) -> usize {
        self.state.values().map(|s| s[0].len()).sum::<usize>() / self.batch
    }

    pub fn fixed(mut self) -> Sessions {
        self.flip = false;
        self
    }

    pub fn segments(&self, stream: &Arc<CudaStream>) -> Vec<(u64, u64, usize)> {
        self.state.values().map(|p| (p[0].device_ptr(stream).0, p[1].device_ptr(stream).0, p[0].len() / self.batch)).collect()
    }
}

pub struct FrameBufs {
    plan: LaunchPlan,
    bufs: Vec<CudaSlice<f32>>,
    ptrs: Vec<u64>,
    pub batch: usize,
}

impl FrameBufs {
    pub fn input(&mut self) -> &mut CudaSlice<f32> {
        &mut self.bufs[self.plan.input]
    }

    pub fn input_ptr(&self) -> u64 {
        self.ptrs[self.plan.input]
    }

    pub fn output(&self) -> &CudaSlice<f32> {
        &self.bufs[self.plan.output]
    }

    pub fn output_floats(&self) -> usize {
        self.batch * self.plan.t_out * self.plan.out_channels
    }
}

pub struct CudaCodec {
    stream: Arc<CudaStream>,
    target: kir::Target,
    fns: BTreeMap<String, CudaFunction>,
    kernels: BTreeMap<String, ConvKernel>,
    pub encoder: GpuStack,
    pub decoder: GpuStack,
    pub frame_samples: usize,
    pub latent_dim: usize,
}

fn upload(stream: &Arc<CudaStream>, h: &HostStack) -> Result<GpuStack> {
    let mut weights = BTreeMap::new();
    for (i, p) in h.params.iter().enumerate() {
        let mut put = |part: Part, w: &[f32], b: &[f32]| -> Result<()> {
            weights.insert((i, part), Dev { w: stream.clone_htod(w)?, b: stream.clone_htod(if b.is_empty() { &[0.0f32][..] } else { b })? });
            Ok(())
        };
        match p {
            Param::Conv(c) => put(Part::Main, &c.w, &c.b)?,
            Param::Linear { w, b } | Param::Norm { w, b } => put(Part::Main, w, b)?,
            Param::Vec(g) => put(Part::Main, g, &[])?,
            Param::Attention { qkv, out } => {
                let c = out.len().isqrt();
                put(Part::Main, qkv, &vec![0.0; 3 * c])?;
                put(Part::Out, out, &vec![0.0; c])?;
            }
        }
    }
    Ok(GpuStack { stack: h.stack.clone(), weights })
}

fn rows(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn state_floats(op: &CodecOp) -> usize {
    match op {
        CodecOp::Conv { k, stride, cin, .. } => (k - stride) * cin,
        CodecOp::ConvUp { cin, .. } => *cin,
        CodecOp::Attention { .. } => op.state_floats(),
        _ => 0,
    }
}

impl CudaCodec {
    pub fn load(ctx: &Arc<CudaContext>, jit: &Jit, host: &HostCodec, shapes: &[(usize, usize)]) -> Result<CudaCodec> {
        Self::load_on(&ctx.default_stream(), jit, host, shapes)
    }

    pub fn load_on(stream: &Arc<CudaStream>, jit: &Jit, host: &HostCodec, shapes: &[(usize, usize)]) -> Result<CudaCodec> {
        let ctx = stream.context();
        let target = crate::codegen::cuda_target(ctx)?;
        let mut all = Vec::new();
        for &(steps, batch) in shapes {
            if host.plan.has_encoder() {
                all.extend(plan(&host.plan.encoder, steps * host.plan.frame_samples, batch)?.kernels(&target)?);
            }
            all.extend(plan(&host.plan.decoder, steps, batch)?.kernels(&target)?);
        }
        all.sort_by_key(ConvKernel::name);
        all.dedup_by_key(|k| k.name());
        let m = module(&all)?;
        let loaded = jit.load(&m)?;
        let mut fns = BTreeMap::new();
        for name in &m.kernels {
            fns.insert(name.to_string(), loaded.function(name)?);
        }
        use cudarc::driver::sys::CUfunction_attribute as A;
        for k in &all {
            fns[&k.name()].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes() as i32)?;
        }
        fns["line_ring_attention"].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, target.smem_limit as i32)?;
        let stream = stream.clone();
        let encoder = upload(&stream, &host.encoder)?;
        let decoder = upload(&stream, &host.decoder)?;
        let kernels = all.into_iter().map(|k| (k.name(), k)).collect();
        Ok(CudaCodec { stream, target, fns, kernels, encoder, decoder, frame_samples: host.plan.frame_samples, latent_dim: host.plan.latent_dim })
    }

    pub fn sessions(&self, which: &GpuStack, batch: usize) -> Result<Sessions> {
        let mut state = BTreeMap::new();
        for (i, op) in which.stack.ops.iter().enumerate() {
            let n = batch * state_floats(op);
            if n > 0 {
                state.insert(i, [self.stream.alloc_zeros(n)?, self.stream.alloc_zeros(n)?]);
            }
        }
        Ok(Sessions { batch, state, flip: true })
    }

    pub fn frame_buffers(&self, which: &GpuStack, steps: usize, batch: usize) -> Result<FrameBufs> {
        let p = plan(&which.stack, steps, batch)?;
        let bufs: Vec<CudaSlice<f32>> = p.buffer_floats.iter().map(|n| self.stream.alloc_zeros((*n).max(1))).collect::<Result<_, _>>()?;
        let ptrs: Vec<u64> = bufs.iter().map(|v| v.device_ptr(&self.stream).0).collect();
        Ok(FrameBufs { plan: p, bufs, ptrs, batch })
    }

    pub fn launch_frame(&self, which: &GpuStack, f: &FrameBufs, sessions: &mut Sessions) -> Result<()> {
        ensure!(sessions.batch == f.batch, "frame buffers for {} sessions, state for {}", f.batch, sessions.batch);
        self.launch_all(which, &f.plan, &f.ptrs, sessions)
    }

    pub fn run(&self, which: &GpuStack, input: &[f32], sessions: &mut Sessions) -> Result<Vec<f32>> {
        let b = sessions.batch;
        let steps = input.len() / (b * which.stack.in_channels);
        ensure!(steps * b * which.stack.in_channels == input.len() && steps > 0, "{} values are not whole steps for {b} sessions", input.len());
        let p = plan(&which.stack, steps, b)?;
        let s = &self.stream;
        let mut bufs: Vec<CudaSlice<f32>> = p.buffer_floats.iter().map(|n| s.alloc_zeros((*n).max(1))).collect::<Result<_, _>>()?;
        s.memcpy_htod(input, &mut bufs[p.input].slice_mut(0..input.len()))?;
        let ptrs: Vec<u64> = bufs.iter().map(|v| v.device_ptr(s).0).collect();
        self.launch_all(which, &p, &ptrs, sessions)?;
        let n = b * p.t_out * p.out_channels;
        Ok(s.clone_dtoh(&bufs[p.output].slice(0..n))?)
    }

    fn launch_all(&self, which: &GpuStack, p: &LaunchPlan, ptrs: &[u64], sessions: &mut Sessions) -> Result<()> {
        let s = &self.stream;
        let b = sessions.batch as i32;
        let dptr = |v: &CudaSlice<f32>| v.device_ptr(s).0;
        for l in &p.launches {
            let (x, y) = (ptrs[l.input], ptrs[l.output]);
            let t = l.t_in as i32;
            let ring = || -> Result<u64> { Ok(dptr(&sessions.state.get(&l.op).context("an attention launch has ring state")?[0])) };
            match &l.kind {
                Kind::Conv { shape, gelu, residual, rows, act } => {
                    let k = &self.kernels[&super::compile::conv_kernel(&self.target, *shape, *gelu, *residual, *rows, *act)?.name()];
                    let dev = &which.weights[&(l.op, l.part)];
                    let st = match l.layer.and_then(|i| sessions.state.get(&i)) {
                        Some(pair) if shape.context() > 0 => dptr(&pair[0]),
                        _ => dptr(&dev.b),
                    };
                    let scale = match l.scale {
                        Some(sc) => dptr(&which.weights[&(sc, Part::Main)].w),
                        None => dptr(&dev.b),
                    };
                    let (w, bias, to) = (dptr(&dev.w), dptr(&dev.b), l.t_out as i32);
                    let (gx, gy, gz) = k.line_grid(sessions.batch * l.t_out);
                    let mut lb = s.launch_builder(&self.fns[&k.name()]);
                    lb.arg(&x).arg(&st).arg(&w).arg(&bias).arg(&scale).arg(&y).arg(&b).arg(&t).arg(&to);
                    unsafe { lb.launch(LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() }) }?;
                }
                Kind::Depthwise { channels, taps } => {
                    let dev = &which.weights[&(l.op, Part::Main)];
                    let st = dptr(&sessions.state[&l.op][0]);
                    let scale = dptr(&which.weights[&(l.scale.context("a depthwise launch folds its scale")?, Part::Main)].w);
                    let (w, bias, c, kk) = (dptr(&dev.w), dptr(&dev.b), *channels as i32, *taps as i32);
                    let mut lb = s.launch_builder(&self.fns["line_dwconv"]);
                    lb.arg(&x).arg(&st).arg(&w).arg(&bias).arg(&scale).arg(&y).arg(&b).arg(&t).arg(&c).arg(&kk);
                    unsafe { lb.launch(rows(sessions.batch * l.t_out * channels)) }?;
                }
                Kind::DwUp { channels, stride } => {
                    let dev = &which.weights[&(l.op, Part::Main)];
                    let st = dptr(&sessions.state[&l.op][0]);
                    let (w, bias, c, ss) = (dptr(&dev.w), dptr(&dev.b), *channels as i32, *stride as i32);
                    let mut lb = s.launch_builder(&self.fns["line_dwup"]);
                    lb.arg(&x).arg(&st).arg(&w).arg(&bias).arg(&y).arg(&b).arg(&t).arg(&c).arg(&ss);
                    unsafe { lb.launch(rows(sessions.batch * l.t_in * stride * channels)) }?;
                }
                Kind::Rms { channels, eps } | Kind::Layer { channels, eps } => {
                    let dev = &which.weights[&(l.op, Part::Main)];
                    let (g, beta) = (dptr(&dev.w), dptr(&dev.b));
                    let (pp, c, e) = ((sessions.batch * l.t_in) as i32, *channels as i32, *eps);
                    let per = 256 / 32;
                    let cfg = LaunchConfig { grid_dim: ((sessions.batch * l.t_in).div_ceil(per) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    if matches!(l.kind, Kind::Rms { .. }) {
                        let mut lb = s.launch_builder(&self.fns["line_rms"]);
                        lb.arg(&x).arg(&y).arg(&g).arg(&pp).arg(&c).arg(&e);
                        unsafe { lb.launch(cfg) }?;
                    } else {
                        let mut lb = s.launch_builder(&self.fns["line_layernorm"]);
                        lb.arg(&x).arg(&y).arg(&g).arg(&beta).arg(&pp).arg(&c).arg(&e);
                        unsafe { lb.launch(cfg) }?;
                    }
                }
                Kind::Rope { heads, head_dim, context, theta } => {
                    let r = ring()?;
                    let (h, d, st) = (*heads as i32, *head_dim as i32, (1 + 2 * context * heads * head_dim) as i32);
                    let mut lb = s.launch_builder(&self.fns["line_rope"]);
                    lb.arg(&x).arg(&r).arg(&b).arg(&t).arg(&h).arg(&d).arg(&st).arg(theta);
                    unsafe { lb.launch(rows(sessions.batch * l.t_in * heads * head_dim)) }?;
                }
                Kind::Ring { heads, head_dim, context } => {
                    let r = ring()?;
                    let (h, d, w) = (*heads as i32, *head_dim as i32, *context as i32);
                    let scale = 1.0f32 / (*head_dim as f32).sqrt();
                    let mut lb = s.launch_builder(&self.fns["line_ring_attention"]);
                    lb.arg(&x).arg(&r).arg(&y).arg(&b).arg(&t).arg(&h).arg(&d).arg(&w).arg(&scale);
                    let smem = kir::conv::ring_attention_smem(*context, *head_dim);
                    unsafe { lb.launch(LaunchConfig { grid_dim: ((sessions.batch * l.t_in * heads) as u32, 1, 1), block_dim: (RING_THREADS, 1, 1), shared_mem_bytes: smem }) }?;
                }
                Kind::RingStore { channels, context } => {
                    let r = ring()?;
                    let (dm, w) = (*channels as i32, *context as i32);
                    let mut lb = s.launch_builder(&self.fns["line_ring_store"]);
                    lb.arg(&x).arg(&r).arg(&b).arg(&t).arg(&dm).arg(&w);
                    unsafe { lb.launch(rows(sessions.batch * l.t_in.min(*context) * channels)) }?;
                }
                Kind::Advance { context, channels } => {
                    let r = ring()?;
                    let st = (1 + 2 * context * channels) as i32;
                    let mut lb = s.launch_builder(&self.fns["line_advance"]);
                    lb.arg(&r).arg(&b).arg(&t).arg(&st);
                    unsafe { lb.launch(rows(sessions.batch)) }?;
                }
                Kind::Shift { channels, context } => {
                    let layer = l.layer.context("a shift has a layer")?;
                    let pair = sessions.state.get_mut(&layer).context("a shift has state")?;
                    let (old, new) = (dptr(&pair[0]), dptr(&pair[1]));
                    let (tp, c, ctx) = (l.t_pad as i32, *channels as i32, *context as i32);
                    let mut lb = s.launch_builder(&self.fns["line_shift"]);
                    lb.arg(&x).arg(&old).arg(&new).arg(&b).arg(&t).arg(&tp).arg(&c).arg(&ctx);
                    unsafe { lb.launch(rows(sessions.batch * context * channels)) }?;
                    if sessions.flip {
                        pair.swap(0, 1);
                    } else {
                        let [old, new] = pair;
                        s.memcpy_dtod(&*new, old)?;
                    }
                }
            }
        }
        Ok(())
    }
}
