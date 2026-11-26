use super::compile::{Kind, LaunchPlan, plan};
use super::graph::{CodecOp, Stack};
use super::host::{HostCodec, HostStack, Param};
use crate::codegen::Module;
use crate::jit::Jit;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use kir::conv::{ConvKernel, LINE_ROW_KERNELS, THREADS};
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
    weights: BTreeMap<usize, Dev>,
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
    for (i, (op, p)) in h.stack.ops.iter().zip(&h.params).enumerate() {
        let dev = match (op, p) {
            (_, Param::Conv(c)) => Dev { w: stream.clone_htod(&c.w)?, b: stream.clone_htod(&c.b)? },
            (_, Param::Linear { w, b }) => Dev { w: stream.clone_htod(w)?, b: stream.clone_htod(b)? },
            (CodecOp::RmsNorm { .. } | CodecOp::ScaleAdd { .. }, Param::Vec(g)) => Dev { w: stream.clone_htod(g)?, b: stream.alloc_zeros(1)? },
            _ => continue,
        };
        weights.insert(i, dev);
    }
    Ok(GpuStack { stack: h.stack.clone(), weights })
}

fn rows(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
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
            all.extend(plan(&host.plan.encoder, steps * host.plan.frame_samples, batch)?.kernels(&target)?);
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
        for k in &all {
            use cudarc::driver::sys::CUfunction_attribute as A;
            fns[&k.name()].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes() as i32)?;
        }
        let stream = stream.clone();
        let encoder = upload(&stream, &host.encoder)?;
        let decoder = upload(&stream, &host.decoder)?;
        let kernels = all.into_iter().map(|k| (k.name(), k)).collect();
        Ok(CudaCodec { stream, target, fns, kernels, encoder, decoder, frame_samples: host.plan.frame_samples, latent_dim: host.plan.latent_dim })
    }

    pub fn sessions(&self, which: &GpuStack, batch: usize) -> Result<Sessions> {
        let mut state = BTreeMap::new();
        for (i, op) in which.stack.ops.iter().enumerate() {
            let cols = match op {
                CodecOp::Conv { k, stride, .. } => k - stride,
                CodecOp::ConvUp { .. } => 1,
                _ => 0,
            };
            if cols > 0 {
                let n = batch * cols * op_cin(op);
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
            let (t, to) = (l.t_in as i32, l.t_out as i32);
            match &l.kind {
                Kind::Conv { shape, gelu, residual, rows } => {
                    let k = &self.kernels[&super::compile::conv_kernel(&self.target, *shape, *gelu, *residual, *rows)?.name()];
                    let dev = &which.weights[&l.op];
                    let st = match l.layer.and_then(|i| sessions.state.get(&i)) {
                        Some(pair) => dptr(&pair[0]),
                        None => dptr(&dev.b),
                    };
                    let scale = match l.scale {
                        Some(sc) => dptr(&which.weights[&sc].w),
                        None => dptr(&dev.b),
                    };
                    let (w, bias) = (dptr(&dev.w), dptr(&dev.b));
                    let (gx, gy, gz) = k.line_grid(sessions.batch * l.t_out);
                    let mut lb = s.launch_builder(&self.fns[&k.name()]);
                    lb.arg(&x).arg(&st).arg(&w).arg(&bias).arg(&scale).arg(&y).arg(&b).arg(&t).arg(&to);
                    unsafe { lb.launch(LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() }) }?;
                }
                Kind::Depthwise { channels, taps } => {
                    let dev = &which.weights[&l.op];
                    let st = dptr(&sessions.state[&l.op][0]);
                    let scale = dptr(&which.weights[&l.scale.context("a depthwise launch folds its scale")?].w);
                    let (w, bias, c, kk) = (dptr(&dev.w), dptr(&dev.b), *channels as i32, *taps as i32);
                    let mut lb = s.launch_builder(&self.fns["line_dwconv"]);
                    lb.arg(&x).arg(&st).arg(&w).arg(&bias).arg(&scale).arg(&y).arg(&b).arg(&t).arg(&c).arg(&kk);
                    unsafe { lb.launch(rows(sessions.batch * l.t_out * channels)) }?;
                }
                Kind::Rms { channels, eps } => {
                    let g = dptr(&which.weights[&l.op].w);
                    let (pp, c, e) = ((sessions.batch * l.t_in) as i32, *channels as i32, *eps);
                    let per = 256 / 32;
                    let mut lb = s.launch_builder(&self.fns["line_rms"]);
                    lb.arg(&x).arg(&y).arg(&g).arg(&pp).arg(&c).arg(&e);
                    unsafe { lb.launch(LaunchConfig { grid_dim: ((sessions.batch * l.t_in).div_ceil(per) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
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

fn op_cin(op: &CodecOp) -> usize {
    match op {
        CodecOp::Conv { cin, .. } | CodecOp::ConvUp { cin, .. } | CodecOp::Linear { cin, .. } => *cin,
        CodecOp::RmsNorm { channels, .. } | CodecOp::ScaleAdd { channels, .. } => *channels,
    }
}
