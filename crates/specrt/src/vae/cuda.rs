use super::compile::{Compiled, Dims, Pro, Step, compile, conv_shape};
use super::graph::{Act, VaeOp, VaePlan};
use super::host::{Grid, HostVae};
use crate::codegen::Module;
use crate::jit::{Jit, Loaded};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg};
use kir::conv::{ConvKernel, Epilogue, Prologue, ROW_THREADS, THREADS};
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn kernels(c: &Compiled, target: &kir::Target) -> Result<BTreeMap<usize, ConvKernel>> {
    let mut out = BTreeMap::new();
    for st in &c.schedule.steps {
        let (op, up, apply, residual, stats) = match st {
            Step::Conv { op, prologue, residual, stats, .. } => {
                let apply = match prologue {
                    Pro::Apply { op: n, .. } => match c.plan.ops[*n] {
                        VaeOp::GroupNorm { groups, act, .. } => Some((groups, act == Act::Silu)),
                        _ => None,
                    },
                    _ => None,
                };
                let g = stats.map(|(n, _)| match c.plan.ops[n] {
                    VaeOp::GroupNorm { groups, .. } => groups,
                    _ => 0,
                });
                (*op, *prologue == Pro::Upsample, apply, residual.is_some(), g)
            }
            Step::TimeUp { op, .. } => (*op, false, None, false, None),
            _ => continue,
        };
        let shape = conv_shape(&c.plan.ops[op], up).context("a conv step has a conv op")?;
        let prologue = apply.map_or(Prologue::None, |(groups, silu)| Prologue::GroupApply { groups, silu });
        let k = ConvKernel::new(target, shape, prologue, Epilogue { residual, stats, gelu: false }).map_err(anyhow::Error::msg)?;
        out.insert(op * 4 + usize::from(up), k);
    }
    Ok(out)
}

pub fn module(all: &[ConvKernel]) -> Result<Module> {
    let source = kir::cuda::emit_vae(all).map_err(anyhow::Error::msg)?;
    let mut names: Vec<&'static str> = kir::conv::ROW_KERNELS.to_vec();
    for k in all {
        names.push(Box::leak(k.name().into_boxed_str()));
        names.push(Box::leak(format!("{}_clamp", k.name()).into_boxed_str()));
    }
    names.sort_unstable();
    names.dedup();
    let hash = kir::fold_hashes(all.iter().map(ConvKernel::hash));
    Ok(Module { source, kernels: names, smem: BTreeMap::new(), arch_features: false, kir_hash: hash })
}

struct Dev {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

pub struct CudaVae {
    stream: Arc<CudaStream>,
    loaded: Loaded,
    target: kir::Target,
    host: HostVae,
    weights: BTreeMap<usize, Dev>,
    fns: BTreeMap<String, CudaFunction>,
    pub fuse: bool,
}

pub struct FrameCache {
    hist: BTreeMap<usize, (CudaSlice<f32>, usize)>,
    started: BTreeMap<usize, bool>,
}

fn grid1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(ROW_THREADS).max(1), 1, 1), block_dim: (ROW_THREADS, 1, 1), shared_mem_bytes: 0 }
}

fn strided(n: usize, sms: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(ROW_THREADS).clamp(1, 8 * sms), 1, 1), block_dim: (ROW_THREADS, 1, 1), shared_mem_bytes: 0 }
}

impl CudaVae {
    pub fn load(ctx: &Arc<CudaContext>, jit: &Jit, vae_dir: &std::path::Path, plans: &[(Dims, bool)], fuse: bool) -> Result<CudaVae> {
        let host = HostVae::open(vae_dir)?;
        let target = crate::codegen::cuda_target(ctx)?;
        let mut all: Vec<ConvKernel> = Vec::new();
        for (d, first) in plans {
            let c = compile(&host.plan, *d, &target, fuse, *first)?;
            all.extend(kernels(&c, &target)?.into_values());
        }
        all.sort_by_key(ConvKernel::name);
        all.dedup_by_key(|k| k.name());
        let m = module(&all)?;
        let loaded = jit.load(&m)?;
        let stream = ctx.default_stream();
        let mut fns = BTreeMap::new();
        for name in &m.kernels {
            let f = loaded.function(name)?;
            fns.insert(name.to_string(), f);
        }
        for k in &all {
            use cudarc::driver::sys::CUfunction_attribute as A;
            for n in [k.name(), format!("{}_clamp", k.name())] {
                fns[&n].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes() as i32)?;
            }
        }
        {
            use cudarc::driver::sys::CUfunction_attribute as A;
            fns["vae_attention"].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, target.smem_limit as i32)?;
        }
        let mut weights = BTreeMap::new();
        for (i, op) in host.plan.ops.iter().enumerate() {
            if let Some(c) = host.conv_weights(i)
                && matches!(op, VaeOp::Conv { .. } | VaeOp::TimeUp { .. })
            {
                weights.insert(i, Dev { w: stream.clone_htod(&c.taps_major())?, b: stream.clone_htod(&c.b)? });
            }
            if let Some((g, b)) = host.norm_weights(i) {
                let beta = if b.is_empty() { vec![0.0; g.len()] } else { b.to_vec() };
                weights.insert(i, Dev { w: stream.clone_htod(g)?, b: stream.clone_htod(&beta)? });
            }
        }
        Ok(CudaVae { stream, loaded, target, host, weights, fns, fuse })
    }

    pub fn plan(&self) -> &VaePlan {
        &self.host.plan
    }

    pub fn cache_hit(&self) -> bool {
        self.loaded.cache_hit
    }

    pub fn new_stream(&self) -> FrameCache {
        FrameCache { hist: BTreeMap::new(), started: BTreeMap::new() }
    }

    pub fn decode(&self, latent: &Grid) -> Result<Grid> {
        let mut s = self.new_stream();
        if !self.host.plan.temporal {
            return self.run(latent, &mut s, true);
        }
        let mut out: Option<Grid> = None;
        for f in 0..latent.t {
            let y = self.run(&latent.frames(f, 1), &mut s, f == 0)?;
            match &mut out {
                Some(o) => o.append_frames(&y),
                None => out = Some(y),
            }
        }
        out.context("an empty latent")
    }

    pub fn arena_of(&self, d: Dims, first: bool) -> Result<u64> {
        Ok(compile(&self.host.plan, d, &self.target, self.fuse, first)?.arena.size)
    }

    pub fn run(&self, latent: &Grid, stream: &mut FrameCache, first: bool) -> Result<Grid> {
        let d = Dims { t: latent.t, h: latent.h, w: latent.w };
        let c = compile(&self.host.plan, d, &self.target, self.fuse, first)?;
        let ks = kernels(&c, &self.target)?;
        let s = &self.stream;
        let words = (c.arena.size as usize).div_ceil(4).max(1);
        let mut arena: CudaSlice<f32> = s.alloc_zeros(words)?;
        let count = c.lowered.count;
        let dims = |v: usize| -> Dims {
            match c.schedule.values[v].rows {
                crate::signature::Rows::Grid(l) => c.shapes.levels[l],
                _ => Dims { t: 1, h: 1, w: 1 },
            }
        };
        let off = |v: usize| -> Result<(usize, usize)> {
            let (o, n) = c.arena.offsets.get(&v).copied().with_context(|| format!("value {v} has no arena slot"))?;
            Ok((o as usize / 4, n as usize / 4))
        };
        let elems = |v: usize| count.rows(c.schedule.values[v].rows) * c.schedule.values[v].cols;
        let (io, _) = off(c.schedule.input)?;
        s.memcpy_htod(&latent.data, &mut arena.slice_mut(io..io + latent.data.len()))?;
        let null_f: CudaSlice<f32> = s.alloc_zeros(1)?;
        let null_d: CudaSlice<f64> = s.alloc_zeros(1)?;
        let ptr = |v: &CudaSlice<f32>, from: usize| -> u64 {
            use cudarc::driver::DevicePtr;
            v.device_ptr(s).0 + 4 * from as u64
        };
        let dptr = |v: &CudaSlice<f32>, from: usize| -> u64 { ptr(v, from) };
        let sms = self.target.sm_count;
        for st in &c.schedule.steps {
            match st {
                Step::Conv { op, input, prologue, residual, stats, clamp, output } => {
                    let k = &ks[&(op * 4 + usize::from(*prologue == Pro::Upsample))];
                    let od = dims(*output);
                    let (xi, _) = off(*input)?;
                    let (yo, _) = off(*output)?;
                    let dev = &self.weights[op];
                    let (sin, gamma, beta, eps, cnt) = match prologue {
                        Pro::Apply { op: n, stats: sv } => {
                            let VaeOp::GroupNorm { eps, groups, channels, .. } = self.host.plan.ops[*n] else { unreachable!() };
                            let (so, _) = off(*sv)?;
                            let nd = dims(*input);
                            (dptr(&arena, so), ptr(&self.weights[n].w, 0), ptr(&self.weights[n].b, 0), eps, (nd.pixels() * channels / groups) as f64)
                        }
                        _ => (ptr(&null_f, 0), ptr(&null_f, 0), ptr(&null_f, 0), 0.0f32, 1.0f64),
                    };
                    let resid = match residual {
                        Some(r) => ptr(&arena, off(*r)?.0),
                        None => ptr(&null_f, 0),
                    };
                    let sout = match stats {
                        Some((_, sv)) => {
                            let (so, n) = off(*sv)?;
                            self.zero(&arena, so, n)?;
                            dptr(&arena, so)
                        }
                        None => {
                            use cudarc::driver::DevicePtr;
                            null_d.device_ptr(s).0
                        }
                    };
                    let (lo, hi) = match clamp {
                        Some(cl) => match self.host.plan.ops[*cl] {
                            VaeOp::Clamp { lo, hi, .. } => (lo, hi),
                            _ => unreachable!(),
                        },
                        None => (0.0, 0.0),
                    };
                    let (hist, hframes) = self.history(stream, *op, k);
                    let name = if clamp.is_some() { format!("{}_clamp", k.name()) } else { k.name() };
                    let (t, h, w) = (od.t as i32, od.h as i32, od.w as i32);
                    let (gx, gy, gz) = k.grid(od.t, od.h, od.w);
                    let cfg = LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() };
                    let (xp, wp, bp, yp) = (ptr(&arena, xi), ptr(&dev.w, 0), ptr(&dev.b, 0), ptr(&arena, yo));
                    let mut b = s.launch_builder(&self.fns[&name]);
                    b.arg(&xp).arg(&hist).arg(&hframes).arg(&wp).arg(&bp).arg(&resid).arg(&yp).arg(&sin).arg(&gamma).arg(&beta).arg(&eps).arg(&cnt).arg(&sout).arg(&lo).arg(&hi).arg(&t).arg(&h).arg(&w);
                    unsafe { b.launch(cfg) }?;
                    self.keep_history(stream, *op, k, &arena, xi, dims(*input))?;
                }
                Step::TimeUp { op, input, output } => {
                    let id = dims(*input);
                    let (xi, _) = off(*input)?;
                    let n = elems(*input);
                    let (yo, _) = off(*output)?;
                    if !stream.started.get(op).copied().unwrap_or(false) {
                        stream.started.insert(*op, true);
                        let (xp, yp, nn, lo, hi) = (ptr(&arena, xi), ptr(&arena, yo), n as i32, f32::NEG_INFINITY, f32::INFINITY);
                        let mut b = s.launch_builder(&self.fns["vae_clamp"]);
                        b.arg(&xp).arg(&yp).arg(&nn).arg(&lo).arg(&hi);
                        unsafe { b.launch(grid1(n)) }?;
                        continue;
                    }
                    let k = &ks[&(op * 4)];
                    let (hist, hframes) = self.history(stream, *op, k);
                    let dev = &self.weights[op];
                    let (t, h, w) = (id.t as i32, id.h as i32, id.w as i32);
                    let (gx, gy, gz) = k.grid(id.t, id.h, id.w);
                    let cfg = LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() };
                    let (xp, wp, bp, yp, z) = (ptr(&arena, xi), ptr(&dev.w, 0), ptr(&dev.b, 0), ptr(&arena, yo), ptr(&null_f, 0));
                    let nd = {
                        use cudarc::driver::DevicePtr;
                        null_d.device_ptr(s).0
                    };
                    let (zero_f, one) = (0.0f32, 1.0f64);
                    let mut b = s.launch_builder(&self.fns[&k.name()]);
                    b.arg(&xp).arg(&hist).arg(&hframes).arg(&wp).arg(&bp).arg(&z).arg(&yp).arg(&z).arg(&z).arg(&z).arg(&zero_f).arg(&one).arg(&nd).arg(&zero_f).arg(&zero_f).arg(&t).arg(&h).arg(&w);
                    unsafe { b.launch(cfg) }?;
                    self.keep_history(stream, *op, k, &arena, xi, id)?;
                }
                Step::Stats { op, input, output } => {
                    let VaeOp::GroupNorm { groups, channels, .. } = self.host.plan.ops[*op] else { unreachable!() };
                    let (xi, _) = off(*input)?;
                    let (so, n) = off(*output)?;
                    self.zero(&arena, so, n)?;
                    let (xp, sp, p, cc, g) = (ptr(&arena, xi), dptr(&arena, so), dims(*input).pixels() as i32, channels as i32, groups as i32);
                    let mut b = s.launch_builder(&self.fns["vae_group_stats"]);
                    b.arg(&xp).arg(&sp).arg(&p).arg(&cc).arg(&g);
                    unsafe { b.launch(strided(elems(*input), sms)) }?;
                }
                Step::Apply { op, input, stats: sv, output } => {
                    let VaeOp::GroupNorm { groups, channels, eps, act, .. } = self.host.plan.ops[*op] else { unreachable!() };
                    let (xi, _) = off(*input)?;
                    let (so, _) = off(*sv)?;
                    let (yo, _) = off(*output)?;
                    let px = dims(*input).pixels();
                    let (xp, yp, sp, gp, bp) = (ptr(&arena, xi), ptr(&arena, yo), dptr(&arena, so), ptr(&self.weights[op].w, 0), ptr(&self.weights[op].b, 0));
                    let (p, cc, g, cnt, silu) = (px as i32, channels as i32, groups as i32, (px * channels / groups) as f64, i32::from(act == Act::Silu));
                    let mut b = s.launch_builder(&self.fns["vae_group_apply"]);
                    b.arg(&xp).arg(&yp).arg(&sp).arg(&gp).arg(&bp).arg(&p).arg(&cc).arg(&g).arg(&eps).arg(&cnt).arg(&silu);
                    unsafe { b.launch(strided(elems(*input), sms)) }?;
                }
                Step::Rms { op, input, output } => {
                    let VaeOp::RmsNorm { channels, act, .. } = self.host.plan.ops[*op] else { unreachable!() };
                    let (xi, _) = off(*input)?;
                    let (yo, _) = off(*output)?;
                    let px = dims(*input).pixels();
                    let (xp, yp, gp, p, cc, silu) = (ptr(&arena, xi), ptr(&arena, yo), ptr(&self.weights[op].w, 0), px as i32, channels as i32, i32::from(act == Act::Silu));
                    let rows = (ROW_THREADS / 32) as usize;
                    let mut b = s.launch_builder(&self.fns["vae_rms"]);
                    b.arg(&xp).arg(&yp).arg(&gp).arg(&p).arg(&cc).arg(&silu);
                    unsafe { b.launch(LaunchConfig { grid_dim: (px.div_ceil(rows) as u32, 1, 1), block_dim: (ROW_THREADS, 1, 1), shared_mem_bytes: 0 }) }?;
                }
                Step::Upsample { input, output, .. } => {
                    let od = dims(*output);
                    let (xi, _) = off(*input)?;
                    let (yo, _) = off(*output)?;
                    let cc = c.schedule.values[*output].cols as i32;
                    let (xp, yp, t, h, w) = (ptr(&arena, xi), ptr(&arena, yo), od.t as i32, od.h as i32, od.w as i32);
                    let mut b = s.launch_builder(&self.fns["vae_upsample"]);
                    b.arg(&xp).arg(&yp).arg(&t).arg(&h).arg(&w).arg(&cc);
                    unsafe { b.launch(strided(elems(*output), sms)) }?;
                }
                Step::Attention { op, input, output } => {
                    let VaeOp::Attention { channels, .. } = self.host.plan.ops[*op] else { unreachable!() };
                    let id = dims(*input);
                    let keys = id.h * id.w;
                    ensure!(kir::conv::attention_fits(&self.target, keys), "vae attention over {keys} keys needs {} bytes of shared memory", kir::conv::attention_smem(keys));
                    let (xi, _) = off(*input)?;
                    let (yo, _) = off(*output)?;
                    let (xp, yp, sq, cc, scale) = (ptr(&arena, xi), ptr(&arena, yo), keys as i32, channels as i32, 1.0f32 / (channels as f32).sqrt());
                    let mut b = s.launch_builder(&self.fns["vae_attention"]);
                    b.arg(&xp).arg(&yp).arg(&sq).arg(&cc).arg(&scale);
                    unsafe { b.launch(LaunchConfig { grid_dim: (keys as u32, id.t as u32, 1), block_dim: (kir::conv::ATTENTION_THREADS, 1, 1), shared_mem_bytes: kir::conv::attention_smem(keys) as u32 }) }?;
                }
                Step::Add { a, b: bv, output, .. } => {
                    let (ao, _) = off(*a)?;
                    let (bo, _) = off(*bv)?;
                    let (yo, _) = off(*output)?;
                    let n = elems(*output);
                    let (ap, bp, yp, nn) = (ptr(&arena, ao), ptr(&arena, bo), ptr(&arena, yo), n as i32);
                    let mut b = s.launch_builder(&self.fns["vae_add"]);
                    b.arg(&ap).arg(&bp).arg(&yp).arg(&nn);
                    unsafe { b.launch(grid1(n)) }?;
                }
                Step::Clamp { op, input, output } => {
                    let VaeOp::Clamp { lo, hi, .. } = self.host.plan.ops[*op] else { unreachable!() };
                    let (xi, _) = off(*input)?;
                    let (yo, _) = off(*output)?;
                    let n = elems(*output);
                    let (xp, yp, nn) = (ptr(&arena, xi), ptr(&arena, yo), n as i32);
                    let mut b = s.launch_builder(&self.fns["vae_clamp"]);
                    b.arg(&xp).arg(&yp).arg(&nn).arg(&lo).arg(&hi);
                    unsafe { b.launch(grid1(n)) }?;
                }
            }
        }
        let (oo, _) = off(c.schedule.output)?;
        let od = dims(c.schedule.output);
        let cc = c.schedule.values[c.schedule.output].cols;
        let n = od.pixels() * cc;
        let data = s.clone_dtoh(&arena.slice(oo..oo + n))?;
        Ok(Grid { c: cc, t: od.t, h: od.h, w: od.w, data })
    }

    fn zero(&self, arena: &CudaSlice<f32>, from: usize, words: usize) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let p = arena.device_ptr(&self.stream).0 + 4 * from as u64;
        let n = (words / 2) as i32;
        let mut b = self.stream.launch_builder(&self.fns["vae_zero"]);
        b.arg(&p).arg(&n);
        unsafe { b.launch(grid1(words / 2)) }?;
        Ok(())
    }

    fn history(&self, stream: &FrameCache, op: usize, k: &ConvKernel) -> (u64, i32) {
        use cudarc::driver::DevicePtr;
        if k.shape.kt == 1 {
            return (0, 0);
        }
        match stream.hist.get(&op) {
            Some((buf, frames)) => (buf.device_ptr(&self.stream).0, *frames as i32),
            None => (0, 0),
        }
    }

    fn keep_history(&self, stream: &mut FrameCache, op: usize, k: &ConvKernel, arena: &CudaSlice<f32>, from: usize, d: Dims) -> Result<()> {
        if k.shape.kt == 1 {
            return Ok(());
        }
        let keep = k.shape.kt - 1;
        let frame = d.h * d.w * k.shape.cin;
        let prev = stream.hist.remove(&op);
        let (old_frames, old) = match &prev {
            Some((b, f)) => (*f, Some(b)),
            None => (0, None),
        };
        let total = old_frames + d.t;
        let start = total.saturating_sub(keep);
        let frames = total - start;
        let mut buf: CudaSlice<f32> = self.stream.alloc_zeros((frames * frame).max(1))?;
        for f in 0..frames {
            let src = start + f;
            let mut dst = unsafe_view(&mut buf, f * frame, frame);
            if src < old_frames {
                let o = old.unwrap();
                self.stream.memcpy_dtod(&o.slice(src * frame..(src + 1) * frame), &mut dst)?;
            } else {
                let at = from + (src - old_frames) * frame;
                self.stream.memcpy_dtod(&arena.slice(at..at + frame), &mut dst)?;
            }
        }
        stream.hist.insert(op, (buf, frames));
        Ok(())
    }
}

fn unsafe_view(buf: &mut CudaSlice<f32>, from: usize, n: usize) -> CudaViewMut<'_, f32> {
    buf.slice_mut(from..from + n)
}

pub fn conv_once(ctx: &Arc<CudaContext>, jit: &Jit, x: &Grid, c: &super::host::ConvWeights, up: bool, interleave: bool) -> Result<Grid> {
    let target = crate::codegen::cuda_target(ctx)?;
    let shape = kir::conv::ConvShape::grid(c.cin, c.cout, c.k, c.kt, up, interleave);
    let k = ConvKernel::new(&target, shape, Prologue::None, Epilogue { residual: false, stats: None, gelu: false }).map_err(anyhow::Error::msg)?;
    let m = module(std::slice::from_ref(&k))?;
    let loaded = jit.load(&m)?;
    let f = loaded.function(&k.name())?;
    {
        use cudarc::driver::sys::CUfunction_attribute as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes() as i32)?;
    }
    let s = ctx.default_stream();
    let (h, w) = if up { (2 * x.h, 2 * x.w) } else { (x.h, x.w) };
    let out_t = if interleave { 2 * x.t } else { x.t };
    let out_c = if interleave { c.cout / 2 } else { c.cout };
    let xd = s.clone_htod(&x.data)?;
    let wd = s.clone_htod(&c.taps_major())?;
    let bd = s.clone_htod(&c.b)?;
    let mut yd: CudaSlice<f32> = s.alloc_zeros(out_t * h * w * out_c)?;
    let zf: CudaSlice<f32> = s.alloc_zeros(1)?;
    let zd: CudaSlice<f64> = s.alloc_zeros(1)?;
    let (frames, t, hh, ww, eps, cnt, lo, hi) = (0i32, x.t as i32, h as i32, w as i32, 0.0f32, 1.0f64, 0.0f32, 0.0f32);
    let (gx, gy, gz) = k.grid(x.t, h, w);
    let mut b = s.launch_builder(&f);
    b.arg(&xd).arg(&zf).arg(&frames).arg(&wd).arg(&bd).arg(&zf).arg(&mut yd).arg(&zd).arg(&zf).arg(&zf).arg(&eps).arg(&cnt).arg(&zd).arg(&lo).arg(&hi).arg(&t).arg(&hh).arg(&ww);
    unsafe { b.launch(LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() }) }?;
    let data = s.clone_dtoh(&yd)?;
    Ok(Grid { c: out_c, t: out_t, h, w, data })
}
