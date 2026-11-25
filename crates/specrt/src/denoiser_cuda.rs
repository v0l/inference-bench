use crate::codegen::Module;
use crate::denoiser::{DenoiserGraph, DenoiserSpec, Shapes, Signer, kernel_key, weight_names};
use crate::jit::Jit;
use crate::plan::{DitOp, Op, Plan, Stream};
use crate::weights::Checkpoint;
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use kir::vision::{FLASH_BQ, FLASH_THREADS, GEMM_BM, GEMM_BN, GEMM_SMEM, GEMM_THREADS};
use std::collections::BTreeMap;
use std::sync::Arc;

struct DeviceLinear {
    w: CudaSlice<u16>,
    n: usize,
    k: usize,
}

struct Buffer {
    f32: Option<CudaSlice<f32>>,
    hi: Option<CudaSlice<u16>>,
    lo: Option<CudaSlice<u16>>,
    cols: usize,
    rows: String,
}

pub struct Arena {
    bufs: BTreeMap<&'static str, Buffer>,
    planes: CudaSlice<u16>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    seg_lo: CudaSlice<i32>,
    seg_hi: CudaSlice<i32>,
    shapes: Shapes,
    seqs: usize,
    pub bytes: usize,
}

impl Arena {
    pub fn seqs(&self) -> usize {
        self.seqs
    }
}

pub struct Denoiser {
    stream: Arc<CudaStream>,
    graph: &'static DenoiserGraph,
    pub spec: DenoiserSpec,
    pub plan: Plan,
    hoisted: Vec<bool>,
    linears: BTreeMap<String, DeviceLinear>,
    norms: BTreeMap<String, CudaSlice<f32>>,
    vision: BTreeMap<&'static str, CudaFunction>,
    compose: BTreeMap<String, CudaFunction>,
    shapes: Shapes,
    head_dim: usize,
    pub fused_step: bool,
    pub weight_bytes: usize,
}

#[derive(Default)]
pub struct Trace {
    pub blocks: Vec<(String, Vec<f32>)>,
}

fn grid(n: usize, per: usize) -> u32 {
    n.div_ceil(per).max(1) as u32
}

pub const MAX_ROWS: usize = 65535;

fn rows_of(rows: &str, sh: &Shapes, seqs: usize) -> usize {
    match rows {
        "joint" => seqs * sh.joint(),
        "image" => seqs * sh.image,
        "text" => seqs * sh.text,
        "latent" => sh.image,
        "seqs" => seqs,
        _ => 1,
    }
}

pub fn arena_bytes(graph: &DenoiserGraph, spec: &DenoiserSpec, sh: &Shapes, seqs: usize) -> Result<usize> {
    let head_dim = spec.size("head_dim")?;
    let mut bytes = 0usize;
    for d in graph.buffer.iter().chain(&graph.input).chain(&graph.output).filter(|d| !d.int) {
        bytes += 4 * rows_of(&d.rows, sh, seqs) * spec.size(&d.cols)?;
    }
    let plane = (spec.hidden() / head_dim) * seqs * sh.joint() * kir::vision::head_pad(head_dim);
    Ok(bytes + 12 * plane + 8 * seqs * sh.joint() * (head_dim + 1))
}

pub fn weight_bytes(dir: &std::path::Path) -> Result<u64> {
    let ck = Checkpoint::open(&dir.join("transformer"))?;
    let mut bytes = 0u64;
    for name in ck.names_under("") {
        let n: usize = ck.shape_of(&name)?.iter().product();
        bytes += 2 * n as u64;
    }
    Ok(bytes)
}

impl Denoiser {
    pub fn load(stream: &Arc<CudaStream>, jit: &Jit, dir: &std::path::Path, shapes: Shapes) -> Result<Denoiser> {
        let rule = crate::sampler::Pipeline::from_dir(dir)?.step_rule(false)?;
        Denoiser::load_with(stream, jit, dir, shapes, &rule, false)
    }

    pub fn load_with(stream: &Arc<CudaStream>, jit: &Jit, dir: &std::path::Path, shapes: Shapes, rule: &kir::expr::Expr, guided: bool) -> Result<Denoiser> {
        let graph = DenoiserGraph::embedded();
        let spec = DenoiserSpec::from_pipeline(dir)?;
        let plan = graph.sampling_plan(rule, guided)?;
        crate::denoiser::validate(graph, &spec, &plan)?;
        let hoisted = crate::denoiser::hoisted(graph, &spec, &plan)?;
        let ck = Checkpoint::open(&dir.join("transformer"))?;
        let target = crate::codegen::cuda_target(&stream.context())?;
        let head_dim = spec.size("head_dim")?;
        let vk = kir::vision::VisionKernel::new(&target, head_dim, true).map_err(anyhow::Error::msg)?.joint();
        let mut vision_names = kir::vision::KERNELS.to_vec();
        vision_names.extend_from_slice(kir::vision::JOINT_KERNELS);
        let fused_step = !guided;
        let source = if fused_step {
            vision_names.push("vis_mma_step");
            kir::cuda::emit_vision_step(&vk, rule).map_err(anyhow::Error::msg)?
        } else {
            kir::cuda::emit_vision(&vk).map_err(anyhow::Error::msg)?
        };
        let hash = format!("{}{}", vk.hash(), if fused_step { rule.to_string() } else { String::new() });
        let vision_mod = jit.load(&Module { source, kernels: vision_names.clone(), smem: BTreeMap::new(), arch_features: false, kir_hash: kir::fnv_hash(&hash) })?;
        let mut vision = BTreeMap::new();
        for n in ["vis_mma", "vis_flash", "vis_axis_rope"].into_iter().chain(fused_step.then_some("vis_mma_step")) {
            vision.insert(n, vision_mod.function(n)?);
        }
        {
            use cudarc::driver::sys::CUfunction_attribute as A;
            for (n, f) in &vision {
                if n.starts_with("vis_mma") {
                    f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, GEMM_SMEM as i32)?;
                }
            }
            vision["vis_flash"].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, vk.flash_smem() as i32)?;
        }
        let kernels = crate::denoiser::described_kernels(graph, &spec, &plan, &shapes, &target)?;
        let names: Vec<&'static str> = kernels.iter().map(|k| k.name).collect();
        let source = kir::cuda::emit_module(&kernels).map_err(anyhow::Error::msg)?;
        let loaded = jit.load(&Module { source, kernels: names.clone(), smem: BTreeMap::new(), arch_features: false, kir_hash: kir::module_hash(&kernels) })?;
        let mut compose = BTreeMap::new();
        for n in names {
            compose.insert(n.to_string(), loaded.function(n)?);
        }
        let signer = Signer { graph, spec: &spec };
        let mut linears = BTreeMap::new();
        let mut norms = BTreeMap::new();
        let mut weight_bytes = 0usize;
        let mut each = |ops: &[Op], block: Option<usize>| -> Result<()> {
            for op in ops {
                let Op::Dit(d) = op else { continue };
                match *d {
                    DitOp::Linear { weight, input, output, .. } => {
                        let key = match block {
                            Some(i) => weight.replace("{i}", &i.to_string()),
                            None => weight.to_string(),
                        };
                        if linears.contains_key(&key) {
                            continue;
                        }
                        let (n, k) = signer.linear_shape(weight, input, output)?;
                        let mut w: Vec<f32> = Vec::with_capacity(n * k);
                        for (name, range) in weight_names(weight, block) {
                            let shape = ck.shape_of(&name)?;
                            let full = ck.f32_values(&name, &shape)?;
                            match range {
                                Some((a, b)) => {
                                    let (a, b) = (spec.size(&a)?, spec.size(&b)?);
                                    let stride = shape[1];
                                    for r in 0..shape[0] {
                                        w.extend_from_slice(&full[r * stride + a..r * stride + b]);
                                    }
                                }
                                None => w.extend(full),
                            }
                        }
                        ensure!(w.len() == n * k, "{key}: {} values, expected {n} x {k}", w.len());
                        ensure!(k.is_multiple_of(8), "{key}: k = {k} is not a multiple of 8");
                        let bits: Vec<u16> = w.iter().map(|&v| crate::weights::f32_to_bf16(v)).collect();
                        weight_bytes += 2 * bits.len();
                        linears.insert(key, DeviceLinear { w: stream.clone_htod(&bits)?, n, k });
                    }
                    DitOp::AxisRope { text, image, .. } => {
                        for base in [text, image] {
                            for qk in ["q", "k"] {
                                let name = format!("{}_{qk}.weight", base.replace("{i}", &block.unwrap_or(0).to_string()));
                                if norms.contains_key(&name) {
                                    continue;
                                }
                                let shape = ck.shape_of(&name)?;
                                let v: Vec<f32> = ck.bf16(&name, &shape)?.iter().map(|&b| crate::weights::bf16_to_f32(b)).collect();
                                weight_bytes += 4 * v.len();
                                norms.insert(name, stream.clone_htod(&v)?);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        };
        each(&plan.head, None)?;
        for (b, g) in plan.blocks.iter().zip(&graph.block) {
            for i in 0..spec.size(&g.count)? {
                each(&b.ops, Some(i))?;
            }
        }
        each(&plan.tail, None)?;
        stream.synchronize()?;
        Ok(Denoiser { stream: stream.clone(), graph, spec, plan, hoisted, linears, norms, vision, compose, shapes, head_dim, fused_step, weight_bytes })
    }

    fn rows_of(&self, rows: &str, sh: &Shapes, seqs: usize) -> usize {
        rows_of(rows, sh, seqs)
    }

    pub fn arena_bytes(&self, seqs: usize) -> Result<usize> {
        arena_bytes(self.graph, &self.spec, &self.shapes, seqs)
    }

    pub fn arena(&self) -> Result<Arena> {
        self.arena_for(1)
    }

    pub fn arena_for(&self, seqs: usize) -> Result<Arena> {
        ensure!((1..=2).contains(&seqs), "a forward runs one sequence, or two with guidance, not {seqs}");
        let s = &self.stream;
        let sh = &self.shapes;
        let mut bufs = BTreeMap::new();
        for d in self.graph.buffer.iter().chain(&self.graph.input).chain(&self.graph.output) {
            if d.int {
                continue;
            }
            let rows = self.rows_of(&d.rows, sh, seqs);
            let cols = self.spec.size(&d.cols)?;
            let n = rows * cols;
            let b = if d.planes {
                Buffer { f32: None, hi: Some(s.alloc_zeros::<u16>(n)?), lo: Some(s.alloc_zeros::<u16>(n)?), cols, rows: d.rows.clone() }
            } else {
                Buffer { f32: Some(s.alloc_zeros::<f32>(n)?), hi: None, lo: None, cols, rows: d.rows.clone() }
            };
            bufs.insert(crate::denoiser::intern_name(&d.name), b);
        }
        let h = self.spec.hidden();
        let hdp = kir::vision::head_pad(self.head_dim);
        let joint = sh.joint();
        let plane = (h / self.head_dim) * seqs * joint * hdp;
        let planes = s.alloc_zeros::<u16>(6 * plane)?;
        let ids = crate::denoiser_host::grid_ids(sh.text, sh.patch_h, sh.patch_w, self.spec.axes.len());
        let (cos, sin) = crate::denoiser_host::rope_tables(&self.spec.axes, &ids, self.spec.get("rope_theta")?);
        let (cos, sin) = (cos.repeat(seqs), sin.repeat(seqs));
        let seg_lo: Vec<i32> = (0..seqs * joint).map(|r| (r / joint * joint) as i32).collect();
        let seg_hi: Vec<i32> = seg_lo.iter().map(|l| l + joint as i32).collect();
        Ok(Arena { bufs, planes, cos: s.clone_htod(&cos)?, sin: s.clone_htod(&sin)?, seg_lo: s.clone_htod(&seg_lo)?, seg_hi: s.clone_htod(&seg_hi)?, shapes: sh.clone(), seqs, bytes: self.arena_bytes(seqs)? })
    }

    fn span(&self, a: &Arena, name: &str, rows: Stream, seq: usize) -> (usize, usize) {
        let decl = a.bufs.get(name).map_or("joint", |b| b.rows.as_str());
        let sh = &a.shapes;
        let (j, i, t) = (sh.joint(), sh.image, sh.text);
        match (rows, decl) {
            (Stream::Seqs, _) | (_, "seqs") => (seq, seq + 1),
            (Stream::Latent, _) | (_, "latent") => (0, i),
            (Stream::Text, "joint") => (seq * j, seq * j + t),
            (Stream::Image, "joint") => (seq * j + t, (seq + 1) * j),
            (Stream::Joint, "joint") => (seq * j, (seq + 1) * j),
            (_, "text") => (seq * t, (seq + 1) * t),
            (_, _) => (seq * i, (seq + 1) * i),
        }
    }

    pub fn prepare(&self, a: &mut Arena, texts: &[&[f32]]) -> Result<()> {
        ensure!(texts.len() == a.seqs, "{} text embeddings for {} sequences", texts.len(), a.seqs);
        let joined: Vec<f32> = texts.concat();
        self.stream.memcpy_htod(&joined, a.bufs.get_mut("text").context("no text input")?.f32.as_mut().unwrap())?;
        for (op, hoist) in self.plan.head.iter().zip(&self.hoisted) {
            if *hoist {
                self.run(op, None, a)?;
            }
        }
        Ok(())
    }

    pub fn set_latent(&self, a: &mut Arena, latent: &[f32]) -> Result<()> {
        let name = self.graph.latent()?;
        Ok(self.stream.memcpy_htod(latent, a.bufs.get_mut(name).with_context(|| format!("no {name} input"))?.f32.as_mut().unwrap())?)
    }

    pub fn latent(&self, a: &Arena) -> Result<Vec<f32>> {
        self.read(a, self.graph.latent()?)
    }

    fn network(&self, a: &mut Arena, sigma: f32, trace: Option<&mut Trace>) -> Result<()> {
        self.stream.memcpy_htod(&vec![sigma; a.seqs], a.bufs.get_mut("timestep").context("no timestep input")?.f32.as_mut().unwrap())?;
        for (op, hoist) in self.plan.head.iter().zip(&self.hoisted) {
            if !*hoist {
                self.run(op, None, a)?;
            }
        }
        let mut trace = trace;
        for (b, g) in self.plan.blocks.iter().zip(&self.graph.block) {
            for i in 0..self.spec.size(&g.count)? {
                for op in &b.ops {
                    self.run(op, Some(i), a)?;
                }
                if let Some(t) = trace.as_deref_mut() {
                    t.blocks.push((format!("{}{i}", g.name), self.read(a, "x")?));
                }
            }
        }
        Ok(())
    }

    fn tail(&self, a: &mut Arena, step: bool) -> Result<()> {
        for op in &self.plan.tail {
            match op {
                Op::Dit(DitOp::Step { .. }) if !step => {}
                Op::Dit(DitOp::Linear { output, .. }) if step && self.fused_step && Some(*output) == self.step_input() => self.run_fused(op, a)?,
                Op::Dit(DitOp::Step { .. }) if self.fused_step => {}
                _ => self.run(op, None, a)?,
            }
        }
        Ok(())
    }

    fn step_input(&self) -> Option<&'static str> {
        self.plan.tail.iter().find_map(|o| match o {
            Op::Dit(DitOp::Step { from, .. }) => Some(*from),
            _ => None,
        })
    }

    pub fn forward(&self, a: &mut Arena, latent: &[f32], sigma: f32, trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        self.set_latent(a, latent)?;
        self.network(a, sigma, trace)?;
        self.tail(a, false)?;
        self.read(a, "velocity")
    }

    pub fn step(&self, a: &mut Arena, s0: f32, s1: f32, guidance: f32) -> Result<()> {
        self.stream.memcpy_htod(&[s0, s1, guidance], a.bufs.get_mut("step").context("no step input")?.f32.as_mut().unwrap())?;
        self.network(a, s0, None)?;
        self.tail(a, true)
    }

    pub fn step_unfused(&self, a: &mut Arena, s0: f32, s1: f32, guidance: f32) -> Result<()> {
        self.stream.memcpy_htod(&[s0, s1, guidance], a.bufs.get_mut("step").context("no step input")?.f32.as_mut().unwrap())?;
        self.network(a, s0, None)?;
        self.tail(a, false)?;
        let step = self.plan.tail.iter().find(|o| matches!(o, Op::Dit(DitOp::Step { .. }))).context("the plan has no step")?;
        self.run(step, None, a)
    }

    pub fn read(&self, a: &Arena, name: &str) -> Result<Vec<f32>> {
        let b = a.bufs.get(name).with_context(|| format!("no buffer {name}"))?;
        let d = b.f32.as_ref().with_context(|| format!("{name} is held as planes"))?;
        let mut out = vec![0f32; d.len()];
        self.stream.memcpy_dtoh(d, &mut out)?;
        Ok(out)
    }

    fn compose_fn(&self, d: &DitOp) -> Result<&CudaFunction> {
        let (name, _, _) = kernel_key(d, self.graph, &self.spec)?.context("not a composed op")?;
        self.compose.get(&name).with_context(|| format!("kernel {name} is not loaded"))
    }

    fn ptr_f32(&self, a: &Arena, name: &str, row: usize) -> Result<u64> {
        let b = a.bufs.get(name).with_context(|| format!("no buffer {name}"))?;
        let d = b.f32.as_ref().with_context(|| format!("{name} has no f32 form"))?;
        Ok(d.device_ptr(&self.stream).0 + (row * b.cols * 4) as u64)
    }

    fn ptr_planes(&self, a: &Arena, name: &str, row: usize) -> Result<(u64, u64)> {
        let b = a.bufs.get(name).with_context(|| format!("no buffer {name}"))?;
        let hi = b.hi.as_ref().with_context(|| format!("{name} has no planes"))?;
        let lo = b.lo.as_ref().unwrap();
        let off = (row * b.cols * 2) as u64;
        Ok((hi.device_ptr(&self.stream).0 + off, lo.device_ptr(&self.stream).0 + off))
    }

    fn row_grid(cols: usize, rows: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (grid(cols, 256), rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    fn linear(&self, f: &CudaFunction, l: &DeviceLinear, hi: u64, lo: u64, resid: u64, out: u64, m: usize) -> Result<()> {
        let null = 0u64;
        let (m32, n32, k32, act) = (m as i32, l.n as i32, l.k as i32, 0i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&hi).arg(&lo).arg(&l.w).arg(&null).arg(&resid).arg(&out).arg(&null).arg(&null).arg(&m32).arg(&n32).arg(&k32).arg(&act);
        unsafe { b.launch(LaunchConfig { grid_dim: (grid(l.n, GEMM_BN as usize), grid(m, GEMM_BM as usize), 1), block_dim: (GEMM_THREADS, 1, 1), shared_mem_bytes: GEMM_SMEM }) }?;
        Ok(())
    }

    fn run_fused(&self, op: &Op, a: &Arena) -> Result<()> {
        let Op::Dit(DitOp::Linear { weight, input, rows, add: false, .. }) = *op else { bail!("{} is not the step's producer", op.label()) };
        ensure!(a.seqs == 1, "the step epilogue runs without guidance");
        let l = self.linears.get(weight).with_context(|| format!("linear {weight} is not loaded"))?;
        let (r0, r1) = self.span(a, input, rows, 0);
        let (hi, lo) = self.ptr_planes(a, input, r0)?;
        let latent = self.ptr_f32(a, self.graph.latent()?, 0)?;
        let sigmas = self.ptr_f32(a, "step", 0)?;
        self.linear(&self.vision["vis_mma_step"], l, hi, lo, sigmas, latent, r1 - r0)
    }

    fn run(&self, op: &Op, block: Option<usize>, a: &mut Arena) -> Result<()> {
        self.run_inner(op, block, a).with_context(|| format!("{} at {}x{} latent patches", op.label(), a.shapes.patch_h, a.shapes.patch_w))
    }

    fn run_inner(&self, op: &Op, block: Option<usize>, a: &mut Arena) -> Result<()> {
        let Op::Dit(d) = op else { bail!("{} is not a denoiser op", op.label()) };
        if let DitOp::JointAttention { .. } | DitOp::AxisRope { .. } | DitOp::Step { .. } | DitOp::TimestepEmbed { .. } = d {
            return self.run_seq(d, block, a, 0);
        }
        let shared_input = |name: &str| a.bufs.get(name).is_some_and(|b| b.rows == "latent");
        let once = match *d {
            DitOp::Split { from, .. } | DitOp::Linear { input: from, .. } => shared_input(from),
            _ => false,
        };
        for seq in 0..a.seqs {
            if once && seq > 0 {
                self.copy_seq(d, a, seq)?;
                continue;
            }
            self.run_seq(d, block, a, seq)?;
        }
        Ok(())
    }

    fn copy_seq(&self, d: &DitOp, a: &Arena, seq: usize) -> Result<()> {
        let (DitOp::Split { into, rows, .. } | DitOp::Linear { output: into, rows, .. }) = *d else { bail!("{} has no shared input", d.label()) };
        let b = &a.bufs[into];
        let (r0, r1) = self.span(a, into, rows, 0);
        let (d0, _) = self.span(a, into, rows, seq);
        let n = (r1 - r0) * b.cols;
        let elem = if b.f32.is_some() { 4 } else { 2 };
        let copy = |base: u64| -> Result<()> {
            let (src, dst) = (base + (r0 * b.cols * elem) as u64, base + (d0 * b.cols * elem) as u64);
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, n * elem, self.stream.cu_stream()) }?;
            Ok(())
        };
        match (&b.f32, &b.hi, &b.lo) {
            (Some(f), _, _) => copy(f.device_ptr(&self.stream).0)?,
            (None, Some(hi), Some(lo)) => {
                copy(hi.device_ptr(&self.stream).0)?;
                copy(lo.device_ptr(&self.stream).0)?;
            }
            _ => bail!("{into} has no storage"),
        }
        Ok(())
    }

    fn run_seq(&self, d: &DitOp, block: Option<usize>, a: &mut Arena, seq: usize) -> Result<()> {
        let s = &self.stream;
        let h = self.spec.hidden();
        match *d {
            DitOp::Split { from, into, rows, .. } => {
                let (r0, r1) = self.span(a, from, rows, seq);
                let (o0, _) = self.span(a, into, rows, seq);
                let src = self.ptr_f32(a, from, r0)?;
                let (hi, lo) = self.ptr_planes(a, into, o0)?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::row_grid(a.bufs[from].cols, r1 - r0)) }?;
            }
            DitOp::Copy { from, into, rows } => {
                let (r0, r1) = self.span(a, from, rows, seq);
                let (o0, _) = self.span(a, into, rows, seq);
                let (src, dst) = (self.ptr_f32(a, from, r0)?, self.ptr_f32(a, into, o0)?);
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&src).arg(&dst);
                unsafe { b.launch(Self::row_grid(a.bufs[from].cols, r1 - r0)) }?;
            }
            DitOp::TimestepEmbed { from, into } => {
                let src = self.ptr_f32(a, from, 0)?;
                let (hi, lo) = self.ptr_planes(a, into, 0)?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::row_grid(a.bufs[into].cols, a.seqs)) }?;
            }
            DitOp::Patchify { from, into } | DitOp::Unpatchify { from, into, .. } => {
                let n = a.shapes.image * self.spec.size("in_channels")?;
                let src = self.ptr_f32(a, from, 0)?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                let (hi, lo) = if matches!(d, DitOp::Patchify { .. }) { self.ptr_planes(a, into, 0)? } else { (self.ptr_f32(a, into, 0)?, 0) };
                b.arg(&src).arg(&hi);
                if matches!(d, DitOp::Patchify { .. }) {
                    b.arg(&lo);
                }
                unsafe { b.launch(Self::row_grid(n, 1)) }?;
            }
            DitOp::Linear { weight, input, output, rows, add, .. } => {
                let key = match block {
                    Some(i) => weight.replace("{i}", &i.to_string()),
                    None => weight.to_string(),
                };
                let l = self.linears.get(&key).with_context(|| format!("linear {key} is not loaded"))?;
                let (r0, r1) = self.span(a, input, rows, seq);
                let (o0, _) = self.span(a, output, rows, seq);
                let in_cols = a.bufs[input].cols;
                ensure!(in_cols == l.k || weight.contains('['), "{key}: input has {in_cols} columns, the weight {}", l.k);
                let (hi, lo) = self.ptr_planes(a, input, r0)?;
                let out = self.ptr_f32(a, output, o0)?;
                ensure!(a.bufs[output].cols == l.n, "{key}: output has {} columns, the weight {}", a.bufs[output].cols, l.n);
                self.linear(&self.vision["vis_mma"], l, hi, lo, if add { out } else { 0 }, out, r1 - r0)?;
            }
            DitOp::Modulate { from, into, rows, set, shift, scale } => {
                let (r0, r1) = self.span(a, from, rows, seq);
                let x = self.ptr_f32(a, from, r0)?;
                let base = self.ptr_f32(a, set, seq)?;
                let (sh, sc) = (base + (shift as usize * h * 4) as u64, base + (scale as usize * h * 4) as u64);
                let (hi, lo) = self.ptr_planes(a, into, r0)?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&x).arg(&sh).arg(&sc).arg(&hi).arg(&lo);
                unsafe { b.launch(LaunchConfig { grid_dim: ((r1 - r0) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            DitOp::GatedAdd { from, into, rows, set, gate } => {
                let (r0, r1) = self.span(a, into, rows, seq);
                let x = self.ptr_f32(a, into, r0)?;
                let br = self.ptr_f32(a, from, r0)?;
                let g = self.ptr_f32(a, set, seq)? + (gate as usize * h * 4) as u64;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&x).arg(&br).arg(&g);
                unsafe { b.launch(Self::row_grid(h, r1 - r0)) }?;
            }
            DitOp::Glu { from, into, rows, .. } => {
                let (r0, r1) = self.span(a, from, rows, seq);
                let src = self.ptr_f32(a, from, r0)?;
                let (hi, lo) = self.ptr_planes(a, into, r0)?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::row_grid(a.bufs[into].cols, r1 - r0)) }?;
            }
            DitOp::AxisRope { from, text, image } => {
                let i = block.unwrap_or(0).to_string();
                let w = |base: &str, qk: &str| -> Result<u64> {
                    let name = format!("{}_{qk}.weight", base.replace("{i}", &i));
                    Ok(self.norms.get(&name).with_context(|| format!("norm {name} is not loaded"))?.device_ptr(s).0)
                };
                let (qt, kt, qi, ki) = (w(text, "q")?, w(text, "k")?, w(image, "q")?, w(image, "k")?);
                let src = self.ptr_f32(a, from, 0)?;
                let joint = a.shapes.joint();
                let rows = a.seqs * joint;
                let heads = h / self.head_dim;
                let (t32, h32, hd32, stride, text_rows, eps, seq_rows) = (rows as i32, heads as i32, self.head_dim as i32, a.bufs[from].cols as i32, a.shapes.text as i32, self.spec.get("norm_eps")? as f32, joint as i32);
                let mut b = s.launch_builder(&self.vision["vis_axis_rope"]);
                b.arg(&src).arg(&qt).arg(&kt).arg(&qi).arg(&ki).arg(&a.cos).arg(&a.sin).arg(&mut a.planes).arg(&t32).arg(&h32).arg(&hd32).arg(&stride).arg(&text_rows).arg(&eps).arg(&seq_rows);
                unsafe { b.launch(LaunchConfig { grid_dim: (rows as u32, heads as u32, 1), block_dim: (kir::vision::head_pad(self.head_dim).max(32) as u32, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            DitOp::JointAttention { into, .. } => {
                let rows = a.seqs * a.shapes.joint();
                let heads = h / self.head_dim;
                let (hi, lo) = self.ptr_planes(a, into, 0)?;
                let (t32, h32, hd32) = (rows as i32, heads as i32, self.head_dim as i32);
                let scale_log2 = (1.0 / (self.head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
                let smem = kir::vision::VisionKernel { target: crate::codegen::default_target(), head_dim: self.head_dim, precise: true, tiled: false, joint: true }.flash_smem();
                let mut b = s.launch_builder(&self.vision["vis_flash"]);
                b.arg(&a.planes).arg(&a.seg_lo).arg(&a.seg_hi).arg(&hi).arg(&lo).arg(&t32).arg(&h32).arg(&hd32).arg(&scale_log2);
                unsafe { b.launch(LaunchConfig { grid_dim: (grid(rows, FLASH_BQ as usize), heads as u32, 1), block_dim: (FLASH_THREADS, 1, 1), shared_mem_bytes: smem }) }?;
            }
            DitOp::Step { from, into, guided, .. } => {
                ensure!(guided == (a.seqs == 2), "a {} step on an arena of {} sequences", if guided { "guided" } else { "plain" }, a.seqs);
                let x = self.ptr_f32(a, into, 0)?;
                let c = self.ptr_f32(a, from, 0)?;
                let rows = a.shapes.image;
                let st = self.ptr_f32(a, "step", 0)?;
                let (s0, s1, g) = (st, st + 4, st + 8);
                let u = if guided { self.ptr_f32(a, from, rows)? } else { 0 };
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&x);
                if guided {
                    b.arg(&c).arg(&u).arg(&s0).arg(&s1).arg(&g);
                } else {
                    b.arg(&c).arg(&s0).arg(&s1);
                }
                unsafe { b.launch(Self::row_grid(a.bufs[into].cols, rows)) }?;
            }
            other => bail!("{} has no kernel in the joint denoiser runtime", other.label()),
        }
        Ok(())
    }
}
