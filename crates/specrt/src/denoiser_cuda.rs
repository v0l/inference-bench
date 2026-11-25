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
    joint: bool,
}

pub struct Arena {
    bufs: BTreeMap<&'static str, Buffer>,
    planes: CudaSlice<u16>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    seg_lo: CudaSlice<i32>,
    seg_hi: CudaSlice<i32>,
    shapes: Shapes,
    pub bytes: usize,
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
}

#[derive(Default)]
pub struct Trace {
    pub blocks: Vec<(String, Vec<f32>)>,
}

fn grid(n: usize, per: usize) -> u32 {
    n.div_ceil(per).max(1) as u32
}

impl Denoiser {
    pub fn load(stream: &Arc<CudaStream>, jit: &Jit, dir: &std::path::Path, shapes: Shapes) -> Result<Denoiser> {
        let graph = DenoiserGraph::embedded();
        let spec = DenoiserSpec::from_pipeline(dir)?;
        let plan = graph.plan()?;
        crate::denoiser::validate(graph, &spec, &plan)?;
        let hoisted = crate::denoiser::hoisted(graph, &spec, &plan)?;
        let ck = Checkpoint::open(&dir.join("transformer"))?;
        let target = crate::codegen::cuda_target(&stream.context())?;
        let head_dim = spec.size("head_dim")?;
        let vk = kir::vision::VisionKernel::new(&target, head_dim, true).map_err(anyhow::Error::msg)?.joint();
        let mut vision_names = kir::vision::KERNELS.to_vec();
        vision_names.extend_from_slice(kir::vision::JOINT_KERNELS);
        let vision_mod = jit.load(&Module { source: kir::cuda::emit_vision(&vk).map_err(anyhow::Error::msg)?, kernels: vision_names.clone(), smem: BTreeMap::new(), arch_features: false, kir_hash: vk.hash() })?;
        let mut vision = BTreeMap::new();
        for n in ["vis_mma", "vis_flash", "vis_axis_rope"] {
            vision.insert(n, vision_mod.function(n)?);
        }
        {
            use cudarc::driver::sys::CUfunction_attribute as A;
            vision["vis_mma"].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, GEMM_SMEM as i32)?;
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
        Ok(Denoiser { stream: stream.clone(), graph, spec, plan, hoisted, linears, norms, vision, compose, shapes, head_dim })
    }

    pub fn arena(&self) -> Result<Arena> {
        let s = &self.stream;
        let sh = &self.shapes;
        let mut bufs = BTreeMap::new();
        let mut bytes = 0usize;
        for d in self.graph.buffer.iter().chain(&self.graph.input).chain(&self.graph.output) {
            if d.int {
                continue;
            }
            let rows = match d.rows.as_str() {
                "joint" => sh.joint(),
                "image" => sh.image,
                "text" => sh.text,
                _ => 1,
            };
            let cols = self.spec.size(&d.cols)?;
            let n = rows * cols;
            let b = if d.planes {
                bytes += 4 * n;
                Buffer { f32: None, hi: Some(s.alloc_zeros::<u16>(n)?), lo: Some(s.alloc_zeros::<u16>(n)?), cols, joint: d.rows == "joint" }
            } else {
                bytes += 4 * n;
                Buffer { f32: Some(s.alloc_zeros::<f32>(n)?), hi: None, lo: None, cols, joint: d.rows == "joint" }
            };
            bufs.insert(crate::denoiser::intern_name(&d.name), b);
        }
        let h = self.spec.hidden();
        let hdp = kir::vision::head_pad(self.head_dim);
        let plane = (h / self.head_dim) * sh.joint() * hdp;
        bytes += 6 * plane * 2;
        let planes = s.alloc_zeros::<u16>(6 * plane)?;
        let ids = crate::denoiser_host::grid_ids(sh.text, sh.patch_h, sh.patch_w, self.spec.axes.len());
        let (cos, sin) = crate::denoiser_host::rope_tables(&self.spec.axes, &ids, self.spec.get("rope_theta")?);
        let seg_lo = vec![0i32; sh.joint()];
        let seg_hi = vec![sh.joint() as i32; sh.joint()];
        Ok(Arena { bufs, planes, cos: s.clone_htod(&cos)?, sin: s.clone_htod(&sin)?, seg_lo: s.clone_htod(&seg_lo)?, seg_hi: s.clone_htod(&seg_hi)?, shapes: sh.clone(), bytes })
    }

    fn span(&self, a: &Arena, name: &str, rows: Stream) -> (usize, usize) {
        let joint = a.bufs.get(name).is_some_and(|b| b.joint);
        let sh = &a.shapes;
        match (rows, joint) {
            (Stream::Text, _) => (0, sh.text),
            (Stream::Image, true) => (sh.text, sh.joint()),
            (Stream::Image, false) => (0, sh.image),
            (Stream::Joint, _) => (0, sh.joint()),
            (Stream::Seqs, _) => (0, 1),
        }
    }

    pub fn prepare(&self, a: &mut Arena, text: &[f32]) -> Result<()> {
        self.stream.memcpy_htod(text, a.bufs.get_mut("text").context("no text input")?.f32.as_mut().unwrap())?;
        for (op, hoist) in self.plan.head.iter().zip(&self.hoisted) {
            if *hoist {
                self.run(op, None, a)?;
            }
        }
        Ok(())
    }

    pub fn forward(&self, a: &mut Arena, latent: &[f32], sigma: f32, trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        self.stream.memcpy_htod(latent, a.bufs.get_mut("latent").context("no latent input")?.f32.as_mut().unwrap())?;
        self.stream.memcpy_htod(&[sigma], a.bufs.get_mut("timestep").context("no timestep input")?.f32.as_mut().unwrap())?;
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
        for op in &self.plan.tail {
            self.run(op, None, a)?;
        }
        self.read(a, "velocity")
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

    fn run(&self, op: &Op, block: Option<usize>, a: &mut Arena) -> Result<()> {
        let Op::Dit(d) = op else { bail!("{} is not a denoiser op", op.label()) };
        let s = &self.stream;
        let h = self.spec.hidden();
        match *d {
            DitOp::Split { from, into, rows, .. } => {
                let (r0, r1) = self.span(a, from, rows);
                let (o0, _) = self.span(a, into, rows);
                let src = self.ptr_f32(a, from, r0)?;
                let (hi, lo) = self.ptr_planes(a, into, o0)?;
                let f = self.compose_fn(d)?;
                let mut b = s.launch_builder(f);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::row_grid(a.bufs[from].cols, r1 - r0)) }?;
            }
            DitOp::Copy { from, into, rows } => {
                let (r0, r1) = self.span(a, from, rows);
                let (o0, _) = self.span(a, into, rows);
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
                unsafe { b.launch(Self::row_grid(a.bufs[into].cols, 1)) }?;
            }
            DitOp::Patchify { from, into } => {
                let src = self.ptr_f32(a, from, 0)?;
                let (hi, lo) = self.ptr_planes(a, into, 0)?;
                let n = a.shapes.image * self.spec.size("in_channels")?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::row_grid(n, 1)) }?;
            }
            DitOp::Unpatchify { from, into } => {
                let (src, dst) = (self.ptr_f32(a, from, 0)?, self.ptr_f32(a, into, 0)?);
                let n = a.shapes.image * self.spec.size("in_channels")?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&src).arg(&dst);
                unsafe { b.launch(Self::row_grid(n, 1)) }?;
            }
            DitOp::Linear { weight, input, output, rows, add } => {
                let key = match block {
                    Some(i) => weight.replace("{i}", &i.to_string()),
                    None => weight.to_string(),
                };
                let l = self.linears.get(&key).with_context(|| format!("linear {key} is not loaded"))?;
                let (r0, r1) = self.span(a, input, rows);
                let (o0, _) = self.span(a, output, rows);
                let m = r1 - r0;
                let in_cols = a.bufs[input].cols;
                ensure!(in_cols == l.k || weight.contains('['), "{key}: input has {in_cols} columns, the weight {}", l.k);
                let (hi, lo) = self.ptr_planes(a, input, r0)?;
                let out = self.ptr_f32(a, output, o0)?;
                let out_cols = a.bufs[output].cols;
                ensure!(out_cols == l.n, "{key}: output has {out_cols} columns, the weight {}", l.n);
                let null = 0u64;
                let resid = if add { out } else { 0u64 };
                let (m32, n32, k32, act) = (m as i32, l.n as i32, l.k as i32, 0i32);
                let mut b = s.launch_builder(&self.vision["vis_mma"]);
                b.arg(&hi).arg(&lo).arg(&l.w).arg(&null).arg(&resid).arg(&out).arg(&null).arg(&null).arg(&m32).arg(&n32).arg(&k32).arg(&act);
                unsafe { b.launch(LaunchConfig { grid_dim: (grid(l.n, GEMM_BN as usize), grid(m, GEMM_BM as usize), 1), block_dim: (GEMM_THREADS, 1, 1), shared_mem_bytes: GEMM_SMEM }) }?;
            }
            DitOp::Modulate { from, into, rows, set, shift, scale } => {
                let (r0, r1) = self.span(a, from, rows);
                let x = self.ptr_f32(a, from, r0)?;
                let base = self.ptr_f32(a, set, 0)?;
                let (sh, sc) = (base + (shift as usize * h * 4) as u64, base + (scale as usize * h * 4) as u64);
                let (hi, lo) = self.ptr_planes(a, into, r0)?;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&x).arg(&sh).arg(&sc).arg(&hi).arg(&lo);
                unsafe { b.launch(LaunchConfig { grid_dim: ((r1 - r0) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            DitOp::GatedAdd { from, into, rows, set, gate } => {
                let (r0, r1) = self.span(a, into, rows);
                let x = self.ptr_f32(a, into, r0)?;
                let br = self.ptr_f32(a, from, r0)?;
                let g = self.ptr_f32(a, set, 0)? + (gate as usize * h * 4) as u64;
                let mut b = s.launch_builder(self.compose_fn(d)?);
                b.arg(&x).arg(&br).arg(&g);
                unsafe { b.launch(Self::row_grid(h, r1 - r0)) }?;
            }
            DitOp::Glu { from, into, rows, .. } => {
                let (r0, r1) = self.span(a, from, rows);
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
                let heads = h / self.head_dim;
                let (t32, h32, hd32, stride, text_rows, eps) = (joint as i32, heads as i32, self.head_dim as i32, a.bufs[from].cols as i32, a.shapes.text as i32, self.spec.get("norm_eps")? as f32);
                let mut b = s.launch_builder(&self.vision["vis_axis_rope"]);
                b.arg(&src).arg(&qt).arg(&kt).arg(&qi).arg(&ki).arg(&a.cos).arg(&a.sin).arg(&mut a.planes).arg(&t32).arg(&h32).arg(&hd32).arg(&stride).arg(&text_rows).arg(&eps);
                unsafe { b.launch(LaunchConfig { grid_dim: (joint as u32, heads as u32, 1), block_dim: (kir::vision::head_pad(self.head_dim).max(32) as u32, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            DitOp::JointAttention { into } => {
                let joint = a.shapes.joint();
                let heads = h / self.head_dim;
                let (hi, lo) = self.ptr_planes(a, into, 0)?;
                let (t32, h32, hd32) = (joint as i32, heads as i32, self.head_dim as i32);
                let scale_log2 = (1.0 / (self.head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
                let smem = kir::vision::VisionKernel { target: crate::codegen::default_target(), head_dim: self.head_dim, precise: true, tiled: false, joint: true }.flash_smem();
                let mut b = s.launch_builder(&self.vision["vis_flash"]);
                b.arg(&a.planes).arg(&a.seg_lo).arg(&a.seg_hi).arg(&hi).arg(&lo).arg(&t32).arg(&h32).arg(&hd32).arg(&scale_log2);
                unsafe { b.launch(LaunchConfig { grid_dim: (grid(joint, FLASH_BQ as usize), heads as u32, 1), block_dim: (FLASH_THREADS, 1, 1), shared_mem_bytes: smem }) }?;
            }
        }
        Ok(())
    }
}
