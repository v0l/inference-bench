use crate::codegen::Module;
use crate::denoiser::{DenoiserGraph, kernel_key, map_description, map_params};
use crate::flow_head::{FlowHead, Level, load_weights};
use crate::jit::Jit;
use crate::plan::{DitOp, Op};
use crate::weights::Checkpoint;
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use kir::gemv::{BA_ROWS_MAX_TOKENS, BA_ROWS_THREADS, Gemv, GemvKernel};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

pub const STORE: &str = "acc";
pub const GATED: &str = "old + g * acc";

fn dit(op: &Op) -> Result<&DitOp> {
    match op {
        Op::Dit(d) => Ok(d),
        other => bail!("{} is not a flow head op", other.label()),
    }
}

pub fn rows_name(n: usize, k: usize, epilogue: &str) -> String {
    format!("head_rows_{n}_{k}_{}", if epilogue == STORE { "store" } else { "gated" })
}

pub struct Kernels {
    pub compose: Vec<kir::Kernel>,
    pub rows: Vec<GemvKernel>,
}

impl Kernels {
    pub fn source(&self) -> Result<kir::KirSource> {
        Ok(kir::KirSource::join([&kir::cuda::emit_module(&self.compose).map_err(anyhow::Error::msg)?, &kir::gemv::emit_gemvs(&self.rows).map_err(anyhow::Error::msg)?], "\n"))
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.compose.iter().map(|k| k.name).chain(self.rows.iter().map(|k| crate::denoiser::intern_name(&k.name))).collect()
    }

    pub fn hash(&self) -> String {
        kir::fold_hashes([kir::module_hash(&self.compose), kir::gemv::gemv_hash(&self.rows)])
    }
}

fn all_ops(head: &FlowHead) -> impl Iterator<Item = &Op> {
    head.start.iter().chain(&head.plan.head).chain(head.plan.blocks.iter().flat_map(|b| &b.ops)).chain(&head.plan.tail).chain(&head.finish)
}

pub fn kernels(head: &FlowHead, target: &kir::Target, fused: &[BTreeSet<usize>]) -> Result<Kernels> {
    let registry = crate::kir_kernel::Registry::active();
    let graph: &DenoiserGraph = head.graph;
    let base = head.spec.kir_shape();
    let signer = head.signer();
    let mut compose: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut rows: BTreeMap<String, GemvKernel> = BTreeMap::new();
    for op in all_ops(head) {
        let d = dit(op)?;
        if let DitOp::Linear { weight, input, output, .. } = *d {
            let (n, k) = signer.linear_shape(weight, input, output)?;
            let name = rows_name(n, k, STORE);
            if !rows.contains_key(&name) {
                rows.insert(name.clone(), GemvKernel::new(target, &name, Gemv::Rows { n, k, epilogue: STORE.into() }).map_err(anyhow::Error::msg)?);
            }
            continue;
        }
        let Some((name, desc, extra)) = kernel_key(d, graph, &head.spec)? else { bail!("{} has no kernel", d.label()) };
        if compose.contains_key(&name) {
            continue;
        }
        let mut shape = base.clone();
        for (k, v) in &extra {
            shape = shape.num(k, *v);
        }
        let k = if desc == "dit_map" {
            kir::desc::Description::parse(&map_description(d, graph, &head.spec)?).map_err(anyhow::Error::msg)?.build(target, &shape).map_err(anyhow::Error::msg)?
        } else {
            registry.get(&desc).with_context(|| format!("no kernel description {desc}"))?.build(target, &shape).map_err(anyhow::Error::msg)?
        };
        compose.insert(name.clone(), k.renamed(&name));
    }
    for (b, f) in head.plan.blocks.iter().zip(fused) {
        for &i in f {
            let DitOp::Linear { weight, input, output, .. } = *dit(&b.ops[i])? else { bail!("a fused gated add follows a linear") };
            let (n, k) = signer.linear_shape(weight, input, output)?;
            let name = rows_name(n, k, GATED);
            if !rows.contains_key(&name) {
                rows.insert(name.clone(), GemvKernel::new(target, &name, Gemv::Rows { n, k, epilogue: GATED.into() }).map_err(anyhow::Error::msg)?);
            }
        }
    }
    Ok(Kernels { compose: compose.into_values().collect(), rows: rows.into_values().collect() })
}

struct Buf {
    f32: Option<CudaSlice<f32>>,
    hi: Option<CudaSlice<u16>>,
    lo: Option<CudaSlice<u16>>,
    cols: usize,
}

pub struct CudaHead {
    pub head: FlowHead,
    stream: Arc<CudaStream>,
    fns: BTreeMap<String, CudaFunction>,
    linears: BTreeMap<String, (CudaSlice<u16>, usize, usize)>,
    vectors: BTreeMap<String, CudaSlice<f32>>,
    fused: Vec<BTreeSet<usize>>,
    levels: Vec<Level>,
    layers: Vec<usize>,
    pub weight_bytes: usize,
}

pub struct CudaSession {
    rows: usize,
    bufs: BTreeMap<&'static str, Buf>,
    guidance: Vec<f32>,
    pub launches: usize,
}

impl CudaHead {
    pub fn load(stream: &Arc<CudaStream>, jit: &Jit, head: FlowHead, dir: &Path, fuse: bool) -> Result<CudaHead> {
        let target = crate::codegen::cuda_target(&stream.context())?;
        let fused = if fuse { head.gated_epilogues(1, &target)? } else { vec![BTreeSet::new(); head.plan.blocks.len()] };
        let ks = kernels(&head, &target, &fused)?;
        let names = ks.names();
        let loaded = jit.load(&Module { source: ks.source()?, kernels: names.clone(), smem: BTreeMap::new(), arch_features: false, kir_hash: ks.hash() })?;
        let mut fns = BTreeMap::new();
        for n in names {
            fns.insert(n.to_string(), loaded.function(n)?);
        }
        let ck = Checkpoint::open(dir)?;
        let mut linears = BTreeMap::new();
        let mut vectors = BTreeMap::new();
        let weight_bytes = std::cell::Cell::new(0usize);
        load_weights(&head, &ck, |key, w, n, k| {
            let bits: Vec<u16> = w.iter().map(|&v| crate::weights::f32_to_bf16(v)).collect();
            weight_bytes.set(weight_bytes.get() + 2 * bits.len());
            linears.insert(key, (stream.clone_htod(&bits)?, n, k));
            Ok(())
        }, |key, v| {
            weight_bytes.set(weight_bytes.get() + 4 * v.len());
            vectors.insert(key, stream.clone_htod(&v)?);
            Ok(())
        })?;
        let weight_bytes = weight_bytes.get();
        let levels = head.levels()?;
        let layers = head.layers()?;
        Ok(CudaHead { head, stream: stream.clone(), fns, linears, vectors, fused, levels, layers, weight_bytes })
    }

    pub fn session(&self, guidance: &[f32]) -> Result<CudaSession> {
        let rows = guidance.len();
        ensure!(rows > 0, "a head batch has at least one session");
        let mut bufs = BTreeMap::new();
        for d in self.head.graph.buffer.iter().chain(&self.head.graph.input).chain(&self.head.graph.output) {
            let cols = self.head.spec.size(&d.cols)?;
            let n = rows * cols;
            let b = if d.planes { Buf { f32: None, hi: Some(self.stream.alloc_zeros(n)?), lo: Some(self.stream.alloc_zeros(n)?), cols } } else { Buf { f32: Some(self.stream.alloc_zeros(n)?), hi: None, lo: None, cols } };
            bufs.insert(crate::denoiser::intern_name(&d.name), b);
        }
        let mut s = CudaSession { rows, bufs, guidance: Vec::new(), launches: 0 };
        self.set_guidance(&mut s, guidance)?;
        Ok(s)
    }

    fn upload(&self, s: &mut CudaSession, name: &str, v: &[f32]) -> Result<()> {
        let b = s.bufs.get_mut(name).with_context(|| format!("no buffer {name}"))?;
        Ok(self.stream.memcpy_htod(v, b.f32.as_mut().context("an upload target is f32")?)?)
    }

    pub fn set_guidance(&self, s: &mut CudaSession, guidance: &[f32]) -> Result<()> {
        ensure!(guidance.len() == s.rows, "{} guidance values for {} sessions", guidance.len(), s.rows);
        s.guidance = guidance.to_vec();
        for (name, v) in self.head.scalar_values(0, guidance)? {
            self.upload(s, name, &v)?;
        }
        s.launches = 0;
        for (op, l) in self.head.plan.head.iter().zip(&self.levels) {
            if *l == Level::Session {
                self.run(dit(op)?, None, false, s)?;
            }
        }
        Ok(())
    }

    pub fn frame(&self, s: &mut CudaSession, hidden: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        s.launches = 0;
        self.upload(s, "hidden", hidden)?;
        self.upload(s, "noise", noise)?;
        for op in &self.head.start {
            self.run(dit(op)?, None, false, s)?;
        }
        for (op, l) in self.head.plan.head.iter().zip(&self.levels) {
            if *l == Level::Frame {
                self.run(dit(op)?, None, false, s)?;
            }
        }
        let moving = self.head.step_scalars();
        for step in 0..self.head.steps {
            let guidance = s.guidance.clone();
            for (name, v) in self.head.scalar_values(step, &guidance)? {
                if moving.contains(&name) {
                    self.upload(s, name, &v)?;
                }
            }
            for (op, l) in self.head.plan.head.iter().zip(&self.levels) {
                if *l == Level::Step {
                    self.run(dit(op)?, None, false, s)?;
                }
            }
            for ((b, n), fused) in self.head.plan.blocks.iter().zip(&self.layers).zip(&self.fused) {
                for i in 0..*n {
                    let mut skip = false;
                    for (j, op) in b.ops.iter().enumerate() {
                        if std::mem::take(&mut skip) {
                            continue;
                        }
                        let gated = fused.contains(&j);
                        self.run(dit(op)?, Some(i), gated, s)?;
                        skip = gated;
                    }
                }
            }
            for op in &self.head.plan.tail {
                self.run(dit(op)?, None, false, s)?;
            }
        }
        for op in &self.head.finish {
            self.run(dit(op)?, None, false, s)?;
        }
        let out = &s.bufs[self.head.graph.output[0].name.as_str()];
        Ok(self.stream.clone_dtoh(out.f32.as_ref().unwrap())?)
    }

    fn f32_ptr(&self, s: &CudaSession, name: &str) -> Result<u64> {
        let b = s.bufs.get(name).with_context(|| format!("no buffer {name}"))?;
        Ok(b.f32.as_ref().with_context(|| format!("{name} is held as planes"))?.device_ptr(&self.stream).0)
    }

    fn planes(&self, s: &CudaSession, name: &str) -> Result<(u64, u64)> {
        let b = s.bufs.get(name).with_context(|| format!("no buffer {name}"))?;
        let hi = b.hi.as_ref().with_context(|| format!("{name} is not held as planes"))?;
        Ok((hi.device_ptr(&self.stream).0, b.lo.as_ref().unwrap().device_ptr(&self.stream).0))
    }

    fn compose(&self, d: &DitOp) -> Result<&CudaFunction> {
        let (name, _, _) = kernel_key(d, self.head.graph, &self.head.spec)?.context("not a composed op")?;
        self.fns.get(&name).with_context(|| format!("kernel {name} is not loaded"))
    }

    fn grid(cols: usize, rows: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (cols.div_ceil(256).max(1) as u32, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    fn run(&self, d: &DitOp, block: Option<usize>, gated: bool, s: &mut CudaSession) -> Result<()> {
        self.launch(d, block, gated, s).with_context(|| d.label())?;
        s.launches += 1;
        Ok(())
    }

    fn launch(&self, d: &DitOp, block: Option<usize>, gated: bool, s: &mut CudaSession) -> Result<()> {
        let st = &self.stream;
        let rows = s.rows;
        let h = self.head.spec.hidden();
        let at = |w: &str| block.map_or(w.to_string(), |i| w.replace("{i}", &i.to_string()));
        match *d {
            DitOp::Linear { weight, input, output, .. } => {
                let (w, n, k) = self.linears.get(&at(weight)).with_context(|| format!("linear {} is not loaded", at(weight)))?;
                let (xh, xl) = self.planes(s, input)?;
                let out = self.f32_ptr(s, output)?;
                let (aux, aux_stride, out) = if gated {
                    let (set, gate, into) = self.head.plan.blocks.iter().flat_map(|b| &b.ops).find_map(|o| match o {
                        Op::Dit(DitOp::GatedAdd { set, gate, into, .. }) => Some((*set, *gate, *into)),
                        _ => None,
                    }).context("a gated epilogue has a gated add")?;
                    (self.f32_ptr(s, set)? + (gate as usize * h * 4) as u64, s.bufs[set].cols as i32, self.f32_ptr(s, into)?)
                } else {
                    (0, 0, out)
                };
                let f = &self.fns[&rows_name(*n, *k, if gated { GATED } else { STORE })];
                let (t, os) = (rows as i32, s.bufs[output].cols as i32);
                let mut b = st.launch_builder(f);
                b.arg(w).arg(&xh).arg(&xl).arg(&out).arg(&aux).arg(&t).arg(&os).arg(&aux_stride);
                unsafe { b.launch(LaunchConfig { grid_dim: (*n as u32, rows.div_ceil(BA_ROWS_MAX_TOKENS) as u32, 1), block_dim: (BA_ROWS_THREADS, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            DitOp::TimestepEmbed { from, into } => {
                let src = self.f32_ptr(s, from)?;
                let (hi, lo) = self.planes(s, into)?;
                let mut b = st.launch_builder(self.compose(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::grid(s.bufs[into].cols, rows)) }?;
            }
            DitOp::Split { from, into, .. } => {
                let src = self.f32_ptr(s, from)?;
                let (hi, lo) = self.planes(s, into)?;
                let mut b = st.launch_builder(self.compose(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::grid(s.bufs[from].cols, rows)) }?;
            }
            DitOp::Copy { from, into, .. } => {
                let (src, dst) = (self.f32_ptr(s, from)?, self.f32_ptr(s, into)?);
                let mut b = st.launch_builder(self.compose(d)?);
                b.arg(&src).arg(&dst);
                unsafe { b.launch(Self::grid(s.bufs[from].cols, rows)) }?;
            }
            DitOp::Glu { from, into, .. } => {
                let src = self.f32_ptr(s, from)?;
                let (hi, lo) = self.planes(s, into)?;
                let mut b = st.launch_builder(self.compose(d)?);
                b.arg(&src).arg(&hi).arg(&lo);
                unsafe { b.launch(Self::grid(s.bufs[into].cols, rows)) }?;
            }
            DitOp::Modulate { from, into, set, shift, scale, weight, .. } => {
                let x = self.f32_ptr(s, from)?;
                let base = self.f32_ptr(s, set)?;
                let (sh, sc) = (base + (shift as usize * h * 4) as u64, base + (scale as usize * h * 4) as u64);
                let (hi, lo) = self.planes(s, into)?;
                let mut b = st.launch_builder(self.compose(d)?);
                b.arg(&x).arg(&sh).arg(&sc);
                let gamma = match weight {
                    Some(w) => Some(self.vectors.get(&format!("{}.weight", at(w))).context("modulate weight is not loaded")?.device_ptr(st).0),
                    None => None,
                };
                if let Some(g) = &gamma {
                    b.arg(g);
                }
                b.arg(&hi).arg(&lo);
                unsafe { b.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            DitOp::GatedAdd { from, into, set, gate, .. } => {
                let (x, br) = (self.f32_ptr(s, into)?, self.f32_ptr(s, from)?);
                let g = self.f32_ptr(s, set)? + (gate as usize * h * 4) as u64;
                let mut b = st.launch_builder(self.compose(d)?);
                b.arg(&x).arg(&br).arg(&g);
                unsafe { b.launch(Self::grid(h, rows)) }?;
            }
            DitOp::Map { into, .. } => {
                let mut ptrs: Vec<u64> = Vec::new();
                match s.bufs[into].f32.is_some() {
                    true => ptrs.push(self.f32_ptr(s, into)?),
                    false => {
                        let (hi, lo) = self.planes(s, into)?;
                        ptrs.extend([hi, lo]);
                    }
                }
                for n in map_params(d, self.head.graph, &self.head.spec)? {
                    ptrs.push(match self.vectors.get(n) {
                        Some(v) => v.device_ptr(st).0,
                        None => self.f32_ptr(s, n)?,
                    });
                }
                let mut b = st.launch_builder(self.compose(d)?);
                for p in &ptrs {
                    b.arg(p);
                }
                unsafe { b.launch(Self::grid(s.bufs[into].cols, rows)) }?;
            }
            ref other => bail!("{} is not a flow head op", other.label()),
        }
        Ok(())
    }
}
