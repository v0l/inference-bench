use super::compile::{Step, conv_kernel, fuse, kernels, steps};
use super::graph::SpeechOp;
use super::host::HostSpeech;
use crate::codegen::Module;
use crate::jit::Jit;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use kir::conv::ConvKernel;
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn module(host: &HostSpeech, target: &kir::Target) -> Result<(Module, Vec<Step>, Vec<ConvKernel>)> {
    let fused = fuse(&host.plan, target, true)?;
    let steps = steps(&host.plan, &fused)?;
    let convs = kernels(&steps, target)?;
    let mut source = kir::cuda::emit_line(&convs).map_err(anyhow::Error::msg)?;
    source.append("\n", &kir::cuda::emit_mel(&host.plan.mel).map_err(anyhow::Error::msg)?);
    let mut names: Vec<&'static str> = host.plan.mel.names().into_iter().map(|n| &*Box::leak(n.into_boxed_str())).collect();
    for k in &convs {
        names.push(Box::leak(k.name().into_boxed_str()));
    }
    names.sort_unstable();
    names.dedup();
    let kir_hash = kir::fold_hashes(convs.iter().map(ConvKernel::hash).chain([host.plan.mel.hash()]));
    Ok((Module { source, kernels: names, smem: BTreeMap::new(), arch_features: false, kir_hash }, steps, convs))
}

pub struct CudaCrossRows {
    stream: Arc<CudaStream>,
    f: CudaFunction,
    heads: usize,
    head_dim: usize,
    rows: usize,
}

impl CudaCrossRows {
    pub fn load(stream: &Arc<CudaStream>, jit: &Jit, spec: &crate::spec::Spec) -> Result<CudaCrossRows> {
        let c = spec.cross.as_ref().context("the spec has no [cross] section")?;
        let target = crate::codegen::cuda_target(&stream.context())?;
        let vk = kir::vision::VisionKernel::new(&target, spec.attention.head_dim, true).map_err(anyhow::Error::msg)?.joint();
        let mut names = kir::vision::KERNELS.to_vec();
        names.extend_from_slice(kir::vision::JOINT_KERNELS);
        let source = kir::cuda::emit_vision(&vk).map_err(anyhow::Error::msg)?;
        let loaded = jit.load(&Module { source, kernels: names, smem: BTreeMap::new(), arch_features: false, kir_hash: kir::fnv_hash(&vk.hash()) })?;
        let f = loaded.function("vis_flash_kv_rows")?;
        use cudarc::driver::sys::CUfunction_attribute as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kir::vision::cross_rows_smem(c.rows) as i32)?;
        Ok(CudaCrossRows { stream: stream.clone(), f, heads: spec.attention.heads, head_dim: spec.attention.head_dim, rows: c.rows })
    }

    pub fn run(&self, q: &[f32], cross_kv: &[&[f32]]) -> Result<Vec<f32>> {
        let s = &self.stream;
        let width = self.heads * self.head_dim;
        let seqs = cross_kv.len();
        ensure!(q.len() == seqs * width, "{} query values for {seqs} rows of {width}", q.len());
        let stride = self.rows * 2 * width;
        ensure!(cross_kv.iter().all(|k| k.len() == stride), "a sequence has no cross_kv state of {stride} values");
        let packed: Vec<f32> = cross_kv.iter().flat_map(|k| k.iter().copied()).collect();
        let (qd, kvd) = (s.clone_htod(q)?, s.clone_htod(&packed)?);
        let out = s.alloc_zeros::<f32>(seqs * width)?;
        let (qp, kp, op) = (qd.device_ptr(s).0, kvd.device_ptr(s).0, out.device_ptr(s).0);
        let (r, h, hd, st) = (self.rows as i32, self.heads as i32, self.head_dim as i32, stride as i64);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut b = s.launch_builder(&self.f);
        b.arg(&qp).arg(&kp).arg(&op).arg(&r).arg(&h).arg(&hd).arg(&st).arg(&scale);
        unsafe { b.launch(LaunchConfig { grid_dim: (self.heads as u32, seqs as u32, 1), block_dim: (kir::vision::ROWS_THREADS, 1, 1), shared_mem_bytes: kir::vision::cross_rows_smem(self.rows) }) }?;
        Ok(s.clone_dtoh(&out)?)
    }
}

struct Gemm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

pub struct CudaSpeech {
    stream: Arc<CudaStream>,
    target: kir::Target,
    mel: kir::mel::LogMel,
    fns: BTreeMap<String, CudaFunction>,
    steps: Vec<Step>,
    gemms: BTreeMap<usize, Gemm>,
    pub launches: usize,
}

fn flat(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(kir::mel::THREADS as usize).max(1) as u32, 1, 1), block_dim: (kir::mel::THREADS, 1, 1), shared_mem_bytes: 0 }
}

impl CudaSpeech {
    pub fn load(stream: &Arc<CudaStream>, jit: &Jit, host: &HostSpeech) -> Result<CudaSpeech> {
        let target = crate::codegen::cuda_target(&stream.context())?;
        let (m, steps, convs) = module(host, &target)?;
        let loaded = jit.load(&m)?;
        let mut fns = BTreeMap::new();
        for name in &m.kernels {
            fns.insert(name.to_string(), loaded.function(name)?);
        }
        for k in &convs {
            use cudarc::driver::sys::CUfunction_attribute as A;
            fns[&k.name()].set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes() as i32)?;
        }
        let mel = &host.plan.mel;
        let mut gemms = BTreeMap::new();
        let mut convs_host = host.convs.iter();
        for (i, op) in host.plan.ops.iter().enumerate() {
            let (w, b) = match op {
                SpeechOp::Dft => (mel.dft_table(), vec![0.0; 2 * mel.freqs()]),
                SpeechOp::Bank => (mel.mel_bank(), vec![0.0; mel.bins]),
                SpeechOp::Conv { .. } => {
                    let c = convs_host.next().unwrap();
                    (c.w.clone(), c.b.clone())
                }
                _ => continue,
            };
            gemms.insert(i, Gemm { w: stream.clone_htod(&w)?, b: stream.clone_htod(&b)? });
        }
        let launches = steps.len();
        Ok(CudaSpeech { stream: stream.clone(), target, mel: mel.clone(), fns, steps, gemms, launches })
    }

    pub fn run(&self, pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let s = &self.stream;
        let m = &self.mel;
        let (fr, freqs, bins) = (m.frames, m.freqs(), m.bins);
        let len = pcm.len().min(m.samples);
        ensure!(len > 0, "no audio");
        let audio = s.clone_htod(&pcm[..len])?;
        let mut cur: CudaSlice<f32> = s.alloc_zeros(1)?;
        let mut mel_out: Option<Vec<f32>> = None;
        let top: CudaSlice<f32> = s.alloc_zeros(1)?;
        let dptr = |v: &CudaSlice<f32>| v.device_ptr(s).0;
        let zero = s.alloc_zeros::<f32>(1)?;
        for step in &self.steps {
            match step {
                Step::Frames => {
                    let y = s.alloc_zeros::<f32>(fr * m.n_fft)?;
                    let (a, n, yp) = (dptr(&audio), len as i32, dptr(&y));
                    let mut lb = s.launch_builder(&self.fns[&m.name("mel_frames")]);
                    lb.arg(&a).arg(&n).arg(&yp);
                    unsafe { lb.launch(flat(fr * m.n_fft)) }?;
                    cur = y;
                }
                Step::Gemm { op, shape, gelu, t_in, t_out } => {
                    let k = conv_kernel(&self.target, *shape, *gelu)?;
                    let g = &self.gemms[op];
                    let y = s.alloc_zeros::<f32>(t_out * shape.cout)?;
                    let (x, st, w, b, yp) = (dptr(&cur), dptr(&zero), dptr(&g.w), dptr(&g.b), dptr(&y));
                    let (bb, t, to) = (1i32, *t_in as i32, *t_out as i32);
                    let (gx, gy, gz) = k.line_grid(*t_out);
                    let mut lb = s.launch_builder(&self.fns[&k.name()]);
                    lb.arg(&x).arg(&st).arg(&w).arg(&b).arg(&st).arg(&yp).arg(&bb).arg(&t).arg(&to);
                    unsafe { lb.launch(LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (kir::conv::THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() }) }?;
                    cur = y;
                }
                Step::Power => {
                    let y = s.alloc_zeros::<f32>(fr * freqs)?;
                    let (x, yp) = (dptr(&cur), dptr(&y));
                    let mut lb = s.launch_builder(&self.fns[&m.name("mel_power")]);
                    lb.arg(&x).arg(&yp);
                    unsafe { lb.launch(flat(fr * freqs)) }?;
                    cur = y;
                }
                Step::Log => {
                    let y = s.alloc_zeros::<f32>(fr * bins)?;
                    let (x, yp) = (dptr(&cur), dptr(&y));
                    let mut lb = s.launch_builder(&self.fns[&m.name("mel_log")]);
                    lb.arg(&x).arg(&yp);
                    unsafe { lb.launch(flat(fr * bins)) }?;
                    cur = y;
                }
                Step::Max => {
                    let (x, t) = (dptr(&cur), dptr(&top));
                    let mut lb = s.launch_builder(&self.fns[&m.name("mel_max")]);
                    lb.arg(&x).arg(&t);
                    unsafe { lb.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (kir::mel::THREADS, 1, 1), shared_mem_bytes: 0 }) }?;
                }
                Step::Scale => {
                    let y = s.alloc_zeros::<f32>(fr * bins)?;
                    let (x, t, yp) = (dptr(&cur), dptr(&top), dptr(&y));
                    let mut lb = s.launch_builder(&self.fns[&m.name("mel_scale")]);
                    lb.arg(&x).arg(&t).arg(&yp);
                    unsafe { lb.launch(flat(fr * bins)) }?;
                    mel_out = Some(s.clone_dtoh(&y)?);
                    cur = y;
                }
                Step::Gelu { .. } => anyhow::bail!("an unfused gelu has no speech kernel"),
            }
        }
        let stem = s.clone_dtoh(&cur)?;
        Ok((mel_out.unwrap_or_default(), stem))
    }
}
