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

pub struct CudaRowPlans {
    stream: Arc<CudaStream>,
    edit: CudaFunction,
    allow: CudaFunction,
    vocab: usize,
}

impl CudaRowPlans {
    pub fn load(stream: &Arc<CudaStream>, jit: &Jit, spec: &crate::spec::Spec) -> Result<CudaRowPlans> {
        let shape = kir::duplex::DuplexShape { hidden: spec.dims.hidden, vocab: spec.dims.vocab, latent: 1 };
        let source = kir::cuda::emit_duplex(&shape).map_err(anyhow::Error::msg)?;
        let loaded = jit.load(&Module { source, kernels: kir::duplex::KERNELS.to_vec(), smem: BTreeMap::new(), arch_features: false, kir_hash: kir::fnv_hash(&shape.hash()) })?;
        Ok(CudaRowPlans { stream: stream.clone(), edit: loaded.function("duplex_edit_logits")?, allow: loaded.function("duplex_allow")?, vocab: spec.dims.vocab })
    }

    pub fn run(&self, logits: &[f32], plans: &[crate::duplex::sampling::RowPlan]) -> Result<Vec<f32>> {
        let s = &self.stream;
        let rows = plans.len();
        ensure!(logits.len() == rows * self.vocab, "{} logits for {rows} rows of {}", logits.len(), self.vocab);
        let es = plans.iter().map(|p| p.edits.len()).max().unwrap_or(0).max(1);
        let al = plans.iter().map(|p| p.allow.len()).max().unwrap_or(0).max(1);
        let mut tok = vec![0i32; rows * es];
        let mut add = vec![0f32; rows * es];
        let mut allow = vec![0i32; rows * al];
        for (r, p) in plans.iter().enumerate() {
            for (i, (t, a)) in p.edits.iter().enumerate() {
                tok[r * es + i] = *t as i32;
                add[r * es + i] = *a;
            }
            for (i, t) in p.allow.iter().enumerate() {
                allow[r * al + i] = *t as i32;
            }
        }
        let ecount: Vec<i32> = plans.iter().map(|p| p.edits.len() as i32).collect();
        let acount: Vec<i32> = plans.iter().map(|p| p.allow.len() as i32).collect();
        let l = s.clone_htod(logits)?;
        let (td, ad, ed, ald, acd) = (s.clone_htod(&tok)?, s.clone_htod(&add)?, s.clone_htod(&ecount)?, s.clone_htod(&allow)?, s.clone_htod(&acount)?);
        let lp = l.device_ptr(s).0;
        let (tp, ap, ep, alp, acp) = (td.device_ptr(s).0, ad.device_ptr(s).0, ed.device_ptr(s).0, ald.device_ptr(s).0, acd.device_ptr(s).0);
        let (es, al) = (es as i32, al as i32);
        let mut b = s.launch_builder(&self.edit);
        b.arg(&lp).arg(&tp).arg(&ap).arg(&ep).arg(&es);
        unsafe { b.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 }) }?;
        let mut b = s.launch_builder(&self.allow);
        b.arg(&lp).arg(&alp).arg(&acp).arg(&al);
        let y = (self.vocab as u32).div_ceil(256).min(64);
        unsafe { b.launch(LaunchConfig { grid_dim: (rows as u32, y, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(s.clone_dtoh(&l)?)
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
    first_conv: usize,
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
        let first_conv = host.plan.ops.iter().position(|o| matches!(o, SpeechOp::Conv { .. })).unwrap_or(host.plan.ops.len());
        Ok(CudaSpeech { stream: stream.clone(), target, mel: mel.clone(), fns, steps, gemms, launches, first_conv })
    }

    pub fn run(&self, pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let (mel, stem) = self.exec(pcm, false)?;
        Ok((mel, self.stream.clone_dtoh(&stem)?))
    }

    pub fn mel(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        Ok(self.exec(pcm, true)?.0)
    }

    pub fn stem(&self, mel: &[f32]) -> Result<CudaSlice<f32>> {
        let m = &self.mel;
        ensure!(mel.len() == m.frames * m.bins, "{} mel values, the plan reads {} frames of {} bins", mel.len(), m.frames, m.bins);
        let mut cur = self.stream.clone_htod(mel)?;
        for step in &self.steps {
            if let Step::Gemm { op, shape, gelu, t_in, t_out } = step
                && *op >= self.first_conv
            {
                cur = self.gemm(&cur, *op, *shape, *gelu, *t_in, *t_out)?;
            }
        }
        Ok(cur)
    }

    fn gemm(&self, cur: &CudaSlice<f32>, op: usize, shape: kir::conv::ConvShape, gelu: bool, t_in: usize, t_out: usize) -> Result<CudaSlice<f32>> {
        let s = &self.stream;
        let k = conv_kernel(&self.target, shape, gelu)?;
        let g = &self.gemms[&op];
        let y = s.alloc_zeros::<f32>(t_out * shape.cout)?;
        let zero = s.alloc_zeros::<f32>(1)?;
        let dptr = |v: &CudaSlice<f32>| v.device_ptr(s).0;
        let (x, st, w, b, yp) = (dptr(cur), dptr(&zero), dptr(&g.w), dptr(&g.b), dptr(&y));
        let (bb, t, to) = (1i32, t_in as i32, t_out as i32);
        let (gx, gy, gz) = k.line_grid(t_out);
        let mut lb = s.launch_builder(&self.fns[&k.name()]);
        lb.arg(&x).arg(&st).arg(&w).arg(&b).arg(&st).arg(&yp).arg(&bb).arg(&t).arg(&to);
        unsafe { lb.launch(LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (kir::conv::THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() }) }?;
        Ok(y)
    }

    fn exec(&self, pcm: &[f32], mel_only: bool) -> Result<(Vec<f32>, CudaSlice<f32>)> {
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
                    if mel_only && *op >= self.first_conv {
                        break;
                    }
                    cur = self.gemm(&cur, *op, *shape, *gelu, *t_in, *t_out)?;
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
        let stem = cur;
        Ok((mel_out.unwrap_or_default(), stem))
    }
}
