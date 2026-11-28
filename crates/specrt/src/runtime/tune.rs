use super::{DeviceMatrix, DeviceMixer, GemmOut, Planes, Runtime};
use crate::codegen::{KernelChoice, WeightKind};
use anyhow::{Context, Result};
use cudarc::driver::sys::CUevent_flags;
use std::path::{Path, PathBuf};

pub const TUNE_FILE: &str = "gemm-select.tsv";
const TUNE_REPS: usize = 3;
const TUNE_CHAIN: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub struct TunedPick {
    pub linear: &'static str,
    pub weight: WeightKind,
    pub rows: usize,
    pub kernel: String,
    pub micros: f64,
    pub cached: bool,
}

pub fn tune_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(TUNE_FILE)
}

pub fn load_picks(path: &Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else { return 0 };
    let mut n = 0;
    for line in text.lines() {
        if let Some((key, suffix)) = line.split_once('\t') {
            kir::gemm::record_pick(key.to_string(), suffix.to_string());
            n += 1;
        }
    }
    n
}

fn append_picks(path: &Path, rows: &[(String, String)]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).with_context(|| path.display().to_string())?;
    for (k, v) in rows {
        writeln!(f, "{k}\t{v}")?;
    }
    Ok(())
}

impl Runtime {
    pub(super) fn fill_tune_inputs(&mut self) -> Result<()> {
        let widest = kir::gemm::TUNE_BUCKETS[kir::gemm::TUNE_BUCKETS.len() - 1] * self.spec.dims.hidden.max(self.spec.max_intermediate()).max(self.spec.q_dim());
        let mut x = 0x9e37_79b9u32;
        let stream = self.stream.clone();
        for p in [&mut self.pre.xn_planes, &mut self.pre.attn_planes, &mut self.pre.act_planes] {
            let n = p.hi.len().min(widest);
            let host: Vec<u16> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    crate::weights::f32_to_bf16((x >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0)
                })
                .collect();
            stream.memcpy_htod(&host, &mut p.hi.slice_mut(..n))?;
        }
        Ok(())
    }
    fn tune_weights(&self, linear: &str, kind: WeightKind) -> Vec<&DeviceMatrix> {
        self.layers
            .iter()
            .filter_map(|l| match (&l.mixer, linear) {
                (DeviceMixer::Attention(a), "qkv") => Some(&a.qkv),
                (DeviceMixer::Attention(a), "o") => Some(&a.o),
                (DeviceMixer::Linear(g), "lin_in") => Some(&g.in_proj),
                (DeviceMixer::Linear(g), "lin_out") => Some(&g.out),
                (_, "gate_up") if l.moe.is_none() => Some(&l.gate_up),
                (_, "down") if l.moe.is_none() => Some(&l.down),
                _ => None,
            })
            .filter(|w| w.kind() == kind)
            .collect()
    }

    fn time_choice(&self, choice: &KernelChoice, ws: &[&DeviceMatrix], out: GemmOut, x: &Planes, rows: usize, tokens: usize) -> Result<f64> {
        let ev = || -> Result<cudarc::driver::CudaEvent> {
            let e = self.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            e.record(&self.stream)?;
            Ok(e)
        };
        let run = |i: usize| self.tc_choice(choice, ws[i % ws.len()], None, x.all(), out_again(&out), rows, tokens);
        for i in 0..ws.len().min(TUNE_CHAIN) {
            run(i)?;
        }
        self.stream.synchronize()?;
        let mut t = Vec::with_capacity(TUNE_REPS);
        let mut at = 0;
        for _ in 0..TUNE_REPS {
            let a = ev()?;
            for _ in 0..TUNE_CHAIN {
                run(at)?;
                at += 1;
            }
            let b = ev()?;
            t.push(a.elapsed_ms(&b)? as f64 * 1e3 / TUNE_CHAIN as f64);
        }
        t.sort_by(f64::total_cmp);
        Ok(t[0])
    }

    pub fn tune_gemms(&self) -> Result<Vec<TunedPick>> {
        let spec = &self.spec;
        let t = &self.target;
        let cache_dir = spec.runtime.cache_dir.clone().unwrap_or_else(crate::jit::default_cache_dir);
        let path = tune_path(&cache_dir);
        load_picks(&path);
        let mut fresh = Vec::new();
        let mut out = Vec::new();
        let mut linears: Vec<(&'static str, Vec<&DeviceMatrix>)> = Vec::new();
        for l in kir::gemm::BATCH_LINEARS {
            for kind in [WeightKind::Fp4, WeightKind::Fp8] {
                let ws = self.tune_weights(l, kind);
                if !ws.is_empty() {
                    linears.push((l, ws));
                }
            }
        }
        for (linear, ws) in linears {
            let w = ws[0];
            let kind = w.kind();
            let act = !matches!(w, DeviceMatrix::Fp4 { act: false, .. });
            let ws: Vec<&DeviceMatrix> = ws.into_iter().filter(|m| act == !matches!(m, DeviceMatrix::Fp4 { act: false, .. })).collect();
            let policy = crate::gemm::policy_on(spec, t, act);
            let req = crate::gemm::request(spec, linear, 1, kind);
            for bucket in kir::gemm::TUNE_BUCKETS {
                if bucket > self.chunk {
                    break;
                }
                let r = kir::gemm::tuned_request(&req, bucket);
                if !kir::gemm::tuned(t, &policy, &r) {
                    continue;
                }
                let full = kir::gemm::eligible(t, &policy, &r);
                let all: Vec<KernelChoice> = full.iter().filter(|c| self.kernels.contains_key(c.name.as_str()) && c.quant.is_none_or(|q| self.kernels.contains_key(q)) && c.reduce.as_ref().is_none_or(|q| self.kernels.contains_key(q.as_str()))).cloned().collect();
                if all.len() < 2 {
                    continue;
                }
                let key = kir::gemm::pick_key(t, &policy, &r, bucket, &full);
                if let Some(suffix) = kir::gemm::measured_pick(&key) {
                    out.push(TunedPick { linear, weight: kind, rows: bucket, kernel: format!("{linear}{suffix}"), micros: 0.0, cached: true });
                    continue;
                }
                let (n, k) = (r.shape.n, r.shape.k);
                let gu = matches!(r.shape.epi, kir::gemm::Epilogue::Act | kir::gemm::Epilogue::Map);
                let inputs = if gu { vec![&self.pre.xn_planes] } else { vec![&self.pre.xn_planes, &self.pre.attn_planes, &self.pre.act_planes] };
                let Some(planes) = inputs.into_iter().find(|p| p.hi.len() >= bucket * k) else { continue };
                let out_buf = || if gu { GemmOut::Planes(&self.pre.act_planes) } else { GemmOut::F32(&self.pre.qkv) };
                let room = if gu { self.pre.act_planes.hi.len() >= bucket * n / 2 } else { self.pre.qkv.len() >= bucket * n };
                if !room {
                    continue;
                }
                let mut best: Option<(f64, &KernelChoice)> = None;
                for c in &all {
                    let us = self.time_choice(c, &ws, out_buf(), planes, n, bucket)?;
                    if best.is_none_or(|(b, _)| us < b) {
                        best = Some((us, c));
                    }
                }
                let Some((us, c)) = best else { continue };
                let suffix = kir::gemm::pick_suffix(&r, &c.name).to_string();
                kir::gemm::record_pick(key.clone(), suffix.clone());
                fresh.push((key, suffix));
                out.push(TunedPick { linear, weight: kind, rows: bucket, kernel: c.name.clone(), micros: us, cached: false });
            }
        }
        self.stream.synchronize()?;
        let _ = std::fs::create_dir_all(&cache_dir);
        if let Err(e) = append_picks(&path, &fresh) {
            tracing::warn!("gemm selection cache not written: {e:#}");
        }
        Ok(out)
    }
}

fn out_again<'a>(o: &GemmOut<'a>) -> GemmOut<'a> {
    match o {
        GemmOut::F32(f) => GemmOut::F32(f),
        GemmOut::Planes(p) => GemmOut::Planes(p),
    }
}
