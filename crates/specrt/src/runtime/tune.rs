use super::{DeviceMatrix, DeviceMixer, GemmOut, Planes, Runtime};
use crate::codegen::{BATCH_VARIANTS, BatchShape, GemmPath, WeightKind};
use crate::spec::TuneMode;
use crate::tune::{Identity, ShapeKey, Tuner, candidate_splits};
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

const LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
const WARMUP: usize = 3;
const SAMPLES: usize = 8;
const REPEATS: usize = 10;

pub(super) fn driver_version() -> String {
    let mut v = 0;
    let rc = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v) };
    format!("{v}/{rc:?}")
}

impl Runtime {
    pub(super) fn splits_for(&self, kernel: &str, shape: &BatchShape, rows: usize, width: usize) -> usize {
        let key = ShapeKey::new(kernel, rows, width, shape.tokens);
        if let Some(&s) = self.tuned_splits.read().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return s;
        }
        shape.splits(&self.target, rows, width)
    }

    fn matrix_for(&self, linear: &str, rows: usize, width: usize) -> Option<&DeviceMatrix> {
        let fits = |m: &DeviceMatrix| match m {
            DeviceMatrix::Fp4 { packed, .. } => packed.len() * 2 == rows * width,
            DeviceMatrix::Fp8 { values, .. } => values.len() == rows * width,
            _ => false,
        };
        self.layers
            .iter()
            .filter_map(|l| match (linear, &l.mixer) {
                ("gate_up", _) => Some(&l.gate_up),
                ("down", _) => Some(&l.down),
                ("qkv", DeviceMixer::Attention(a)) => Some(&a.qkv),
                ("o", DeviceMixer::Attention(a)) => Some(&a.o),
                ("lin_in", DeviceMixer::Linear(g)) => Some(&g.in_proj),
                ("lin_out", DeviceMixer::Linear(g)) => Some(&g.out),
                _ => None,
            })
            .find(|m| fits(m))
    }

    pub(super) fn tune_batched(&self, mode: TuneMode, dir: &Path, module_hash: &str) -> Result<usize> {
        if mode == TuneMode::Off || !self.use_mma {
            return Ok(0);
        }
        let identity = Identity { module_hash: module_hash.to_string(), gpu: self.ctx.name()?, driver: driver_version() };
        let mut tuner = Tuner::open(mode, dir, identity);
        let mut tuned = 0;
        for linear in LINEARS {
            let (rows, width, ..) = crate::codegen::linear_dims(&self.spec, linear);
            if rows == 0 {
                continue;
            }
            let Some(w) = self.matrix_for(linear, rows, width) else { continue };
            let x = Planes::new(&self.stream, BATCH_VARIANTS[BATCH_VARIANTS.len() - 1] * width)?;
            for &tokens in &BATCH_VARIANTS {
                let choice = crate::codegen::select(&self.spec, &self.target, linear, tokens, w.kind())?;
                if !matches!(choice.path, GemmPath::F4b | GemmPath::F8b) {
                    continue;
                }
                let shape = BatchShape::for_tokens(tokens);
                let part = match w.kind() {
                    WeightKind::Fp4 => self.f4b_part.len(),
                    _ => self.f8b_part.len(),
                };
                let formula = shape.splits(&self.target, rows, width);
                let most = (width / kir::gemm::BATCH_KC / 4).max(1);
                let candidates: Vec<usize> =
                    candidate_splits(formula, most).into_iter().filter(|s| s * rows * tokens <= part).collect();
                if candidates.len() < 2 {
                    continue;
                }
                let key = ShapeKey::new(&choice.name, rows, width, shape.tokens);
                let planes_out = Planes::new(&self.stream, tokens * self.spec.gate_up_out(rows).max(rows))?;
                let f32_out = self.stream.alloc_zeros::<f32>(tokens * rows)?;
                let run = || -> Result<()> {
                    let out = if linear == "gate_up" { GemmOut::Planes(&planes_out) } else { GemmOut::F32(&f32_out) };
                    match w.kind() {
                        WeightKind::Fp4 => self.tc_f4b(&choice, w, x.all(), out, rows, tokens),
                        _ => self.tc_f8b(&choice, w, None, x.all(), out, rows, tokens),
                    }
                };
                let before = tuner.len();
                let won = tuner.choose(&key, formula, &candidates, |s| {
                    self.tuned_splits.write().unwrap_or_else(|e| e.into_inner()).insert(key.clone(), s);
                    self.time_samples(&run)
                });
                self.tuned_splits.write().unwrap_or_else(|e| e.into_inner()).insert(key, won);
                tracing::info!(linear, tokens, formula, won, "tuned batched splits");
                if tuner.len() > before {
                    tuned += 1;
                }
            }
        }
        if tuned > 0 {
            let _ = tuner.save();
        }
        Ok(tuned)
    }

    fn time_samples(&self, run: &impl Fn() -> Result<()>) -> Vec<f64> {
        for _ in 0..WARMUP {
            if run().is_err() {
                return Vec::new();
            }
        }
        let mut out = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            if self.stream.synchronize().is_err() {
                return Vec::new();
            }
            let started = Instant::now();
            for _ in 0..REPEATS {
                if run().is_err() {
                    return Vec::new();
                }
            }
            if self.stream.synchronize().is_err() {
                return Vec::new();
            }
            out.push(started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64);
        }
        out
    }
}
