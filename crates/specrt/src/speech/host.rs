use super::graph::{SpeechOp, SpeechPlan};
use crate::vae::host::Weights;
use anyhow::{Result, ensure};

pub fn gelu(x: f32) -> f32 {
    crate::codec::host::gelu(x)
}

pub fn frames(plan: &SpeechPlan, pcm: &[f32]) -> Vec<f32> {
    let m = &plan.mel;
    let len = pcm.len().min(m.samples);
    let mut out = vec![0f32; m.frames * m.n_fft];
    for f in 0..m.frames {
        for t in 0..m.n_fft {
            let i = (f * m.hop + t) as isize - (m.n_fft / 2) as isize;
            out[f * m.n_fft + t] = m.source(i).filter(|j| *j < len).map_or(0.0, |j| pcm[j]);
        }
    }
    out
}

pub fn matmul(x: &[f32], rows: usize, cin: usize, w: &[f32], cout: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0f32; rows * cout];
    for r in 0..rows {
        let xr = &x[r * cin..(r + 1) * cin];
        for n in 0..cout {
            let acc: f32 = w[n * cin..(n + 1) * cin].iter().zip(xr).map(|(a, b)| a * b).sum();
            y[r * cout + n] = acc + bias.map_or(0.0, |b| b[n]);
        }
    }
    y
}

fn eval(e: &kir::expr::Expr, names: &[(&str, f32)]) -> Result<f32> {
    let env = |n: &str| names.iter().find(|(k, _)| *k == n).map(|(_, v)| f64::from(*v));
    let v = e.eval(&env).ok_or_else(|| anyhow::anyhow!("the expression {e} does not evaluate on the host"))?;
    Ok(v as f32)
}

pub fn log_mel(plan: &SpeechPlan, pcm: &[f32]) -> Result<Vec<f32>> {
    let m = &plan.mel;
    let (fr, freqs) = (m.frames, m.freqs());
    let spec = matmul(&frames(plan, pcm), fr, m.n_fft, &m.dft_table(), 2 * freqs, None);
    let power: Vec<f32> = spec.chunks(2).map(|c| c[0] * c[0] + c[1] * c[1]).collect();
    let mel = matmul(&power, fr, freqs, &m.mel_bank(), m.bins, None);
    let logged: Vec<f32> = mel.iter().map(|v| eval(&m.log, &[("m", *v)])).collect::<Result<_>>()?;
    let top = logged.iter().copied().fold(f32::MIN, f32::max);
    logged.iter().map(|v| eval(&m.out, &[("v", eval(&m.floor, &[("v", *v), ("top", top)])?)])).collect()
}

pub struct Conv {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub fn padded_conv(x: &[f32], t_in: usize, cin: usize, c: &Conv, cout: usize, k: usize, stride: usize, pad: usize, t_out: usize) -> Vec<f32> {
    let mut y = vec![0f32; t_out * cout];
    for o in 0..t_out {
        for n in 0..cout {
            let mut acc = c.b[n];
            for tap in 0..k {
                let p = (o * stride + tap) as isize - pad as isize;
                if p < 0 || p as usize >= t_in {
                    continue;
                }
                let row = &x[p as usize * cin..(p as usize + 1) * cin];
                acc += c.w[(n * k + tap) * cin..(n * k + tap + 1) * cin].iter().zip(row).map(|(a, b)| a * b).sum::<f32>();
            }
            y[o * cout + n] = acc;
        }
    }
    y
}

pub struct HostSpeech {
    pub plan: SpeechPlan,
    pub convs: Vec<Conv>,
}

impl HostSpeech {
    pub fn load(plan: SpeechPlan, weights: &dyn Weights) -> Result<HostSpeech> {
        let mut convs = Vec::new();
        for op in &plan.ops {
            if let SpeechOp::Conv { weight, cin, cout, k, .. } = op {
                let w = weights.floats(&format!("{}.{weight}.weight", plan.root))?;
                let b = weights.floats(&format!("{}.{weight}.bias", plan.root))?;
                ensure!(w.len() == cout * cin * k && b.len() == *cout, "{} has {} weights", op.label(), w.len());
                convs.push(Conv { w: crate::codec::host::taps_major(&w, *cout, *cin, *k), b });
            }
        }
        Ok(HostSpeech { plan, convs })
    }

    pub fn open(dir: &std::path::Path) -> Result<HostSpeech> {
        let plan = super::graph::plan_dir(dir)?;
        let ck = crate::weights::Checkpoint::open(dir)?;
        plan.check_tensors(&|n| ck.shape_of(n))?;
        HostSpeech::load(plan, &ck)
    }

    pub fn stem(&self, mel: &[f32]) -> Result<Vec<f32>> {
        ensure!(mel.len() == self.plan.mel.frames * self.plan.mel.bins, "{} mel values, the plan reads {} frames of {} bins", mel.len(), self.plan.mel.frames, self.plan.mel.bins);
        let mut x = mel.to_vec();
        let mut convs = self.convs.iter();
        for op in &self.plan.ops {
            match op {
                SpeechOp::Conv { cin, cout, k, stride, pad, t_in, t_out, .. } => x = padded_conv(&x, *t_in, *cin, convs.next().unwrap(), *cout, *k, *stride, *pad, *t_out),
                SpeechOp::Gelu { .. } => x.iter_mut().for_each(|v| *v = gelu(*v)),
                _ => {}
            }
        }
        Ok(x)
    }

    pub fn run(&self, pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mel = log_mel(&self.plan, pcm)?;
        let stem = self.stem(&mel)?;
        Ok((mel, stem))
    }
}

pub struct HostEncoder {
    pub graph: crate::denoiser::DenoiserGraph,
    pub spec: crate::denoiser::DenoiserSpec,
    pub plan: crate::plan::Plan,
    pub rows: usize,
}

impl HostEncoder {
    pub fn new(config: &serde_json::Value) -> Result<HostEncoder> {
        let (graph, spec) = super::graph::encoder(config)?;
        let plan = graph.plan()?;
        crate::denoiser::validate(&graph, &spec, &plan)?;
        let rows = spec.size("positions")?;
        Ok(HostEncoder { graph, spec, plan, rows })
    }

    pub fn encode(&self, ck: &crate::weights::Checkpoint, stem: &[f32]) -> Result<Vec<f32>> {
        let host = crate::denoiser_host::HostDenoiser::load(&self.graph, &self.spec, &self.plan, ck, false)?;
        ensure!(stem.len() == self.rows * self.spec.hidden(), "{} stem values, the encoder reads {} rows of {}", stem.len(), self.rows, self.spec.hidden());
        let x = crate::denoiser_host::Inputs { latent: &[], text: &[], timestep: 0.0, ids: &[], image: self.rows, text_rows: 0, patch_h: 0, patch_w: 0, frames: 0 };
        host.run(&x, &[("stem", stem.to_vec())], "states", &mut Default::default())
    }
}
