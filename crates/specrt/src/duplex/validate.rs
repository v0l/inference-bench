use super::engine::{Engine, RecordedNoise, StartOptions};
use super::prompt::Prompt;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const VOICED_DBFS: f32 = -40.0;

pub struct Trace {
    pub dir: PathBuf,
    pub index: Value,
}

impl Trace {
    pub fn open(dir: &Path) -> Result<Trace> {
        let text = std::fs::read_to_string(dir.join("index.json")).with_context(|| format!("{}/index.json", dir.display()))?;
        Ok(Trace { dir: dir.to_path_buf(), index: serde_json::from_str(&text)? })
    }

    pub fn meta(&self) -> &Value {
        &self.index["meta"]
    }

    pub fn has(&self, key: &str) -> bool {
        self.index["files"].get(key).is_some()
    }

    pub fn get(&self, key: &str) -> Result<Vec<f32>> {
        let f = self.index["files"].get(key).with_context(|| format!("the trace has no {key}"))?;
        let bytes = std::fs::read(self.dir.join(f["file"].as_str().context("file")?))?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    pub fn prompt(&self) -> Result<Prompt> {
        Ok(Prompt {
            tokens: self.get("prompt.tokens")?.iter().map(|&t| t as u32).collect(),
            model_latents: self.get("prompt.model_latents")?,
            user_latents: self.get("prompt.user_latents")?,
            model_mask: self.get("prompt.model_mask")?,
            user_mask: self.get("prompt.user_mask")?,
        })
    }

    pub fn noise(&self) -> Result<RecordedNoise> {
        let user = if self.has("frame.user_noise") { self.get("frame.user_noise")? } else { Vec::new() };
        let user_scale = if self.has("frame.user_scale") { self.get("frame.user_scale")? } else { Vec::new() };
        Ok(RecordedNoise { head: self.get("frame.head_noise")?, user, user_scale })
    }

    pub fn greedy(&self) -> bool {
        self.meta()["greedy"].as_bool().unwrap_or(false)
    }

    pub fn frames(&self) -> usize {
        self.meta()["frames"].as_u64().unwrap_or(0) as usize
    }

    pub fn layers(&self) -> Vec<usize> {
        self.meta()["layers"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()).unwrap_or_default()
    }
}

pub fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    if aa == 0.0 && bb == 0.0 { 1.0 } else { ab / (aa.sqrt() * bb.sqrt()).max(1e-30) }
}

pub fn kl_top(want: &[f32], got: &[f32]) -> f64 {
    let soft = |v: &[f32]| -> Vec<f64> {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        if !m.is_finite() {
            return vec![1.0 / v.len() as f64; v.len()];
        }
        let e: Vec<f64> = v.iter().map(|x| ((*x as f64) - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    };
    let (p, q) = (soft(want), soft(got));
    p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-30)).ln()).sum()
}

pub fn voiced(audio: &[f32], frame: usize) -> Vec<bool> {
    audio.chunks(frame).map(|f| {
        let rms = (f.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / f.len().max(1) as f64 + 1e-12).sqrt();
        20.0 * rms.log10() > VOICED_DBFS as f64
    }).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |m| m.0)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FrameRow {
    pub frame: usize,
    pub user_latent: f32,
    pub embeds: f32,
    pub layers: Vec<f32>,
    pub layers_cos: Vec<f64>,
    pub hidden: f32,
    pub hidden_cos: f64,
    pub top1: bool,
    pub kl_top10: f64,
    pub pad_eop: [f32; 2],
    pub latent: f32,
    pub audio: f32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub frames: usize,
    pub top1_rate: f64,
    pub kl_mean: f64,
    pub kl_max: f64,
    pub hidden_cos_min: f64,
    pub layer_cos_min: Vec<f64>,
    pub layer_rel_max: Vec<f32>,
    pub hidden_rel_max: f32,
    pub hidden_rel_mean: f64,
    pub latent_rel_mean: f64,
    pub latent_rel_max: f32,
    pub audio_rel: f32,
    pub audio_cos: f64,
    pub user_latent_max: f32,
    pub first_off_floor: Option<(usize, String, f32)>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    pub rows: Vec<FrameRow>,
    pub summary: Summary,
    pub tokens: Vec<u32>,
    pub audio: Vec<f32>,
}

pub const FLOOR: f32 = 3e-4;

fn start(e: &mut Engine, t: &Trace, seed: u64) -> Result<(super::SessionId, super::Frame)> {
    let p = t.prompt()?;
    let o = StartOptions { greedy: t.greedy(), seed, noise: Some(t.noise()?), speak_first: t.meta()["speak_first"].as_bool().unwrap_or(false), ..StartOptions::default() };
    if let Some(temp) = t.meta()["temperature"].as_f64().filter(|v| *v > 0.0) {
        e.duplex.sampling.temperature = temp as f32;
    }
    e.start_prompt(&p, o)
}

pub fn teacher_forced(e: &mut Engine, t: &Trace, frames: Option<usize>) -> Result<Report> {
    let layers = e.opts().trace_layers.clone();
    ensure!(layers == t.layers(), "the engine traces layers {layers:?}, the reference {:?}", t.layers());
    let (h, d, f) = (e.spec.dims.hidden, e.duplex.latent_dim, e.duplex.frame_samples);
    let n = frames.unwrap_or(t.frames()).min(t.frames());
    let (tokens, latents, hidden, top_ids, top_logits, pad_eop, audio, user) = (t.get("frame.token")?, t.get("frame.latent")?, t.get("frame.hidden")?, t.get("frame.top_ids")?, t.get("frame.top_logits")?, t.get("frame.pad_eop")?, t.get("frame.audio")?, t.get("frame.user_audio")?);
    let (ref_layers, ref_embeds, ref_user) = (t.get("frame.layers")?, if n > 1 { t.get("frame.embeds")? } else { Vec::new() }, if n > 1 { t.get("frame.user_latent")? } else { Vec::new() });
    let top = t.meta()["top"].as_u64().unwrap_or(10) as usize;
    let (pad, eop) = (e.duplex.pad as usize, e.duplex.eop as usize);
    let (id, first) = start(e, t, 0)?;
    let mut report = Report::default();
    let row_of = |e: &mut Engine, i: usize, fr: &super::Frame, traced: bool| -> Result<FrameRow> {
        let logits = e.row_logits(0)?;
        let ids: Vec<usize> = top_ids[i * top..(i + 1) * top].iter().map(|&v| v as usize).collect();
        let got: Vec<f32> = ids.iter().map(|&j| logits[j]).collect();
        let normed = e.rt.normed_rows(1)?;
        let want_h = &hidden[i * h..(i + 1) * h];
        let mut row = FrameRow {
            frame: i,
            hidden: max_rel(&normed, want_h),
            hidden_cos: cosine(&normed, want_h),
            top1: argmax(&logits) == ids[0],
            kl_top10: kl_top(&top_logits[i * top..(i + 1) * top], &got),
            pad_eop: [logits[pad] - pad_eop[2 * i], logits[eop] - pad_eop[2 * i + 1]],
            latent: max_rel(&fr.latent, &latents[i * d..(i + 1) * d]),
            audio: max_rel(&fr.audio, &audio[i * f..(i + 1) * f].iter().map(|v| v.clamp(-1.0, 1.0)).collect::<Vec<_>>()),
            ..FrameRow::default()
        };
        if traced {
            let tr = e.layer_trace(1)?;
            row.embeds = max_rel(&tr[..h], &ref_embeds[(i - 1) * h..i * h]);
            for (k, _) in layers.iter().enumerate() {
                let (g, w) = (&tr[(k + 1) * h..(k + 2) * h], &ref_layers[(i * layers.len() + k) * h..(i * layers.len() + k + 1) * h]);
                row.layers.push(max_rel(g, w));
                row.layers_cos.push(cosine(g, w));
            }
            row.user_latent = max_rel(&e.user_latent_rows(1)?, &ref_user[(i - 1) * d..i * d]);
        }
        Ok(row)
    };
    report.rows.push(row_of(e, 0, &first, false)?);
    report.tokens.push(first.token);
    report.audio.extend(&first.audio);
    for i in 1..n {
        e.set_feedback(id, tokens[i - 1] as u32, &latents[(i - 1) * d..i * d])?;
        let fr = e.step(&[(id, user[(i - 1) * f..i * f].to_vec())])?.remove(0);
        report.rows.push(row_of(e, i, &fr, true)?);
        report.tokens.push(fr.token);
        report.audio.extend(&fr.audio);
    }
    e.end(id)?;
    let want_audio: Vec<f32> = audio[..n * f].iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    report.summary = summarise(&report.rows, layers.len(), &report.audio, &want_audio);
    Ok(report)
}

fn summarise(rows: &[FrameRow], layers: usize, got_audio: &[f32], want_audio: &[f32]) -> Summary {
    let n = rows.len();
    let mut s = Summary { frames: n, layer_cos_min: vec![1.0; layers], layer_rel_max: vec![0.0; layers], hidden_cos_min: 1.0, ..Summary::default() };
    for r in rows {
        s.top1_rate += r.top1 as u8 as f64 / n as f64;
        s.kl_mean += r.kl_top10 / n as f64;
        s.kl_max = s.kl_max.max(r.kl_top10);
        s.hidden_cos_min = s.hidden_cos_min.min(r.hidden_cos);
        s.hidden_rel_max = s.hidden_rel_max.max(r.hidden);
        s.hidden_rel_mean += r.hidden as f64 / n as f64;
        s.latent_rel_mean += r.latent as f64 / n as f64;
        s.latent_rel_max = s.latent_rel_max.max(r.latent);
        s.user_latent_max = s.user_latent_max.max(r.user_latent);
        for (k, (rel, cos)) in r.layers.iter().zip(&r.layers_cos).enumerate() {
            s.layer_rel_max[k] = s.layer_rel_max[k].max(*rel);
            s.layer_cos_min[k] = s.layer_cos_min[k].min(*cos);
        }
        if s.first_off_floor.is_none() {
            let named: Vec<(String, f32)> = std::iter::once(("embeds".to_string(), r.embeds)).chain(r.layers.iter().enumerate().map(|(k, v)| (format!("layer slot {k}"), *v))).chain(std::iter::once(("hidden".to_string(), r.hidden))).collect();
            if let Some((name, v)) = named.into_iter().find(|(_, v)| *v > FLOOR) {
                s.first_off_floor = Some((r.frame, name, v));
            }
        }
    }
    s.audio_rel = max_rel(got_audio, want_audio);
    s.audio_cos = cosine(got_audio, want_audio);
    s
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FreeRun {
    pub tokens: Vec<u32>,
    pub want_tokens: Vec<u32>,
    pub first_divergence: Option<usize>,
    pub hidden_until_divergence: Vec<f32>,
    pub voiced_frames: usize,
    pub want_voiced_frames: usize,
    pub voiced_agreement: f64,
    pub text: String,
    #[serde(skip)]
    pub audio: Vec<f32>,
}

pub fn free_running(e: &mut Engine, t: &Trace, seed: u64) -> Result<FreeRun> {
    let (h, f) = (e.spec.dims.hidden, e.duplex.frame_samples);
    let n = t.frames();
    let (want_tokens, hidden, want_audio, user) = (t.get("frame.token")?, t.get("frame.hidden")?, t.get("frame.audio")?, t.get("frame.user_audio")?);
    let (id, first) = start(e, t, seed)?;
    let mut run = FreeRun { want_tokens: want_tokens.iter().map(|&v| v as u32).collect(), ..FreeRun::default() };
    let mut frames = vec![first];
    run.hidden_until_divergence.push(max_rel(&e.rt.normed_rows(1)?, &hidden[..h]));
    for i in 1..n {
        let fr = e.step(&[(id, user[(i - 1) * f..i * f].to_vec())])?.remove(0);
        if run.first_divergence.is_none() {
            run.hidden_until_divergence.push(max_rel(&e.rt.normed_rows(1)?, &hidden[i * h..(i + 1) * h]));
        }
        frames.push(fr);
        if run.first_divergence.is_none() && frames[i - 1].token != run.want_tokens[i - 1] {
            run.first_divergence = Some(i - 1);
        }
    }
    e.end(id)?;
    if run.first_divergence.is_none() && frames.last().map(|fr| fr.token) != run.want_tokens.last().copied() {
        run.first_divergence = Some(n - 1);
    }
    run.tokens = frames.iter().map(|fr| fr.token).collect();
    run.text = frames.iter().map(|fr| fr.text.as_str()).collect::<String>().trim().to_string();
    run.audio = frames.iter().flat_map(|fr| fr.audio.iter().copied()).collect();
    let (a, b) = (voiced(&run.audio, f), voiced(&want_audio, f));
    run.voiced_frames = a.iter().filter(|v| **v).count();
    run.want_voiced_frames = b.iter().filter(|v| **v).count();
    run.voiced_agreement = a.iter().zip(&b).filter(|(x, y)| x == y).count() as f64 / a.len().max(1) as f64;
    Ok(run)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PromptCheck {
    pub tokens_equal: bool,
    pub masks_equal: bool,
    pub engine_len: usize,
    pub reference_len: usize,
    pub reference_samples: [usize; 2],
    pub reference_audio: f32,
    pub reference_mean: f32,
    pub reference_mean_worst_frame: usize,
    pub reference_mean_but_last: f32,
    pub silence_rows: f32,
}

pub fn prompt_check(e: &Engine, t: &Trace) -> Result<PromptCheck> {
    let m = t.meta();
    let card = match m["card"].as_str() {
        Some(p) => std::fs::read_to_string(p)?.trim().to_string(),
        None => String::new(),
    };
    let reference = match m["voice"].as_str() {
        Some(p) => Some(super::wav::read_wav(&std::fs::read(p)?)?.0),
        None => None,
    };
    let speak_first = m["speak_first"].as_bool().unwrap_or(false);
    let ours = e.prompt_for(&StartOptions { card: card.clone(), reference: reference.clone(), speak_first, ..StartOptions::default() })?;
    let want = t.prompt()?;
    let mut c = PromptCheck {
        tokens_equal: ours.tokens == want.tokens,
        masks_equal: ours.model_mask == want.model_mask && ours.user_mask == want.user_mask,
        engine_len: ours.len(),
        reference_len: want.len(),
        ..PromptCheck::default()
    };
    if let Some(r) = &reference {
        let trimmed = super::prompt::trim_reference(&e.duplex, r);
        let want_ref = t.get("prompt.reference")?;
        c.reference_samples = [trimmed.len(), want_ref.len()];
        if trimmed.len() == want_ref.len() {
            c.reference_audio = max_rel(&trimmed, &want_ref);
        }
        let mean = e.encode_clip(&trimmed)?;
        let want_mean = t.get("prompt.ref_mean")?;
        if mean.len() == want_mean.len() {
            c.reference_mean = max_rel(&mean, &want_mean);
            let d = e.duplex.latent_dim;
            let per: Vec<f32> = mean.chunks(d).zip(want_mean.chunks(d)).map(|(a, b)| max_rel(a, b)).collect();
            c.reference_mean_worst_frame = per.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |m| m.0);
            c.reference_mean_but_last = max_rel(&mean[..mean.len() - d], &want_mean[..mean.len() - d]);
        }
    }
    let d = e.duplex.latent_dim;
    let rows: Vec<usize> = (0..want.len()).filter(|&i| want.user_mask[i] > 0.0).collect();
    for r in rows {
        c.silence_rows = c.silence_rows.max(max_rel(&ours.user_latents[r * d..(r + 1) * d], &want.user_latents[r * d..(r + 1) * d]));
    }
    Ok(c)
}

pub fn markdown(r: &Report, layers: &[usize]) -> String {
    let s = &r.summary;
    let mut out = String::new();
    out.push_str(&format!("frames {} | top 1 {:.4} | KL top 10 mean {:.2e} max {:.2e} | hidden cos min {:.6} rel max {:.2e} mean {:.2e} | latent rel mean {:.2e} max {:.2e} | audio rel {:.2e} cos {:.5} | user latent {:.2e} | first off floor {:?}\n\n", s.frames, s.top1_rate, s.kl_mean, s.kl_max, s.hidden_cos_min, s.hidden_rel_max, s.hidden_rel_mean, s.latent_rel_mean, s.latent_rel_max, s.audio_rel, s.audio_cos, s.user_latent_max, s.first_off_floor));
    out.push_str("| layer | max rel | min cos |\n|---|---|---|\n");
    for (k, l) in layers.iter().enumerate() {
        out.push_str(&format!("| {l} | {:.2e} | {:.6} |\n", s.layer_rel_max[k], s.layer_cos_min[k]));
    }
    out
}
