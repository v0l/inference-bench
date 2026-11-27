use crate::duplex::sampling::RowPlan;
use crate::family::{SpeechDecode, read_params};
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    pub sot: u32,
    pub eot: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    pub no_speech: u32,
    pub prev: Option<u32>,
    pub vocab: usize,
    pub max_target: usize,
    pub prev_cutoff: usize,
    pub max_length: usize,
    pub max_initial_timestamp: Option<u32>,
    pub suppress: Vec<u32>,
    pub begin_suppress: Vec<u32>,
    pub languages: BTreeMap<String, u32>,
    pub tasks: BTreeMap<String, u32>,
    pub forced: Vec<(usize, Option<u32>)>,
    pub multilingual: bool,
    pub decode: SpeechDecode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub language: Option<String>,
    pub task: Option<String>,
    pub timestamps: bool,
    pub temperatures: Vec<f64>,
    pub compression_ratio_threshold: Option<f64>,
    pub logprob_threshold: Option<f64>,
    pub no_speech_threshold: Option<f64>,
    pub condition_on_prev: bool,
    pub max_new_tokens: Option<usize>,
    pub seed: u64,
    pub prompt: Vec<u32>,
}

impl Options {
    pub fn greedy() -> Options {
        Options { language: None, task: None, timestamps: false, temperatures: vec![0.0], compression_ratio_threshold: None, logprob_threshold: None, no_speech_threshold: None, condition_on_prev: false, max_new_tokens: None, seed: 0, prompt: Vec::new() }
    }

    pub fn from_rules(r: &Rules) -> Options {
        let d = &r.decode;
        Options { temperatures: d.temperatures.clone(), compression_ratio_threshold: d.compression_ratio_threshold, logprob_threshold: d.logprob_threshold, no_speech_threshold: d.no_speech_threshold, condition_on_prev: true, timestamps: true, ..Options::greedy() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    pub compression_ratio: f64,
    pub avg_logprob: f64,
    pub no_speech_prob: Option<f64>,
    pub needs_fallback: bool,
    pub should_skip: bool,
}

fn ids(v: &serde_json::Value, key: &str) -> Result<Vec<u32>> {
    match v.get(key) {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(a) => a.as_array().with_context(|| format!("{key} is not a list"))?.iter().map(|t| t.as_u64().map(|t| t as u32).with_context(|| format!("{key} holds a non token id"))).collect(),
    }
}

fn table(v: &serde_json::Value, key: &str) -> Result<BTreeMap<String, u32>> {
    let Some(m) = v.get(key).and_then(|m| m.as_object()) else { return Ok(BTreeMap::new()) };
    m.iter().map(|(k, t)| Ok((k.clone(), t.as_u64().with_context(|| format!("{key}.{k} is not a token id"))? as u32))).collect()
}

fn read_json(p: &Path) -> Result<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?).with_context(|| p.display().to_string())
}

impl Rules {
    pub fn open(dir: &Path, decode: &SpeechDecode) -> Result<Rules> {
        let config = read_json(&dir.join("config.json"))?;
        let gen_path = dir.join("generation_config.json");
        let generation = if gen_path.exists() { read_json(&gen_path)? } else { serde_json::json!({}) };
        Rules::from_configs(&config, &generation, decode)
    }

    pub fn from_configs(config: &serde_json::Value, generation: &serde_json::Value, decode: &SpeechDecode) -> Result<Rules> {
        let mut merged = config.as_object().cloned().unwrap_or_default();
        for (k, v) in generation.as_object().into_iter().flatten() {
            merged.insert(k.clone(), v.clone());
        }
        let merged = serde_json::Value::Object(merged);
        let p = read_params(&decode.params, &merged).context("[speech.decode.params]")?;
        let get = |k: &str| p.get(k).copied().with_context(|| format!("[speech.decode.params] has no {k}"));
        let whole = |k: &str| -> Result<Option<u32>> {
            let v = get(k)?;
            if v < 0.0 {
                return Ok(None);
            }
            ensure!(v.fract() == 0.0, "[speech.decode.params] {k} = {v} is not a whole number");
            Ok(Some(v as u32))
        };
        let need = |k: &str| whole(k)?.with_context(|| format!("[speech.decode.params] {k} is negative"));
        let forced = match merged.get(&decode.forced).and_then(|f| f.as_array()) {
            None => Vec::new(),
            Some(rows) => rows.iter().map(|r| {
                let pos = r.get(0).and_then(|p| p.as_u64()).context("a forced decoder id has no position")? as usize;
                Ok((pos, r.get(1).and_then(|t| t.as_u64()).map(|t| t as u32)))
            }).collect::<Result<_>>()?,
        };
        let vocab = merged.get("vocab_size").and_then(|v| v.as_u64()).context("config.json has no vocab_size")? as usize;
        let r = Rules {
            sot: need("sot")?,
            eot: need("eot")?,
            no_timestamps: need("no_timestamps")?,
            timestamp_begin: need("timestamp_begin")?,
            no_speech: need("no_speech")?,
            prev: whole("prev")?,
            vocab,
            max_target: need("max_target")? as usize,
            prev_cutoff: need("prev_cutoff")? as usize,
            max_length: need("max_length")? as usize,
            max_initial_timestamp: whole("max_initial_timestamp")?,
            suppress: ids(&merged, &decode.suppress)?,
            begin_suppress: ids(&merged, &decode.begin_suppress)?,
            languages: table(&merged, &decode.languages)?,
            tasks: table(&merged, &decode.tasks)?,
            forced,
            multilingual: merged.get(&decode.multilingual).and_then(|m| m.as_bool()).unwrap_or(false),
            decode: decode.clone(),
        };
        ensure!((r.timestamp_begin as usize) < r.vocab && (r.eot as usize) < r.vocab, "the timestamp and end tokens lie outside the vocab of {}", r.vocab);
        Ok(r)
    }

    pub fn language_id(&self, language: &str) -> Result<u32> {
        let l = language.to_lowercase();
        let key = if l.starts_with("<|") { l.clone() } else { format!("<|{l}|>") };
        self.languages.get(&key).copied().with_context(|| format!("language {language} is not one this model knows ({} languages)", self.languages.len()))
    }

    pub fn language_row(&self) -> RowPlan {
        RowPlan { edits: Vec::new(), allow: self.languages.values().copied().collect() }
    }

    pub fn forced_row(token: u32) -> RowPlan {
        RowPlan { edits: Vec::new(), allow: vec![token] }
    }

    pub fn needs_detection(&self, o: &Options) -> bool {
        o.language.is_none() && !self.languages.is_empty() && {
            let t = self.forced_prefix(o);
            t.len() <= 1 || t[1].is_none()
        }
    }

    fn forced_prefix(&self, o: &Options) -> Vec<Option<u32>> {
        let mut init = vec![Some(self.sot)];
        if o.task.is_none() && o.language.is_none() && self.forced.first().is_some_and(|f| f.0 == 1) {
            for (i, (pos, t)) in self.forced.iter().enumerate() {
                if *pos != i + 1 {
                    break;
                }
                init.push(*t);
            }
        }
        init
    }

    pub fn init_tokens(&self, o: &Options, detected: Option<u32>) -> Result<Vec<u32>> {
        ensure!(self.multilingual || (o.task.is_none() && o.language.is_none()), "an English only model takes no language or task");
        let mut init = self.forced_prefix(o);
        let lang = match &o.language {
            Some(l) => Some(self.language_id(l)?),
            None if self.needs_detection(o) => Some(detected.context("the language must be detected first")?),
            None => None,
        };
        if let Some(l) = lang {
            if init.len() > 1 {
                init[1] = Some(l);
            } else {
                init.push(Some(l));
            }
        }
        let task_ids: BTreeSet<u32> = self.tasks.values().copied().collect();
        if let Some(task) = &o.task {
            let id = *self.tasks.get(task).with_context(|| format!("task {task} is not one of {:?}", self.tasks.keys()))?;
            if init.iter().flatten().any(|t| task_ids.contains(t)) {
                init.iter_mut().filter(|t| t.is_some_and(|t| task_ids.contains(&t))).for_each(|t| *t = Some(id));
            } else {
                init.push(Some(id));
            }
        } else if o.language.is_some() && !self.tasks.is_empty() && !init.iter().flatten().any(|t| task_ids.contains(t)) {
            init.push(Some(*self.tasks.get("transcribe").context("the task table has no transcribe")?));
        }
        if !o.timestamps && init.last() != Some(&Some(self.no_timestamps)) {
            init.push(Some(self.no_timestamps));
        } else if o.timestamps && init.last() == Some(&Some(self.no_timestamps)) {
            init.pop();
        }
        Ok(init.into_iter().flatten().collect())
    }

    pub fn row(&self, generated: &[u32], timestamps: bool) -> RowPlan {
        let tb = self.timestamp_begin;
        let mut bans: BTreeSet<u32> = self.suppress.iter().copied().collect();
        if generated.is_empty() {
            bans.extend(&self.begin_suppress);
        }
        let (mut lo, mut hi) = (0u32, self.vocab as u32);
        if timestamps {
            bans.insert(self.no_timestamps);
            let n = generated.len();
            let last = n >= 1 && generated[n - 1] >= tb;
            let penultimate = n < 2 || generated[n - 2] >= tb;
            if last {
                if penultimate {
                    hi = hi.min(tb);
                } else {
                    lo = lo.max(self.eot);
                }
            }
            let stamps: Vec<u32> = generated.iter().copied().filter(|t| *t >= tb).collect();
            let floor = match stamps.last() {
                None => tb,
                Some(&t) if last && !penultimate => t,
                Some(&t) => t + 1,
            };
            bans.extend(tb..floor.min(self.vocab as u32));
            if generated.is_empty() {
                lo = lo.max(tb);
                if let Some(m) = self.max_initial_timestamp {
                    hi = hi.min(tb + m + 1);
                }
            }
        }
        let allow = if (lo, hi) == (0, self.vocab as u32) { Vec::new() } else { (lo..hi.max(lo)).filter(|t| !bans.contains(t)).collect() };
        let allow = if allow.is_empty() && (lo, hi) != (0, self.vocab as u32) { vec![lo.min(self.vocab as u32 - 1)] } else { allow };
        RowPlan { edits: bans.into_iter().map(|t| (t, f32::NEG_INFINITY)).collect(), allow }
    }

    pub fn timestamp_rule(&self, row: &mut [f32]) {
        let tb = self.timestamp_begin as usize;
        let lp = log_softmax(row);
        let stamps = logsumexp(&lp[tb..]);
        let text = lp[..tb].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if stamps > text {
            row[..tb].iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
        }
    }

    pub fn process(&self, row: &[f32], generated: &[u32], timestamps: bool) -> Vec<f32> {
        let mut out = row.to_vec();
        apply(&mut out, &self.row(generated, timestamps));
        if timestamps {
            self.timestamp_rule(&mut out);
        }
        out
    }

    pub fn detect(&self, sot_logits: &[f32]) -> u32 {
        let mut row = sot_logits.to_vec();
        apply(&mut row, &self.language_row());
        argmax(&row)
    }

    pub fn no_speech_prob(&self, sot_logits: &[f32]) -> f64 {
        log_softmax(sot_logits)[self.no_speech as usize].exp()
    }

    pub fn compression_ratio(&self, tokens: &[u32]) -> Result<f64> {
        let width = ((self.vocab as f64).log2() / 8.0) as usize + 1;
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| u64::from(*t).to_le_bytes().into_iter().take(width)).collect();
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
        z.write_all(&bytes)?;
        Ok(bytes.len() as f64 / z.finish()?.len() as f64)
    }

    pub fn avg_logprob(scores: &[Vec<f32>], tokens: &[u32], temperature: f64) -> f64 {
        let scale = if temperature > 0.0 { temperature } else { 1.0 };
        let (scores, tokens) = if scores.len() > tokens.len() { (&scores[..tokens.len()], tokens) } else { (scores, &tokens[tokens.len() - scores.len()..]) };
        let sum: f64 = scores.iter().zip(tokens).map(|(s, t)| {
            let scaled: Vec<f32> = s.iter().map(|v| (f64::from(*v) * scale) as f32).collect();
            log_softmax(&scaled)[*t as usize]
        }).sum();
        sum / tokens.len() as f64
    }

    pub fn decide(&self, o: &Options, tokens: &[u32], scores: &[Vec<f32>], temperature: f64, no_speech_prob: Option<f64>) -> Result<Decision> {
        let compression_ratio = self.compression_ratio(tokens)?;
        let avg_logprob = Rules::avg_logprob(scores, tokens, temperature);
        let mut needs_fallback = o.compression_ratio_threshold.is_some_and(|t| compression_ratio > t);
        let low = o.logprob_threshold.is_some_and(|t| avg_logprob < t);
        needs_fallback |= low;
        let mut should_skip = false;
        if let Some(t) = o.no_speech_threshold {
            ensure!(o.logprob_threshold.is_some(), "a no speech threshold needs a log probability threshold");
            let p = no_speech_prob.context("no speech probability was not measured")?;
            if low && p > t {
                needs_fallback = false;
                should_skip = true;
            }
        }
        Ok(Decision { compression_ratio, avg_logprob, no_speech_prob, needs_fallback, should_skip })
    }

    pub fn warp(&self, row: &mut [f32], temperature: f64) {
        if temperature <= 0.0 {
            return;
        }
        if temperature != 1.0 {
            row.iter_mut().for_each(|v| *v = (f64::from(*v) / temperature) as f32);
        }
        let k = self.decode.top_k;
        if k > 0 && k < row.len() {
            let mut sorted: Vec<f32> = row.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let kth = sorted[k - 1];
            row.iter_mut().filter(|v| **v < kth).for_each(|v| *v = f32::NEG_INFINITY);
        }
    }

    pub fn pick(&self, row: &[f32], temperature: f64, seed: u64, draw: u64) -> u32 {
        if temperature <= 0.0 {
            return argmax(row);
        }
        let lp = log_softmax(row);
        let u = crate::sampler::uniform(seed, draw);
        let mut acc = 0.0;
        for (i, l) in lp.iter().enumerate() {
            acc += l.exp();
            if u < acc {
                return i as u32;
            }
        }
        argmax(row)
    }

    pub fn condition_tokens(&self, segments: &[Vec<u32>]) -> Result<Vec<u32>> {
        let prev = self.prev.context("the generation config names no previous text token")?;
        let tb = self.timestamp_begin;
        let mut all: Vec<u32> = Vec::new();
        for s in segments {
            if s.len() > 2 && s[s.len() - 2] >= tb {
                all.extend_from_slice(&s[..s.len() - 1]);
            } else {
                all.extend_from_slice(s);
            }
        }
        let cut = all.len().saturating_sub(self.prev_cutoff);
        let mut out = vec![prev];
        out.extend_from_slice(&all[cut..]);
        Ok(out)
    }
}

pub fn apply(row: &mut [f32], plan: &RowPlan) {
    for &(t, a) in &plan.edits {
        let v = &mut row[t as usize];
        *v = if a == f32::NEG_INFINITY { a } else { *v + a };
    }
    if !plan.allow.is_empty() {
        let keep: BTreeSet<u32> = plan.allow.iter().copied().collect();
        for (i, v) in row.iter_mut().enumerate() {
            if !keep.contains(&(i as u32)) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

pub fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

pub fn log_softmax(row: &[f32]) -> Vec<f64> {
    let v: Vec<f64> = row.iter().map(|x| f64::from(*x)).collect();
    let z = logsumexp(&v);
    v.iter().map(|x| x - z).collect()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub tokens: Vec<u32>,
}

pub fn segments(r: &Rules, tokens: &[u32], time_offset: f64, seek_frames: usize, frames_per_position: usize, feature_precision: f64) -> Result<(Vec<Segment>, usize)> {
    let tb = r.timestamp_begin;
    let tp = r.decode.time_precision;
    let is_ts: Vec<bool> = tokens.iter().map(|t| *t >= tb).collect();
    let single_ending = is_ts.len() >= 2 && is_ts[is_ts.len() - 2..] == [false, true];
    let mut slices: Vec<usize> = (0..is_ts.len().saturating_sub(1)).filter(|&i| is_ts[i] && is_ts[i + 1]).map(|i| i + 1).collect();
    if slices.is_empty() {
        let mut end = (seek_frames as f64 * feature_precision / tp).trunc();
        if let Some(i) = is_ts.iter().rposition(|t| *t)
            && tokens[i] != tb
        {
            end = f64::from(tokens[i] - tb);
        }
        return Ok((vec![Segment { start: time_offset, end: time_offset + end * tp, tokens: tokens.to_vec() }], seek_frames));
    }
    if single_ending {
        slices.push(tokens.len());
    } else {
        *slices.last_mut().unwrap() += 1;
    }
    let mut out = Vec::new();
    let mut last = 0;
    for (i, &cur) in slices.iter().enumerate() {
        let is_last = i == slices.len() - 1;
        let part = &tokens[last..cur];
        ensure!(!part.is_empty(), "an empty segment at {last}");
        let start = f64::from(part[0].checked_sub(tb).context("a segment starts with text")?);
        let end_tok = if !is_last || single_ending { part[part.len() - 1] } else { part[part.len() - 2] };
        let end = f64::from(end_tok.checked_sub(tb).context("a segment ends with text")?);
        out.push(Segment { start: time_offset + start * tp, end: time_offset + end * tp, tokens: part.to_vec() });
        last = cur;
    }
    let offset = if single_ending {
        seek_frames
    } else {
        let Some(t) = last.checked_sub(2).map(|i| tokens[i]) else { bail!("no timestamp to seek to") };
        (t - tb) as usize * frames_per_position
    };
    Ok((out, offset))
}
