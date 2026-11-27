pub mod plan;
pub mod service;
pub mod text;
pub mod voices;

use super::decoder::{HostDecoder, Sequence};
use crate::flow_head::{FlowHead, HostHead};
use crate::spec::{Spec, Tts};
use crate::weights::Checkpoint;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::path::Path;

pub fn is_tts_config(config: &str) -> bool {
    Spec::from_hf_config_tts(config).is_ok()
}

pub fn sine_embedding(position: f64, dim: usize, max_period: f64) -> Result<Vec<f32>> {
    ensure!(dim >= 4 && dim.is_multiple_of(2), "a sinusoidal condition needs an even width of at least 4, not {dim}");
    let half = dim / 2;
    let phase: Vec<f64> = (0..half).map(|i| position / max_period.powf(i as f64 / (half - 1) as f64)).collect();
    Ok(phase.iter().map(|p| p.cos() as f32).chain(phase.iter().map(|p| p.sin() as f32)).collect())
}

fn matvec(w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let k = x.len();
    (0..rows).map(|r| w[r * k..(r + 1) * k].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

pub fn conditions(tts: &Tts, ck: &Checkpoint, hidden: usize) -> Result<Vec<f32>> {
    let c = &tts.condition;
    let p = |n: &str| c.params.get(n).copied().with_context(|| format!("[tts.condition.params] has no {n}"));
    let lut_shape = ck.shape_of(&c.lut)?;
    let row = p("lut_row")? as usize;
    ensure!(lut_shape.len() == 2 && row < lut_shape[0], "{} has no row {row}", c.lut);
    let lut = ck.f32_values(&c.lut, &lut_shape)?;
    let lut_row = &lut[row * lut_shape[1]..(row + 1) * lut_shape[1]];
    let mut out = matvec(&ck.f32_values(&c.lut_proj, &ck.shape_of(&c.lut_proj)?)?, lut_row, hidden);
    let sine_shape = ck.shape_of(&c.sine_proj)?;
    let sine = sine_embedding(p("sine_scale")? * p("sine_value")?, sine_shape[1], p("sine_period")?)?;
    for (o, v) in out.iter_mut().zip(matvec(&ck.f32_values(&c.sine_proj, &sine_shape)?, &sine, hidden)) {
        *o += v;
    }
    Ok(out)
}

pub struct Voice {
    pub frames: usize,
    pub rows: Vec<f32>,
}

impl Voice {
    pub fn open(path: &Path, tts: &Tts, width: usize) -> Result<Voice> {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
        let st = safetensors::SafeTensors::deserialize(&bytes).with_context(|| path.display().to_string())?;
        let t = st.tensor(&tts.voice_tensor).with_context(|| format!("{} holds no {}", path.display(), tts.voice_tensor))?;
        let v: Vec<f32> = match t.dtype() {
            safetensors::Dtype::F32 => t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            safetensors::Dtype::BF16 => t.data().as_chunks::<2>().0.iter().map(|b| crate::weights::bf16_to_f32(u16::from_le_bytes(*b))).collect(),
            other => anyhow::bail!("{} is {other:?}, a voice is f32 or bf16", tts.voice_tensor),
        };
        Voice::from_tensor(&v, t.shape(), tts, width)
    }

    pub fn from_tensor(v: &[f32], shape: &[usize], tts: &Tts, width: usize) -> Result<Voice> {
        let (c, t) = match (tts.voice_layout.as_str(), shape) {
            ("channels_first", [1, c, t] | [c, t]) => (*c, *t),
            ("frames_first", [1, t, c] | [t, c]) => (*c, *t),
            (layout, _) => anyhow::bail!("a voice of shape {shape:?} does not fit the {layout} layout"),
        };
        ensure!(c == width, "the voice has {c} channels, the model reads {width}");
        let rows = match tts.voice_layout.as_str() {
            "channels_first" => (0..t).flat_map(|f| (0..c).map(move |ch| v[ch * t + f])).collect(),
            _ => v.to_vec(),
        };
        Ok(Voice { frames: t, rows })
    }
}

pub struct Frame {
    pub hidden: Vec<f32>,
    pub eos: f32,
    pub latent: Vec<f32>,
    pub codec: Vec<f32>,
}

pub struct HostTts {
    pub spec: Spec,
    pub tts: Tts,
    pub decoder: HostDecoder,
    pub head: HostHead,
    pub rules: plan::Rules,
    bos: Vec<f32>,
    latent_dim: usize,
    voice_dim: usize,
}

pub struct Session {
    seq: Sequence,
    head: crate::flow_head::HostSession,
    previous: Vec<f32>,
    pub frame: usize,
}

impl HostTts {
    pub fn open(dir: &Path) -> Result<HostTts> {
        let json = std::fs::read_to_string(dir.join("config.json")).with_context(|| dir.display().to_string())?;
        let spec = Spec::from_hf_config_tts(&json)?;
        let tts = spec.tts.clone().context("no [tts]")?;
        let raw = Checkpoint::open(dir)?;
        let hidden = spec.dims.hidden;
        let mut extra = BTreeMap::new();
        extra.insert(tts.conditions.clone(), conditions(&tts, &raw, hidden)?);
        let bos = raw.f32_values(&tts.bos, &raw.shape_of(&tts.bos)?)?;
        let ck = raw.renamed(&spec.tensors);
        let decoder = HostDecoder::load_with(spec.clone(), &ck, false, extra)?;
        let head = HostHead::load(FlowHead::from_dir(dir, None)?, dir)?;
        let inputs = spec.inputs.as_ref().context("no [inputs]")?;
        let latent_dim = inputs.width_of(&tts.latent).context("[tts] latent is not an input stream")?;
        let voice_dim = inputs.width_of(&tts.voice).context("[tts] voice is not an input stream")?;
        ensure!(head.head.input_width("noise")? == latent_dim && head.head.input_width("hidden")? == hidden, "the flow head reads {} wide latents and {} wide hidden rows, the backbone gives {latent_dim} and {hidden}", head.head.input_width("noise")?, head.head.input_width("hidden")?);
        ensure!(bos.len() == latent_dim, "{} has {} values for a {latent_dim} wide latent", tts.bos, bos.len());
        let rules = plan::Rules::from_spec(&tts)?;
        Ok(HostTts { spec, tts, decoder, head, rules, bos, latent_dim, voice_dim })
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    fn streams(&self, rows: &[(Option<&[f32]>, Option<&[f32]>)]) -> BTreeMap<String, Vec<f32>> {
        let n = rows.len();
        let (mut latent, mut latent_mask) = (vec![0.0; n * self.latent_dim], vec![0.0; n]);
        let (mut voice, mut voice_mask) = (vec![0.0; n * self.voice_dim], vec![0.0; n]);
        for (r, (l, v)) in rows.iter().enumerate() {
            if let Some(l) = l {
                latent[r * self.latent_dim..(r + 1) * self.latent_dim].copy_from_slice(l);
                latent_mask[r] = 1.0;
            }
            if let Some(v) = v {
                voice[r * self.voice_dim..(r + 1) * self.voice_dim].copy_from_slice(v);
                voice_mask[r] = 1.0;
            }
        }
        let t = &self.tts;
        BTreeMap::from([(t.latent.clone(), latent), (t.latent_mask.clone(), latent_mask), (t.voice.clone(), voice), (t.voice_mask.clone(), voice_mask)])
    }

    pub fn prompt(&self, voice: Option<&Voice>, text: &[u32]) -> Result<(Vec<u32>, BTreeMap<String, Vec<f32>>)> {
        let t = &self.tts;
        let mut tokens = Vec::new();
        let mut rows: Vec<(Option<&[f32]>, Option<&[f32]>)> = Vec::new();
        if let Some(v) = voice {
            for f in 0..v.frames {
                tokens.push(t.token("prompt")?);
                rows.push((None, Some(&v.rows[f * self.voice_dim..(f + 1) * self.voice_dim])));
            }
        }
        for &id in text {
            ensure!((id as usize) < self.spec.dims.vocab, "token {id} is past the {} entry table: the tokenizer is not this model's", self.spec.dims.vocab);
            tokens.push(id);
            rows.push((None, None));
        }
        tokens.push(t.token("bos")?);
        rows.push((Some(&self.bos), None));
        Ok((tokens, self.streams(&rows)))
    }

    pub fn start(&self, voice: Option<&Voice>, text: &[u32]) -> Result<(Session, Vec<f32>, f32)> {
        let (tokens, streams) = self.prompt(voice, text)?;
        let mut seq = self.decoder.new_sequence();
        let out = self.decoder.prefill_with(&mut seq, &tokens, streams)?;
        self.session_from(seq, &out)
    }

    pub fn voice_prefix(&self, voice: &Voice) -> Result<Sequence> {
        let (tokens, streams) = self.prompt(Some(voice), &[])?;
        let n = tokens.len() - 1;
        let streams = streams.into_iter().map(|(k, v)| {
            let w = v.len() / tokens.len();
            (k, v[..n * w].to_vec())
        }).collect();
        let mut seq = self.decoder.new_sequence();
        self.decoder.prefill_with(&mut seq, &tokens[..n], streams)?;
        Ok(seq)
    }

    pub fn start_after(&self, prefix: &Sequence, text: &[u32]) -> Result<(Session, Vec<f32>, f32)> {
        let (tokens, streams) = self.prompt(None, text)?;
        let mut seq = prefix.clone();
        let out = self.decoder.prefill_with(&mut seq, &tokens, streams)?;
        self.session_from(seq, &out)
    }

    fn session_from(&self, seq: Sequence, out: &super::decoder::Out) -> Result<(Session, Vec<f32>, f32)> {
        let h = self.spec.dims.hidden;
        let last = out.hidden[out.hidden.len() - h..].to_vec();
        let eos = *out.logits.last().context("the prefill wrote no head row")?;
        let head = self.head.session(&[1.0])?;
        Ok((Session { seq, head, previous: Vec::new(), frame: 0 }, last, eos))
    }

    pub fn prefill_hidden(&self, voice: Option<&Voice>, text: &[u32]) -> Result<Vec<f32>> {
        let (tokens, streams) = self.prompt(voice, text)?;
        let mut seq = self.decoder.new_sequence();
        Ok(self.decoder.prefill_with(&mut seq, &tokens, streams)?.hidden)
    }

    pub fn sample(&self, s: &mut Session, hidden: &[f32], eos: f32, noise: &[f32]) -> Result<Frame> {
        ensure!(noise.len() == self.latent_dim, "{} noise values for a {} wide latent", noise.len(), self.latent_dim);
        let latent = self.head.frame(&mut s.head, hidden, noise)?;
        let codec = self.head.output(&s.head, "codec")?.to_vec();
        s.previous = latent.clone();
        s.frame += 1;
        Ok(Frame { hidden: hidden.to_vec(), eos, latent, codec })
    }

    pub fn advance(&self, s: &mut Session, latent: &[f32]) -> Result<(Vec<f32>, f32)> {
        ensure!(latent.len() == self.latent_dim, "{} latent values for a {} wide latent", latent.len(), self.latent_dim);
        let streams = self.streams(&[(Some(latent), None)]);
        let out = self.decoder.step_with(&mut [&mut s.seq], &[self.tts.token("latent")?], streams)?;
        Ok((out.hidden, out.logits[0]))
    }

    pub fn noise_scale(&self, temperature: f32) -> f32 {
        temperature.max(0.0).sqrt()
    }

    pub fn generate(&self, voice: Option<&Voice>, text: &[u32], tail: usize, budget: usize, mut noise: impl FnMut(usize) -> Vec<f32>) -> Result<Vec<Frame>> {
        let (mut s, mut hidden, mut eos) = self.start(voice, text)?;
        let mut rule = self.rules.eos(tail);
        let mut frames = Vec::new();
        for f in 0..budget {
            let frame = self.sample(&mut s, &hidden, eos, &noise(f))?;
            let stop = rule.should_stop(frame.eos);
            let latent = frame.latent.clone();
            frames.push(frame);
            if stop {
                break;
            }
            (hidden, eos) = self.advance(&mut s, &latent)?;
        }
        Ok(frames)
    }
}
