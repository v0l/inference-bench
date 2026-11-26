use super::decode::{Decision, Options, Rules, Segment, segments};
use super::decoder::{HostDecoder, Sequence};
use super::host::{HostEncoder, HostSpeech};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::path::Path;

pub struct HostTranscriber {
    pub speech: HostSpeech,
    pub encoder: HostEncoder,
    pub decoder: HostDecoder,
    pub rules: Rules,
    ck: crate::weights::Checkpoint,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Attempt {
    pub temperature: f64,
    pub tokens: Vec<u32>,
    pub decision: Decision,
    #[serde(skip)]
    pub scores: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Window {
    pub seek: usize,
    pub decoder_input: Vec<u32>,
    pub attempts: Vec<Attempt>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Transcript {
    pub language: Option<u32>,
    pub sequence: Vec<u32>,
    pub segments: Vec<Segment>,
    pub windows: Vec<Window>,
}

pub fn log_mel_clip(plan: &super::SpeechPlan, pcm: &[f32]) -> Result<Vec<f32>> {
    let mut p = plan.clone();
    p.mel.samples = pcm.len().max(plan.mel.samples);
    p.mel.frames = p.mel.samples / p.mel.hop;
    super::host::log_mel(&p, pcm)
}

pub type Picker<'a> = dyn FnMut(usize, usize, usize, &[f32], f64) -> Option<u32> + 'a;

impl HostTranscriber {
    pub fn open(dir: &Path, bf16: bool) -> Result<HostTranscriber> {
        let speech = HostSpeech::open(dir)?;
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let encoder = HostEncoder::new(&config)?;
        let decoder = HostDecoder::open(dir, bf16)?;
        let decode = decoder.spec.speech_decode.clone().with_context(|| format!("the {} family declares no [speech.decode] section", decoder.spec.name))?;
        let rules = Rules::open(dir, &decode)?;
        ensure!(rules.vocab == decoder.spec.dims.vocab, "generation rules for {} tokens, the decoder has {}", rules.vocab, decoder.spec.dims.vocab);
        let ck = crate::weights::Checkpoint::open(dir)?;
        Ok(HostTranscriber { speech, encoder, decoder, rules, ck })
    }

    pub fn window_frames(&self) -> usize {
        self.speech.plan.mel.frames
    }

    pub fn stride(&self) -> usize {
        self.speech.plan.mel.frames / self.speech.plan.positions
    }

    pub fn feature_precision(&self) -> f64 {
        self.speech.plan.mel.hop as f64 / self.speech.plan.mel.rate
    }

    pub fn encode(&self, mel: &[f32], seek: usize, frames: usize) -> Result<Vec<f32>> {
        let bins = self.speech.plan.mel.bins;
        let mut window = vec![0f32; self.window_frames() * bins];
        window[..frames * bins].copy_from_slice(&mel[seek * bins..(seek + frames) * bins]);
        let stem = self.speech.stem(&window)?;
        self.encoder.encode(&self.ck, &stem)
    }

    fn sequence(&self, states: &[f32]) -> Result<Sequence> {
        let mut seq = self.decoder.new_sequence();
        self.decoder.window(&mut seq, states)?;
        Ok(seq)
    }

    pub fn detect(&self, states: &[f32]) -> Result<(u32, Vec<f32>)> {
        let mut seq = self.sequence(states)?;
        let logits = self.decoder.prefill(&mut seq, &[self.rules.sot])?;
        Ok((self.rules.detect(&logits), logits))
    }

    fn generate(&self, states: &[f32], prompt: &[u32], sot_at: usize, o: &Options, temperature: f64, draw: &mut u64, at: (usize, usize), picker: &mut Picker) -> Result<(Vec<u32>, Vec<Vec<f32>>, f64)> {
        let vocab = self.rules.vocab;
        let mut seq = self.sequence(states)?;
        let logits = self.decoder.prefill(&mut seq, prompt)?;
        let no_speech = self.rules.no_speech_prob(&logits[sot_at * vocab..(sot_at + 1) * vocab]);
        let total = (self.rules.max_length + prompt.len().min(self.rules.prev_cutoff)).min(self.rules.max_target);
        let budget = o.max_new_tokens.map_or(total.saturating_sub(prompt.len()), |m| m.min(self.rules.max_target - prompt.len()));
        let mut row = logits[(prompt.len() - 1) * vocab..].to_vec();
        let (mut tokens, mut scores) = (Vec::new(), Vec::new());
        while tokens.len() < budget {
            let mut processed = self.rules.process(&row, &tokens, o.timestamps);
            self.rules.warp(&mut processed, temperature);
            let next = match picker(at.0, at.1, tokens.len(), &processed, temperature) {
                Some(t) => t,
                None => self.rules.pick(&processed, temperature, o.seed, *draw),
            };
            *draw += 1;
            scores.push(processed);
            tokens.push(next);
            if next == self.rules.eot || tokens.len() == budget {
                break;
            }
            row = self.decoder.step(&mut [&mut seq], &[next])?;
        }
        Ok((tokens, scores, no_speech))
    }

    pub fn transcribe(&self, pcm: &[f32], o: &Options) -> Result<Transcript> {
        let mel = log_mel_clip(&self.speech.plan, pcm)?;
        self.transcribe_mel(&mel, o, &mut |_, _, _, _, _| None)
    }

    pub fn transcribe_mel(&self, mel: &[f32], o: &Options, picker: &mut Picker) -> Result<Transcript> {
        let bins = self.speech.plan.mel.bins;
        let total = mel.len() / bins;
        let span = self.window_frames();
        ensure!(o.timestamps || total <= span, "audio over one window needs timestamps to seek");
        let mut out = Transcript::default();
        let first = self.encode(mel, 0, total.min(span))?;
        if self.rules.needs_detection(o) {
            out.language = Some(self.detect(&first)?.0);
        }
        let init = self.rules.init_tokens(o, out.language)?;
        let (mut seek, mut condition, mut draw) = (0usize, o.condition_on_prev, 0u64);
        let mut cached = Some(first);
        let mut done: Vec<Vec<u32>> = Vec::new();
        while seek < total {
            let frames = (total - seek).min(span);
            let states = match cached.take() {
                Some(s) => s,
                None => self.encode(mel, seek, frames)?,
            };
            let mut prompt = Vec::new();
            if condition && !done.is_empty() {
                prompt = self.rules.condition_tokens(&done)?;
            }
            let sot_at = prompt.len();
            prompt.extend_from_slice(&init);
            let mut window = Window { seek, decoder_input: prompt.clone(), attempts: Vec::new() };
            let mut chosen = None;
            for (i, &t) in o.temperatures.iter().enumerate() {
                let (tokens, scores, no_speech) = self.generate(&states, &prompt, sot_at, o, t, &mut draw, (out.windows.len(), i), picker)?;
                let decision = self.rules.decide(o, &tokens, &scores, t, Some(no_speech))?;
                let last = i + 1 == o.temperatures.len();
                let stop = !decision.needs_fallback || last;
                window.attempts.push(Attempt { temperature: t, tokens, decision, scores });
                if stop {
                    chosen = Some(window.attempts.len() - 1);
                    break;
                }
            }
            let a = &window.attempts[chosen.context("no temperatures to decode with")?];
            condition = o.condition_on_prev && a.temperature < self.rules.decode.condition_below;
            let skip = a.decision.should_skip;
            let mut tokens = a.tokens.clone();
            if tokens.last() == Some(&self.rules.eot) {
                tokens.pop();
            }
            out.windows.push(window);
            if skip {
                seek += frames;
                continue;
            }
            let offset_time = seek as f64 * self.rules.decode.time_precision / self.stride() as f64;
            if !o.timestamps {
                out.sequence.extend_from_slice(&tokens);
                out.segments.push(Segment { start: offset_time, end: offset_time + frames as f64 * self.feature_precision(), tokens: tokens.clone() });
                done.push(tokens);
                seek += frames;
                continue;
            }
            let (segs, advance) = segments(&self.rules, &tokens, offset_time, frames, self.stride(), self.feature_precision())?;
            for s in &segs {
                out.sequence.extend_from_slice(&s.tokens);
                done.push(s.tokens.clone());
            }
            out.segments.extend(segs);
            seek += advance.max(1);
        }
        Ok(out)
    }
}
