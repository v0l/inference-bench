use super::decode::Options;
use super::transcribe::{Engine, HostTranscriber, Transcriber, Transcript};
use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{Sender, channel};

pub fn is_speech_config(config: &str) -> bool {
    crate::spec::Spec::from_hf_config_speech(config).is_ok_and(|s| s.speech_decode.is_some())
}

pub fn resolve_dir(model: &str) -> Option<PathBuf> {
    let config = crate::weights::find_in_snapshot(model, "config.json").ok()?;
    is_speech_config(&std::fs::read_to_string(&config).ok()?).then(|| config.parent().map(Path::to_path_buf)).flatten()
}

pub fn resample(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || pcm.is_empty() || from == 0 || to == 0 {
        return pcm.to_vec();
    }
    let ratio = f64::from(from) / f64::from(to);
    let len = (pcm.len() as f64 / ratio).floor() as usize;
    if ratio > 1.0 {
        let half = ratio / 2.0;
        return (0..len)
            .map(|i| {
                let centre = (i as f64 + 0.5) * ratio;
                let lo = ((centre - half).floor().max(0.0) as usize).min(pcm.len() - 1);
                let hi = ((centre + half).ceil() as usize).clamp(lo + 1, pcm.len());
                pcm[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
            })
            .collect();
    }
    (0..len)
        .map(|i| {
            let at = i as f64 * ratio;
            let lo = (at as usize).min(pcm.len() - 1);
            let hi = (lo + 1).min(pcm.len() - 1);
            let t = (at - lo as f64) as f32;
            pcm[lo] * (1.0 - t) + pcm[hi] * t
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Request {
    pub language: Option<String>,
    pub prompt: String,
    pub temperature: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Segment {
    pub id: usize,
    pub seek: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Heard {
    pub text: String,
    pub language: Option<String>,
    pub duration: f64,
    pub segments: Vec<Segment>,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ServiceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub window_seconds: f64,
    pub device: String,
    pub multilingual: bool,
}

pub trait Transcribe {
    fn info(&self) -> ServiceInfo;
    fn transcribe(&mut self, pcm: &[f32], request: &Request) -> Result<Heard>;
}

pub struct Text {
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl Text {
    pub fn open(dir: &Path) -> Result<Text> {
        let path = dir.join("tokenizer.json");
        let tokenizer = match path.exists() {
            true => Some(tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("{}: {e}", path.display()))?),
            false => None,
        };
        Ok(Text { tokenizer })
    }

    pub fn decode(&self, tokens: &[u32], eot: u32) -> Result<String> {
        let words: Vec<u32> = tokens.iter().copied().filter(|t| *t < eot).collect();
        match &self.tokenizer {
            Some(t) => t.decode(&words, true).map_err(|e| anyhow!("tokenizer: {e}")),
            None => Ok(String::new()),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let t = self.tokenizer.as_ref().context("this model ships no tokenizer.json, so a prompt cannot be read")?;
        Ok(t.encode(format!(" {}", text.trim()), false).map_err(|e| anyhow!("tokenizer: {e}"))?.get_ids().to_vec())
    }
}

pub struct Hosted<E: Engine> {
    pub transcriber: Transcriber<E>,
    pub text: Text,
    pub name: String,
    pub device: String,
}

impl<E: Engine> Hosted<E> {
    pub fn options(&self, request: &Request, prompt: Vec<u32>, samples: usize) -> Options {
        let rules = &self.transcriber.rules;
        let language = request.language.clone().filter(|l| rules.multilingual && !l.trim().is_empty());
        Options {
            task: language.as_ref().map(|_| "transcribe".to_string()),
            language,
            timestamps: samples > self.transcriber.plan.mel.samples,
            temperatures: vec![request.temperature.unwrap_or(0.0).max(0.0)],
            prompt,
            ..Options::greedy()
        }
    }

    pub fn heard(&self, got: &Transcript, samples: usize) -> Result<Heard> {
        let rules = &self.transcriber.rules;
        let mut segments = Vec::new();
        for (s, &w) in got.segments.iter().zip(&got.segment_windows) {
            let window = &got.windows[w];
            let a = window.attempts.last().context("a window without an attempt")?;
            segments.push(Segment {
                id: segments.len(),
                seek: window.seek,
                start: s.start,
                end: s.end,
                text: self.text.decode(&s.tokens, rules.eot)?,
                tokens: s.tokens.clone(),
                temperature: a.temperature,
                avg_logprob: a.decision.avg_logprob,
                compression_ratio: a.decision.compression_ratio,
                no_speech_prob: a.decision.no_speech_prob.unwrap_or(0.0),
            });
        }
        let language = got.language.or_else(|| self.forced_language()).and_then(|id| rules.languages.iter().find(|(_, v)| **v == id).map(|(k, _)| k.trim_start_matches("<|").trim_end_matches("|>").to_string()));
        Ok(Heard {
            text: self.text.decode(&got.sequence, rules.eot)?.trim().to_string(),
            language: language.or_else(|| (!rules.multilingual).then(|| "en".to_string())),
            duration: samples as f64 / self.transcriber.plan.mel.rate,
            segments,
            tokens: got.sequence.clone(),
        })
    }

    fn forced_language(&self) -> Option<u32> {
        self.transcriber.rules.forced.iter().find(|f| f.0 == 1).and_then(|f| f.1)
    }

    pub fn run(&self, pcm: &[f32], request: &Request) -> Result<Heard> {
        let prompt = self.text.encode(&request.prompt)?;
        let o = self.options(request, prompt, pcm.len());
        let got = self.transcriber.transcribe(pcm, &o)?;
        let mut heard = self.heard(&got, pcm.len())?;
        if heard.language.is_none() {
            heard.language = o.language.clone();
        }
        Ok(heard)
    }

    pub fn describe(&self) -> ServiceInfo {
        let m = &self.transcriber.plan.mel;
        ServiceInfo { name: self.name.clone(), sample_rate: m.rate as u32, window_seconds: m.samples as f64 / m.rate, device: self.device.clone(), multilingual: self.transcriber.rules.multilingual }
    }
}

impl<E: Engine> Transcribe for Hosted<E> {
    fn info(&self) -> ServiceInfo {
        self.describe()
    }

    fn transcribe(&mut self, pcm: &[f32], request: &Request) -> Result<Heard> {
        self.run(pcm, request)
    }
}

fn family_name(dir: &Path) -> Result<String> {
    let text = std::fs::read_to_string(dir.join("config.json")).with_context(|| format!("{}/config.json", dir.display()))?;
    Ok(crate::spec::Spec::from_hf_config_speech(&text)?.name)
}

pub fn open_host(dir: &Path) -> Result<Box<dyn Transcribe>> {
    let transcriber = HostTranscriber::open(dir, false)?;
    Ok(Box::new(Hosted { transcriber, text: Text::open(dir)?, name: family_name(dir)?, device: "CPU".into() }))
}

pub struct GpuHosted {
    inner: Hosted<super::gpu::GpuEngine>,
    _jit: crate::jit::Jit,
}

impl Transcribe for GpuHosted {
    fn info(&self) -> ServiceInfo {
        self.inner.describe()
    }

    fn transcribe(&mut self, pcm: &[f32], request: &Request) -> Result<Heard> {
        self.inner.run(pcm, request)
    }
}

pub fn open_gpu(dir: &Path, device: usize) -> Result<Box<dyn Transcribe>> {
    let ctx = cudarc::driver::CudaContext::new(device)?;
    let jit = crate::jit::Jit::new(ctx.clone(), crate::jit::default_cache_dir())?;
    let transcriber = super::gpu::GpuTranscriber::open(&ctx.default_stream(), &jit, dir)?;
    let inner = Hosted { transcriber, text: Text::open(dir)?, name: family_name(dir)?, device: "GPU".into() };
    Ok(Box::new(GpuHosted { inner, _jit: jit }))
}

pub fn open(dir: &Path, device: usize) -> Result<Box<dyn Transcribe>> {
    match open_gpu(dir, device) {
        Ok(t) => Ok(t),
        Err(e) if cudarc::driver::CudaContext::new(device).is_err() => {
            tracing::warn!(error = %e, "no CUDA device for speech, transcribing on the CPU");
            open_host(dir)
        }
        Err(e) => Err(e),
    }
}

type Job = (Vec<f32>, Request, Sender<Result<Heard>>);

pub struct Service {
    jobs: Mutex<Option<Sender<Job>>>,
    pub info: ServiceInfo,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Service {
    pub fn spawn(make: impl FnOnce() -> Result<Box<dyn Transcribe>> + Send + 'static) -> Result<std::sync::Arc<Service>> {
        let (jobs, inbox) = channel::<Job>();
        let (ready_tx, ready) = channel();
        let thread = std::thread::Builder::new().name("speech-engine".into()).spawn(move || {
            let mut engine = match make() {
                Ok(e) => {
                    let _ = ready_tx.send(Ok(e.info()));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            while let Ok((pcm, request, reply)) = inbox.recv() {
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.transcribe(&pcm, &request)));
                let _ = reply.send(out.unwrap_or_else(|_| Err(anyhow!("the speech engine panicked"))));
            }
        })?;
        let info = ready.recv().map_err(|_| anyhow!("the speech engine stopped while loading"))??;
        Ok(std::sync::Arc::new(Service { jobs: Mutex::new(Some(jobs)), info, thread: Mutex::new(Some(thread)) }))
    }

    pub fn load(dir: &Path, device: usize) -> Result<std::sync::Arc<Service>> {
        let dir = dir.to_path_buf();
        let sink = crate::progress::current();
        Service::spawn(move || crate::progress::with_sink(sink, || open(&dir, device)))
    }

    pub fn load_on_host(dir: &Path) -> Result<std::sync::Arc<Service>> {
        let dir = dir.to_path_buf();
        Service::spawn(move || open_host(&dir))
    }

    pub fn transcribe(&self, pcm: Vec<f32>, rate: u32, request: Request) -> Result<Heard> {
        let pcm = resample(&pcm, rate, self.info.sample_rate);
        let (tx, rx) = channel();
        self.jobs.lock().unwrap().as_ref().context("the speech engine has stopped")?.send((pcm, request, tx)).map_err(|_| anyhow!("the speech engine has stopped"))?;
        rx.recv().map_err(|_| anyhow!("the speech engine has stopped"))?
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.jobs.lock().unwrap().take();
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_the_length_ratio_and_a_constant_level() {
        let pcm = vec![0.5f32; 48_000];
        let down = resample(&pcm, 48_000, 16_000);
        assert_eq!(down.len(), 16_000);
        assert!(down.iter().all(|v| (v - 0.5).abs() < 1e-6));
        let up = resample(&vec![0.25f32; 8_000], 8_000, 16_000);
        assert_eq!(up.len(), 16_000);
        assert!(up.iter().all(|v| (v - 0.25).abs() < 1e-6));
        assert_eq!(resample(&pcm[..10], 16_000, 16_000), pcm[..10].to_vec());
    }
}
