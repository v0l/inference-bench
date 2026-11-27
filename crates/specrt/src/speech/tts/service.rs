use super::voices::{Catalogue, Entry, HUB};
use super::{HostTts, Voice, plan, text};
use crate::codec::HostCodec;
use crate::codec::cuda::CudaCodec;
use crate::flow_head::{FlowHead, frame_noise};
use crate::flow_head_cuda::CudaHead;
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

pub fn resolve_dir(model: &str) -> Option<PathBuf> {
    let config = crate::weights::find_in_snapshot(model, "config.json").ok()?;
    super::is_tts_config(&std::fs::read_to_string(&config).ok()?).then(|| config.parent().map(Path::to_path_buf)).flatten()
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct SpeakerInfo {
    pub name: String,
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub device: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Job {
    pub text: String,
    pub voice: Option<PathBuf>,
    pub seed: u64,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Spoken {
    pub samples: usize,
    pub frames: usize,
    pub chunks: usize,
    pub stopped: bool,
}

pub trait Speak {
    fn info(&self) -> SpeakerInfo;
    fn catalogue(&self) -> Catalogue;
    fn speak(&mut self, job: &Job, emit: &mut dyn FnMut(&[f32]) -> bool) -> Result<Spoken>;
}

enum Back {
    Host(HostCodec),
    Cuda { head: CudaHead, codec: CudaCodec, _jit: crate::jit::Jit },
}

pub struct Speaker {
    tts: HostTts,
    tok: tokenizers::Tokenizer,
    norm: text::Normalize,
    back: Back,
    voices: BTreeMap<PathBuf, Voice>,
    catalogue: Catalogue,
    info: SpeakerInfo,
    voice_width: usize,
    scale: f32,
}

impl Speaker {
    fn with(dir: &Path, back: Back, frame_samples: usize, device: &str) -> Result<Speaker> {
        let tts = HostTts::open(dir)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{}/tokenizer.json: {e}", dir.display()))?;
        let norm = text::Normalize::for_lang(tts.tts.language.parse::<text::Lang>().map_err(anyhow::Error::msg)?);
        let voice_width = tts.spec.inputs.as_ref().context("no [inputs]")?.width_of(&tts.tts.voice).context("[tts] voice is not an input stream")?;
        let sample_rate = (frame_samples as f64 * tts.rules.frame_rate).round() as u32;
        let scale = tts.noise_scale(tts.tts.param("temperature")? as f32);
        let catalogue = Catalogue::find(dir, &tts.tts.voices, &tts.tts.language);
        let info = SpeakerInfo { name: tts.spec.name.clone(), sample_rate, frame_samples, device: device.to_string() };
        Ok(Speaker { tts, tok, norm, back, voices: BTreeMap::new(), catalogue, info, voice_width, scale })
    }

    pub fn open_host(dir: &Path) -> Result<Speaker> {
        let codec = HostCodec::open(dir)?;
        let frame = codec.plan.frame_samples;
        Speaker::with(dir, Back::Host(codec), frame, "CPU")
    }

    pub fn open_gpu(dir: &Path, device: usize) -> Result<Speaker> {
        let ctx = cudarc::driver::CudaContext::new(device)?;
        let stream = ctx.default_stream();
        let jit = crate::jit::Jit::new(ctx.clone(), crate::jit::default_cache_dir())?;
        let head = CudaHead::load(&stream, &jit, FlowHead::from_dir(dir, None)?, dir, true)?;
        let host = HostCodec::open(dir)?;
        let codec = CudaCodec::load(&ctx, &jit, &host, &[(1, 1)])?;
        let frame = host.plan.frame_samples;
        Speaker::with(dir, Back::Cuda { head, codec, _jit: jit }, frame, "GPU")
    }

    fn voice(&mut self, path: &Path) -> Result<&Voice> {
        if !self.voices.contains_key(path) {
            let v = Voice::open(path, &self.tts.tts, self.voice_width)?;
            self.voices.insert(path.to_path_buf(), v);
        }
        Ok(&self.voices[path])
    }

    fn chunk(&self, voice: Option<&Voice>, chunk: &plan::Chunk, seed: u64, frame: &mut u64, emit: &mut dyn FnMut(&[f32]) -> bool) -> Result<(usize, bool)> {
        let t = &self.tts;
        let d = t.latent_dim();
        let (mut s, mut hidden, mut eos) = t.start(voice, &chunk.tokens)?;
        let mut rule = t.rules.eos(chunk.frames_after_eos);
        let mut samples = 0;
        let noise = |frame: &mut u64| -> Vec<f32> {
            let n = frame_noise(seed, *frame, d).iter().map(|v| v * self.scale).collect();
            *frame += 1;
            n
        };
        match &self.back {
            Back::Host(codec) => {
                let mut state = codec.decoder.new_state();
                for _ in 0..chunk.frame_budget {
                    let f = t.sample(&mut s, &hidden, eos, &noise(frame))?;
                    let pcm = codec.decoder.run(&f.codec, &mut state)?;
                    samples += pcm.len();
                    if !emit(&pcm) {
                        return Ok((samples, true));
                    }
                    if rule.should_stop(eos) {
                        break;
                    }
                    (hidden, eos) = t.advance(&mut s, &f.latent)?;
                }
            }
            Back::Cuda { head, codec, .. } => {
                let mut hs = head.session(&[1.0])?;
                let mut cs = codec.sessions(&codec.decoder, 1)?;
                for _ in 0..chunk.frame_budget {
                    let latent = head.frame(&mut hs, &hidden, &noise(frame))?;
                    let z = head.read(&hs, "codec")?;
                    let pcm = codec.run(&codec.decoder, &z, &mut cs)?;
                    samples += pcm.len();
                    if !emit(&pcm) {
                        return Ok((samples, true));
                    }
                    if rule.should_stop(eos) {
                        break;
                    }
                    (hidden, eos) = t.advance(&mut s, &latent)?;
                }
            }
        }
        Ok((samples, false))
    }
}

impl Speak for Speaker {
    fn info(&self) -> SpeakerInfo {
        self.info.clone()
    }

    fn catalogue(&self) -> Catalogue {
        self.catalogue.clone()
    }

    fn speak(&mut self, job: &Job, emit: &mut dyn FnMut(&[f32]) -> bool) -> Result<Spoken> {
        if let Some(p) = &job.voice {
            self.voice(p)?;
        }
        let voice = job.voice.as_ref().map(|p| &self.voices[p]);
        let normalised = self.norm.apply(&job.text).into_owned();
        let chunks: Vec<plan::Chunk> = plan::chunks(&self.tok, &normalised, &self.tts.rules)?.into_iter().filter(|c| !c.tokens.is_empty()).collect();
        let mut out = Spoken::default();
        let mut frame = 0u64;
        for c in &chunks {
            out.chunks += 1;
            let (n, stopped) = self.chunk(voice, c, job.seed, &mut frame, emit)?;
            out.samples += n;
            if stopped {
                out.stopped = true;
                break;
            }
        }
        out.frames = frame as usize;
        Ok(out)
    }
}

pub fn open(dir: &Path, device: usize) -> Result<Box<dyn Speak>> {
    match Speaker::open_gpu(dir, device) {
        Ok(s) => Ok(Box::new(s)),
        Err(e) if cudarc::driver::CudaContext::new(device).is_err() => {
            tracing::warn!(error = %e, "no CUDA device for text to speech, speaking on the CPU");
            Ok(Box::new(Speaker::open_host(dir)?))
        }
        Err(e) => Err(e),
    }
}

pub type Emit = Box<dyn FnMut(&[f32]) -> bool + Send>;

type Work = (Job, Emit, Sender<Result<Spoken>>);

pub struct Service {
    jobs: Mutex<Option<Sender<Work>>>,
    pub info: SpeakerInfo,
    catalogue: Mutex<Catalogue>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    pub hub: String,
}

impl Service {
    pub fn spawn(make: impl FnOnce() -> Result<Box<dyn Speak>> + Send + 'static) -> Result<Arc<Service>> {
        let (jobs, inbox) = channel::<Work>();
        let (ready_tx, ready) = channel();
        let thread = std::thread::Builder::new().name("speech-voice".into()).spawn(move || {
            let mut engine = match make() {
                Ok(e) => {
                    let _ = ready_tx.send(Ok((e.info(), e.catalogue())));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            while let Ok((job, mut emit, reply)) = inbox.recv() {
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.speak(&job, &mut *emit)));
                drop(emit);
                let _ = reply.send(out.unwrap_or_else(|_| Err(anyhow!("the speech engine panicked"))));
            }
        })?;
        let (info, catalogue) = ready.recv().map_err(|_| anyhow!("the speech engine stopped while loading"))??;
        Ok(Arc::new(Service { jobs: Mutex::new(Some(jobs)), info, catalogue: Mutex::new(catalogue), thread: Mutex::new(Some(thread)), hub: HUB.to_string() }))
    }

    pub fn load(dir: &Path, device: usize) -> Result<Arc<Service>> {
        let dir = dir.to_path_buf();
        let sink = crate::progress::current();
        Service::spawn(move || crate::progress::with_sink(sink, || open(&dir, device)))
    }

    pub fn load_on_host(dir: &Path) -> Result<Arc<Service>> {
        let dir = dir.to_path_buf();
        Service::spawn(move || Ok(Box::new(Speaker::open_host(&dir)?) as Box<dyn Speak>))
    }

    fn fetched(&self) -> std::sync::MutexGuard<'_, Catalogue> {
        let mut c = self.catalogue.lock().unwrap_or_else(|e| e.into_inner());
        if c.voices.is_empty()
            && let Err(e) = c.fetch_catalogue(&self.hub)
        {
            tracing::warn!("could not fetch the voice catalogue: {e:#}");
        }
        c
    }

    pub fn voices(&self) -> Vec<(Entry, bool)> {
        let c = self.fetched();
        c.voices.iter().map(|v| (v.clone(), c.installed(v))).collect()
    }

    pub fn pick(&self, asked: &str) -> Result<Option<(Entry, PathBuf)>> {
        let c = self.fetched();
        let Some(v) = c.pick(asked).cloned() else { return Ok(None) };
        let file = c.voice_file(&v, &self.hub)?;
        Ok(Some((v, file)))
    }

    pub fn start(&self, job: Job, emit: Emit) -> Result<Receiver<Result<Spoken>>> {
        let (tx, rx) = channel();
        self.jobs.lock().unwrap().as_ref().context("the speech engine has stopped")?.send((job, emit, tx)).map_err(|_| anyhow!("the speech engine has stopped"))?;
        Ok(rx)
    }

    pub fn say(&self, job: Job) -> Result<(Vec<f32>, Spoken)> {
        let pcm = Arc::new(Mutex::new(Vec::new()));
        let sink = pcm.clone();
        let rx = self.start(job, Box::new(move |f| {
            sink.lock().unwrap().extend_from_slice(f);
            true
        }))?;
        let spoken = rx.recv().map_err(|_| anyhow!("the speech engine has stopped"))??;
        let pcm = std::mem::take(&mut *pcm.lock().unwrap());
        Ok((pcm, spoken))
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
