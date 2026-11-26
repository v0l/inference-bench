use super::clock::{self, Clock, Control, Event, FrameEngine, Out, Reply, SessionId, StartRequest};
use super::{Engine, EngineOptions, Frame};
use crate::sessions::SessionStore;
use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceOptions {
    pub slots: usize,
    pub max_ctx: usize,
    pub max_queue: usize,
    pub store_dir: Option<PathBuf>,
    pub keep_idle_secs: u64,
    pub graphs: bool,
}

impl Default for ServiceOptions {
    fn default() -> ServiceOptions {
        ServiceOptions { slots: 4, max_ctx: 8192, max_queue: 8, store_dir: None, keep_idle_secs: 0, graphs: true }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ServiceInfo {
    pub name: String,
    pub sample_rate: usize,
    pub frame_samples: usize,
    pub slots: usize,
    pub max_ctx: usize,
    pub device: usize,
    pub snapshots: bool,
}

thread_local! {
    static OPTIONS: std::cell::RefCell<Option<ServiceOptions>> = const { std::cell::RefCell::new(None) };
}

pub fn with_options<T>(options: Option<ServiceOptions>, f: impl FnOnce() -> T) -> T {
    let before = OPTIONS.with(|o| o.replace(options));
    let out = f();
    OPTIONS.with(|o| *o.borrow_mut() = before);
    out
}

pub fn options() -> Option<ServiceOptions> {
    OPTIONS.with(|o| o.borrow().clone())
}

pub fn resolve_dir(model: &str) -> Option<PathBuf> {
    let config = crate::weights::find_in_snapshot(model, "config.json").ok()?;
    let text = std::fs::read_to_string(&config).ok()?;
    crate::spec::Spec::from_hf_config_duplex(&text).ok()?;
    config.parent().map(Path::to_path_buf)
}

pub fn is_duplex_config(config: &str) -> bool {
    crate::spec::Spec::from_hf_config_duplex(config).is_ok()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Footprint {
    pub weights: u64,
    pub per_session: u64,
    pub frame_seconds: f64,
    pub sample_rate: usize,
    pub card_fields: Vec<String>,
}

impl Footprint {
    pub fn total(&self, sessions: usize) -> u64 {
        self.weights + self.per_session * sessions as u64
    }
}

pub fn footprint(dir: &Path, max_ctx: usize) -> Result<Footprint> {
    let text = std::fs::read_to_string(dir.join("config.json")).with_context(|| format!("{}/config.json", dir.display()))?;
    let spec = crate::spec::Spec::from_hf_config_duplex(&text)?;
    let d = spec.duplex.as_ref().context("no [duplex] section")?;
    let weights: u64 = std::fs::read_dir(dir)?.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    let config: serde_json::Value = serde_json::from_str(&text)?;
    let codec = crate::codec::graph::plan(&config)?;
    let codec_f32_extra = ((codec.encoder.params() + codec.decoder.params()) * 2) as u64;
    let weights = weights + codec_f32_extra;
    let pages = max_ctx.div_ceil(crate::codegen::PAGE);
    let kv = (pages * crate::runtime::kv_block_bytes(&spec, 2)) as u64;
    let per_session = kv + crate::serve::slot_state_bytes(&spec) + (2 << 20);
    Ok(Footprint { weights, per_session, frame_seconds: d.frame_seconds(), sample_rate: d.sample_rate, card_fields: d.card.fields.iter().map(|f| f.name.clone()).collect() })
}

pub struct Service {
    events: Mutex<Sender<Event>>,
    pub info: ServiceInfo,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

pub struct Session {
    pub id: SessionId,
    pub first: Out,
    frames: Mutex<Option<Receiver<Out>>>,
    events: Sender<Event>,
    closed: bool,
}

impl Session {
    pub fn frames(&mut self) -> Option<Receiver<Out>> {
        self.frames.lock().unwrap().take()
    }

    pub fn handle(&self) -> SessionHandle {
        SessionHandle { id: self.id, events: self.events.clone() }
    }

    pub fn audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.handle().audio(pcm)
    }

    pub fn control(&self, control: Control) -> Result<Reply> {
        self.handle().control(control)
    }

    pub fn park(mut self) -> Result<String> {
        self.closed = true;
        let (tx, rx) = channel();
        self.events.send(Event::Park(self.id, tx)).map_err(|_| anyhow!("the duplex engine has stopped"))?;
        rx.recv().map_err(|_| anyhow!("the duplex engine has stopped"))?
    }

    pub fn close(mut self) {
        self.closed = true;
        let _ = self.events.send(Event::Close(self.id));
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.events.send(Event::Close(self.id));
        }
    }
}

#[derive(Clone)]
pub struct SessionHandle {
    pub id: SessionId,
    events: Sender<Event>,
}

impl SessionHandle {
    pub fn audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.events.send(Event::Audio(self.id, pcm)).map_err(|_| anyhow!("the duplex engine has stopped"))
    }

    pub fn control(&self, control: Control) -> Result<Reply> {
        let (tx, rx) = channel();
        self.events.send(Event::Control(self.id, control, tx)).map_err(|_| anyhow!("the duplex engine has stopped"))?;
        rx.recv().map_err(|_| anyhow!("the duplex engine has stopped"))?
    }
}

impl Service {
    pub fn spawn<F>(frame: Duration, samples: usize, max_queue: usize, build: F) -> Result<Arc<Service>>
    where
        F: FnOnce() -> Result<(Box<dyn FrameEngine>, ServiceInfo)> + Send + 'static,
    {
        let (events, inbox) = channel::<Event>();
        let (ready_tx, ready_rx) = channel();
        let thread = std::thread::Builder::new()
            .name("duplex-clock".into())
            .spawn(move || {
                let (mut engine, info) = match build() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let clock = Clock { frame, samples, max_queue };
                let _ = ready_tx.send(Ok(info));
                let (out, _unrouted) = channel();
                if let Err(e) = clock::run(engine.as_mut(), &clock, inbox, out, None) {
                    tracing::error!(error = %format!("{e:#}"), "the duplex clock stopped");
                }
            })
            .context("start the duplex clock thread")?;
        let info = ready_rx.recv().map_err(|_| anyhow!("the duplex engine stopped while loading"))??;
        Ok(Arc::new(Service { events: Mutex::new(events), info, thread: Mutex::new(Some(thread)) }))
    }

    pub fn load(dir: &Path, opts: ServiceOptions, device: usize) -> Result<Arc<Service>> {
        let text = std::fs::read_to_string(dir.join("config.json")).with_context(|| format!("{}/config.json", dir.display()))?;
        let duplex = crate::spec::Spec::from_hf_config_duplex(&text)?.duplex.context("no [duplex] section")?;
        let dir = dir.to_path_buf();
        let frame = Duration::from_secs_f64(duplex.frame_seconds());
        let max_queue = opts.max_queue.max(1);
        let sink = crate::progress::current();
        Service::spawn(frame, duplex.frame_samples, max_queue, move || crate::progress::with_sink(sink, || {
            let engine = Engine::load(&dir, EngineOptions { slots: opts.slots.max(1), max_ctx: opts.max_ctx, graphs: opts.graphs, device, ..EngineOptions::default() })?;
            let store = match (&opts.store_dir, opts.keep_idle_secs) {
                (Some(root), keep) if keep > 0 => Some(SessionStore::open(root, engine.rt.session_compat())?),
                _ => None,
            };
            let info = ServiceInfo {
                name: engine.spec.name.clone(),
                sample_rate: engine.duplex.sample_rate,
                frame_samples: engine.duplex.frame_samples,
                slots: opts.slots.max(1),
                max_ctx: opts.max_ctx,
                device,
                snapshots: store.is_some(),
            };
            let mut hosted = Hosted { engine, store, keep_idle_secs: opts.keep_idle_secs };
            hosted.warm()?;
            Ok((Box::new(hosted) as Box<dyn FrameEngine>, info))
        }))
    }

    pub fn start(&self, request: StartRequest) -> Result<Session> {
        let events = self.events.lock().unwrap().clone();
        let (frames_tx, frames) = channel();
        let (tx, rx) = channel();
        events.send(Event::Start(request, frames_tx, tx)).map_err(|_| anyhow!("the duplex engine has stopped"))?;
        let (id, first) = rx.recv().map_err(|_| anyhow!("the duplex engine has stopped"))??;
        Ok(Session { id, first, frames: Mutex::new(Some(frames)), events, closed: false })
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let (dead, _) = channel();
        drop(std::mem::replace(&mut *self.events.lock().unwrap(), dead));
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

pub fn out_of(f: Frame) -> Out {
    Out { audio: f.audio, token: Some(f.token), text: f.text, index: f.index, late: false, done: f.done, error: None, input: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_footprint_counts_the_f32_codec_and_grows_with_context() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/haru/tiny");
        let small = footprint(&dir, 256).unwrap();
        let big = footprint(&dir, 4096).unwrap();
        let file = std::fs::metadata(dir.join("model.safetensors")).unwrap().len();
        assert!(small.weights > file, "the codec runs in f32, so it holds more than the bf16 file");
        assert!(big.per_session > small.per_session && big.weights == small.weights);
        assert_eq!(small.total(3), small.weights + 3 * small.per_session);
        assert_eq!((small.sample_rate, small.card_fields.len()), (24000, 7));
    }
}

pub struct Hosted {
    pub engine: Engine,
    pub store: Option<SessionStore>,
    pub keep_idle_secs: u64,
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Hosted {
    pub fn sweep(&self, now: u64) -> usize {
        let Some(store) = &self.store else { return 0 };
        let old: Vec<String> = store.list().into_iter().filter(|e| now.saturating_sub(e.last_used) > self.keep_idle_secs).map(|e| e.id).collect();
        old.iter().filter(|id| store.delete(id).is_ok()).count()
    }

    pub fn warm(&mut self) -> Result<()> {
        let (id, _) = self.engine.start(super::StartOptions::default())?;
        let silence = vec![0.0f32; self.engine.duplex.frame_samples];
        let warmed = self.engine.step(&[(id, silence)]);
        self.engine.end(id)?;
        warmed.map(|_| ())
    }
}

impl FrameEngine for Hosted {
    fn step(&mut self, frames: &[(SessionId, Vec<f32>)]) -> Result<Vec<(SessionId, Out)>> {
        let out = self.engine.step(frames)?;
        Ok(frames.iter().map(|(id, _)| *id).zip(out.into_iter().map(out_of)).collect())
    }

    fn start(&mut self, request: StartRequest) -> Result<(SessionId, Out)> {
        if let Some(key) = request.resume {
            let store = self.store.as_ref().context("this server keeps no idle sessions: set duplex.keep_idle_secs")?;
            self.sweep(now());
            let id = self.engine.resume(store, &key, now())?;
            let _ = store.delete(&key);
            let index = self.engine.live(id)?.frame;
            return Ok((id, Out { audio: Vec::new(), token: None, text: String::new(), index, late: false, done: false, error: None, input: None }));
        }
        let mut options = request.options;
        if let Some(p) = request.prompt {
            let (id, first) = self.engine.start_prompt(&p, options)?;
            return Ok((id, out_of(first)));
        }
        if !request.card_fields.is_empty() {
            anyhow::ensure!(options.card.is_empty(), "give the card as text or as fields, not both");
            options.card = self.engine.duplex.card.render(&request.card_fields)?;
        }
        let (id, first) = self.engine.start(options)?;
        Ok((id, out_of(first)))
    }

    fn control(&mut self, id: SessionId, control: Control) -> Result<Reply> {
        match control {
            Control::Say(text) => self.engine.say(id, &text).map(Reply::Queued),
            Control::Mode(m) => self.engine.set_mode(id, m).map(|_| Reply::Done),
            Control::Interrupt => self.engine.interrupt(id).map(|(r, said)| Reply::Said { said, unsaid: r.unsaid.len() }),
        }
    }

    fn end(&mut self, id: SessionId) -> Result<()> {
        self.engine.end(id)
    }

    fn park(&mut self, id: SessionId) -> Result<String> {
        let store = self.store.as_ref().context("this server keeps no idle sessions")?;
        let key = self.engine.park(id, store, now())?;
        self.sweep(now());
        Ok(key)
    }
}
