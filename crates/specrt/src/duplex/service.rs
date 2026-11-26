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

pub struct Service {
    events: Mutex<Sender<Event>>,
    pub info: ServiceInfo,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

pub struct Session {
    pub id: SessionId,
    pub first: Out,
    pub frames: Receiver<Out>,
    events: Sender<Event>,
    closed: bool,
}

impl Session {
    pub fn audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.events.send(Event::Audio(self.id, pcm)).map_err(|_| anyhow!("the duplex engine has stopped"))
    }

    pub fn control(&self, control: Control) -> Result<Reply> {
        let (tx, rx) = channel();
        self.events.send(Event::Control(self.id, control, tx)).map_err(|_| anyhow!("the duplex engine has stopped"))?;
        rx.recv().map_err(|_| anyhow!("the duplex engine has stopped"))?
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
        Service::spawn(frame, duplex.frame_samples, max_queue, move || {
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
        })
    }

    pub fn start(&self, request: StartRequest) -> Result<Session> {
        let events = self.events.lock().unwrap().clone();
        let (frames_tx, frames) = channel();
        let (tx, rx) = channel();
        events.send(Event::Start(request, frames_tx, tx)).map_err(|_| anyhow!("the duplex engine has stopped"))?;
        let (id, first) = rx.recv().map_err(|_| anyhow!("the duplex engine has stopped"))??;
        Ok(Session { id, first, frames, events, closed: false })
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
    Out { audio: f.audio, token: Some(f.token), text: f.text, index: f.index, late: false, done: f.done }
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
            return Ok((id, Out { audio: Vec::new(), token: None, text: String::new(), index, late: false, done: false }));
        }
        let (id, first) = self.engine.start(request.options)?;
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
