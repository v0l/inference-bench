use crate::chat::ChatRequest;
use crate::live::{History, HISTORY_SAMPLES, Meter, Reading, RequestRecord};
use crate::native::{Factory, Native, specrt_factory};
use crate::result::{BackendError, ErrorKind, Generation};
use crate::target::Target;
use specrt::serve::{ServeOptions, Stats};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

pub const MAX_PARALLEL: usize = 64;
pub const SAMPLE_EVERY: Duration = Duration::from_secs(1);

pub type ServingFactory = Arc<dyn Fn(ServeOptions) -> Option<Box<dyn Target>> + Send + Sync>;

pub fn specrt_serving_factory() -> ServingFactory {
    Arc::new(|opts| Some(Box::new(crate::serving::ServingTarget::new(opts)) as Box<dyn Target>))
}

type StatsSource = Arc<Mutex<Vec<(String, Box<dyn Target>)>>>;
type DuplexList = Arc<Mutex<Vec<(String, Arc<specrt::duplex::service::Service>)>>>;
type TranscriberList = Arc<Mutex<Vec<(String, Arc<specrt::speech::service::Service>)>>>;
type LoadedList = Arc<Mutex<Vec<Loaded>>>;
type LiveList = Arc<Mutex<Vec<Live>>>;

#[derive(Clone)]
pub struct Live {
    pub model_id: String,
    pub meter: Arc<Meter>,
    pub history: Arc<Mutex<History>>,
}

impl Live {
    fn new(model_id: &str) -> Self {
        Self { model_id: model_id.to_string(), meter: Arc::default(), history: Arc::new(Mutex::new(History::new(HISTORY_SAMPLES))) }
    }
}

pub const DEFAULT_MAX_LOADED: usize = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub model_id: String,
    pub display: String,
    pub load_secs: f64,
    pub vision: bool,
    pub ctx: usize,
    pub max_num_seqs: usize,
    pub served_names: Vec<String>,
    pub parallel_note: Option<String>,
    pub engine: specrt::options::EngineChoice,
    pub speculation: Option<String>,
    pub thinking: specrt::chat::Thinking,
    pub tools: specrt::chat::ToolFormat,
    pub vram_bytes: Option<u64>,
    pub landed: Vec<(String, u64)>,
    pub kind: specrt::media::ModelKind,
    pub media: Option<specrt::media::MediaInfo>,
}

impl Loaded {
    pub fn answers_to(&self, name: &str) -> bool {
        self.model_id == name
            || self.display == name
            || self.served_names.iter().any(|n| n == name)
            || crate::library::short_alias(&self.display).is_some_and(|a| a == name)
    }
}

pub enum Command {
    Load { model_id: String, display: String, max_ctx: Option<usize>, max_num_seqs: Option<usize>, kv_cache_bytes: Option<u64>, served_names: Vec<String>, engine: specrt::options::EngineChoice, alongside: bool },
    Unload { model: Option<String> },
    Suspend { model: Option<String>, done: Sender<bool> },
    Resume { model: Option<String>, done: Sender<Result<(), BackendError>> },
    Chat { model: Option<String>, request: ChatRequest, events: Sender<ChatEvent>, queued: Instant, meter: Arc<Meter> },
    Media { model: Option<String>, job: specrt::media::MediaJob, events: Sender<MediaUpdate>, cancel: Arc<AtomicBool> },
    Shutdown,
}

#[derive(Debug, Clone)]
pub enum MediaUpdate {
    Event(specrt::media::MediaEvent),
    Done(specrt::media::MediaOutput),
    Failed(BackendError),
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Text(String),
    Done(Generation),
    Failed(BackendError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    Loading(String),
    LoadProgress { stage: String, done: u64, total: u64 },
    Loaded(Loaded),
    LoadFailed(BackendError),
    Unloaded,
}

pub type Wake = Arc<dyn Fn() + Send + Sync>;
pub type GpuSampler = Arc<dyn Fn() -> Vec<crate::gpu::GpuStats> + Send + Sync>;

pub fn no_gpu_sampler() -> GpuSampler {
    Arc::new(Vec::new)
}

pub struct Engine {
    commands: Sender<Command>,
    pub events: Receiver<EngineEvent>,
    loaded: LoadedList,
    suspended: LoadedList,
    lives: LiveList,
    idle: Live,
    stats: StatsSource,
    watching: Arc<AtomicBool>,
    stop_sampler: Arc<AtomicBool>,
    max_loaded: Arc<AtomicUsize>,
    duplexes: DuplexList,
    transcribers: TranscriberList,
    duplex_options: Arc<Mutex<specrt::duplex::service::ServiceOptions>>,
    duplex_loader: Arc<Mutex<crate::native::DuplexLoader>>,
}

#[derive(Clone)]
pub struct Handle {
    commands: Sender<Command>,
    loaded: LoadedList,
    suspended: LoadedList,
    lives: LiveList,
    idle: Live,
    stats: StatsSource,
    watching: Arc<AtomicBool>,
    duplexes: DuplexList,
    transcribers: TranscriberList,
}

impl Handle {
    pub fn transcriber_of(&self, model: Option<&str>) -> Option<Arc<specrt::speech::service::Service>> {
        let id = match model {
            Some(name) => self.loaded_named(name)?.model_id,
            None => self.loaded()?.model_id,
        };
        self.transcribers.lock().unwrap().iter().find(|(m, _)| *m == id).map(|(_, s)| s.clone())
    }

    pub fn duplex_of(&self, model: Option<&str>) -> Option<Arc<specrt::duplex::service::Service>> {
        let id = match model {
            Some(name) => self.loaded_named(name)?.model_id,
            None => self.loaded()?.model_id,
        };
        self.duplexes.lock().unwrap().iter().find(|(m, _)| *m == id).map(|(_, s)| s.clone())
    }

    pub fn loaded(&self) -> Option<Loaded> {
        self.loaded.lock().unwrap().first().cloned()
    }

    pub fn loaded_all(&self) -> Vec<Loaded> {
        self.loaded.lock().unwrap().clone()
    }

    pub fn loaded_named(&self, name: &str) -> Option<Loaded> {
        self.loaded.lock().unwrap().iter().find(|l| l.answers_to(name)).cloned()
    }

    pub fn thinking(&self) -> specrt::chat::Thinking {
        self.loaded().map(|l| l.thinking).unwrap_or_default()
    }

    pub fn tools(&self) -> specrt::chat::ToolFormat {
        self.loaded().map(|l| l.tools).unwrap_or_default()
    }

    pub fn tools_of(&self, model: &str) -> specrt::chat::ToolFormat {
        self.loaded_named(model).map(|l| l.tools).unwrap_or_default()
    }

    pub fn suspended(&self) -> Option<Loaded> {
        self.suspended.lock().unwrap().first().cloned()
    }

    pub fn suspended_all(&self) -> Vec<Loaded> {
        self.suspended.lock().unwrap().clone()
    }

    pub fn suspend(&self) -> bool {
        self.suspend_model(None)
    }

    pub fn suspend_model(&self, model: Option<&str>) -> bool {
        let (done, rx) = channel();
        self.commands.send(Command::Suspend { model: model.map(str::to_string), done }).is_ok() && rx.recv().unwrap_or(false)
    }

    pub fn resume(&self) -> Result<(), BackendError> {
        self.resume_model(None)
    }

    pub fn resume_model(&self, model: Option<&str>) -> Result<(), BackendError> {
        let (done, rx) = channel();
        self.commands.send(Command::Resume { model: model.map(str::to_string), done }).map_err(|_| BackendError::failed("the engine has stopped"))?;
        rx.recv().unwrap_or_else(|_| Err(BackendError::failed("the engine has stopped")))
    }

    pub fn chat(&self, request: ChatRequest) -> Receiver<ChatEvent> {
        self.chat_on(None, request)
    }

    pub fn generate_media(&self, model: Option<&str>, job: specrt::media::MediaJob, cancel: Arc<AtomicBool>) -> Receiver<MediaUpdate> {
        let (tx, rx) = channel();
        if self.commands.send(Command::Media { model: model.map(str::to_string), job, events: tx.clone(), cancel }).is_err() {
            let _ = tx.send(MediaUpdate::Failed(BackendError::failed("the engine has stopped")));
        }
        rx
    }

    pub fn chat_on(&self, model: Option<&str>, request: ChatRequest) -> Receiver<ChatEvent> {
        let (tx, rx) = channel();
        let meter = self.live_of(model).map_or_else(|| self.idle.meter.clone(), |l| l.meter);
        let queued = meter.submitted();
        if self.commands.send(Command::Chat { model: model.map(str::to_string), request, events: tx.clone(), queued, meter: meter.clone() }).is_err() {
            meter.withdrawn();
            let _ = tx.send(ChatEvent::Failed(BackendError::failed("the engine has stopped")));
        }
        rx
    }

    fn with_target<R>(&self, model: Option<&str>, f: impl FnOnce(&dyn Target) -> Option<R>) -> Option<R> {
        let id = match model {
            Some(name) => Some(self.loaded_named(name)?.model_id),
            None => None,
        };
        let stats = self.stats.lock().unwrap();
        let (_, target) = match &id {
            Some(id) => stats.iter().find(|(m, _)| m == id)?,
            None => stats.first()?,
        };
        f(target.as_ref())
    }

    pub fn serving_stats(&self) -> Option<Stats> {
        self.with_target(None, |t| t.stats())
    }

    pub fn serving_stats_of(&self, model: &str) -> Option<Stats> {
        self.with_target(Some(model), |t| t.stats())
    }

    pub fn sessions(&self) -> Option<Arc<specrt::sessions::Shared>> {
        self.with_target(None, |t| t.sessions())
    }

    pub fn serving_metrics(&self) -> Option<String> {
        self.with_target(None, |t| t.metrics())
    }

    pub fn serving_metrics_of(&self, model: &str) -> Option<String> {
        self.with_target(Some(model), |t| t.metrics())
    }

    pub fn failure(&self) -> Option<String> {
        self.with_target(None, |t| t.failure())
    }

    pub fn failure_of(&self, model: &str) -> Option<String> {
        self.with_target(Some(model), |t| t.failure())
    }

    pub fn live_of(&self, model: Option<&str>) -> Option<Live> {
        let id = match model {
            Some(name) => self.loaded_named(name)?.model_id,
            None => self.loaded()?.model_id,
        };
        self.lives.lock().unwrap().iter().find(|l| l.model_id == id).cloned()
    }

    pub fn reading(&self) -> Reading {
        self.reading_of(None)
    }

    pub fn reading_of(&self, model: Option<&str>) -> Reading {
        let loaded = match model {
            Some(name) => self.loaded_named(name),
            None => self.loaded(),
        };
        let max = loaded.as_ref().map_or(1, |l| l.max_num_seqs);
        let meter = self.live_of(model).map_or_else(|| self.idle.meter.clone(), |l| l.meter);
        let stats = loaded.and_then(|l| self.serving_stats_of(&l.model_id));
        meter.read(max, stats)
    }

    pub fn recent_requests(&self) -> Vec<RequestRecord> {
        self.recent_requests_of(None)
    }

    pub fn recent_requests_of(&self, model: Option<&str>) -> Vec<RequestRecord> {
        self.live_of(model).map_or_else(Vec::new, |l| l.meter.recent())
    }

    pub fn history(&self) -> Arc<Mutex<History>> {
        self.history_of(None)
    }

    pub fn history_of(&self, model: Option<&str>) -> Arc<Mutex<History>> {
        self.live_of(model).map_or_else(|| self.idle.history.clone(), |l| l.history)
    }

    pub fn sessions_of(&self, model: &str) -> Option<Arc<specrt::sessions::Shared>> {
        self.with_target(Some(model), |t| t.sessions())
    }

    pub fn watch(&self, on: bool) {
        self.watching.store(on, Ordering::Relaxed);
    }
}

impl Engine {
    pub fn spawn(wake: Wake) -> Self {
        Self::with_factories(wake, specrt_factory(), specrt_serving_factory())
    }

    pub fn with_factory(wake: Wake, factory: Factory) -> Self {
        Self::with_factories_sampled(wake, factory, Arc::new(|_| None), no_gpu_sampler())
    }

    pub fn with_factories(wake: Wake, factory: Factory, serving: ServingFactory) -> Self {
        Self::with_factories_sampled(wake, factory, serving, Arc::new(crate::gpu::sample))
    }

    pub fn with_factories_sampled(wake: Wake, factory: Factory, serving: ServingFactory, gpu: GpuSampler) -> Self {
        let (commands, inbox) = channel();
        let (events_tx, events) = channel();
        let loaded: LoadedList = Arc::default();
        let suspended: LoadedList = Arc::default();
        let lives: LiveList = Arc::default();
        let idle = Live::new("");
        let stats: StatsSource = Arc::default();
        let duplexes: DuplexList = Arc::default();
        let transcribers: TranscriberList = Arc::default();
        let duplex_options = Arc::new(Mutex::new(specrt::duplex::service::ServiceOptions::default()));
        let duplex_loader = Arc::new(Mutex::new(crate::native::specrt_duplex_loader()));
        let max_loaded = Arc::new(AtomicUsize::new(DEFAULT_MAX_LOADED));
        let watching = Arc::new(AtomicBool::new(false));
        let stop_sampler = Arc::new(AtomicBool::new(false));
        let shared = Shared { loaded: loaded.clone(), suspended: suspended.clone(), lives: lives.clone(), stats: stats.clone(), max_loaded: max_loaded.clone(), duplexes: duplexes.clone(), transcribers: transcribers.clone(), duplex_options: duplex_options.clone(), duplex_loader: duplex_loader.clone() };
        let this = Self { commands, events, loaded, suspended, lives, idle, stats, watching, stop_sampler, max_loaded, duplexes, transcribers, duplex_options, duplex_loader };
        let handle = this.handle();
        let (stop, wake_sampler) = (this.stop_sampler.clone(), wake.clone());
        std::thread::spawn(move || run(inbox, events_tx, shared, wake, factory, serving, gpu));
        std::thread::spawn(move || sample(handle, stop, wake_sampler));
        this
    }

    pub fn handle(&self) -> Handle {
        Handle {
            commands: self.commands.clone(),
            loaded: self.loaded.clone(),
            suspended: self.suspended.clone(),
            lives: self.lives.clone(),
            idle: self.idle.clone(),
            stats: self.stats.clone(),
            watching: self.watching.clone(),
            duplexes: self.duplexes.clone(),
            transcribers: self.transcribers.clone(),
        }
    }

    pub fn set_duplex_options(&self, options: specrt::duplex::service::ServiceOptions) {
        *self.duplex_options.lock().unwrap() = options;
    }

    pub fn set_duplex_loader(&self, loader: crate::native::DuplexLoader) {
        *self.duplex_loader.lock().unwrap() = loader;
    }

    pub fn reading(&self) -> Reading {
        self.handle().reading()
    }

    pub fn watch(&self, on: bool) {
        self.watching.store(on, Ordering::Relaxed);
    }

    pub fn send(&self, command: Command) {
        let _ = self.commands.send(command);
    }

    pub fn set_max_loaded(&self, n: usize) {
        self.max_loaded.store(n.max(1), Ordering::Relaxed);
    }

    pub fn max_loaded(&self) -> usize {
        self.max_loaded.load(Ordering::Relaxed)
    }

    pub fn loaded(&self) -> Option<Loaded> {
        self.loaded.lock().unwrap().first().cloned()
    }

    pub fn loaded_all(&self) -> Vec<Loaded> {
        self.loaded.lock().unwrap().clone()
    }

    pub fn vision(&self) -> bool {
        self.loaded().is_some_and(|l| l.vision)
    }

    pub fn thinking(&self) -> specrt::chat::Thinking {
        self.loaded().map(|l| l.thinking).unwrap_or_default()
    }

    pub fn tools(&self) -> specrt::chat::ToolFormat {
        self.loaded().map(|l| l.tools).unwrap_or_default()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop_sampler.store(true, Ordering::Relaxed);
        let _ = self.commands.send(Command::Shutdown);
    }
}

struct Shared {
    loaded: LoadedList,
    suspended: LoadedList,
    lives: LiveList,
    stats: StatsSource,
    max_loaded: Arc<AtomicUsize>,
    duplexes: DuplexList,
    transcribers: TranscriberList,
    duplex_options: Arc<Mutex<specrt::duplex::service::ServiceOptions>>,
    duplex_loader: Arc<Mutex<crate::native::DuplexLoader>>,
}

struct Slot {
    info: Loaded,
    backend: Backend,
    reload: Command,
}

fn sample(handle: Handle, stop: Arc<AtomicBool>, wake: Wake) {
    let slice = Duration::from_millis(50);
    let mut since = Duration::ZERO;
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(slice);
        since += slice;
        let lives: Vec<Live> = handle.lives.lock().unwrap().clone();
        let fresh = lives.iter().any(|l| l.history.lock().unwrap().is_empty());
        if since < SAMPLE_EVERY && !fresh {
            continue;
        }
        since = Duration::ZERO;
        for live in &lives {
            let reading = handle.reading_of(Some(&live.model_id));
            live.history.lock().unwrap().push(Instant::now(), reading);
        }
        if handle.watching.load(Ordering::Relaxed) {
            wake();
        }
    }
}

struct Running {
    cancel: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

enum Backend {
    Single(Native),
    Batched { target: Box<dyn Target>, running: Vec<Running> },
}

impl Backend {
    fn media(&self) -> Option<specrt::media::MediaInfo> {
        match self {
            Backend::Single(n) => n.media_info(),
            Backend::Batched { target, .. } => target.media_info(),
        }
    }

    fn stop(self) {
        if let Backend::Batched { running, .. } = self {
            for r in &running {
                r.cancel.store(true, Ordering::Relaxed);
            }
            for r in running {
                let _ = r.thread.join();
            }
        }
    }
}

pub fn serve_options(max_ctx: Option<usize>, seqs: usize, kv_cache_bytes: Option<u64>, engine: &specrt::options::EngineChoice) -> Result<ServeOptions, BackendError> {
    if let Some(solved) = &engine.solved {
        let base = engine.apply_overrides(ServeOptions::default()).map_err(|e| BackendError::new(crate::result::ErrorKind::InvalidRequest, format!("engine options: {e:#}")))?;
        return ServeOptions::from_solved(solved, base).map_err(|e| BackendError::new(crate::result::ErrorKind::InvalidRequest, format!("solved placement: {e:#}")));
    }
    let base = ServeOptions {
        max_model_len: max_ctx,
        max_num_seqs: seqs,
        kv_cache_bytes,
        kv_cache_tokens: None,
        ..ServeOptions::default()
    };
    engine.apply(base).map_err(|e| BackendError::new(crate::result::ErrorKind::InvalidRequest, format!("engine options: {e:#}")))
}

struct Opened {
    backend: Backend,
    vision: bool,
    thinking: specrt::chat::Thinking,
    tools: specrt::chat::ToolFormat,
    seqs: usize,
    note: Option<String>,
    speculation: Option<String>,
    share: Option<Box<dyn Target>>,
}

fn open(
    model_id: &str,
    max_ctx: Option<usize>,
    seqs: usize,
    kv_cache_bytes: Option<u64>,
    engine: &specrt::options::EngineChoice,
    factory: &Factory,
    serving: &ServingFactory,
    progress: &specrt::progress::Sink,
    duplex: &specrt::duplex::service::ServiceOptions,
    duplex_loader: &crate::native::DuplexLoader,
) -> Result<Opened, BackendError> {
    if specrt::duplex::service::resolve_dir(model_id).is_some() || specrt::speech::service::resolve_dir(model_id).is_some() {
        let device = engine.overrides.get(specrt::media::DEVICE_KEY).and_then(|v| v.as_u64()).map(|d| d as usize);
        let mut opts = duplex.clone();
        if let Some(c) = max_ctx {
            opts.max_ctx = c;
        }
        let mut native = Native::with_factory(max_ctx, factory.clone()).on_device(device).with_duplex_loader(duplex_loader.clone());
        native.set_duplex(opts.clone());
        native.set_progress(Some(progress.clone()));
        native.load(model_id)?;
        let seqs = if native.transcriber().is_some() { 1 } else { opts.slots };
        return Ok(Opened { backend: Backend::Single(native), vision: false, thinking: Default::default(), tools: Default::default(), seqs, note: None, speculation: None, share: None });
    }
    if specrt::media::resolve_dir(model_id).is_some() {
        let device = engine.overrides.get(specrt::media::DEVICE_KEY).and_then(|v| v.as_u64()).map(|d| d as usize);
        let mut native = Native::with_factory(max_ctx, factory.clone()).on_device(device);
        native.set_progress(Some(progress.clone()));
        native.load(model_id)?;
        let caps = native.capabilities();
        return Ok(Opened { backend: Backend::Single(native), vision: caps.vision, thinking: caps.thinking, tools: caps.tools, seqs: 1, note: None, speculation: None, share: None });
    }
    serve_options(max_ctx, seqs, kv_cache_bytes, engine)?;
    if let Some(solved) = &engine.solved
        && let Some(mut target) = serving(serve_options(max_ctx, solved.slots, None, engine)?)
    {
        target.set_progress(Some(progress.clone()));
        target.load(model_id)?;
        let caps = target.capabilities();
        let share = target.share();
        let speculation = target.speculation();
        return Ok(Opened { backend: Backend::Batched { target, running: Vec::new() }, vision: caps.vision, thinking: caps.thinking, tools: caps.tools, seqs: solved.slots, note: None, speculation, share });
    }
    let mut note = None;
    let placed = seqs <= 1 && serving(serve_options(max_ctx, 1, kv_cache_bytes, engine)?).is_some_and(|t| t.requires_serving());
    let attempts = if seqs > 1 { vec![seqs, 1] } else if placed { vec![1] } else { Vec::new() };
    for (n, attempt) in attempts.into_iter().enumerate() {
        let pool = if attempt > 1 { kv_cache_bytes } else { None };
        let Some(mut target) = serving(serve_options(max_ctx, attempt, pool, engine)?) else { break };
        target.set_progress(Some(progress.clone()));
        match target.load(model_id) {
            Ok(()) => {
                let caps = target.capabilities();
                let share = target.share();
                let speculation = target.speculation();
                return Ok(Opened { backend: Backend::Batched { target, running: Vec::new() }, vision: caps.vision, thinking: caps.thinking, tools: caps.tools, seqs: attempt, note, speculation, share });
            }
            Err(e) => {
                tracing::warn!(model = model_id, seqs = attempt, error = %e, "the serving engine could not load");
                if target.requires_serving() {
                    return Err(e);
                }
                if n == 0 {
                    note = Some(format!("parallel requests off: {}", e.message));
                }
            }
        }
    }
    let mut native = Native::with_factory(max_ctx, factory.clone());
    native.set_progress(Some(progress.clone()));
    native.load(model_id)?;
    let caps = native.capabilities();
    Ok(Opened { backend: Backend::Single(native), vision: caps.vision, thinking: caps.thinking, tools: caps.tools, seqs: 1, note, speculation: None, share: None })
}

struct Parked {
    info: Loaded,
    load: Command,
}

fn slot_index(slots: &[Slot], model: Option<&str>) -> Option<usize> {
    match model {
        Some(name) => slots.iter().position(|s| s.info.answers_to(name)),
        None => (!slots.is_empty()).then_some(0),
    }
}

fn publish(shared: &Shared, slots: &[Slot], parked: &[Parked]) {
    *shared.loaded.lock().unwrap() = slots.iter().map(|s| s.info.clone()).collect();
    *shared.suspended.lock().unwrap() = parked.iter().map(|p| p.info.clone()).collect();
}

fn drop_stats(shared: &Shared, model_id: &str) {
    shared.stats.lock().unwrap().retain(|(id, _)| id != model_id);
    shared.lives.lock().unwrap().retain(|l| l.model_id != model_id);
    shared.duplexes.lock().unwrap().retain(|(id, _)| id != model_id);
    shared.transcribers.lock().unwrap().retain(|(id, _)| id != model_id);
}

fn run(
    inbox: Receiver<Command>,
    events: Sender<EngineEvent>,
    shared: Shared,
    wake: Wake,
    factory: Factory,
    serving: ServingFactory,
    gpu: GpuSampler,
) {
    let mut slots: Vec<Slot> = Vec::new();
    let mut parked: Vec<Parked> = Vec::new();
    let emit = |e: EngineEvent| {
        let _ = events.send(e);
        wake();
    };
    let mut resume_reply: Option<Sender<Result<(), BackendError>>> = None;
    let mut next: Option<Command> = None;
    loop {
        let command = match next.take() {
            Some(c) => c,
            None => match inbox.recv() {
                Ok(c) => c,
                Err(_) => break,
            },
        };
        match command {
            Command::Shutdown => break,
            Command::Unload { model } => {
                match model.as_deref() {
                    None => {
                        for s in slots.drain(..) {
                            drop_stats(&shared, &s.info.model_id);
                            s.backend.stop();
                        }
                        parked.clear();
                    }
                    Some(name) => {
                        if let Some(i) = slot_index(&slots, Some(name)) {
                            let s = slots.remove(i);
                            drop_stats(&shared, &s.info.model_id);
                            s.backend.stop();
                        }
                        parked.retain(|p| !p.info.answers_to(name));
                    }
                }
                publish(&shared, &slots, &parked);
                emit(EngineEvent::Unloaded);
            }
            Command::Suspend { model, done } => {
                let targets: Vec<usize> = match model.as_deref() {
                    None => (0..slots.len()).collect(),
                    Some(name) => slot_index(&slots, Some(name)).into_iter().collect(),
                };
                let any = !targets.is_empty();
                for i in targets.into_iter().rev() {
                    let s = slots.remove(i);
                    drop_stats(&shared, &s.info.model_id);
                    s.backend.stop();
                    parked.push(Parked { info: s.info, load: s.reload });
                }
                if any {
                    publish(&shared, &slots, &parked);
                    emit(EngineEvent::Unloaded);
                }
                let _ = done.send(any);
            }
            Command::Resume { model, done } => {
                let at = match model.as_deref() {
                    Some(name) => parked.iter().position(|p| p.info.answers_to(name)),
                    None => (!parked.is_empty()).then_some(0),
                };
                match at {
                    Some(i) => {
                        resume_reply = Some(done);
                        next = Some(parked.remove(i).load);
                    }
                    None => {
                        let loaded = match model.as_deref() {
                            Some(name) => slot_index(&slots, Some(name)).is_some(),
                            None => !slots.is_empty(),
                        };
                        let _ = done.send(match loaded {
                            true => Ok(()),
                            false => Err(BackendError::new(ErrorKind::ModelNotFound, "no model is loaded in the app")),
                        });
                    }
                }
            }
            Command::Load { model_id, display, max_ctx, max_num_seqs, kv_cache_bytes, served_names, engine, alongside } => {
                let reload = Command::Load { model_id: model_id.clone(), display: display.clone(), max_ctx, max_num_seqs, kv_cache_bytes, served_names: served_names.clone(), engine: engine.clone(), alongside: true };
                let reply_failure = |e: BackendError, resume_reply: &mut Option<Sender<Result<(), BackendError>>>| {
                    if let Some(done) = resume_reply.take() {
                        let _ = done.send(Err(e.clone()));
                    }
                    emit(EngineEvent::LoadFailed(e));
                };
                if alongside {
                    if let Some(i) = slots.iter().position(|s| s.info.model_id == model_id) {
                        let s = slots.remove(i);
                        drop_stats(&shared, &s.info.model_id);
                        s.backend.stop();
                    }
                    parked.retain(|p| !matches!(&p.load, Command::Load { model_id: m, .. } if *m == model_id));
                    let limit = shared.max_loaded.load(Ordering::Relaxed).max(1);
                    if slots.len() >= limit {
                        publish(&shared, &slots, &parked);
                        reply_failure(
                            BackendError::new(ErrorKind::Unavailable, format!("{display} was not loaded: {} model(s) are loaded and the limit is {limit}. Unload one first", slots.len())),
                            &mut resume_reply,
                        );
                        continue;
                    }
                } else {
                    for s in slots.drain(..) {
                        drop_stats(&shared, &s.info.model_id);
                        s.backend.stop();
                    }
                    parked.clear();
                }
                publish(&shared, &slots, &parked);
                emit(EngineEvent::Loading(display.clone()));
                let started = Instant::now();
                let before = gpu();
                let vram_before = crate::gpu::used_total(&before).map(|(used, _)| used);
                let seqs = max_num_seqs.unwrap_or(1).max(1);
                let sink = {
                    let (events, wake) = (events.clone(), wake.clone());
                    specrt::progress::Sink::new(move |p| {
                        let _ = events.send(EngineEvent::LoadProgress { stage: p.stage.label().to_string(), done: p.done, total: p.total });
                        wake();
                    })
                };
                let mut duplex_options = shared.duplex_options.lock().unwrap().clone();
                if let Some(n) = max_num_seqs {
                    duplex_options.slots = n.max(1);
                }
                let duplex_loader = shared.duplex_loader.lock().unwrap().clone();
                match open(&model_id, max_ctx, seqs, kv_cache_bytes, &engine, &factory, &serving, &sink, &duplex_options, &duplex_loader) {
                    Ok(o) => {
                        let ctx = engine.solved.as_ref().map(|s| s.context).or(max_ctx).unwrap_or(specrt::engine::DEFAULT_CTX);
                        let after = gpu();
                        let vram_bytes = vram_before.zip(crate::gpu::used_total(&after).map(|(used, _)| used)).map(|(b, a)| a.saturating_sub(b)).filter(|v| *v > 0);
                        let landed = crate::gpu::landed(&before, &after);
                        let info = Loaded {
                            model_id: model_id.clone(),
                            display,
                            load_secs: started.elapsed().as_secs_f64(),
                            vision: o.vision,
                            ctx,
                            max_num_seqs: o.seqs,
                            served_names,
                            parallel_note: o.note,
                            engine,
                            speculation: o.speculation,
                            thinking: o.thinking,
                            tools: o.tools,
                            vram_bytes,
                            landed,
                            kind: match &o.backend { Backend::Single(n) if n.duplex().is_some() => specrt::media::ModelKind::Duplex, Backend::Single(n) if n.transcriber().is_some() => specrt::media::ModelKind::Transcription, b => b.media().map(|m| m.kind).unwrap_or_default() },
                            media: o.backend.media(),
                        };
                        if let Some(share) = o.share {
                            shared.stats.lock().unwrap().push((model_id.clone(), share));
                        }
                        if let Backend::Single(n) = &o.backend
                            && let Some(d) = n.duplex()
                        {
                            shared.duplexes.lock().unwrap().push((model_id.clone(), d));
                        }
                        if let Backend::Single(n) = &o.backend
                            && let Some(t) = n.transcriber()
                        {
                            shared.transcribers.lock().unwrap().push((model_id.clone(), t));
                        }
                        shared.lives.lock().unwrap().push(Live::new(&model_id));
                        slots.push(Slot { info: info.clone(), backend: o.backend, reload });
                        publish(&shared, &slots, &parked);
                        if let Some(done) = resume_reply.take() {
                            let _ = done.send(Ok(()));
                        }
                        emit(EngineEvent::Loaded(info));
                    }
                    Err(e) => reply_failure(e, &mut resume_reply),
                }
            }
            Command::Media { model, job, events: out, cancel } => {
                let send = |out: &Sender<MediaUpdate>, u: MediaUpdate| {
                    let _ = out.send(u);
                    wake();
                };
                let Some(at) = slot_index(&slots, model.as_deref()) else {
                    let message = match &model {
                        Some(name) => format!("model {name} is not loaded"),
                        None => "no model is loaded".to_string(),
                    };
                    send(&out, MediaUpdate::Failed(BackendError::new(ErrorKind::ModelNotFound, message)));
                    continue;
                };
                let result = match &mut slots[at].backend {
                    Backend::Single(b) => b.generate_media(&job, &mut |e| {
                        send(&out, MediaUpdate::Event(e));
                        !cancel.load(Ordering::SeqCst)
                    }),
                    Backend::Batched { target, .. } => target.generate_media(&job, &mut |e| {
                        send(&out, MediaUpdate::Event(e));
                        !cancel.load(Ordering::SeqCst)
                    }),
                };
                send(&out, match result {
                    Ok(o) => MediaUpdate::Done(o),
                    Err(e) => MediaUpdate::Failed(e),
                });
            }
            Command::Chat { model, mut request, events: out, queued, meter } => {
                let send = {
                    let wake = wake.clone();
                    move |out: &Sender<ChatEvent>, e: ChatEvent| {
                        let _ = out.send(e);
                        wake();
                    }
                };
                let Some(at) = slot_index(&slots, model.as_deref()) else {
                    meter.withdrawn();
                    let message = match &model {
                        Some(name) => format!("model {name} is not loaded"),
                        None => "no model is loaded".to_string(),
                    };
                    send(&out, ChatEvent::Failed(BackendError::failed(message)));
                    continue;
                };
                match &mut slots[at].backend {
                    Backend::Single(b) => {
                        let mut tracked = meter.begin(queued);
                        let result = b.chat(&request, &mut |t| {
                            tracked.piece();
                            send(&out, ChatEvent::Text(t.to_string()));
                        });
                        send(
                            &out,
                            match result {
                                Ok(g) => {
                                    tracked.finish(&g);
                                    ChatEvent::Done(g)
                                }
                                Err(e) => ChatEvent::Failed(e),
                            },
                        );
                    }
                    Backend::Batched { target, running } => {
                        if let Some(reason) = target.failure() {
                            meter.withdrawn();
                            send(&out, ChatEvent::Failed(BackendError::new(ErrorKind::Unavailable, format!("the model failed: {reason}. Load it again from the model bar"))));
                            continue;
                        }
                        running.retain(|r| !r.thread.is_finished());
                        let Some(mut shared_target) = target.share() else {
                            meter.withdrawn();
                            send(&out, ChatEvent::Failed(BackendError::failed("the batched engine has stopped")));
                            continue;
                        };
                        let cancel = request.cancel.clone().unwrap_or_default();
                        request.cancel = Some(cancel.clone());
                        let mut tracked = meter.begin(queued);
                        let send = send.clone();
                        let thread = std::thread::spawn(move || {
                            let result = shared_target.chat(&request, &mut |t| {
                                tracked.piece();
                                send(&out, ChatEvent::Text(t.to_string()));
                            });
                            send(
                                &out,
                                match result {
                                    Ok(g) => {
                                        tracked.finish(&g);
                                        ChatEvent::Done(g)
                                    }
                                    Err(e) => ChatEvent::Failed(e),
                                },
                            );
                        });
                        running.push(Running { cancel, thread });
                    }
                }
            }
        }
    }
    for s in slots.drain(..) {
        s.backend.stop();
    }
    shared.stats.lock().unwrap().clear();
    shared.lives.lock().unwrap().clear();
    publish(&shared, &[], &[]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Message, Role, Sampling};
    use crate::fake::{Script, ScriptedModel};
    use crate::target::{Capabilities, Request};
    use std::time::Duration;

    fn script() -> Script {
        Script {
            models: vec!["m".into()],
            chunks: vec!["Hel".into(), "lo ".into(), "there".into()],
            chunk_delay_ms: 5,
            first_delay_ms: 20,
            usage: Some((7, 3)),
            fail_status: None,
            offload: None,
            vision: false,
            files: Default::default(),
            sessions: Default::default(),
        }
    }

    fn load(model: &str, display: &str) -> Command {
        Command::Load { model_id: model.into(), display: display.into(), max_ctx: None, max_num_seqs: None, kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default(), alongside: false }
    }

    fn engine() -> Engine {
        Engine::with_factory(Arc::new(|| {}), ScriptedModel::factory(script()))
    }

    fn wait_event(e: &Engine) -> EngineEvent {
        e.events.recv_timeout(Duration::from_secs(10)).unwrap()
    }

    fn image_engine(step_ms: u64) -> (Engine, Arc<std::sync::atomic::AtomicUsize>) {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let f = crate::fake::FakeImage::factory(step_ms, runs.clone(), Some(ScriptedModel::factory(script())));
        (Engine::with_factories_sampled(Arc::new(|| {}), f, Arc::new(|_| None), no_gpu_sampler()), runs)
    }

    #[test]
    fn an_image_model_loads_as_its_own_kind_and_streams_step_progress() {
        let (e, _) = image_engine(1);
        let dir = crate::fake::tiny_pipeline().display().to_string();
        e.send(load(&dir, "tiny-klein"));
        assert!(matches!(wait_event(&e), EngineEvent::Loading(_)));
        let EngineEvent::Loaded(l) = wait_event(&e) else { panic!("not loaded") };
        assert_eq!(l.kind, specrt::media::ModelKind::Image);
        assert_eq!(l.media.as_ref().unwrap().family, "flux2_klein");
        let job = specrt::media::MediaJob { n: 2, steps: Some(3), ..specrt::media::MediaJob::image("a fox", 32, 32, 5) };
        let rx = e.handle().generate_media(None, job, Default::default());
        let mut steps = Vec::new();
        let out = loop {
            match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
                MediaUpdate::Event(specrt::media::MediaEvent::Progress { image, step, steps: n }) => steps.push((image, step, n)),
                MediaUpdate::Event(_) => {}
                MediaUpdate::Done(o) => break o,
                MediaUpdate::Failed(err) => panic!("{err}"),
            }
        };
        assert_eq!(steps, vec![(0, 1, 3), (0, 2, 3), (0, 3, 3), (1, 1, 3), (1, 2, 3), (1, 3, 3)]);
        assert_eq!(out.images.iter().map(|p| p.seed).collect::<Vec<_>>(), [5, 6]);
        let text = ask(&e.handle(), "hi");
        assert!(text.2.unwrap().kind == ErrorKind::InvalidRequest, "a chat request to an image model is a 400");
    }

    #[test]
    fn cancelling_a_media_job_stops_it_between_steps() {
        let (e, _) = image_engine(20);
        e.send(load(&crate::fake::tiny_pipeline().display().to_string(), "tiny-klein"));
        wait_event(&e);
        wait_event(&e);
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = e.handle().generate_media(None, specrt::media::MediaJob { steps: Some(50), ..specrt::media::MediaJob::image("a fox", 32, 32, 1) }, cancel.clone());
        let mut seen = 0;
        let failed = loop {
            match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
                MediaUpdate::Event(_) => {
                    seen += 1;
                    if seen == 2 {
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
                MediaUpdate::Done(_) => panic!("ran to the end"),
                MediaUpdate::Failed(err) => break err,
            }
        };
        assert_eq!(failed.kind, ErrorKind::Cancelled);
        assert!(seen < 10, "{seen} steps ran after the cancel");
    }

    #[test]
    fn an_invalid_media_job_fails_as_an_invalid_request_and_a_text_model_refuses_one() {
        let (e, runs) = image_engine(1);
        e.send(load(&crate::fake::tiny_pipeline().display().to_string(), "tiny-klein"));
        wait_event(&e);
        wait_event(&e);
        let rx = e.handle().generate_media(None, specrt::media::MediaJob::image("a fox", 30, 32, 1), Default::default());
        let MediaUpdate::Failed(err) = rx.recv_timeout(Duration::from_secs(10)).unwrap() else { panic!("not refused") };
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(err.message.contains("multiple of 4"), "{}", err.message);
        assert_eq!(runs.load(Ordering::SeqCst), 0, "refused before any step");
        e.send(load("m", "m"));
        wait_event(&e);
        let EngineEvent::Loaded(l) = wait_event(&e) else { panic!() };
        assert_eq!(l.kind, specrt::media::ModelKind::Text);
        let rx = e.handle().generate_media(None, specrt::media::MediaJob::image("a fox", 32, 32, 1), Default::default());
        let MediaUpdate::Failed(err) = rx.recv_timeout(Duration::from_secs(10)).unwrap() else { panic!() };
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
    }

    fn ask(h: &Handle, text: &str) -> (String, Option<Generation>, Option<BackendError>) {
        let rx = h.chat(ChatRequest::new(vec![Message::new(Role::User, text)], Sampling::default()));
        let (mut text, mut done, mut failed) = (String::new(), None, None);
        loop {
            match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
                ChatEvent::Text(t) => text.push_str(&t),
                ChatEvent::Done(g) => {
                    done = Some(g);
                    break;
                }
                ChatEvent::Failed(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        (text, done, failed)
    }

    #[test]
    fn a_duplex_model_loads_on_the_chosen_device_with_its_sessions_and_refuses_text() {
        let loads = Arc::new(AtomicUsize::new(0));
        let e = engine();
        e.set_duplex_loader(crate::fake::echo_duplex_loader(loads.clone()));
        let dir = crate::fake::tiny_haru().display().to_string();
        let mut engine_choice = specrt::options::EngineChoice::default();
        engine_choice.overrides.insert(specrt::media::DEVICE_KEY.into(), serde_json::json!(1));
        e.send(Command::Load { model_id: dir.clone(), display: "tiny-haru".into(), max_ctx: Some(1024), max_num_seqs: Some(3), kv_cache_bytes: None, served_names: Vec::new(), engine: engine_choice, alongside: false });
        assert!(matches!(wait_event(&e), EngineEvent::Loading(_)));
        let EngineEvent::Loaded(l) = wait_event(&e) else { panic!("not loaded") };
        assert_eq!((l.kind, l.max_num_seqs, l.ctx), (specrt::media::ModelKind::Duplex, 3, 1024));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        let s = e.handle().duplex_of(Some("tiny-haru")).expect("the service is reachable by the display name");
        assert_eq!((s.info.slots, s.info.max_ctx, s.info.device, s.info.frame_samples), (3, 1024, 1, 3200));
        let (_, _, failed) = ask(&e.handle(), "hi");
        assert!(failed.unwrap().message.contains("/v1/duplex"));
        e.send(Command::Unload { model: None });
        assert!(matches!(wait_event(&e), EngineEvent::Unloaded));
        assert!(e.handle().duplex_of(None).is_none());
    }

    #[test]
    fn a_loaded_model_streams_a_reply_with_stats() {
        let e = engine();
        e.send(load("m", "M"));
        assert_eq!(wait_event(&e), EngineEvent::Loading("M".into()));
        assert!(matches!(wait_event(&e), EngineEvent::Loaded(l) if l.display == "M"));
        assert_eq!(e.loaded().unwrap().model_id, "m");
        let (text, done, _) = ask(&e.handle(), "hi");
        assert_eq!(text, "Hello there");
        let g = done.unwrap();
        assert_eq!((g.prompt_tokens, g.completion_tokens), (7, 3));
        assert!(g.ttft_secs >= 0.02);
    }

    #[test]
    fn chat_without_a_model_fails_and_unload_clears_the_state() {
        let e = engine();
        let (_, _, failed) = ask(&e.handle(), "hi");
        assert!(failed.unwrap().message.contains("no model"));
        e.send(load("m", "M"));
        wait_event(&e);
        wait_event(&e);
        e.send(Command::Unload { model: None });
        assert_eq!(wait_event(&e), EngineEvent::Unloaded);
        assert!(e.loaded().is_none());
    }

    #[test]
    fn a_scripted_load_samples_the_injected_gpu_twice() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let sampler: GpuSampler = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Vec::new()
        });
        let e = Engine::with_factories_sampled(Arc::new(|| {}), ScriptedModel::factory(script()), Arc::new(|_| None), sampler);
        e.send(load("m", "M"));
        wait_event(&e);
        assert!(matches!(wait_event(&e), EngineEvent::Loaded(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_model_that_is_not_on_disk_is_a_load_failure() {
        let e = engine();
        e.send(load("other", "O"));
        wait_event(&e);
        match wait_event(&e) {
            EngineEvent::LoadFailed(err) => assert_eq!(err.kind, crate::result::ErrorKind::ModelNotFound),
            other => panic!("{other:?}"),
        }
        assert!(e.loaded().is_none());
    }

    fn slow_script() -> Script {
        Script { chunks: (0..30).map(|i| format!("t{i} ")).collect(), chunk_delay_ms: 10, first_delay_ms: 20, usage: Some((12, 30)), ..script() }
    }

    fn batched(seqs: Option<usize>) -> Engine {
        let e = crate::fake::engine(Arc::new(|| {}), slow_script());
        e.send(Command::Load { model_id: "m".into(), display: "M".into(), max_ctx: Some(4096), max_num_seqs: seqs, kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default(), alongside: false });
        wait_event(&e);
        assert!(matches!(wait_event(&e), EngineEvent::Loaded(_)));
        e
    }

    #[test]
    fn serve_options_pass_the_pool_size_and_not_context_times_seqs() {
        let o = serve_options(Some(8192), 4, Some(3 << 30), &Default::default()).unwrap();
        assert_eq!((o.kv_cache_bytes, o.kv_cache_tokens, o.max_model_len, o.max_num_seqs), (Some(3 << 30), None, Some(8192), 4));
    }

    #[test]
    fn an_invalid_engine_option_fails_the_load_naming_it() {
        let mut bad = specrt::options::EngineChoice::default();
        bad.set("speculative.draft_tokens", serde_json::json!(0));
        let err = serve_options(Some(4096), 4, None, &bad).unwrap_err();
        assert!(err.message.contains("draft_tokens"), "{}", err.message);
    }

    fn ask_all(h: &Handle, n: usize) -> Vec<Receiver<ChatEvent>> {
        (0..n).map(|i| h.chat(ChatRequest::new(vec![Message::new(Role::User, format!("q{i}"))], Sampling::default()))).collect()
    }

    fn drain(rx: Receiver<ChatEvent>) -> Generation {
        loop {
            match rx.recv_timeout(Duration::from_secs(20)).unwrap() {
                ChatEvent::Done(g) => return g,
                ChatEvent::Failed(e) => panic!("{e}"),
                ChatEvent::Text(_) => {}
            }
        }
    }

    #[test]
    fn four_parallel_requests_run_together_when_the_engine_loads_with_four_sequences() {
        let e = batched(Some(4));
        assert_eq!(e.loaded().unwrap().max_num_seqs, 4);
        let h = e.handle();
        let rxs = ask_all(&h, 4);
        let mut peak = 0;
        let until = Instant::now() + Duration::from_secs(10);
        while Instant::now() < until && peak < 4 {
            peak = peak.max(h.reading().running);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(peak, 4);
        for rx in rxs {
            assert_eq!(drain(rx).completion_tokens, 30);
        }
        assert_eq!(h.reading().running, 0);
        assert_eq!(h.recent_requests().len(), 4);
    }

    #[test]
    fn one_sequence_serves_requests_one_after_another() {
        let e = batched(Some(1));
        let h = e.handle();
        assert!(h.serving_stats().is_none());
        let started = Instant::now();
        let rxs = ask_all(&h, 2);
        let mut peak = 0;
        let done = std::thread::spawn(move || rxs.into_iter().map(drain).count());
        while !done.is_finished() {
            peak = peak.max(h.reading().running);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(done.join().unwrap(), 2);
        assert_eq!(peak, 1);
        assert!(started.elapsed() >= Duration::from_millis(2 * 300));
    }

    #[test]
    fn more_requests_than_sequences_wait_and_unloading_stops_a_running_batch() {
        let e = batched(Some(2));
        let h = e.handle();
        let rxs = ask_all(&h, 4);
        let until = Instant::now() + Duration::from_secs(10);
        let mut waited = false;
        while Instant::now() < until && !waited {
            let r = h.reading();
            waited = r.running == 2 && r.waiting >= 1;
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(waited);
        e.send(Command::Unload { model: None });
        assert_eq!(wait_event(&e), EngineEvent::Unloaded);
        assert!(e.loaded().is_none());
        assert!(h.serving_stats().is_none());
        drop(rxs);
    }

    struct NoParallel;

    impl Target for NoParallel {
        fn name(&self) -> &str {
            "no-parallel"
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn load(&mut self, _: &str) -> Result<(), BackendError> {
            Err(BackendError::failed("the GPU ran out of memory: KV cache"))
        }

        fn generate(&mut self, _: &Request, _: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
            unreachable!()
        }
    }

    #[test]
    fn a_failed_parallel_load_falls_back_to_the_serving_engine_with_one_sequence() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let inner = crate::fake::ScriptedServing::factory(slow_script());
        let log = seen.clone();
        let serving: ServingFactory = Arc::new(move |opts| {
            log.lock().unwrap().push((opts.max_num_seqs, opts.kv_cache_bytes));
            if opts.max_num_seqs > 1 { Some(Box::new(NoParallel) as Box<dyn Target>) } else { inner(opts) }
        });
        let e = Engine::with_factories(Arc::new(|| {}), ScriptedModel::factory(slow_script()), serving);
        e.send(Command::Load { model_id: "m".into(), display: "M".into(), max_ctx: Some(4096), max_num_seqs: Some(4), kv_cache_bytes: Some(1 << 30), served_names: Vec::new(), engine: Default::default(), alongside: false });
        wait_event(&e);
        let EngineEvent::Loaded(l) = wait_event(&e) else { panic!("not loaded") };
        assert_eq!(l.max_num_seqs, 1);
        assert!(l.parallel_note.unwrap().starts_with("parallel requests off: the GPU ran out of memory"));
        assert_eq!(*seen.lock().unwrap(), vec![(4, Some(1 << 30)), (1, None)]);
        assert!(e.handle().serving_stats().is_some());
    }

    struct StageOom;

    impl Target for StageOom {
        fn name(&self) -> &str {
            "stage-oom"
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn requires_serving(&self) -> bool {
            true
        }

        fn load(&mut self, _: &str) -> Result<(), BackendError> {
            Err(BackendError::failed("the GPU ran out of memory: stage 1 (layers 36..64) on GPU 1: KV cache"))
        }

        fn generate(&mut self, _: &Request, _: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
            unreachable!()
        }
    }

    #[test]
    fn a_placed_load_that_fails_reports_the_stage_and_never_retries_on_one_gpu() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let serving: ServingFactory = Arc::new(move |opts| {
            log.lock().unwrap().push(opts.max_num_seqs);
            Some(Box::new(StageOom) as Box<dyn Target>)
        });
        let e = Engine::with_factories(Arc::new(|| {}), ScriptedModel::factory(slow_script()), serving);
        e.send(Command::Load { model_id: "m".into(), display: "M".into(), max_ctx: Some(4096), max_num_seqs: Some(21), kv_cache_bytes: Some(1 << 30), served_names: Vec::new(), engine: Default::default(), alongside: false });
        wait_event(&e);
        match wait_event(&e) {
            EngineEvent::LoadFailed(err) => assert!(err.message.contains("stage 1 (layers 36..64) on GPU 1"), "{}", err.message),
            other => panic!("{other:?}"),
        }
        assert_eq!(*seen.lock().unwrap(), vec![21]);
        assert!(e.loaded().is_none());
    }

    #[test]
    fn the_scripted_fake_cannot_load_unknown_models_batched_either() {
        let e = crate::fake::engine(Arc::new(|| {}), slow_script());
        e.send(Command::Load { model_id: "other".into(), display: "O".into(), max_ctx: None, max_num_seqs: Some(4), kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default(), alongside: false });
        wait_event(&e);
        match wait_event(&e) {
            EngineEvent::Loaded(l) => panic!("{l:?}"),
            EngineEvent::LoadFailed(_) => {}
            other => panic!("{other:?}"),
        }
    }

    fn two_models() -> Engine {
        let e = Engine::with_factory(Arc::new(|| {}), ScriptedModel::factory(Script { models: vec!["a".into(), "b".into()], ..script() }));
        e.set_max_loaded(2);
        e
    }

    fn alongside(model: &str, display: &str) -> Command {
        match load(model, display) {
            Command::Load { model_id, display, max_ctx, max_num_seqs, kv_cache_bytes, served_names, engine, .. } => Command::Load { model_id, display, max_ctx, max_num_seqs, kv_cache_bytes, served_names, engine, alongside: true },
            other => other,
        }
    }

    fn loaded_ok(e: &Engine) {
        loop {
            match wait_event(e) {
                EngineEvent::Loaded(_) => return,
                EngineEvent::LoadFailed(err) => panic!("{err}"),
                _ => {}
            }
        }
    }

    fn ask_model(h: &Handle, model: &str) -> Result<String, BackendError> {
        let rx = h.chat_on(Some(model), ChatRequest::new(vec![Message::new(Role::User, "hi")], Sampling::default()));
        let mut text = String::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
                ChatEvent::Text(t) => text.push_str(&t),
                ChatEvent::Done(_) => return Ok(text),
                ChatEvent::Failed(e) => return Err(e),
            }
        }
    }

    #[test]
    fn a_model_loaded_alongside_keeps_the_first_and_both_answer_by_name() {
        let e = two_models();
        e.send(alongside("a", "A"));
        loaded_ok(&e);
        e.send(alongside("b", "B"));
        loaded_ok(&e);
        let h = e.handle();
        assert_eq!(h.loaded_all().iter().map(|l| l.display.as_str()).collect::<Vec<_>>(), ["A", "B"]);
        assert_eq!(ask_model(&h, "A").unwrap(), "Hello there");
        assert_eq!(ask_model(&h, "b").unwrap(), "Hello there");
        assert_eq!(ask_model(&h, "B").unwrap(), "Hello there");
        assert!(ask_model(&h, "C").unwrap_err().message.contains("not loaded"));
    }

    #[test]
    fn unloading_one_model_leaves_the_other_serving() {
        let e = two_models();
        e.send(alongside("a", "A"));
        loaded_ok(&e);
        e.send(alongside("b", "B"));
        loaded_ok(&e);
        e.send(Command::Unload { model: Some("A".into()) });
        assert_eq!(wait_event(&e), EngineEvent::Unloaded);
        let h = e.handle();
        assert_eq!(h.loaded_all().len(), 1);
        assert!(ask_model(&h, "A").is_err());
        assert_eq!(ask_model(&h, "B").unwrap(), "Hello there");
    }

    #[test]
    fn a_load_past_the_limit_evicts_nothing() {
        let e = engine();
        e.send(alongside("m", "M"));
        loaded_ok(&e);
        e.send(alongside("other", "O"));
        match wait_event(&e) {
            EngineEvent::LoadFailed(err) => assert!(err.message.contains("limit is 1"), "{}", err.message),
            other => panic!("{other:?}"),
        }
        assert_eq!(e.handle().loaded_all().len(), 1);
        assert_eq!(ask_model(&e.handle(), "M").unwrap(), "Hello there");
    }

    #[test]
    fn a_plain_load_replaces_every_loaded_model() {
        let e = two_models();
        e.send(alongside("a", "A"));
        loaded_ok(&e);
        e.send(alongside("b", "B"));
        loaded_ok(&e);
        e.send(load("a", "A"));
        loaded_ok(&e);
        assert_eq!(e.handle().loaded_all().len(), 1);
    }

    #[test]
    fn suspending_one_model_parks_it_and_resume_brings_it_back() {
        let e = two_models();
        e.send(alongside("a", "A"));
        loaded_ok(&e);
        e.send(alongside("b", "B"));
        loaded_ok(&e);
        let h = e.handle();
        assert!(h.suspend_model(Some("A")));
        assert_eq!(h.suspended_all().len(), 1);
        assert_eq!(h.loaded_all().len(), 1);
        h.resume_model(Some("A")).unwrap();
        assert_eq!(h.loaded_all().len(), 2);
        assert!(h.suspended_all().is_empty());
    }

    #[test]
    fn each_loaded_model_keeps_its_own_request_counts_and_history() {
        let e = two_models();
        e.send(alongside("a", "A"));
        loaded_ok(&e);
        e.send(alongside("b", "B"));
        loaded_ok(&e);
        let h = e.handle();
        ask_model(&h, "A").unwrap();
        ask_model(&h, "A").unwrap();
        ask_model(&h, "B").unwrap();
        assert_eq!(h.recent_requests_of(Some("A")).len(), 2);
        assert_eq!(h.recent_requests_of(Some("B")).len(), 1);
        assert_eq!(h.reading_of(Some("A")).requests, 2);
        assert_eq!(h.reading_of(Some("B")).requests, 1);
        assert!(!Arc::ptr_eq(&h.history_of(Some("A")), &h.history_of(Some("B"))));
        e.send(Command::Unload { model: Some("A".into()) });
        assert_eq!(wait_event(&e), EngineEvent::Unloaded);
        assert!(h.recent_requests_of(Some("A")).is_empty());
        assert_eq!(h.recent_requests_of(Some("B")).len(), 1);
    }
}
