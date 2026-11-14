use crate::chat::ChatRequest;
use crate::live::{History, HISTORY_SAMPLES, Meter, Reading, RequestRecord};
use crate::native::{Factory, Native, specrt_factory};
use crate::result::{BackendError, Generation};
use crate::target::Target;
use specrt::serve::{ServeOptions, Stats};
use std::sync::atomic::{AtomicBool, Ordering};
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

type StatsSource = Arc<Mutex<Option<Box<dyn Target>>>>;

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
}

pub enum Command {
    Load { model_id: String, display: String, max_ctx: Option<usize>, max_num_seqs: Option<usize>, kv_cache_bytes: Option<u64>, served_names: Vec<String>, engine: specrt::options::EngineChoice },
    Unload,
    Chat { request: ChatRequest, events: Sender<ChatEvent>, queued: Instant },
    Shutdown,
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

pub struct Engine {
    commands: Sender<Command>,
    pub events: Receiver<EngineEvent>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    meter: Arc<Meter>,
    stats: StatsSource,
    history: Arc<Mutex<History>>,
    watching: Arc<AtomicBool>,
    stop_sampler: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct Handle {
    commands: Sender<Command>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    meter: Arc<Meter>,
    stats: StatsSource,
    history: Arc<Mutex<History>>,
    watching: Arc<AtomicBool>,
}

impl Handle {
    pub fn loaded(&self) -> Option<Loaded> {
        self.loaded.lock().unwrap().clone()
    }

    pub fn thinking(&self) -> specrt::chat::Thinking {
        self.loaded().map(|l| l.thinking).unwrap_or_default()
    }

    pub fn chat(&self, request: ChatRequest) -> Receiver<ChatEvent> {
        let (tx, rx) = channel();
        let queued = self.meter.submitted();
        if self.commands.send(Command::Chat { request, events: tx.clone(), queued }).is_err() {
            self.meter.withdrawn();
            let _ = tx.send(ChatEvent::Failed(BackendError::failed("the engine has stopped")));
        }
        rx
    }

    pub fn serving_stats(&self) -> Option<Stats> {
        self.stats.lock().unwrap().as_ref().and_then(|t| t.stats())
    }

    pub fn serving_metrics(&self) -> Option<String> {
        self.stats.lock().unwrap().as_ref().and_then(|t| t.metrics())
    }

    pub fn reading(&self) -> Reading {
        let max = self.loaded().map_or(1, |l| l.max_num_seqs);
        self.meter.read(max, self.serving_stats())
    }

    pub fn recent_requests(&self) -> Vec<RequestRecord> {
        self.meter.recent()
    }

    pub fn history(&self) -> Arc<Mutex<History>> {
        self.history.clone()
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
        Self::with_factories(wake, factory, Arc::new(|_| None))
    }

    pub fn with_factories(wake: Wake, factory: Factory, serving: ServingFactory) -> Self {
        let (commands, inbox) = channel();
        let (events_tx, events) = channel();
        let loaded = Arc::new(Mutex::new(None));
        let meter = Arc::new(Meter::default());
        let stats: StatsSource = Arc::new(Mutex::new(None));
        let history = Arc::new(Mutex::new(History::new(HISTORY_SAMPLES)));
        let watching = Arc::new(AtomicBool::new(false));
        let stop_sampler = Arc::new(AtomicBool::new(false));
        let shared = Shared { loaded: loaded.clone(), meter: meter.clone(), stats: stats.clone(), history: history.clone() };
        let this = Self { commands, events, loaded, meter, stats, history, watching, stop_sampler };
        let handle = this.handle();
        let (stop, wake_sampler) = (this.stop_sampler.clone(), wake.clone());
        std::thread::spawn(move || run(inbox, events_tx, shared, wake, factory, serving));
        std::thread::spawn(move || sample(handle, stop, wake_sampler));
        this
    }

    pub fn handle(&self) -> Handle {
        Handle {
            commands: self.commands.clone(),
            loaded: self.loaded.clone(),
            meter: self.meter.clone(),
            stats: self.stats.clone(),
            history: self.history.clone(),
            watching: self.watching.clone(),
        }
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

    pub fn loaded(&self) -> Option<Loaded> {
        self.loaded.lock().unwrap().clone()
    }

    pub fn vision(&self) -> bool {
        self.loaded().is_some_and(|l| l.vision)
    }

    pub fn thinking(&self) -> specrt::chat::Thinking {
        self.loaded().map(|l| l.thinking).unwrap_or_default()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop_sampler.store(true, Ordering::Relaxed);
        let _ = self.commands.send(Command::Shutdown);
    }
}

struct Shared {
    loaded: Arc<Mutex<Option<Loaded>>>,
    meter: Arc<Meter>,
    stats: StatsSource,
    history: Arc<Mutex<History>>,
}

fn sample(handle: Handle, stop: Arc<AtomicBool>, wake: Wake) {
    let slice = Duration::from_millis(50);
    let mut since = Duration::ZERO;
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(slice);
        since += slice;
        let baseline = handle.loaded().is_some() && handle.history.lock().unwrap().is_empty();
        if since < SAMPLE_EVERY && !baseline {
            continue;
        }
        since = Duration::ZERO;
        handle.history.lock().unwrap().push(Instant::now(), handle.reading());
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

pub fn serve_options(max_ctx: Option<usize>, seqs: usize, kv_cache_bytes: Option<u64>, engine: &specrt::options::EngineChoice) -> ServeOptions {
    let base = ServeOptions {
        max_model_len: max_ctx,
        max_num_seqs: seqs,
        kv_cache_bytes,
        kv_cache_tokens: None,
        ..ServeOptions::default()
    };
    engine.apply(base.clone()).unwrap_or(base)
}

struct Opened {
    backend: Backend,
    vision: bool,
    thinking: specrt::chat::Thinking,
    seqs: usize,
    note: Option<String>,
    speculation: Option<String>,
}

fn open(
    model_id: &str,
    max_ctx: Option<usize>,
    seqs: usize,
    kv_cache_bytes: Option<u64>,
    engine: &specrt::options::EngineChoice,
    factory: &Factory,
    serving: &ServingFactory,
    stats: &StatsSource,
    progress: &specrt::progress::Sink,
) -> Result<Opened, BackendError> {
    let mut note = None;
    let attempts = if seqs > 1 { vec![seqs, 1] } else { Vec::new() };
    for (n, attempt) in attempts.into_iter().enumerate() {
        let Some(mut target) = serving(serve_options(max_ctx, attempt, kv_cache_bytes, engine)) else { break };
        target.set_progress(Some(progress.clone()));
        match target.load(model_id) {
            Ok(()) => {
                let caps = target.capabilities();
                *stats.lock().unwrap() = target.share();
                let speculation = target.speculation();
                return Ok(Opened { backend: Backend::Batched { target, running: Vec::new() }, vision: caps.vision, thinking: caps.thinking, seqs: attempt, note, speculation });
            }
            Err(e) => {
                tracing::warn!(model = model_id, seqs = attempt, error = %e, "the serving engine could not load");
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
    Ok(Opened { backend: Backend::Single(native), vision: caps.vision, thinking: caps.thinking, seqs: 1, note, speculation: None })
}

fn run(
    inbox: Receiver<Command>,
    events: Sender<EngineEvent>,
    shared: Shared,
    wake: Wake,
    factory: Factory,
    serving: ServingFactory,
) {
    let mut backend: Option<Backend> = None;
    let emit = |e: EngineEvent| {
        let _ = events.send(e);
        wake();
    };
    let release = |backend: &mut Option<Backend>| {
        if let Some(b) = backend.take() {
            b.stop();
        }
        *shared.stats.lock().unwrap() = None;
        *shared.loaded.lock().unwrap() = None;
        shared.history.lock().unwrap().clear();
    };
    while let Ok(command) = inbox.recv() {
        match command {
            Command::Shutdown => break,
            Command::Unload => {
                release(&mut backend);
                emit(EngineEvent::Unloaded);
            }
            Command::Load { model_id, display, max_ctx, max_num_seqs, kv_cache_bytes, served_names, engine } => {
                release(&mut backend);
                emit(EngineEvent::Loading(display.clone()));
                let started = Instant::now();
                let seqs = max_num_seqs.unwrap_or(1).max(1);
                let sink = {
                    let (events, wake) = (events.clone(), wake.clone());
                    specrt::progress::Sink::new(move |p| {
                        let _ = events.send(EngineEvent::LoadProgress { stage: p.stage.label().to_string(), done: p.done, total: p.total });
                        wake();
                    })
                };
                match open(&model_id, max_ctx, seqs, kv_cache_bytes, &engine, &factory, &serving, &shared.stats, &sink) {
                    Ok(o) => {
                        backend = Some(o.backend);
                        let ctx = max_ctx.unwrap_or(specrt::engine::DEFAULT_CTX);
                        let info = Loaded {
                            model_id,
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
                        };
                        *shared.loaded.lock().unwrap() = Some(info.clone());
                        emit(EngineEvent::Loaded(info));
                    }
                    Err(e) => emit(EngineEvent::LoadFailed(e)),
                }
            }
            Command::Chat { mut request, events: out, queued } => {
                let send = {
                    let wake = wake.clone();
                    move |out: &Sender<ChatEvent>, e: ChatEvent| {
                        let _ = out.send(e);
                        wake();
                    }
                };
                match backend.as_mut() {
                    None => {
                        shared.meter.withdrawn();
                        send(&out, ChatEvent::Failed(BackendError::failed("no model is loaded")));
                    }
                    Some(Backend::Single(b)) => {
                        let mut tracked = shared.meter.begin(queued);
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
                    Some(Backend::Batched { target, running }) => {
                        running.retain(|r| !r.thread.is_finished());
                        let Some(mut shared_target) = target.share() else {
                            shared.meter.withdrawn();
                            send(&out, ChatEvent::Failed(BackendError::failed("the batched engine has stopped")));
                            continue;
                        };
                        let cancel = request.cancel.clone().unwrap_or_default();
                        request.cancel = Some(cancel.clone());
                        let mut tracked = shared.meter.begin(queued);
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
    release(&mut backend);
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
            vision: false,
            files: Default::default(),
        }
    }

    fn load(model: &str, display: &str) -> Command {
        Command::Load { model_id: model.into(), display: display.into(), max_ctx: None, max_num_seqs: None, kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default() }
    }

    fn engine() -> Engine {
        Engine::with_factory(Arc::new(|| {}), ScriptedModel::factory(script()))
    }

    fn wait_event(e: &Engine) -> EngineEvent {
        e.events.recv_timeout(Duration::from_secs(10)).unwrap()
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
        e.send(Command::Unload);
        assert_eq!(wait_event(&e), EngineEvent::Unloaded);
        assert!(e.loaded().is_none());
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
        e.send(Command::Load { model_id: "m".into(), display: "M".into(), max_ctx: Some(4096), max_num_seqs: seqs, kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default() });
        wait_event(&e);
        assert!(matches!(wait_event(&e), EngineEvent::Loaded(_)));
        e
    }

    #[test]
    fn serve_options_pass_the_pool_size_and_not_context_times_seqs() {
        let o = serve_options(Some(8192), 4, Some(3 << 30), &Default::default());
        assert_eq!((o.kv_cache_bytes, o.kv_cache_tokens, o.max_model_len, o.max_num_seqs), (Some(3 << 30), None, Some(8192), 4));
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
        e.send(Command::Unload);
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
        e.send(Command::Load { model_id: "m".into(), display: "M".into(), max_ctx: Some(4096), max_num_seqs: Some(4), kv_cache_bytes: Some(1 << 30), served_names: Vec::new(), engine: Default::default() });
        wait_event(&e);
        let EngineEvent::Loaded(l) = wait_event(&e) else { panic!("not loaded") };
        assert_eq!(l.max_num_seqs, 1);
        assert!(l.parallel_note.unwrap().starts_with("parallel requests off: the GPU ran out of memory"));
        assert_eq!(*seen.lock().unwrap(), vec![(4, Some(1 << 30)), (1, Some(1 << 30))]);
        assert!(e.handle().serving_stats().is_some());
    }

    #[test]
    fn the_scripted_fake_cannot_load_unknown_models_batched_either() {
        let e = crate::fake::engine(Arc::new(|| {}), slow_script());
        e.send(Command::Load { model_id: "other".into(), display: "O".into(), max_ctx: None, max_num_seqs: Some(4), kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default() });
        wait_event(&e);
        match wait_event(&e) {
            EngineEvent::Loaded(l) => panic!("{l:?}"),
            EngineEvent::LoadFailed(_) => {}
            other => panic!("{other:?}"),
        }
    }
}
