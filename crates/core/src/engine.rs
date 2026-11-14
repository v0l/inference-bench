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
    pub parallel_note: Option<String>,
    pub thinking: specrt::chat::Thinking,
}

pub enum Command {
    Load { model_id: String, display: String, max_ctx: Option<usize>, max_num_seqs: Option<usize> },
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

pub fn serve_options(max_ctx: Option<usize>, seqs: usize) -> ServeOptions {
    ServeOptions {
        max_model_len: max_ctx,
        max_num_seqs: seqs,
        kv_cache_tokens: max_ctx.map(|c| c.saturating_mul(seqs)),
        ..ServeOptions::default()
    }
}

struct Opened {
    backend: Backend,
    vision: bool,
    thinking: specrt::chat::Thinking,
    seqs: usize,
    note: Option<String>,
}

fn open(
    model_id: &str,
    max_ctx: Option<usize>,
    seqs: usize,
    factory: &Factory,
    serving: &ServingFactory,
    stats: &StatsSource,
) -> Result<Opened, BackendError> {
    let mut note = None;
    if let Some(mut target) = (seqs > 1).then(|| serving(serve_options(max_ctx, seqs))).flatten() {
        match target.load(model_id) {
            Ok(()) => {
                let caps = target.capabilities();
                *stats.lock().unwrap() = target.share();
                return Ok(Opened { backend: Backend::Batched { target, running: Vec::new() }, vision: caps.vision, thinking: caps.thinking, seqs, note });
            }
            Err(e) => {
                tracing::warn!(model = model_id, seqs, error = %e, "the batched engine could not load, loading with one sequence");
                note = Some(format!("parallel requests off: {}", e.message));
            }
        }
    }
    let mut native = Native::with_factory(max_ctx, factory.clone());
    native.load(model_id)?;
    let caps = native.capabilities();
    Ok(Opened { backend: Backend::Single(native), vision: caps.vision, thinking: caps.thinking, seqs: 1, note })
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
            Command::Load { model_id, display, max_ctx, max_num_seqs } => {
                release(&mut backend);
                emit(EngineEvent::Loading(display.clone()));
                let started = Instant::now();
                let seqs = max_num_seqs.unwrap_or(1).max(1);
                match open(&model_id, max_ctx, seqs, &factory, &serving, &shared.stats) {
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
                            parallel_note: o.note,
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
        Command::Load { model_id: model.into(), display: display.into(), max_ctx: None, max_num_seqs: None }
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
        e.send(Command::Load { model_id: "m".into(), display: "M".into(), max_ctx: Some(4096), max_num_seqs: seqs });
        wait_event(&e);
        assert!(matches!(wait_event(&e), EngineEvent::Loaded(_)));
        e
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

    #[test]
    fn the_scripted_fake_cannot_load_unknown_models_batched_either() {
        let e = crate::fake::engine(Arc::new(|| {}), slow_script());
        e.send(Command::Load { model_id: "other".into(), display: "O".into(), max_ctx: None, max_num_seqs: Some(4) });
        wait_event(&e);
        match wait_event(&e) {
            EngineEvent::Loaded(l) => panic!("{l:?}"),
            EngineEvent::LoadFailed(_) => {}
            other => panic!("{other:?}"),
        }
    }
}
