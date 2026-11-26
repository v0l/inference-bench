use crate::target::{Target, Capabilities, Logits, Request};
use crate::chat::ChatRequest;
use crate::result::{BackendError, ErrorKind, Generation};
use serde_json::Value;
use specrt::engine::{FailureKind, PromptLogits, Reply, failure_kind};
use specrt::runtime::Sampling;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub trait Model {
    fn generate(&mut self, prompt: &str, max_tokens: usize, sampling: Sampling) -> anyhow::Result<Reply>;

    fn chat(
        &mut self,
        messages: &[Value],
        template: Option<&Value>,
        max_tokens: usize,
        sampling: Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<Reply>;

    fn logits(&mut self, prompt: &str) -> anyhow::Result<PromptLogits>;

    fn accepts_media(&self) -> bool {
        false
    }

    fn sampling_defaults(&self) -> specrt::engine::SamplingDefaults {
        Default::default()
    }

    fn thinking(&self) -> specrt::chat::Thinking {
        Default::default()
    }

    fn tool_format(&self) -> specrt::chat::ToolFormat {
        Default::default()
    }

    fn chat_media(
        &mut self,
        _messages: &[Value],
        _template: Option<&Value>,
        _items: &[specrt::vision::MediaItem],
        _budget: specrt::vision::MediaBudget,
        _max_tokens: usize,
        _sampling: Sampling,
        _on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<(Reply, usize)> {
        anyhow::bail!("the model does not accept image or video input")
    }

    fn media_info(&self) -> Option<specrt::media::MediaInfo> {
        None
    }

    fn generate_media(&mut self, _job: &specrt::media::MediaJob, _on_event: &mut dyn FnMut(specrt::media::MediaEvent) -> bool) -> anyhow::Result<specrt::media::MediaOutput> {
        anyhow::bail!(specrt::media::InvalidJob("this is a text model; image and video jobs need an image or video model".into()))
    }
}

pub struct MediaModel(pub specrt::media::ImageModel);

fn not_text(kind: specrt::media::ModelKind) -> anyhow::Error {
    anyhow::anyhow!(specrt::media::InvalidJob(format!("this is an {} model: send it to /v1/images/generations, not a text route", kind.label())))
}

impl Model for MediaModel {
    fn generate(&mut self, _: &str, _: usize, _: Sampling) -> anyhow::Result<Reply> {
        Err(not_text(self.0.info.kind))
    }

    fn chat(&mut self, _: &[Value], _: Option<&Value>, _: usize, _: Sampling, _: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<Reply> {
        Err(not_text(self.0.info.kind))
    }

    fn logits(&mut self, _: &str) -> anyhow::Result<PromptLogits> {
        Err(not_text(self.0.info.kind))
    }

    fn media_info(&self) -> Option<specrt::media::MediaInfo> {
        Some(self.0.info.clone())
    }

    fn generate_media(&mut self, job: &specrt::media::MediaJob, on_event: &mut dyn FnMut(specrt::media::MediaEvent) -> bool) -> anyhow::Result<specrt::media::MediaOutput> {
        self.0.run(job, on_event)
    }
}

impl Model for specrt::Engine {
    fn generate(&mut self, prompt: &str, max_tokens: usize, sampling: Sampling) -> anyhow::Result<Reply> {
        specrt::Engine::generate(self, prompt, max_tokens, sampling)
    }

    fn chat(
        &mut self,
        messages: &[Value],
        template: Option<&Value>,
        max_tokens: usize,
        sampling: Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<Reply> {
        specrt::Engine::chat(self, messages, template, max_tokens, sampling, on_text)
    }

    fn logits(&mut self, prompt: &str) -> anyhow::Result<PromptLogits> {
        specrt::Engine::logits(self, prompt)
    }

    fn accepts_media(&self) -> bool {
        specrt::Engine::accepts_media(self)
    }

    fn sampling_defaults(&self) -> specrt::engine::SamplingDefaults {
        specrt::Engine::sampling_defaults(self)
    }

    fn thinking(&self) -> specrt::chat::Thinking {
        specrt::Engine::thinking(self)
    }

    fn tool_format(&self) -> specrt::chat::ToolFormat {
        specrt::Engine::tool_format(self)
    }

    fn chat_media(
        &mut self,
        messages: &[Value],
        template: Option<&Value>,
        items: &[specrt::vision::MediaItem],
        budget: specrt::vision::MediaBudget,
        max_tokens: usize,
        sampling: Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<(Reply, usize)> {
        specrt::Engine::chat_media(self, messages, template, items, budget, max_tokens, sampling, on_text)
    }
}

pub type Factory = Arc<dyn Fn(&str, Option<usize>) -> anyhow::Result<Box<dyn Model>> + Send + Sync>;

pub fn specrt_factory() -> Factory {
    specrt_factory_with(None)
}

pub fn specrt_factory_with(runtime: Option<specrt::spec::RuntimeOptions>) -> Factory {
    Arc::new(move |model, max_ctx| {
        if specrt::media::resolve_dir(model).is_some() {
            return Ok(Box::new(MediaModel(specrt::media::ImageModel::load(model, runtime.as_ref())?)) as Box<dyn Model>);
        }
        Ok(Box::new(specrt::Engine::load_with(model, max_ctx, runtime.as_ref())?) as Box<dyn Model>)
    })
}

type Job = Box<dyn FnOnce(&mut dyn Model) + Send>;

struct Worker {
    jobs: Option<Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    panic: Arc<Mutex<Option<String>>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub struct Native {
    max_ctx: Option<usize>,
    factory: Factory,
    worker: Option<Worker>,
    vision: bool,
    thinking: specrt::chat::Thinking,
    tools: specrt::chat::ToolFormat,
    progress: Option<specrt::progress::Sink>,
    media: Option<specrt::media::MediaInfo>,
    device: Option<usize>,
    duplex: Option<Arc<specrt::duplex::service::Service>>,
    duplex_options: specrt::duplex::service::ServiceOptions,
}

impl Native {
    pub fn new(max_ctx: Option<usize>) -> Self {
        Self::with_factory(max_ctx, specrt_factory())
    }

    pub fn with_runtime(max_ctx: Option<usize>, runtime: Option<specrt::spec::RuntimeOptions>) -> Self {
        Self::with_factory(max_ctx, specrt_factory_with(runtime))
    }

    pub fn with_factory(max_ctx: Option<usize>, factory: Factory) -> Self {
        Self { max_ctx, factory, worker: None, vision: false, thinking: Default::default(), tools: Default::default(), progress: None, media: None, device: None, duplex: None, duplex_options: Default::default() }
    }

    pub fn on_device(mut self, device: Option<usize>) -> Self {
        self.device = device;
        self
    }

    fn send(&mut self, job: Job) -> Result<(), BackendError> {
        let worker = self.worker.as_ref().ok_or_else(|| BackendError::failed("no model is loaded"))?;
        match worker.jobs.as_ref().map(|j| j.send(job)) {
            Some(Ok(())) => Ok(()),
            _ => Err(self.stopped()),
        }
    }

    fn call<T: Send + 'static>(
        &mut self,
        job: impl FnOnce(&mut dyn Model) -> Result<T, BackendError> + Send + 'static,
    ) -> Result<T, BackendError> {
        let (tx, rx) = channel();
        self.send(Box::new(move |m: &mut dyn Model| {
            let _ = tx.send(job(m));
        }))?;
        rx.recv().map_err(|_| self.stopped())?
    }

    fn stopped(&mut self) -> BackendError {
        let Some(worker) = self.worker.as_mut() else { return BackendError::failed("no model is loaded") };
        if let Some(t) = worker.thread.take() {
            let _ = t.join();
        }
        match worker.panic.lock().unwrap().clone() {
            Some(p) => BackendError::failed(format!("the native engine panicked and was stopped, load the model again: {p}")),
            None => BackendError::failed("the native engine has stopped"),
        }
    }
}

fn not_duplex_text() -> BackendError {
    BackendError::new(ErrorKind::InvalidRequest, "this is a duplex speech model: open a session on the /v1/duplex websocket, not a text route")
}

fn panic_text(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

pub fn engine_error(err: &anyhow::Error) -> BackendError {
    let message = format!("{err:#}");
    if err.downcast_ref::<specrt::media::InvalidJob>().is_some() {
        return BackendError::new(ErrorKind::InvalidRequest, message);
    }
    if err.downcast_ref::<specrt::generate::Cancelled>().is_some() {
        return BackendError::new(ErrorKind::Cancelled, message);
    }
    let lower = message.to_lowercase();
    if lower.contains("out_of_memory") || lower.contains("out of memory") {
        return BackendError::failed(format!("the GPU ran out of memory: {message}"));
    }
    if lower.contains(specrt::serve::HUNG_MARKER) {
        return BackendError::new(ErrorKind::Unavailable, message);
    }
    if lower.contains("chat template:") {
        return BackendError::new(ErrorKind::InvalidRequest, message);
    }
    let kind = match failure_kind(err) {
        FailureKind::UnsupportedArchitecture => ErrorKind::UnsupportedArchitecture,
        FailureKind::ModelNotFound => ErrorKind::ModelNotFound,
        FailureKind::Failed => ErrorKind::Failed,
    };
    BackendError::new(kind, message)
}

fn generation(r: Reply) -> Generation {
    Generation {
        text: r.text,
        prompt_tokens: r.prompt_tokens as u64,
        cached_tokens: 0,
        vision_tokens: 0,
        completion_tokens: r.completion_tokens as u64,
        ttft_secs: r.prefill_secs,
        prefill_secs: r.prefill_secs,
        queue_secs: 0.0,
        decode_secs: r.decode_secs,
        peak_memory_bytes: Some(r.peak_device_bytes),
    }
}

fn run_engine(
    model: String,
    max_ctx: Option<usize>,
    factory: Factory,
    progress: Option<specrt::progress::Sink>,
    device: Option<usize>,
    loaded: Sender<Result<Opened, BackendError>>,
    jobs: std::sync::mpsc::Receiver<Job>,
    panic: Arc<Mutex<Option<String>>>,
) {
    let mut engine = match catch_unwind(AssertUnwindSafe(|| specrt::media::with_device(device, || specrt::progress::with_sink(progress, || factory(&model, max_ctx))))) {
        Ok(Ok(e)) => e,
        Ok(Err(e)) => {
            let _ = loaded.send(Err(engine_error(&e)));
            return;
        }
        Err(p) => {
            let why = panic_text(p.as_ref());
            tracing::error!(model, panic = %why, "the native engine panicked while loading");
            let _ = loaded.send(Err(BackendError::failed(format!("the native engine panicked while loading: {why}"))));
            return;
        }
    };
    let _ = loaded.send(Ok((engine.accepts_media(), engine.thinking(), engine.tool_format(), engine.media_info())));
    while let Ok(job) = jobs.recv() {
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| job(engine.as_mut()))) {
            let why = panic_text(p.as_ref());
            tracing::error!(model, panic = %why, "the native engine panicked, unloading the model");
            *panic.lock().unwrap() = Some(why);
            return;
        }
    }
    drop(engine);
}

enum Streamed {
    Text(String),
    Done(Result<Generation, BackendError>),
}

type Opened = (bool, specrt::chat::Thinking, specrt::chat::ToolFormat, Option<specrt::media::MediaInfo>);

enum MediaStreamed {
    Event(specrt::media::MediaEvent, Sender<bool>),
    Done(Result<specrt::media::MediaOutput, BackendError>),
}

impl Target for Native {
    fn name(&self) -> &str {
        "specrt"
    }

    fn capabilities(&self) -> Capabilities {
        if self.duplex.is_some() {
            return Capabilities { streaming: true, kind: specrt::media::ModelKind::Duplex, ..Capabilities::default() };
        }
        Capabilities { streaming: true, logits: true, memory: true, vision: self.vision, thinking: self.thinking, tools: self.tools, kind: self.media.as_ref().map(|m| m.kind).unwrap_or_default(), ..Capabilities::default() }
    }

    fn duplex(&self) -> Option<Arc<specrt::duplex::service::Service>> {
        self.duplex.clone()
    }

    fn set_duplex(&mut self, options: specrt::duplex::service::ServiceOptions) {
        self.duplex_options = options;
    }

    fn media_info(&self) -> Option<specrt::media::MediaInfo> {
        self.media.clone()
    }

    fn generate_media(&mut self, job: &specrt::media::MediaJob, on_event: &mut dyn FnMut(specrt::media::MediaEvent) -> bool) -> Result<specrt::media::MediaOutput, BackendError> {
        let job = job.clone();
        let (tx, rx) = channel();
        let work: Job = Box::new(move |m: &mut dyn Model| {
            let events = tx.clone();
            let mut forward = |e: specrt::media::MediaEvent| {
                let (back, wait) = channel();
                events.send(MediaStreamed::Event(e, back)).is_ok() && wait.recv().unwrap_or(false)
            };
            let out = m.generate_media(&job, &mut forward).map_err(|e| engine_error(&e));
            let _ = tx.send(MediaStreamed::Done(out));
        });
        self.send(work)?;
        loop {
            match rx.recv() {
                Ok(MediaStreamed::Event(e, back)) => {
                    let _ = back.send(on_event(e));
                }
                Ok(MediaStreamed::Done(out)) => return out,
                Err(_) => return Err(self.stopped()),
            }
        }
    }

    fn set_progress(&mut self, sink: Option<specrt::progress::Sink>) {
        self.progress = sink;
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        self.worker = None;
        self.duplex = None;
        if let Some(dir) = specrt::duplex::service::resolve_dir(model) {
            let opts = self.duplex_options.clone();
            let device = self.device.unwrap_or(0);
            let progress = self.progress.clone();
            let loaded = catch_unwind(AssertUnwindSafe(|| specrt::progress::with_sink(progress, || specrt::duplex::service::Service::load(&dir, opts, device))));
            return match loaded {
                Ok(Ok(s)) => {
                    self.duplex = Some(s);
                    Ok(())
                }
                Ok(Err(e)) => Err(engine_error(&e)),
                Err(p) => Err(BackendError::failed(format!("the duplex engine panicked while loading: {}", panic_text(p.as_ref())))),
            };
        }
        let (jobs, inbox) = channel();
        let (loaded_tx, loaded_rx) = channel();
        let panic = Arc::new(Mutex::new(None));
        let (name, max_ctx, factory, shared, progress, device) =
            (model.to_string(), self.max_ctx, self.factory.clone(), panic.clone(), self.progress.clone(), self.device);
        let thread = std::thread::Builder::new()
            .name("native-engine".into())
            .spawn(move || run_engine(name, max_ctx, factory, progress, device, loaded_tx, inbox, shared))
            .map_err(|e| BackendError::failed(format!("cannot start the engine thread: {e}")))?;
        let worker = Worker { jobs: Some(jobs), thread: Some(thread), panic };
        let outcome = loaded_rx.recv().unwrap_or_else(|_| Err(BackendError::failed("the native engine stopped while loading")));
        match outcome {
            Ok((vision, thinking, tools, media)) => {
                self.worker = Some(worker);
                self.vision = vision;
                self.thinking = thinking;
                self.tools = tools;
                self.media = media;
                Ok(())
            }
            Err(e) => {
                tracing::error!(model, error = %e, "native engine failed to load the model");
                Err(e)
            }
        }
    }

    fn generate(&mut self, request: &Request, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        if self.duplex.is_some() {
            return Err(not_duplex_text());
        }
        let (prompt, max_tokens) = (request.prompt.clone(), request.max_tokens);
        let g = self.call(move |m| {
            m.generate(&prompt, max_tokens, Sampling::GREEDY).map(generation).map_err(|e| engine_error(&e))
        })?;
        on_text(&g.text);
        Ok(g)
    }

    fn chat(&mut self, request: &ChatRequest, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        if self.duplex.is_some() {
            return Err(not_duplex_text());
        }
        let messages = request.messages_json();
        let media = request.media.clone();
        let budget = request.media_budget();
        let (template, s, cancel) = (request.template().cloned(), request.sampling, request.cancel.clone());
        let (tx, rx) = channel();
        let job: Job = Box::new(move |m: &mut dyn Model| {
            let list = messages.as_array().cloned().unwrap_or_default();
            let text = tx.clone();
            let mut send = |t: &str| {
                let _ = text.send(Streamed::Text(t.to_string()));
                !cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            };
            let resolved = s.resolve(m.sampling_defaults());
            let out = if media.is_empty() {
                m.chat(&list, template.as_ref(), s.max_tokens, resolved, &mut send).map(generation)
            } else {
                m.chat_media(&list, template.as_ref(), &media, budget, s.max_tokens, resolved, &mut send)
                    .map(|(reply, vision_tokens)| Generation { vision_tokens: vision_tokens as u64, ..generation(reply) })
            };
            let _ = tx.send(Streamed::Done(out.map_err(|e| engine_error(&e))));
        });
        self.send(job)?;
        loop {
            match rx.recv() {
                Ok(Streamed::Text(t)) => on_text(&t),
                Ok(Streamed::Done(out)) => return out,
                Err(_) => return Err(self.stopped()),
            }
        }
    }

    fn logits(&mut self, prompt: &str) -> Result<Logits, BackendError> {
        let prompt = prompt.to_string();
        self.call(move |m| {
            m.logits(&prompt)
                .map(|l| Logits { tokens: l.tokens, vocab: l.vocab, values: l.values })
                .map_err(|e| engine_error(&e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Message, Role};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    struct Fake {
        dropped: Arc<AtomicUsize>,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn reply(text: String, n: usize) -> Reply {
        Reply { text, prompt_tokens: 3, completion_tokens: n, prefill_secs: 0.01, decode_secs: 0.02, peak_device_bytes: 64 }
    }

    impl Model for Fake {
        fn generate(&mut self, prompt: &str, max_tokens: usize, _: Sampling) -> anyhow::Result<Reply> {
            if prompt == "panic" {
                panic!("kernel exploded");
            }
            if prompt == "oom" {
                anyhow::bail!("alloc kv cache: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")");
            }
            Ok(reply(format!("echo {prompt}"), max_tokens))
        }

        fn chat(
            &mut self,
            _: &[Value],
            _: Option<&Value>,
            max_tokens: usize,
            _: Sampling,
            on_text: &mut dyn FnMut(&str) -> bool,
        ) -> anyhow::Result<Reply> {
            let mut text = String::new();
            for i in 0..max_tokens {
                std::thread::sleep(Duration::from_millis(2));
                let piece = format!("t{i} ");
                text.push_str(&piece);
                if !on_text(&piece) {
                    return Ok(reply(text, i + 1));
                }
            }
            Ok(reply(text, max_tokens))
        }

        fn logits(&mut self, _: &str) -> anyhow::Result<PromptLogits> {
            Ok(PromptLogits { tokens: vec![1, 2], vocab: 2, values: vec![0.0, 1.0, 2.0, 3.0] })
        }
    }

    fn fake(dropped: Arc<AtomicUsize>) -> Native {
        Native::with_factory(
            Some(256),
            Arc::new(move |model, _| match model {
                "missing" => anyhow::bail!("no config.json for missing under /tmp"),
                "explode" => panic!("Unable to dynamically load the \"cuda\" shared library"),
                _ => Ok(Box::new(Fake { dropped: dropped.clone() }) as Box<dyn Model>),
            }),
        )
    }

    fn chat(n: usize) -> ChatRequest {
        let mut r = ChatRequest::new(vec![Message::new(Role::User, "hi")], crate::chat::Sampling::default());
        r.sampling.max_tokens = n;
        r
    }

    struct Recording(Arc<Mutex<Vec<Sampling>>>);

    impl Model for Recording {
        fn generate(&mut self, _: &str, _: usize, _: Sampling) -> anyhow::Result<Reply> {
            Ok(reply(String::new(), 0))
        }

        fn chat(&mut self, _: &[Value], _: Option<&Value>, _: usize, sampling: Sampling, _: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<Reply> {
            self.0.lock().unwrap().push(sampling);
            Ok(reply(String::new(), 0))
        }

        fn logits(&mut self, _: &str) -> anyhow::Result<PromptLogits> {
            anyhow::bail!("no logits")
        }

        fn sampling_defaults(&self) -> specrt::engine::SamplingDefaults {
            specrt::engine::SamplingDefaults { top_k: Some(20), top_p: Some(0.95), min_p: None, ..Default::default() }
        }
    }

    #[test]
    fn the_model_generation_config_fills_the_sampling_a_request_leaves_out() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let shared = seen.clone();
        let mut n = Native::with_factory(None, Arc::new(move |_, _| Ok(Box::new(Recording(shared.clone())) as Box<dyn Model>)));
        n.load("m").unwrap();
        n.chat(&chat(4), &mut |_| {}).unwrap();
        let mut explicit = chat(4);
        explicit.sampling.top_k = Some(1);
        explicit.sampling.top_p = Some(0.5);
        n.chat(&explicit, &mut |_| {}).unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!((seen[0].top_k, seen[0].top_p, seen[0].min_p), (20, 0.95, 0.0));
        assert_eq!((seen[1].top_k, seen[1].top_p), (1, 0.5));
    }

    #[test]
    fn a_missing_model_fails_to_load_cleanly() {
        let mut n = fake(Arc::default());
        let e = n.load("missing").unwrap_err();
        assert_eq!(e.kind, ErrorKind::ModelNotFound, "{e}");
        assert!(n.generate(&Request { prompt: "x".into(), max_tokens: 1 }, &mut |_| {}).unwrap_err().message.contains("no model"));
    }

    #[test]
    fn a_panic_while_loading_is_an_error() {
        let mut n = fake(Arc::default());
        let e = n.load("explode").unwrap_err();
        assert!(e.message.contains("panicked") && e.message.contains("cuda"), "{e}");
    }

    #[test]
    fn generate_chat_and_logits_run_on_the_engine_thread() {
        let mut n = fake(Arc::default());
        n.load("m").unwrap();
        let g = n.generate(&Request { prompt: "p".into(), max_tokens: 4 }, &mut |_| {}).unwrap();
        assert_eq!((g.text.as_str(), g.completion_tokens, g.peak_memory_bytes), ("echo p", 4, Some(64)));
        let mut seen = String::new();
        let g = n.chat(&chat(3), &mut |t| seen.push_str(t)).unwrap();
        assert_eq!((seen.as_str(), g.completion_tokens), ("t0 t1 t2 ", 3));
        assert_eq!(n.logits("p").unwrap().row(1), &[2.0, 3.0]);
    }

    #[test]
    fn cancelling_stops_a_stream_mid_reply() {
        let mut n = fake(Arc::default());
        n.load("m").unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut r = chat(10_000);
        r.cancel = Some(cancel.clone());
        let mut pieces = 0;
        let g = n
            .chat(&r, &mut |_| {
                pieces += 1;
                if pieces == 3 {
                    cancel.store(true, Ordering::Relaxed);
                }
            })
            .unwrap();
        assert!(g.completion_tokens < 10, "kept going for {} tokens", g.completion_tokens);
    }

    #[test]
    fn a_panic_mid_request_is_contained_and_the_model_can_be_reloaded() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut n = fake(dropped.clone());
        n.load("m").unwrap();
        let e = n.generate(&Request { prompt: "panic".into(), max_tokens: 1 }, &mut |_| {}).unwrap_err();
        assert!(e.message.contains("kernel exploded"), "{e}");
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        let again = n.generate(&Request { prompt: "p".into(), max_tokens: 1 }, &mut |_| {}).unwrap_err();
        assert!(again.message.contains("load the model again"), "{again}");
        n.load("m").unwrap();
        assert!(n.generate(&Request { prompt: "p".into(), max_tokens: 1 }, &mut |_| {}).is_ok());
    }

    #[test]
    fn a_chat_template_error_is_an_invalid_request() {
        let e = engine_error(&anyhow::anyhow!("chat template: invalid operation: Unexpected message role"));
        assert_eq!(e.kind, ErrorKind::InvalidRequest, "{e}");
    }

    #[test]
    fn out_of_memory_is_reported_plainly() {
        let mut n = fake(Arc::default());
        n.load("m").unwrap();
        let e = n.generate(&Request { prompt: "oom".into(), max_tokens: 1 }, &mut |_| {}).unwrap_err();
        assert!(e.message.starts_with("the GPU ran out of memory"), "{e}");
        assert!(n.generate(&Request { prompt: "p".into(), max_tokens: 1 }, &mut |_| {}).is_ok());
    }

    #[test]
    fn loading_again_frees_the_previous_model() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut n = fake(dropped.clone());
        n.load("a").unwrap();
        n.load("b").unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        drop(n);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}
