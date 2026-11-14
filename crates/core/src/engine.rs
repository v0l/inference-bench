use crate::chat::ChatRequest;
use crate::native::{Factory, Native, specrt_factory};
use crate::target::Target;
use crate::result::{BackendError, Generation};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub model_id: String,
    pub display: String,
    pub load_secs: f64,
    pub vision: bool,
    pub ctx: usize,
    pub thinking: specrt::chat::Thinking,
}

pub enum Command {
    Load { model_id: String, display: String, max_ctx: Option<usize> },
    Unload,
    Chat { request: ChatRequest, events: Sender<ChatEvent> },
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
}

#[derive(Clone)]
pub struct Handle {
    commands: Sender<Command>,
    loaded: Arc<Mutex<Option<Loaded>>>,
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
        if self.commands.send(Command::Chat { request, events: tx.clone() }).is_err() {
            let _ = tx.send(ChatEvent::Failed(BackendError::failed("the engine has stopped")));
        }
        rx
    }
}

impl Engine {
    pub fn spawn(wake: Wake) -> Self {
        Self::with_factory(wake, specrt_factory())
    }

    pub fn with_factory(wake: Wake, factory: Factory) -> Self {
        let (commands, inbox) = channel();
        let (events_tx, events) = channel();
        let loaded = Arc::new(Mutex::new(None));
        let shared = loaded.clone();
        std::thread::spawn(move || run(inbox, events_tx, shared, wake, factory));
        Self { commands, events, loaded }
    }

    pub fn handle(&self) -> Handle {
        Handle { commands: self.commands.clone(), loaded: self.loaded.clone() }
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
        let _ = self.commands.send(Command::Shutdown);
    }
}

fn run(
    inbox: Receiver<Command>,
    events: Sender<EngineEvent>,
    shared: Arc<Mutex<Option<Loaded>>>,
    wake: Wake,
    factory: Factory,
) {
    let mut backend: Option<Native> = None;
    let emit = |e: EngineEvent| {
        let _ = events.send(e);
        wake();
    };
    while let Ok(command) = inbox.recv() {
        match command {
            Command::Shutdown => break,
            Command::Unload => {
                backend = None;
                *shared.lock().unwrap() = None;
                emit(EngineEvent::Unloaded);
            }
            Command::Load { model_id, display, max_ctx } => {
                backend = None;
                *shared.lock().unwrap() = None;
                emit(EngineEvent::Loading(display.clone()));
                let started = Instant::now();
                let mut native = Native::with_factory(max_ctx, factory.clone());
                match native.load(&model_id) {
                    Ok(()) => {
                        let caps = native.capabilities();
                        let (vision, thinking) = (caps.vision, caps.thinking);
                        backend = Some(native);
                        let ctx = max_ctx.unwrap_or(specrt::engine::DEFAULT_CTX);
                        let info = Loaded { model_id, display, load_secs: started.elapsed().as_secs_f64(), vision, ctx, thinking };
                        *shared.lock().unwrap() = Some(info.clone());
                        emit(EngineEvent::Loaded(info));
                    }
                    Err(e) => emit(EngineEvent::LoadFailed(e)),
                }
            }
            Command::Chat { request, events: out } => {
                let send = |e: ChatEvent| {
                    let _ = out.send(e);
                    wake();
                };
                match backend.as_mut() {
                    None => send(ChatEvent::Failed(BackendError::failed("no model is loaded"))),
                    Some(b) => {
                        let result = b.chat(&request, &mut |t| send(ChatEvent::Text(t.to_string())));
                        send(match result {
                            Ok(g) => ChatEvent::Done(g),
                            Err(e) => ChatEvent::Failed(e),
                        });
                    }
                }
            }
        }
    }
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
        Command::Load { model_id: model.into(), display: display.into(), max_ctx: None }
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
}
