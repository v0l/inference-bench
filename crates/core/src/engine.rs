use crate::backend::{Backend, BackendConfig};
use crate::chat::ChatRequest;
use crate::result::{BackendError, Generation};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub backend: String,
    pub model_id: String,
    pub display: String,
    pub load_secs: f64,
}

pub enum Command {
    Load { backend: String, config: BackendConfig, model_id: String, display: String },
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
        let (commands, inbox) = channel();
        let (events_tx, events) = channel();
        let loaded = Arc::new(Mutex::new(None));
        let shared = loaded.clone();
        std::thread::spawn(move || run(inbox, events_tx, shared, wake));
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
) {
    let mut backend: Option<Box<dyn Backend>> = None;
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
            Command::Load { backend: label, config, model_id, display } => {
                backend = None;
                *shared.lock().unwrap() = None;
                emit(EngineEvent::Loading(display.clone()));
                let started = Instant::now();
                let outcome = config.build().and_then(|mut b| b.load(&model_id).map(|_| b));
                match outcome {
                    Ok(b) => {
                        backend = Some(b);
                        let info = Loaded { backend: label, model_id, display, load_secs: started.elapsed().as_secs_f64() };
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
    use crate::fake::{FakeServer, Script};
    use std::time::Duration;

    fn script() -> Script {
        Script {
            models: vec!["m".into()],
            chunks: vec!["Hel".into(), "lo ".into(), "there".into()],
            chunk_delay_ms: 5,
            first_delay_ms: 20,
            usage: Some((7, 3)),
            fail_status: None,
        }
    }

    fn config(url: &str) -> BackendConfig {
        BackendConfig::Openai { url: url.into(), api_key: None, label: Some("fake".into()) }
    }

    fn engine() -> Engine {
        Engine::spawn(Arc::new(|| {}))
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
        let server = FakeServer::start(script());
        let e = engine();
        e.send(Command::Load { backend: "fake".into(), config: config(&server.url), model_id: "m".into(), display: "M".into() });
        assert_eq!(wait_event(&e), EngineEvent::Loading("M".into()));
        assert!(matches!(wait_event(&e), EngineEvent::Loaded(l) if l.display == "M"));
        assert_eq!(e.loaded().unwrap().backend, "fake");
        let (text, done, _) = ask(&e.handle(), "hi");
        assert_eq!(text, "Hello there");
        let g = done.unwrap();
        assert_eq!((g.prompt_tokens, g.completion_tokens), (7, 3));
        assert!(g.ttft_secs >= 0.02);
    }

    #[test]
    fn chat_without_a_model_fails_and_unload_clears_the_state() {
        let server = FakeServer::start(script());
        let e = engine();
        let (_, _, failed) = ask(&e.handle(), "hi");
        assert!(failed.unwrap().message.contains("no model"));
        e.send(Command::Load { backend: "fake".into(), config: config(&server.url), model_id: "m".into(), display: "M".into() });
        wait_event(&e);
        wait_event(&e);
        e.send(Command::Unload);
        assert_eq!(wait_event(&e), EngineEvent::Unloaded);
        assert!(e.loaded().is_none());
    }

    #[test]
    fn a_model_the_backend_lacks_is_a_load_failure() {
        let server = FakeServer::start(script());
        let e = engine();
        e.send(Command::Load { backend: "fake".into(), config: config(&server.url), model_id: "other".into(), display: "O".into() });
        wait_event(&e);
        match wait_event(&e) {
            EngineEvent::LoadFailed(err) => assert_eq!(err.kind, crate::result::ErrorKind::ModelNotFound),
            other => panic!("{other:?}"),
        }
        assert!(e.loaded().is_none());
    }
}
