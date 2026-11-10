use inference_bench_core::target::{Target, Capabilities, Request};
use inference_bench_core::chat::{ChatRequest, Message, Role, Sampling};
use inference_bench_core::engine::{ChatEvent, Handle};
use inference_bench_core::result::{BackendError, ErrorKind, Generation};

pub struct EngineBackend {
    handle: Handle,
}

impl EngineBackend {
    pub fn new(handle: Handle) -> Self {
        Self { handle }
    }
}

impl Target for EngineBackend {
    fn name(&self) -> &str {
        "engine"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { streaming: true, logits: false, memory: false, vision: false }
    }

    fn load(&mut self, _model: &str) -> Result<(), BackendError> {
        match self.handle.loaded() {
            Some(_) => Ok(()),
            None => Err(BackendError::new(ErrorKind::ModelNotFound, "no model is loaded in the app")),
        }
    }

    fn generate(&mut self, request: &Request, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let sampling = Sampling { temperature: 0.0, max_tokens: request.max_tokens, ..Sampling::default() };
        let mut chat = ChatRequest::new(vec![Message::new(Role::User, request.prompt.clone())], sampling);
        chat.use_template = false;
        self.chat(&chat, on_text)
    }

    fn chat(&mut self, request: &ChatRequest, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let events = self.handle.chat(request.clone());
        loop {
            match events.recv() {
                Ok(ChatEvent::Text(t)) => on_text(&t),
                Ok(ChatEvent::Done(g)) => return Ok(g),
                Ok(ChatEvent::Failed(e)) => return Err(e),
                Err(_) => return Err(BackendError::failed("the engine has stopped")),
            }
        }
    }
}
