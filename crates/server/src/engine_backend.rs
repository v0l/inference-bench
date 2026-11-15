use inference_bench_core::target::{Target, Capabilities, Request};
use inference_bench_core::chat::{ChatRequest, Message, Role, Sampling};
use inference_bench_core::engine::{ChatEvent, Handle};
use inference_bench_core::result::{BackendError, ErrorKind, Generation};

use crate::manager::{ModelSource, SourceModel};

impl ModelSource for Handle {
    fn loaded(&self) -> Vec<SourceModel> {
        Handle::loaded(self)
            .map(|l| {
                let mut aliases = vec![l.model_id];
                aliases.extend(inference_bench_core::library::short_alias(&l.display));
                aliases.extend(l.served_names);
                SourceModel { id: l.display, aliases, slots: Some(l.max_num_seqs) }
            })
            .into_iter()
            .collect()
    }

    fn suspended(&self) -> Vec<SourceModel> {
        Handle::suspended(self)
            .map(|l| {
                let mut aliases = vec![l.model_id];
                aliases.extend(inference_bench_core::library::short_alias(&l.display));
                aliases.extend(l.served_names);
                SourceModel { id: l.display, aliases, slots: Some(l.max_num_seqs) }
            })
            .into_iter()
            .collect()
    }

    fn suspend(&self) -> bool {
        Handle::suspend(self)
    }

    fn resume(&self) -> Result<(), BackendError> {
        Handle::resume(self)
    }
}

pub struct EngineBackend {
    handle: Handle,
}

impl EngineBackend {
    pub fn new(handle: Handle) -> Self {
        Self { handle }
    }
}

impl Target for EngineBackend {
    fn share(&self) -> Option<Box<dyn Target>> {
        Some(Box::new(EngineBackend::new(self.handle.clone())))
    }

    fn metrics(&self) -> Option<String> {
        self.handle.serving_metrics()
    }

    fn failure(&self) -> Option<String> {
        self.handle.failure()
    }

    fn name(&self) -> &str {
        "engine"
    }

    fn capabilities(&self) -> Capabilities {
        let loaded = self.handle.loaded();
        let batched = loaded.as_ref().is_some_and(|l| l.max_num_seqs > 1);
        Capabilities {
            streaming: true,
            vision: loaded.as_ref().is_some_and(|l| l.vision),
            thinking: self.handle.thinking(),
            logprobs: batched,
            batched,
            ..Capabilities::default()
        }
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
