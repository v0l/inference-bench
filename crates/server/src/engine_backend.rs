use inference_bench_core::target::{Target, Capabilities, Request};
use inference_bench_core::chat::{ChatRequest, Message, Role, Sampling};
use inference_bench_core::engine::{ChatEvent, Handle};
use inference_bench_core::result::{BackendError, ErrorKind, Generation};

use crate::manager::{ModelSource, SourceModel};

fn source_model(l: inference_bench_core::engine::Loaded) -> SourceModel {
    let mut aliases = vec![l.model_id];
    aliases.extend(inference_bench_core::library::short_alias(&l.display));
    aliases.extend(l.served_names);
    SourceModel { id: l.display, aliases, slots: Some(l.max_num_seqs) }
}

impl ModelSource for Handle {
    fn loaded(&self) -> Vec<SourceModel> {
        Handle::loaded_all(self).into_iter().map(source_model).collect()
    }

    fn tool_format(&self, id: &str) -> inference_bench_core::specrt_chat::ToolFormat {
        Handle::tools_of(self, id)
    }

    fn suspended(&self) -> Vec<SourceModel> {
        Handle::suspended_all(self).into_iter().map(source_model).collect()
    }

    fn suspend(&self, id: &str) -> bool {
        Handle::suspend_model(self, Some(id))
    }

    fn resume(&self, id: &str) -> Result<(), BackendError> {
        Handle::resume_model(self, Some(id))
    }
}

pub struct EngineBackend {
    handle: Handle,
    model: Option<String>,
}

impl EngineBackend {
    pub fn new(handle: Handle) -> Self {
        Self { handle, model: None }
    }

    pub fn bound(handle: Handle, model: &str) -> Self {
        Self { handle, model: Some(model.to_string()) }
    }

    fn loaded(&self) -> Option<inference_bench_core::engine::Loaded> {
        match &self.model {
            Some(m) => self.handle.loaded_named(m),
            None => self.handle.loaded(),
        }
    }
}

impl Target for EngineBackend {
    fn share(&self) -> Option<Box<dyn Target>> {
        Some(Box::new(EngineBackend { handle: self.handle.clone(), model: self.model.clone() }))
    }

    fn metrics(&self) -> Option<String> {
        match &self.model {
            Some(m) => self.handle.serving_metrics_of(m),
            None => self.handle.serving_metrics(),
        }
    }

    fn failure(&self) -> Option<String> {
        match &self.model {
            Some(m) => self.handle.failure_of(m),
            None => self.handle.failure(),
        }
    }

    fn name(&self) -> &str {
        "engine"
    }

    fn capabilities(&self) -> Capabilities {
        let loaded = self.loaded();
        let batched = loaded.as_ref().is_some_and(|l| l.max_num_seqs > 1);
        Capabilities {
            streaming: true,
            vision: loaded.as_ref().is_some_and(|l| l.vision),
            thinking: loaded.as_ref().map(|l| l.thinking).unwrap_or_default(),
            tools: loaded.as_ref().map(|l| l.tools).unwrap_or_default(),
            logprobs: batched,
            batched,
            ..Capabilities::default()
        }
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        match self.handle.loaded_named(model).or_else(|| self.handle.loaded()) {
            Some(l) => {
                self.model = Some(l.model_id);
                Ok(())
            }
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
        let events = self.handle.chat_on(self.model.as_deref(), request.clone());
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
