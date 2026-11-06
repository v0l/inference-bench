use crate::result::{BackendError, Generation};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub streaming: bool,
    pub logits: bool,
    pub memory: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub prompt: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    pub tokens: Vec<u32>,
    pub vocab: usize,
    pub values: Vec<f32>,
}

impl Logits {
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    pub fn read(path: &std::path::Path, tokens: Vec<u32>, vocab: usize) -> Result<Self, BackendError> {
        let bytes = std::fs::read(path)
            .map_err(|e| BackendError::failed(format!("{}: {e}", path.display())))?;
        let values: Vec<f32> =
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if values.len() != tokens.len() * vocab {
            return Err(BackendError::failed(format!(
                "logits file has {} values, expected {} x {}",
                values.len(),
                tokens.len(),
                vocab
            )));
        }
        Ok(Self { tokens, vocab, values })
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.vocab..(i + 1) * self.vocab]
    }
}

pub trait Backend {
    fn name(&self) -> &str;

    fn capabilities(&self) -> Capabilities;

    fn load(&mut self, model: &str) -> Result<(), BackendError>;

    fn generate(
        &mut self,
        request: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError>;

    fn logits(&mut self, _prompt: &str) -> Result<Logits, BackendError> {
        Err(BackendError::failed(format!("{} does not expose logits", self.name())))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendConfig {
    Openai {
        url: String,
        #[serde(default)]
        api_key: Option<String>,
        #[serde(default)]
        label: Option<String>,
    },
    Transformers {
        #[serde(default)]
        python: Option<String>,
        #[serde(default)]
        device: Option<String>,
    },
    Specrt {
        #[serde(default)]
        worker: Option<PathBuf>,
        #[serde(default)]
        max_ctx: Option<usize>,
    },
}

impl BackendConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Openai { .. } => "openai",
            Self::Transformers { .. } => "transformers",
            Self::Specrt { .. } => "specrt",
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Openai { label: Some(l), .. } => l.clone(),
            Self::Openai { url, .. } => format!("openai {url}"),
            other => other.kind().to_string(),
        }
    }

    pub fn build(&self) -> Result<Box<dyn Backend>, BackendError> {
        match self {
            Self::Openai { url, api_key, label } => Ok(Box::new(crate::openai::OpenAi::new(
                label.clone().unwrap_or_else(|| "openai".into()),
                url,
                api_key.clone(),
            ))),
            Self::Transformers { python, device } => {
                Ok(Box::new(crate::subprocess::Subprocess::transformers(python.clone(), device.clone())))
            }
            Self::Specrt { worker, max_ctx } => {
                Ok(Box::new(crate::subprocess::Subprocess::specrt(worker.clone(), *max_ctx)))
            }
        }
    }
}
