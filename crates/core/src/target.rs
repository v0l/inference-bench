use crate::result::{BackendError, Generation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub streaming: bool,
    pub logits: bool,
    pub memory: bool,
    #[serde(default)]
    pub vision: bool,
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

pub trait Target: Send {
    fn name(&self) -> &str;

    fn capabilities(&self) -> Capabilities;

    fn load(&mut self, model: &str) -> Result<(), BackendError>;

    fn generate(
        &mut self,
        request: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError>;

    fn chat(
        &mut self,
        request: &crate::chat::ChatRequest,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let plain = Request { prompt: request.flatten(), max_tokens: request.sampling.max_tokens };
        self.generate(&plain, on_text)
    }

    fn logits(&mut self, _prompt: &str) -> Result<Logits, BackendError> {
        Err(BackendError::failed(format!("{} does not expose logits", self.name())))
    }

    fn share(&self) -> Option<Box<dyn Target>> {
        None
    }

    fn metrics(&self) -> Option<String> {
        None
    }
}

pub const NATIVE: &str = "specrt";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetConfig {
    #[serde(alias = "specrt")]
    Native {
        #[serde(default)]
        max_ctx: Option<usize>,
    },
    Openai {
        url: String,
        #[serde(default)]
        api_key: Option<String>,
        #[serde(default)]
        label: Option<String>,
    },
}

impl TargetConfig {
    pub fn native(max_ctx: Option<usize>) -> Self {
        Self::Native { max_ctx }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Native { .. } => "native",
            Self::Openai { .. } => "openai",
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Openai { label: Some(l), .. } => l.clone(),
            Self::Openai { url, .. } => format!("openai {url}"),
            Self::Native { .. } => NATIVE.to_string(),
        }
    }

    pub fn build(&self) -> Box<dyn Target> {
        match self {
            Self::Native { max_ctx } => Box::new(crate::native::Native::new(*max_ctx)),
            Self::Openai { url, api_key, label } => Box::new(crate::openai::OpenAi::new(
                label.clone().unwrap_or_else(|| "openai".into()),
                url,
                api_key.clone(),
            )),
        }
    }

    pub fn parse_target(spec: &str) -> Result<Self, String> {
        match spec.split_once(':') {
            _ if spec == "native" || spec == NATIVE => Ok(Self::native(None)),
            Some(("openai", url)) if !url.is_empty() => {
                Ok(Self::Openai { url: url.to_string(), api_key: None, label: None })
            }
            _ => Err(format!("unknown target {spec}: use native or openai:URL")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_parse_from_the_command_line() {
        assert_eq!(TargetConfig::parse_target("native").unwrap(), TargetConfig::native(None));
        assert_eq!(TargetConfig::parse_target("specrt").unwrap().kind(), "native");
        let t = TargetConfig::parse_target("openai:http://127.0.0.1:8080").unwrap();
        assert_eq!(t, TargetConfig::Openai { url: "http://127.0.0.1:8080".into(), api_key: None, label: None });
        assert!(TargetConfig::parse_target("transformers").unwrap_err().contains("unknown target"));
        assert!(TargetConfig::parse_target("openai:").is_err());
    }

    #[test]
    fn the_old_specrt_kind_reads_as_native() {
        let t: TargetConfig = toml::from_str("kind = \"specrt\"\nworker = \"/x\"\nmax_ctx = 4096").unwrap();
        assert_eq!(t, TargetConfig::native(Some(4096)));
    }
}
