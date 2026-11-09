use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    UnsupportedArchitecture,
    ModelNotFound,
    Unreachable,
    Failed,
    Cancelled,
}

impl ErrorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::UnsupportedArchitecture => "unsupported architecture",
            Self::ModelNotFound => "model not found",
            Self::Unreachable => "backend unreachable",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn from_wire(kind: &str) -> Self {
        match kind {
            "unsupported_architecture" => Self::UnsupportedArchitecture,
            "model_not_found" => Self::ModelNotFound,
            "unreachable" => Self::Unreachable,
            _ => Self::Failed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendError {
    pub kind: ErrorKind,
    pub message: String,
}

impl BackendError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Failed, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedArchitecture, message)
    }

    pub fn classify(message: impl Into<String>) -> Self {
        let message = message.into();
        let lower = message.to_lowercase();
        let unsupported = ["unsupported architecture", "unknown model architecture", "architecture"]
            .iter()
            .any(|n| lower.contains(n))
            && ["not supported", "unsupported", "unknown", "does not recognize", "not recognized"]
                .iter()
                .any(|n| lower.contains(n));
        match unsupported {
            true => Self::unsupported(message),
            false => Self::failed(message),
        }
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind.label(), self.message)
    }
}

impl std::error::Error for BackendError {}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_secs: f64,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub backend: String,
    pub model: String,
    pub started: DateTime<Utc>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub ttft_secs: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
    pub text: String,
    pub error: Option<BackendError>,
}

fn rate(tokens: u64, secs: f64) -> Option<f64> {
    (tokens > 0 && secs > 0.0).then(|| tokens as f64 / secs)
}

impl RunResult {
    pub fn ok(backend: &str, model: &str, started: DateTime<Utc>, g: Generation) -> Self {
        Self {
            backend: backend.into(),
            model: model.into(),
            started,
            prompt_tokens: g.prompt_tokens,
            completion_tokens: g.completion_tokens,
            prefill_tok_s: rate(g.prompt_tokens, g.prefill_secs),
            decode_tok_s: rate(g.completion_tokens.saturating_sub(1), g.decode_secs),
            ttft_secs: (g.ttft_secs > 0.0).then_some(g.ttft_secs),
            peak_memory_bytes: g.peak_memory_bytes,
            text: g.text,
            error: None,
        }
    }

    pub fn failed(backend: &str, model: &str, started: DateTime<Utc>, error: BackendError) -> Self {
        Self {
            backend: backend.into(),
            model: model.into(),
            started,
            prompt_tokens: 0,
            completion_tokens: 0,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_secs: None,
            peak_memory_bytes: None,
            text: String::new(),
            error: Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_come_from_tokens_and_seconds() {
        let g = Generation {
            prompt_tokens: 100,
            completion_tokens: 11,
            ttft_secs: 0.5,
            prefill_secs: 0.5,
            decode_secs: 2.0,
            ..Default::default()
        };
        let r = RunResult::ok("b", "m", Utc::now(), g);
        assert_eq!(r.prefill_tok_s, Some(200.0));
        assert_eq!(r.decode_tok_s, Some(5.0));
        assert_eq!(r.ttft_secs, Some(0.5));
    }

    #[test]
    fn one_token_has_no_decode_rate() {
        let g = Generation { completion_tokens: 1, decode_secs: 0.0, ..Default::default() };
        assert_eq!(RunResult::ok("b", "m", Utc::now(), g).decode_tok_s, None);
    }

    #[test]
    fn architecture_errors_are_told_apart() {
        assert_eq!(
            BackendError::classify("The checkpoint you are trying to load has model type `foo` but Transformers does not recognize this architecture").kind,
            ErrorKind::UnsupportedArchitecture
        );
        assert_eq!(
            BackendError::classify("llama_model_load: error loading model: unknown model architecture: 'qwen35'").kind,
            ErrorKind::UnsupportedArchitecture
        );
        assert_eq!(BackendError::classify("connection reset").kind, ErrorKind::Failed);
        assert_eq!(
            BackendError::classify("Model architectures ['X'] failed to be inspected").kind,
            ErrorKind::Failed
        );
    }

    #[test]
    fn result_round_trips_through_json() {
        let r = RunResult::failed("vllm", "m", Utc::now(), BackendError::unsupported("nope"));
        let back: RunResult = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
    }
}
