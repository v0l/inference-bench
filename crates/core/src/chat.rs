use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "system" => Self::System,
            "assistant" => Self::Assistant,
            _ => Self::User,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
    pub max_tokens: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.7, top_p: None, top_k: None, min_p: None, seed: None, max_tokens: 512 }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub sampling: Sampling,
    pub use_template: bool,
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub ignore_eos: bool,
    pub openai_messages: Option<Vec<Value>>,
    pub template_vars: Value,
    pub cache_salt: Option<String>,
    pub media: Vec<crate::media::MediaItem>,
    pub media_max_pixels: Option<usize>,
    pub media_max_video_secs: Option<f64>,
}

impl ChatRequest {
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>, sampling: Sampling) -> Self {
        Self {
            messages,
            sampling,
            use_template: true,
            cancel: None,
            ignore_eos: false,
            openai_messages: None,
            template_vars: json!({"enable_thinking": false}),
            cache_salt: None,
            media: Vec::new(),
            media_max_pixels: None,
            media_max_video_secs: None,
        }
    }

    pub fn template(&self) -> Option<&Value> {
        self.use_template.then_some(&self.template_vars)
    }

    pub fn flatten(&self) -> String {
        match self.messages.as_slice() {
            [only] => only.content.clone(),
            many => {
                let mut out = String::new();
                for m in many {
                    out.push_str(&m.content);
                    out.push_str("\n\n");
                }
                out.trim_end().to_string()
            }
        }
    }

    pub fn messages_json(&self) -> Value {
        if let Some(list) = &self.openai_messages {
            return Value::Array(list.clone());
        }
        Value::Array(
            self.messages.iter().map(|m| json!({"role": m.role.name(), "content": m.content})).collect(),
        )
    }

    pub fn from_openai(body: &Value) -> Option<Self> {
        let defaults = Sampling::default();
        let sampling = Sampling {
            temperature: body["temperature"].as_f64().map(|v| v as f32).unwrap_or(defaults.temperature),
            top_p: body["top_p"].as_f64().map(|v| v as f32).or(defaults.top_p),
            top_k: body["top_k"].as_u64().map(|v| v as u32).or(defaults.top_k),
            min_p: body["min_p"].as_f64().map(|v| v as f32).or(defaults.min_p),
            seed: body["seed"].as_u64(),
            max_tokens: body["max_tokens"].as_u64().map(|v| v as usize).unwrap_or(defaults.max_tokens),
        };
        if let Some(list) = body["messages"].as_array() {
            let messages: Vec<Message> = list
                .iter()
                .map(|m| {
                    Message::new(Role::from_name(m["role"].as_str().unwrap_or("user")), content_text(&m["content"]))
                })
                .collect();
            return (!messages.is_empty()).then(|| Self::new(messages, sampling));
        }
        let prompt = body["prompt"].as_str()?;
        let mut r = Self::new(vec![Message::new(Role::User, prompt)], sampling);
        r.use_template = false;
        Some(r)
    }
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")
        }
        _ => String::new(),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_secs: f64,
    #[serde(default)]
    pub vision_tokens: u64,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
}

impl Stats {
    pub fn of(g: &crate::result::Generation) -> Self {
        let r = crate::result::RunResult::ok("", "", chrono::Utc::now(), g.clone());
        Self {
            prompt_tokens: g.prompt_tokens,
            completion_tokens: g.completion_tokens,
            vision_tokens: g.vision_tokens,
            ttft_secs: g.ttft_secs,
            prefill_tok_s: r.prefill_tok_s,
            decode_tok_s: r.decode_tok_s,
        }
    }
}

pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_openai_chat_body_becomes_a_templated_request() {
        let body = json!({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
            "temperature": 0.2, "seed": 9, "max_tokens": 16});
        let r = ChatRequest::from_openai(&body).unwrap();
        assert!(r.use_template);
        assert_eq!(r.messages.len(), 2);
        assert_eq!(r.messages[0].role, Role::System);
        assert_eq!((r.sampling.temperature, r.sampling.seed, r.sampling.max_tokens), (0.2, Some(9), 16));
    }

    #[test]
    fn an_openai_completion_body_is_a_raw_prompt() {
        let r = ChatRequest::from_openai(&json!({"prompt": "once upon"})).unwrap();
        assert!(!r.use_template);
        assert_eq!(r.flatten(), "once upon");
        assert!(ChatRequest::from_openai(&json!({})).is_none());
    }

    #[test]
    fn content_parts_are_joined() {
        let body = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]});
        assert_eq!(ChatRequest::from_openai(&body).unwrap().messages[0].content, "ab");
    }
}
