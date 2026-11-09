use crate::target::{Target, Capabilities, Request};
use crate::chat::ChatRequest;
use crate::result::{BackendError, ErrorKind, Generation};
use serde_json::{Value, json};
use std::io::BufRead;
use std::time::Instant;

pub struct OpenAi {
    label: String,
    base: String,
    api_key: Option<String>,
    model: String,
    agent: ureq::Agent,
}

impl OpenAi {
    pub fn new(label: String, url: &str, api_key: Option<String>) -> Self {
        let mut base = url.trim_end_matches('/').to_string();
        if let Some(stripped) = base.strip_suffix("/v1") {
            base = stripped.to_string();
        }
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(std::time::Duration::from_secs(1800)))
            .build()
            .into();
        Self { label, base, api_key, model: String::new(), agent }
    }

    fn authorised<B>(&self, mut req: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }
        req
    }

    fn http_error(&self, status: u16, mut body: ureq::Body) -> BackendError {
        let text = body.read_to_string().unwrap_or_default();
        let text_for_log = text.clone();
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v["error"]["message"].as_str().or_else(|| v["error"].as_str()).map(String::from)
            })
            .unwrap_or(text);
        let message = format!("HTTP {status}: {message}");
        tracing::warn!(backend = %self.label, status, body = %crate::logging::preview(&text_for_log, 300), "upstream returned an error");
        match status {
            404 if message.to_lowercase().contains("model") => {
                BackendError::new(ErrorKind::ModelNotFound, message)
            }
            _ => BackendError::classify(message),
        }
    }
}

impl Target for OpenAi {
    fn name(&self) -> &str {
        &self.label
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { streaming: true, logits: false, memory: false }
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        let url = format!("{}/v1/models", self.base);
        let mut resp = self
            .authorised(self.agent.get(&url))
            .call()
            .map_err(|e| BackendError::new(ErrorKind::Unreachable, format!("{url}: {e}")))?;
        let status = resp.status().as_u16();
        if status >= 400 {
            return Err(self.http_error(status, resp.into_body()));
        }
        let listing: Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| BackendError::failed(format!("{url}: bad model list: {e}")))?;
        let ids: Vec<String> = listing["data"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(String::from)).collect())
            .unwrap_or_default();
        if !ids.is_empty() && !ids.iter().any(|id| id == model) {
            return Err(BackendError::new(
                ErrorKind::ModelNotFound,
                format!("{model} is not served at {}, it serves: {}", self.base, ids.join(", ")),
            ));
        }
        self.model = model.to_string();
        self.generate(&Request { prompt: "Hello".into(), max_tokens: 1 }, &mut |_| {}).map(|_| ())
    }

    fn generate(
        &mut self,
        request: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let body = json!({
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "ignore_eos": true,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        self.stream("completions", body, None, on_text)
    }

    fn chat(
        &mut self,
        request: &ChatRequest,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let s = &request.sampling;
        let mut body = json!({
            "model": self.model,
            "max_tokens": s.max_tokens,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "min_p": s.min_p,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if let Some(seed) = s.seed {
            body["seed"] = json!(seed);
        }
        let endpoint = match request.use_template {
            true => {
                body["messages"] = request.messages_json();
                "chat/completions"
            }
            false => {
                body["prompt"] = json!(request.flatten());
                "completions"
            }
        };
        self.stream(endpoint, body, Some(request), on_text)
    }
}

impl OpenAi {
    fn stream(
        &mut self,
        endpoint: &str,
        body: Value,
        request: Option<&ChatRequest>,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let url = format!("{}/v1/{endpoint}", self.base);
        let started = Instant::now();
        let resp = self
            .authorised(self.agent.post(&url))
            .send_json(&body)
            .map_err(|e| {
                tracing::warn!(backend = %self.label, %url, error = %e, "upstream unreachable");
                BackendError::new(ErrorKind::Unreachable, format!("{url}: {e}"))
            })?;
        let status = resp.status().as_u16();
        if status >= 400 {
            return Err(self.http_error(status, resp.into_body()));
        }
        let mut reader = std::io::BufReader::new(resp.into_body().into_reader());
        let mut out = Generation::default();
        let mut first: Option<Instant> = None;
        let mut chunks = 0u64;
        let mut usage: Option<(u64, u64)> = None;
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| BackendError::failed(format!("stream broke: {e}")))?;
            if n == 0 || request.is_some_and(|r| r.cancelled()) {
                if n != 0 {
                    tracing::debug!(backend = %self.label, "generation cancelled, closing upstream stream");
                }
                break;
            }
            let Some(data) = line.trim().strip_prefix("data:") else { continue };
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            let event: Value = serde_json::from_str(data)
                .map_err(|e| BackendError::failed(format!("bad stream event {data}: {e}")))?;
            if let Some(message) = event["error"]["message"].as_str() {
                return Err(BackendError::classify(message));
            }
            if let Some(u) = event.get("usage").filter(|u| u.is_object()) {
                usage = Some((
                    u["prompt_tokens"].as_u64().unwrap_or(0),
                    u["completion_tokens"].as_u64().unwrap_or(0),
                ));
            }
            let choice = &event["choices"][0];
            let text = choice["text"].as_str().or_else(|| choice["delta"]["content"].as_str()).unwrap_or("");
            if !text.is_empty() {
                first.get_or_insert_with(Instant::now);
                chunks += 1;
                out.text.push_str(text);
                on_text(text);
            }
        }
        let total = started.elapsed().as_secs_f64();
        let first = first.ok_or_else(|| BackendError::failed("the server returned no text"))?;
        out.ttft_secs = first.duration_since(started).as_secs_f64();
        out.prefill_secs = out.ttft_secs;
        out.decode_secs = (total - out.ttft_secs).max(0.0);
        let (prompt, completion) = usage.unwrap_or((0, chunks));
        out.prompt_tokens = prompt;
        out.completion_tokens = if completion > 0 { completion } else { chunks };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeServer, Script};

    fn script() -> Script {
        Script {
            models: vec!["qwen3".into()],
            chunks: (0..6).map(|i| format!("w{i} ")).collect(),
            chunk_delay_ms: 20,
            first_delay_ms: 150,
            usage: Some((40, 6)),
            fail_status: None,
            files: Default::default(),
        }
    }

    fn request() -> Request {
        Request { prompt: "hello".into(), max_tokens: 6 }
    }

    #[test]
    fn streams_text_and_times_the_first_token() {
        let server = FakeServer::start(script());
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.load("qwen3").unwrap();
        let mut streamed = String::new();
        let g = b.generate(&request(), &mut |t| streamed.push_str(t)).unwrap();
        assert_eq!(streamed, "w0 w1 w2 w3 w4 w5 ");
        assert_eq!(g.text, streamed);
        assert_eq!((g.prompt_tokens, g.completion_tokens), (40, 6));
        assert!(g.ttft_secs >= 0.15 && g.ttft_secs < 1.0, "ttft {}", g.ttft_secs);
        assert!(g.decode_secs >= 0.08, "decode {}", g.decode_secs);
        let sent = server.requests.lock().unwrap().last().cloned().unwrap();
        let sent: String = sent.split_whitespace().collect();
        assert!(sent.contains("\"model\":\"qwen3\"") && sent.contains("\"stream\":true"), "{sent}");
    }

    #[test]
    fn bench_completions_ignore_eos_so_every_engine_decodes_max_tokens() {
        let server = FakeServer::start(script());
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.load("qwen3").unwrap();
        b.generate(&request(), &mut |_| {}).unwrap();
        let sent = server.requests.lock().unwrap().last().cloned().unwrap();
        let sent: String = sent.split_whitespace().collect();
        assert!(sent.contains("\"ignore_eos\":true"), "{sent}");
    }

    #[test]
    fn load_warms_the_model_so_the_timed_run_excludes_loading() {
        let server = FakeServer::start(script());
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.load("qwen3").unwrap();
        let seen = server.requests.lock().unwrap().clone();
        assert_eq!(seen.iter().filter(|r| r.starts_with("/v1/completions")).count(), 1);
        assert!(seen.last().unwrap().contains("\"max_tokens\": 1"));
    }

    #[test]
    fn a_load_failure_from_the_server_surfaces_at_load() {
        let server = FakeServer::start(Script {
            models: vec!["qwen3".into()],
            fail_status: Some((500, "model name=qwen3 failed to load".into())),
            ..Default::default()
        });
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        assert!(b.load("qwen3").is_err());
    }

    #[test]
    fn chat_posts_messages_and_sampling_to_the_chat_endpoint() {
        use crate::chat::{ChatRequest, Message, Role, Sampling};
        let server = FakeServer::start(script());
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.load("qwen3").unwrap();
        let sampling = Sampling { temperature: 0.3, top_p: Some(0.9), top_k: None, min_p: None, seed: Some(5), max_tokens: 6 };
        let request = ChatRequest::new(vec![Message::new(Role::User, "hi")], sampling);
        let mut streamed = String::new();
        let g = b.chat(&request, &mut |t| streamed.push_str(t)).unwrap();
        assert_eq!(g.completion_tokens, 6);
        let sent = server.requests.lock().unwrap().last().cloned().unwrap();
        assert!(sent.starts_with("/v1/chat/completions"), "{sent}");
        let sent: String = sent.split_whitespace().collect();
        assert!(sent.contains("\"seed\":5") && sent.contains("\"role\":\"user\""), "{sent}");
    }

    #[test]
    fn a_cancelled_chat_stops_reading_the_stream() {
        use crate::chat::{ChatRequest, Message, Role, Sampling};
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        let server = FakeServer::start(script());
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.load("qwen3").unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut request = ChatRequest::new(vec![Message::new(Role::User, "hi")], Sampling::default());
        request.cancel = Some(cancel.clone());
        let mut seen = 0;
        let g = b
            .chat(&request, &mut |_| {
                seen += 1;
                cancel.store(true, Ordering::Relaxed);
            })
            .unwrap();
        assert_eq!(seen, 1);
        assert!(g.completion_tokens <= 2, "{}", g.completion_tokens);
    }

    #[test]
    fn counts_chunks_when_the_server_reports_no_usage() {
        let server = FakeServer::start(Script { usage: None, ..script() });
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.load("qwen3").unwrap();
        let g = b.generate(&request(), &mut |_| {}).unwrap();
        assert_eq!((g.prompt_tokens, g.completion_tokens), (0, 6));
    }

    #[test]
    fn a_model_the_server_does_not_serve_is_reported() {
        let server = FakeServer::start(script());
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        let e = b.load("llama").unwrap_err();
        assert_eq!(e.kind, ErrorKind::ModelNotFound);
        assert!(e.message.contains("qwen3"));
    }

    #[test]
    fn an_unsupported_architecture_error_keeps_the_server_message() {
        let message = "Model architectures ['Qwen3_5ForCausalLM'] are not supported for now.";
        let server = FakeServer::start(Script {
            fail_status: Some((400, message.into())),
            ..script()
        });
        let mut b = OpenAi::new("fake".into(), &server.url, None);
        b.model = "qwen3".into();
        let e = b.generate(&request(), &mut |_| {}).unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedArchitecture);
        assert!(e.message.contains("Qwen3_5ForCausalLM") && e.message.contains("400"));
    }

    #[test]
    fn a_dead_server_is_unreachable() {
        let mut b = OpenAi::new("fake".into(), "http://127.0.0.1:1", None);
        assert_eq!(b.load("m").unwrap_err().kind, ErrorKind::Unreachable);
    }
}
