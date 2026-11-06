use crate::backend::{Backend, Capabilities, Request};
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
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v["error"]["message"].as_str().or_else(|| v["error"].as_str()).map(String::from)
            })
            .unwrap_or(text);
        let message = format!("HTTP {status}: {message}");
        match status {
            404 if message.to_lowercase().contains("model") => {
                BackendError::new(ErrorKind::ModelNotFound, message)
            }
            _ => BackendError::classify(message),
        }
    }
}

impl Backend for OpenAi {
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
        Ok(())
    }

    fn generate(
        &mut self,
        request: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let url = format!("{}/v1/completions", self.base);
        let body = json!({
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        let started = Instant::now();
        let resp = self
            .authorised(self.agent.post(&url))
            .send_json(&body)
            .map_err(|e| BackendError::new(ErrorKind::Unreachable, format!("{url}: {e}")))?;
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
            if n == 0 {
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
            let text = event["choices"][0]["text"].as_str().unwrap_or("");
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
        let sent = server.requests.lock().unwrap().join("\n");
        let sent: String = sent.split_whitespace().collect();
        assert!(sent.contains("\"model\":\"qwen3\"") && sent.contains("\"stream\":true"), "{sent}");
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
