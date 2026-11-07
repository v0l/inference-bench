use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Default)]
pub struct Script {
    pub models: Vec<String>,
    pub chunks: Vec<String>,
    pub chunk_delay_ms: u64,
    pub first_delay_ms: u64,
    pub usage: Option<(u64, u64)>,
    pub fail_status: Option<(u16, String)>,
}

pub struct FakeServer {
    pub url: String,
    pub requests: Arc<Mutex<Vec<String>>>,
}

impl FakeServer {
    pub fn start(script: Script) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let script = script.clone();
                let seen = seen.clone();
                thread::spawn(move || serve(stream, &script, &seen));
            }
        });
        Self { url, requests }
    }
}

fn serve(mut stream: TcpStream, script: &Script, seen: &Mutex<Vec<String>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut length = 0usize;
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).unwrap();
    seen.lock().unwrap().push(format!("{path} {}", String::from_utf8_lossy(&body)));
    if let Some((code, message)) = &script.fail_status {
        let payload = format!("{{\"error\":{{\"message\":{}}}}}", serde_json::to_string(message).unwrap());
        let _ = write!(
            stream,
            "HTTP/1.1 {code} Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        return;
    }
    if path.ends_with("/models") {
        let data: Vec<_> = script.models.iter().map(|m| serde_json::json!({"id": m})).collect();
        let payload = serde_json::json!({"data": data}).to_string();
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        return;
    }
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
    );
    thread::sleep(Duration::from_millis(script.first_delay_ms));
    for chunk in &script.chunks {
        let event = serde_json::json!({"choices": [{"text": chunk}]});
        let _ = write!(stream, "data: {event}\n\n");
        let _ = stream.flush();
        thread::sleep(Duration::from_millis(script.chunk_delay_ms));
    }
    if let Some((prompt, completion)) = script.usage {
        let event = serde_json::json!({
            "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}
        });
        let _ = write!(stream, "data: {event}\n\n");
    }
    let _ = write!(stream, "data: [DONE]\n\n");
}

pub struct ScriptedModel {
    script: Script,
}

impl ScriptedModel {
    pub fn factory(script: Script) -> crate::native::Factory {
        Arc::new(move |model, _| {
            if !script.models.iter().any(|m| m == model) {
                anyhow::bail!("no config.json for {model} under the fake hub");
            }
            Ok(Box::new(ScriptedModel { script: script.clone() }) as Box<dyn crate::native::Model>)
        })
    }

    fn reply(&self, text: String, pieces: usize, started: std::time::Instant, first: f64) -> specrt::engine::Reply {
        let (prompt, completion) = self.script.usage.unwrap_or((0, pieces as u64));
        specrt::engine::Reply {
            text,
            prompt_tokens: prompt as usize,
            completion_tokens: completion.min(pieces as u64) as usize,
            prefill_secs: first,
            decode_secs: started.elapsed().as_secs_f64() - first,
            peak_device_bytes: 0,
        }
    }

    fn stream(&self, on_text: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<specrt::engine::Reply> {
        if let Some((_, message)) = &self.script.fail_status {
            anyhow::bail!("{message}");
        }
        let started = std::time::Instant::now();
        thread::sleep(Duration::from_millis(self.script.first_delay_ms));
        let first = started.elapsed().as_secs_f64();
        let mut text = String::new();
        for (i, chunk) in self.script.chunks.iter().enumerate() {
            text.push_str(chunk);
            if !on_text(chunk) {
                return Ok(self.reply(text, i + 1, started, first));
            }
            thread::sleep(Duration::from_millis(self.script.chunk_delay_ms));
        }
        Ok(self.reply(text, self.script.chunks.len(), started, first))
    }
}

impl crate::native::Model for ScriptedModel {
    fn generate(&mut self, _: &str, _: usize, _: specrt::runtime::Sampling) -> anyhow::Result<specrt::engine::Reply> {
        self.stream(&mut |_| true)
    }

    fn chat(
        &mut self,
        _: &[serde_json::Value],
        _: bool,
        _: usize,
        _: specrt::runtime::Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<specrt::engine::Reply> {
        self.stream(on_text)
    }

    fn logits(&mut self, _: &str) -> anyhow::Result<specrt::engine::PromptLogits> {
        Ok(specrt::engine::PromptLogits { tokens: vec![1, 2], vocab: 2, values: vec![0.0, 1.0, 2.0, 3.0] })
    }
}
