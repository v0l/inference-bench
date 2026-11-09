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
    pub files: Arc<Mutex<Vec<FakeFile>>>,
}

#[derive(Clone, Debug, Default)]
pub struct FakeFile {
    pub path: String,
    pub body: Vec<u8>,
    pub etag: Option<String>,
    pub status: Option<u16>,
}

impl Script {
    pub fn serve_file(&self, file: FakeFile) {
        let mut files = self.files.lock().unwrap();
        files.retain(|f| f.path != file.path);
        files.push(file);
    }
}

pub fn tarball(top: &str, entries: &[(&str, &[u8])], commit: Option<&str>) -> Vec<u8> {
    let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut b = tar::Builder::new(gz);
    if let Some(c) = commit {
        let record = format!(" comment={c}\n");
        let record = format!("{}{record}", record.len() + (record.len() + 2).to_string().len());
        let mut h = tar::Header::new_ustar();
        h.set_entry_type(tar::EntryType::XGlobalHeader);
        h.set_path("pax_global_header").unwrap();
        h.set_size(record.len() as u64);
        h.set_mode(0o666);
        h.set_cksum();
        b.append(&h, record.as_bytes()).unwrap();
    }
    for (path, body) in entries {
        let mut h = tar::Header::new_old();
        let full = format!("{top}/{path}");
        h.as_old_mut().name[..full.len()].copy_from_slice(full.as_bytes());
        h.set_entry_type(if path.ends_with('/') { tar::EntryType::Directory } else { tar::EntryType::Regular });
        h.set_size(body.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        b.append(&h, *body).unwrap();
    }
    b.into_inner().unwrap().finish().unwrap()
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
    let mut if_none_match = None;
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
        if line.to_lowercase().starts_with("if-none-match:") {
            if_none_match = Some(line["if-none-match:".len()..].trim().to_string());
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).unwrap();
    seen.lock().unwrap().push(format!("{path} {}", String::from_utf8_lossy(&body)));
    let file = script.files.lock().unwrap().iter().find(|f| f.path == path).cloned();
    if let Some(f) = file {
        let etag = f.etag.as_ref().map(|e| format!("ETag: {e}\r\n")).unwrap_or_default();
        if f.etag.is_some() && f.etag == if_none_match {
            let _ = write!(stream, "HTTP/1.1 304 Not Modified\r\n{etag}Content-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        }
        let code = f.status.unwrap_or(200);
        let _ = write!(
            stream,
            "HTTP/1.1 {code} X\r\nContent-Type: application/octet-stream\r\n{etag}Content-Length: {}\r\nConnection: close\r\n\r\n",
            f.body.len()
        );
        let _ = stream.write_all(&f.body);
        return;
    }
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
        _: Option<&serde_json::Value>,
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
