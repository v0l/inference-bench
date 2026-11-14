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
    pub vision: bool,
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

pub fn publish_specs(script: &Script, repo: &str, git_ref: &str, commit: &str, etag: &str, files: &[(&str, &[u8])]) {
    let dir = "crates/specrt/specs";
    script.serve_file(FakeFile {
        path: format!("/repos/{repo}/commits/{git_ref}"),
        body: commit.as_bytes().to_vec(),
        etag: Some(format!("\"{etag}\"")),
        status: None,
    });
    let listing: Vec<_> = files
        .iter()
        .map(|(name, body)| serde_json::json!({"name": name, "path": format!("{dir}/{name}"), "type": "file", "size": body.len()}))
        .collect();
    script.serve_file(FakeFile {
        path: format!("/repos/{repo}/contents/{dir}?ref={commit}"),
        body: serde_json::to_vec(&listing).unwrap(),
        etag: None,
        status: None,
    });
    for (name, body) in files {
        script.serve_file(FakeFile { path: format!("/repos/{repo}/contents/{dir}/{name}?ref={commit}"), body: body.to_vec(), etag: None, status: None });
    }
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
    pub vision: bool,
    pub thinking: specrt::chat::Thinking,
    pub templates: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl ScriptedModel {
    pub fn factory(script: Script) -> crate::native::Factory {
        Self::factory_probed(script, Default::default()).0
    }

    pub fn factory_probed(script: Script, thinking: specrt::chat::Thinking) -> (crate::native::Factory, Arc<Mutex<Vec<serde_json::Value>>>) {
        let templates = Arc::new(Mutex::new(Vec::new()));
        let seen = templates.clone();
        let factory: crate::native::Factory = Arc::new(move |model, _| {
            if !script.models.iter().any(|m| m == model) {
                anyhow::bail!("no config.json for {model} under the fake hub");
            }
            Ok(Box::new(ScriptedModel { script: script.clone(), vision: script.vision, thinking, templates: seen.clone() }) as Box<dyn crate::native::Model>)
        });
        (factory, templates)
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
        template: Option<&serde_json::Value>,
        _: usize,
        _: specrt::runtime::Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<specrt::engine::Reply> {
        self.templates.lock().unwrap().push(template.cloned().unwrap_or_default());
        self.stream(on_text)
    }

    fn logits(&mut self, _: &str) -> anyhow::Result<specrt::engine::PromptLogits> {
        Ok(specrt::engine::PromptLogits { tokens: vec![1, 2], vocab: 2, values: vec![0.0, 1.0, 2.0, 3.0] })
    }

    fn accepts_media(&self) -> bool {
        self.vision
    }

    fn thinking(&self) -> specrt::chat::Thinking {
        self.thinking
    }

    fn chat_media(
        &mut self,
        _: &[serde_json::Value],
        _: Option<&serde_json::Value>,
        items: &[specrt::vision::MediaItem],
        _: specrt::vision::MediaBudget,
        _: usize,
        _: specrt::runtime::Sampling,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<(specrt::engine::Reply, usize)> {
        let reply = self.stream(on_text)?;
        Ok((reply, 100 * items.len()))
    }
}

#[derive(Default)]
struct ServingCounters {
    running: std::sync::atomic::AtomicUsize,
    generated: std::sync::atomic::AtomicU64,
    requests: std::sync::atomic::AtomicU64,
    prefilled: std::sync::atomic::AtomicU64,
    ttft: Mutex<Vec<f64>>,
    itl: Mutex<Vec<f64>>,
}

pub struct ScriptedServing {
    script: Script,
    seqs: usize,
    loaded: bool,
    counters: Arc<ServingCounters>,
}

impl ScriptedServing {
    pub fn factory(script: Script) -> crate::engine::ServingFactory {
        Arc::new(move |opts| {
            Some(Box::new(ScriptedServing { script: script.clone(), seqs: opts.max_num_seqs, loaded: false, counters: Default::default() }) as Box<dyn crate::target::Target>)
        })
    }
}

impl crate::target::Target for ScriptedServing {
    fn name(&self) -> &str {
        "scripted-serving"
    }

    fn capabilities(&self) -> crate::target::Capabilities {
        crate::target::Capabilities { streaming: true, vision: self.script.vision, ..Default::default() }
    }

    fn load(&mut self, model: &str) -> Result<(), crate::result::BackendError> {
        if !self.script.models.iter().any(|m| m == model) {
            return Err(crate::result::BackendError::new(crate::result::ErrorKind::ModelNotFound, format!("no config.json for {model} under the fake hub")));
        }
        self.loaded = true;
        Ok(())
    }

    fn generate(&mut self, request: &crate::target::Request, on_text: &mut dyn FnMut(&str)) -> Result<crate::result::Generation, crate::result::BackendError> {
        let chat = crate::chat::ChatRequest::new(vec![crate::chat::Message::new(crate::chat::Role::User, request.prompt.clone())], crate::chat::Sampling::default());
        crate::target::Target::chat(self, &chat, on_text)
    }

    fn chat(&mut self, request: &crate::chat::ChatRequest, on_text: &mut dyn FnMut(&str)) -> Result<crate::result::Generation, crate::result::BackendError> {
        use std::sync::atomic::Ordering::SeqCst;
        let c = &self.counters;
        c.running.fetch_add(1, SeqCst);
        let started = std::time::Instant::now();
        thread::sleep(Duration::from_millis(self.script.first_delay_ms));
        let mut text = String::new();
        let mut last = None;
        let mut pieces = 0u64;
        for chunk in &self.script.chunks {
            if request.cancelled() {
                break;
            }
            let at = std::time::Instant::now();
            match last {
                Some(prev) => c.itl.lock().unwrap().push(at.duration_since(prev).as_secs_f64()),
                None => c.ttft.lock().unwrap().push(at.duration_since(started).as_secs_f64()),
            }
            last = Some(at);
            text.push_str(chunk);
            on_text(chunk);
            pieces += 1;
            c.generated.fetch_add(1, SeqCst);
            thread::sleep(Duration::from_millis(self.script.chunk_delay_ms));
        }
        let (prompt, completion) = self.script.usage.unwrap_or((0, pieces));
        c.prefilled.fetch_add(prompt, SeqCst);
        c.requests.fetch_add(1, SeqCst);
        c.running.fetch_sub(1, SeqCst);
        let elapsed = started.elapsed().as_secs_f64();
        let ttft = c.ttft.lock().unwrap().last().copied().unwrap_or(elapsed);
        Ok(crate::result::Generation {
            text,
            prompt_tokens: prompt,
            completion_tokens: completion.min(pieces),
            ttft_secs: ttft,
            prefill_secs: ttft,
            decode_secs: (elapsed - ttft).max(0.0),
            ..Default::default()
        })
    }

    fn share(&self) -> Option<Box<dyn crate::target::Target>> {
        Some(Box::new(ScriptedServing { script: self.script.clone(), seqs: self.seqs, loaded: self.loaded, counters: self.counters.clone() }))
    }

    fn stats(&self) -> Option<specrt::serve::Stats> {
        use std::sync::atomic::Ordering::SeqCst;
        let c = &self.counters;
        let mut s = specrt::serve::Stats {
            running: c.running.load(SeqCst).min(self.seqs),
            waiting: c.running.load(SeqCst).saturating_sub(self.seqs),
            kv_blocks_total: 100,
            kv_blocks_used: c.running.load(SeqCst) * 2,
            kv_block_bytes: 1 << 20,
            generated_tokens: c.generated.load(SeqCst),
            ..Default::default()
        };
        if c.prefilled.load(SeqCst) > 0 {
            s.prefill_tokens_per_step.observe(c.prefilled.load(SeqCst) as f64);
        }
        for v in c.itl.lock().unwrap().iter() {
            s.inter_token_secs.observe(*v);
        }
        Some(s)
    }

    fn metrics(&self) -> Option<String> {
        self.stats().map(|s| crate::serving::render(&s))
    }
}

pub fn engine(wake: crate::engine::Wake, script: Script) -> crate::engine::Engine {
    crate::engine::Engine::with_factories(wake, ScriptedModel::factory(script.clone()), ScriptedServing::factory(script))
}
