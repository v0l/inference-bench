use crate::chat::ChatRequest;
use crate::backend::{Backend, Capabilities, Logits, Request};
use crate::result::{BackendError, ErrorKind, Generation};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

const TRANSFORMERS_WORKER: &str = include_str!("transformers_worker.py");

pub struct Subprocess {
    name: String,
    program: PathBuf,
    args: Vec<String>,
    caps: Capabilities,
    max_ctx: Option<usize>,
    child: Option<Running>,
}

struct Running {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: Arc<Mutex<String>>,
}

impl Subprocess {
    pub fn new(name: &str, program: impl Into<PathBuf>, args: Vec<String>, caps: Capabilities) -> Self {
        Self {
            name: name.into(),
            program: program.into(),
            args,
            caps,
            max_ctx: None,
            child: None,
        }
    }

    pub fn transformers(python: Option<String>, device: Option<String>) -> Self {
        let script = std::env::temp_dir()
            .join(format!("inference-bench-{}", env!("CARGO_PKG_VERSION")))
            .join("transformers_worker.py");
        let mut args = vec![script.display().to_string()];
        if let Some(d) = device {
            args.push("--device".into());
            args.push(d);
        }
        let mut s = Self::new(
            "transformers",
            python.unwrap_or_else(|| "python3".into()),
            args,
            Capabilities { streaming: true, logits: true, memory: true },
        );
        s.install_script(&script);
        s
    }

    pub fn specrt(worker: Option<PathBuf>, max_ctx: Option<usize>) -> Self {
        let program = worker.unwrap_or_else(default_specrt_worker);
        let mut s = Self::new(
            "specrt",
            program,
            Vec::new(),
            Capabilities { streaming: false, logits: true, memory: true },
        );
        s.max_ctx = max_ctx;
        s
    }

    fn install_script(&mut self, script: &Path) {
        if let Some(dir) = script.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(script, TRANSFORMERS_WORKER);
    }

    fn start(&mut self) -> Result<(), BackendError> {
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                BackendError::new(
                    ErrorKind::Unreachable,
                    format!("cannot start {}: {e}", self.program.display()),
                )
            })?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let stderr = Arc::new(Mutex::new(String::new()));
        let sink = stderr.clone();
        let name = self.name.clone();
        let mut err = BufReader::new(child.stderr.take().unwrap());
        std::thread::spawn(move || {
            let mut line = String::new();
            while err.read_line(&mut line).unwrap_or(0) > 0 {
                tracing::info!(worker = %name, "{}", line.trim_end());
                let mut s = sink.lock().unwrap();
                s.push_str(&line);
                if s.len() > 16384 {
                    let cut = s.len() - 8192;
                    let cut = (cut..s.len()).find(|i| s.is_char_boundary(*i)).unwrap_or(0);
                    s.drain(..cut);
                }
                line.clear();
            }
        });
        tracing::info!(worker = %self.name, program = %self.program.display(), pid = child.id(), "worker started");
        self.child = Some(Running { child, stdin, stdout, stderr });
        Ok(())
    }

    fn send(&mut self, message: Value) -> Result<(), BackendError> {
        let run = self.child.as_mut().ok_or_else(|| BackendError::failed("worker not started"))?;
        writeln!(run.stdin, "{message}")
            .and_then(|_| run.stdin.flush())
            .map_err(|e| self.died(&format!("write failed: {e}")))
    }

    fn died(&self, why: &str) -> BackendError {
        let tail = self
            .child
            .as_ref()
            .map(|r| r.stderr.lock().unwrap().trim().to_string())
            .unwrap_or_default();
        let tail: String = tail.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        BackendError::classify(format!("{} worker stopped ({why}): {tail}", self.name))
    }

    fn next_event(&mut self) -> Result<Value, BackendError> {
        loop {
            let run = self.child.as_mut().ok_or_else(|| BackendError::failed("worker not started"))?;
            let mut line = String::new();
            let read = run.stdout.read_line(&mut line);
            let status = match &read {
                Ok(0) => Some(run.child.wait().map(|s| s.to_string()).unwrap_or_default()),
                _ => None,
            };
            let n = read.map_err(|e| self.died(&e.to_string()))?;
            if n == 0 {
                std::thread::sleep(std::time::Duration::from_millis(50));
                return Err(self.died(&format!("exit {}", status.unwrap_or_default())));
            }
            match serde_json::from_str::<Value>(line.trim()) {
                Ok(v) if v.get("event").is_some() => return Ok(v),
                _ => continue,
            }
        }
    }

    fn worker_error(event: &Value) -> BackendError {
        let message = event["message"].as_str().unwrap_or("worker error").to_string();
        match event["kind"].as_str() {
            Some(k) => BackendError::new(ErrorKind::from_wire(k), message),
            None => BackendError::classify(message),
        }
    }
}

fn default_specrt_worker() -> PathBuf {
    let name = format!("inference-bench-specrt{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(&name)))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(name))
}

impl Drop for Subprocess {
    fn drop(&mut self) {
        if let Some(mut run) = self.child.take() {
            let _ = writeln!(run.stdin, "{}", json!({"op": "quit"}));
            drop(run.stdin);
            let _ = run.child.kill();
            let _ = run.child.wait();
        }
    }
}

impl Backend for Subprocess {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        self.caps
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        if self.child.is_none() {
            self.start()?;
        }
        let started = std::time::Instant::now();
        self.send(json!({"op": "load", "model": model, "max_ctx": self.max_ctx}))?;
        loop {
            let event = self.next_event()?;
            match event["event"].as_str() {
                Some("loaded") => {
                    tracing::info!(worker = %self.name, model, load_secs = started.elapsed().as_secs_f64(), "worker loaded model");
                    return Ok(());
                }
                Some("error") => {
                    let e = Self::worker_error(&event);
                    tracing::error!(worker = %self.name, model, error = %e, "worker failed to load model");
                    return Err(e);
                }
                _ => {}
            }
        }
    }

    fn generate(
        &mut self,
        request: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        self.send(json!({"op": "generate", "prompt": request.prompt, "max_tokens": request.max_tokens, "chat": false}))?;
        self.collect(on_text)
    }

    fn chat(
        &mut self,
        request: &ChatRequest,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let sampling = &request.sampling;
        self.send(json!({
            "op": "chat",
            "messages": request.messages_json(),
            "template": request.use_template,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "seed": sampling.seed.unwrap_or(0),
        }))?;
        self.collect(on_text)
    }

    fn logits(&mut self, prompt: &str) -> Result<Logits, BackendError> {
        self.logits_of(prompt)
    }
}

impl Subprocess {
    fn collect(&mut self, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let mut text = String::new();
        loop {
            let event = self.next_event()?;
            match event["event"].as_str() {
                Some("text") => {
                    let t = event["text"].as_str().unwrap_or("");
                    text.push_str(t);
                    on_text(t);
                }
                Some("done") => {
                    let mut g: Generation = serde_json::from_value(event["generation"].clone())
                        .map_err(|e| BackendError::failed(format!("bad generation: {e}")))?;
                    if g.text.is_empty() {
                        g.text = text;
                    }
                    return Ok(g);
                }
                Some("error") => return Err(Self::worker_error(&event)),
                _ => {}
            }
        }
    }

    fn logits_of(&mut self, prompt: &str) -> Result<Logits, BackendError> {
        let path = std::env::temp_dir().join(format!(
            "inference-bench-logits-{}-{}.bin",
            std::process::id(),
            self.name
        ));
        self.send(json!({"op": "logits", "prompt": prompt, "path": path}))?;
        loop {
            let event = self.next_event()?;
            match event["event"].as_str() {
                Some("logits") => {
                    let tokens: Vec<u32> = event["tokens"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                        .unwrap_or_default();
                    let vocab = event["vocab"].as_u64().unwrap_or(0) as usize;
                    let out = Logits::read(&path, tokens, vocab);
                    let _ = std::fs::remove_file(&path);
                    return out;
                }
                Some("error") => return Err(Self::worker_error(&event)),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAKE: &str = r#"
import json, sys, struct
for line in sys.stdin:
    m = json.loads(line)
    if m["op"] == "load":
        if m["model"] == "bad":
            print(json.dumps({"event": "error", "kind": "unsupported_architecture", "message": "model_type foo is not supported"}), flush=True)
        elif m["model"] == "crash":
            sys.stderr.write("boom unknown model architecture\n"); sys.exit(3)
        else:
            print("noise that is not json", flush=True)
            print(json.dumps({"event": "loaded"}), flush=True)
    elif m["op"] == "generate":
        if m.get("chat", True):
            print(json.dumps({"event": "text", "text": "<templated> "}), flush=True)
        for w in ["a ", "b "]:
            print(json.dumps({"event": "text", "text": w}), flush=True)
        print(json.dumps({"event": "done", "generation": {"text": "", "prompt_tokens": 4, "completion_tokens": 2, "ttft_secs": 0.1, "prefill_secs": 0.1, "decode_secs": 0.5, "peak_memory_bytes": 1024}}), flush=True)
    elif m["op"] == "chat":
        print(json.dumps({"event": "text", "text": m["messages"][-1]["content"] + "!"}), flush=True)
        print(json.dumps({"event": "done", "generation": {"text": "", "prompt_tokens": 3, "completion_tokens": 2, "ttft_secs": 0.1, "prefill_secs": 0.1, "decode_secs": 0.5, "peak_memory_bytes": 1024}}), flush=True)
    elif m["op"] == "logits":
        open(m["path"], "wb").write(struct.pack("<6f", 1, 2, 3, 4, 5, 6))
        print(json.dumps({"event": "logits", "tokens": [7, 8], "vocab": 3}), flush=True)
    elif m["op"] == "quit":
        break
"#;

    fn fake() -> Subprocess {
        let dir = std::env::temp_dir().join(format!("ib-fake-worker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("worker.py");
        std::fs::write(&script, FAKE).unwrap();
        Subprocess::new(
            "fake",
            "python3",
            vec![script.display().to_string()],
            Capabilities { streaming: true, logits: true, memory: true },
        )
    }

    #[test]
    fn loads_generates_and_reads_logits_over_the_worker_protocol() {
        let mut b = fake();
        b.load("good").unwrap();
        let mut seen = String::new();
        let g = b
            .generate(&Request { prompt: "p".into(), max_tokens: 2 }, &mut |t| seen.push_str(t))
            .unwrap();
        assert_eq!(seen, "a b ");
        assert_eq!(g.text, "a b ");
        assert_eq!(g.peak_memory_bytes, Some(1024));
        let l = b.logits("p").unwrap();
        assert_eq!((l.tokens.clone(), l.vocab), (vec![7, 8], 3));
        assert_eq!(l.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn generate_sends_the_raw_prompt_like_an_openai_completion() {
        let mut b = fake();
        b.load("good").unwrap();
        let g = b.generate(&Request { prompt: "p".into(), max_tokens: 2 }, &mut |_| {}).unwrap();
        assert!(!g.text.contains("<templated>"), "{}", g.text);
    }

    #[test]
    fn chat_sends_the_messages_and_collects_the_reply() {
        use crate::chat::{ChatRequest, Message, Role, Sampling};
        let mut b = fake();
        b.load("good").unwrap();
        let request = ChatRequest::new(vec![Message::new(Role::User, "hey")], Sampling::default());
        let g = b.chat(&request, &mut |_| {}).unwrap();
        assert_eq!(g.text, "hey!");
        assert_eq!(g.completion_tokens, 2);
    }

    #[test]
    fn a_worker_error_keeps_its_kind_and_message() {
        let mut b = fake();
        let e = b.load("bad").unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedArchitecture);
        assert!(e.message.contains("foo"));
    }

    #[test]
    fn a_crashed_worker_reports_its_stderr() {
        let mut b = fake();
        let e = b.load("crash").unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedArchitecture);
        assert!(e.message.contains("boom"), "{}", e.message);
    }

    #[test]
    fn a_missing_program_is_unreachable() {
        let mut b = Subprocess::new("x", "/nonexistent/worker", vec![], Capabilities::default());
        assert_eq!(b.load("m").unwrap_err().kind, ErrorKind::Unreachable);
    }
}
