use inference_bench_core::target::{Target, Capabilities, Request};
use inference_bench_core::chat::ChatRequest;
use inference_bench_core::result::{BackendError, ErrorKind, Generation};
use inference_bench_server::{Server, ServerConfig, ServerHandle};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Stats {
    pub loads: AtomicUsize,
    pub cancelled: AtomicUsize,
    pub generated: AtomicUsize,
    pub prompts: Mutex<Vec<String>>,
}

pub struct Fake {
    pub tokens: usize,
    pub first_delay_ms: u64,
    pub token_delay_ms: u64,
    pub stats: Arc<Stats>,
}

impl Target for Fake {
    fn name(&self) -> &str {
        "fake"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        if model == "broken" {
            return Err(BackendError::new(ErrorKind::ModelNotFound, "no such checkpoint"));
        }
        self.stats.loads.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn generate(&mut self, request: &Request, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        self.run(&request.prompt, request.max_tokens, None, on_text)
    }

    fn chat(&mut self, request: &ChatRequest, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        self.run(&request.flatten(), request.sampling.max_tokens, Some(request), on_text)
    }
}

impl Fake {
    fn run(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        request: Option<&ChatRequest>,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        self.stats.prompts.lock().unwrap().push(prompt.to_string());
        let started = Instant::now();
        std::thread::sleep(Duration::from_millis(self.first_delay_ms));
        let ttft = started.elapsed().as_secs_f64();
        let n = self.tokens.min(max_tokens);
        let mut text = String::new();
        for i in 0..n {
            if request.is_some_and(|r| r.cancelled()) {
                self.stats.cancelled.fetch_add(1, Ordering::SeqCst);
                break;
            }
            std::thread::sleep(Duration::from_millis(self.token_delay_ms));
            let piece = format!("w{i} ");
            text.push_str(&piece);
            on_text(&piece);
        }
        if !request.is_some_and(|r| r.cancelled()) {
            self.stats.generated.fetch_add(1, Ordering::SeqCst);
        }
        let completion_tokens = text.split_whitespace().count() as u64;
        Ok(Generation {
            text,
            prompt_tokens: prompt.split_whitespace().count() as u64,
            completion_tokens,
            ttft_secs: ttft,
            prefill_secs: ttft,
            decode_secs: started.elapsed().as_secs_f64() - ttft,
            peak_memory_bytes: None,
        })
    }
}

pub struct Rig {
    pub rt: tokio::runtime::Runtime,
    pub handle: Option<ServerHandle>,
    pub stats: Arc<Stats>,
    pub url: String,
}

pub struct Shape {
    pub tokens: usize,
    pub first_delay_ms: u64,
    pub token_delay_ms: u64,
}

impl Default for Shape {
    fn default() -> Self {
        Self { tokens: 4, first_delay_ms: 0, token_delay_ms: 1 }
    }
}

pub fn config(extra: &str) -> ServerConfig {
    ServerConfig::parse(&format!("bind = \"127.0.0.1:0\"\n{extra}")).unwrap()
}

pub fn start(config: ServerConfig, shape: Shape) -> Rig {
    let stats = Arc::new(Stats::default());
    let s = stats.clone();
    let factory = move |_: Option<usize>| -> Result<Box<dyn Target>, BackendError> {
        Ok(Box::new(Fake {
            tokens: shape.tokens,
            first_delay_ms: shape.first_delay_ms,
            token_delay_ms: shape.token_delay_ms,
            stats: s.clone(),
        }))
    };
    start_with(config, factory, stats)
}

pub fn start_with(config: ServerConfig, factory: impl inference_bench_server::EngineFactory, stats: Arc<Stats>) -> Rig {
    init_logs();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
    let handle = rt
        .block_on(Server::new(config).with_factory(factory).with_gpu_probe(Arc::new(|| Some(1_000_000))).start())
        .unwrap();
    let url = handle.url();
    Rig { rt, handle: Some(handle), stats, url }
}

impl Rig {
    pub fn get(&self, path: &str, key: Option<&str>) -> (u16, String, Vec<(String, String)>) {
        let mut req = agent().get(format!("{}{path}", self.url));
        if let Some(k) = key {
            req = req.header("Authorization", &format!("Bearer {k}"));
        }
        collect(req.call())
    }

    pub fn post(&self, path: &str, key: Option<&str>, body: serde_json::Value) -> (u16, String, Vec<(String, String)>) {
        let mut req = agent().post(format!("{}{path}", self.url));
        if let Some(k) = key {
            req = req.header("Authorization", &format!("Bearer {k}"));
        }
        collect(req.send_json(&body))
    }

    pub fn raw_post(&self, path: &str, body: &str) -> TcpStream {
        let addr = self.handle.as_ref().unwrap().addr;
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write!(
            s,
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        s
    }

    pub fn metrics(&self) -> String {
        self.get("/metrics", None).1
    }

    pub fn stop(mut self) {
        let h = self.handle.take().unwrap();
        self.rt.block_on(h.stop()).unwrap();
    }
}

pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().http_status_as_error(false).build().into()
}

fn collect(r: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> (u16, String, Vec<(String, String)>) {
    let mut resp = r.unwrap();
    let headers = resp
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap(), headers)
}

pub fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
}

pub fn read_until(s: &mut TcpStream, needle: &str) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if String::from_utf8_lossy(&buf).contains(needle) {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&buf).to_string()
}

pub fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("timed out waiting for {what}");
}

static LOGS: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static INIT: Once = Once::new();

#[derive(Clone, Copy)]
struct Sink;

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        LOGS.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Sink {
    type Writer = Sink;

    fn make_writer(&'a self) -> Sink {
        Sink
    }
}

fn init_logs() {
    INIT.call_once(|| {
        let sub = tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(Sink)
            .finish();
        tracing::subscriber::set_global_default(sub).unwrap();
    });
}

pub fn log_lines() -> Vec<serde_json::Value> {
    String::from_utf8_lossy(&LOGS.lock().unwrap())
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("log line is not JSON: {l}: {e}")))
        .collect()
}

pub fn lines_for(request_id: &str) -> Vec<serde_json::Value> {
    log_lines()
        .into_iter()
        .filter(|l| l["spans"].as_array().is_some_and(|s| s.iter().any(|s| s["request_id"] == request_id)))
        .collect()
}

#[allow(dead_code)]
pub fn flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}
