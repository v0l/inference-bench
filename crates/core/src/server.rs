use crate::chat::{ChatRequest, Stats};
use crate::engine::{ChatEvent, Handle};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct LogEntry {
    pub when: chrono::DateTime<chrono::Utc>,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub millis: u64,
    pub stats: Option<Stats>,
}

pub type Log = Arc<Mutex<Vec<LogEntry>>>;

pub struct Server {
    pub port: u16,
    pub log: Log,
    stop: Arc<AtomicBool>,
}

impl Server {
    pub fn start(handle: Handle, port: u16, api_key: &str, lan: bool, wake: crate::engine::Wake) -> std::io::Result<Self> {
        let host = if lan { "0.0.0.0" } else { "127.0.0.1" };
        let listener = TcpListener::bind((host, port))?;
        let port = listener.local_addr()?.port();
        listener.set_nonblocking(true)?;
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (log2, stop2, key) = (log.clone(), stop.clone(), api_key.trim().to_string());
        std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let (h, log, key, wake) = (handle.clone(), log2.clone(), key.clone(), wake.clone());
                        std::thread::spawn(move || {
                            let _ = stream.set_nonblocking(false);
                            let _ = serve(stream, &h, &key, &log);
                            wake();
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(25));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Self { port, log, stop })
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Http {
    method: String,
    path: String,
    authorization: String,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> std::io::Result<Option<Http>> {
    let mut r = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let (mut length, mut authorization) = (0usize, String::new());
    loop {
        line.clear();
        r.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
        let lower = line.to_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("authorization:") {
            authorization = line[14..].trim().to_string();
        }
    }
    let mut body = vec![0u8; length.min(8 << 20)];
    r.read_exact(&mut body)?;
    Ok(Some(Http { method, path, authorization, body }))
}

fn reply(stream: &mut TcpStream, status: u16, body: &Value) -> std::io::Result<()> {
    let text = body.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )
}

fn error_body(message: &str) -> Value {
    json!({"error": {"message": message, "type": "invalid_request_error"}})
}

fn serve(mut stream: TcpStream, h: &Handle, key: &str, log: &Log) -> std::io::Result<()> {
    let started = Instant::now();
    let Some(req) = read_request(&stream)? else { return Ok(()) };
    let path = req.path.split('?').next().unwrap_or("").to_string();
    let authorised = key.is_empty() || req.authorization == format!("Bearer {key}");
    let mut stats = None;
    let status = if !authorised {
        reply(&mut stream, 401, &error_body("invalid api key"))?;
        401
    } else {
        match (req.method.as_str(), path.as_str()) {
            ("GET", "/v1/models") | ("GET", "/models") => {
                let data: Vec<Value> = h
                    .loaded()
                    .map(|l| vec![json!({"id": l.display, "object": "model", "owned_by": "inference-bench"})])
                    .unwrap_or_default();
                reply(&mut stream, 200, &json!({"object": "list", "data": data}))?;
                200
            }
            ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
                complete(&mut stream, h, &req, path.ends_with("chat/completions"), &mut stats)?
            }
            _ => {
                reply(&mut stream, 404, &error_body("not found"))?;
                404
            }
        }
    };
    log.lock().unwrap().push(LogEntry {
        when: chrono::Utc::now(),
        method: req.method,
        path,
        status,
        millis: started.elapsed().as_millis() as u64,
        stats,
    });
    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
}

fn complete(
    stream: &mut TcpStream,
    h: &Handle,
    req: &Http,
    chat: bool,
    stats: &mut Option<Stats>,
) -> std::io::Result<u16> {
    let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
    let Some(mut request) = ChatRequest::from_openai(&body) else {
        reply(stream, 400, &error_body("the body needs messages or a prompt"))?;
        return Ok(400);
    };
    let Some(loaded) = h.loaded() else {
        reply(stream, 400, &error_body("no model is loaded"))?;
        return Ok(400);
    };
    let cancel = Arc::new(AtomicBool::new(false));
    request.cancel = Some(cancel.clone());
    let streaming = body["stream"].as_bool().unwrap_or(false);
    let events = h.chat(request);
    let id = format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis());
    let chunk = |text: &str| {
        let choice = match chat {
            true => json!({"index": 0, "delta": {"content": text}, "finish_reason": null}),
            false => json!({"index": 0, "text": text, "finish_reason": null}),
        };
        json!({"id": id, "object": if chat {"chat.completion.chunk"} else {"text_completion"}, "model": loaded.display, "choices": [choice]})
    };
    if streaming {
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    }
    let mut text = String::new();
    loop {
        match events.recv() {
            Ok(ChatEvent::Text(t)) => {
                text.push_str(&t);
                if streaming && write!(stream, "data: {}\n\n", chunk(&t)).and_then(|_| stream.flush()).is_err() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Ok(ChatEvent::Done(g)) => {
                *stats = Some(Stats::of(&g));
                let usage = json!({"prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens, "total_tokens": g.prompt_tokens + g.completion_tokens});
                if streaming {
                    write!(stream, "data: {}\n\n", json!({"id": id, "model": loaded.display, "choices": [], "usage": usage}))?;
                    write!(stream, "data: [DONE]\n\n")?;
                } else {
                    let full = if g.text.is_empty() { text } else { g.text };
                    let choice = match chat {
                        true => json!({"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}),
                        false => json!({"index": 0, "text": full, "finish_reason": "stop"}),
                    };
                    reply(stream, 200, &json!({"id": id, "object": "chat.completion", "model": loaded.display, "choices": [choice], "usage": usage}))?;
                }
                return Ok(200);
            }
            Ok(ChatEvent::Failed(e)) => {
                if streaming {
                    write!(stream, "data: {}\n\n", error_body(&e.message))?;
                } else {
                    reply(stream, 400, &error_body(&e.message))?;
                }
                return Ok(400);
            }
            Err(_) => return Ok(500),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, BackendConfig, Request};
    use crate::engine::{Command, Engine, EngineEvent};
    use crate::fake::{FakeServer, Script};
    use crate::openai::OpenAi;

    fn loaded_engine() -> (Engine, FakeServer) {
        let fake = FakeServer::start(Script {
            models: vec!["m".into()],
            chunks: vec!["a ".into(), "b ".into(), "c".into()],
            chunk_delay_ms: 2,
            first_delay_ms: 5,
            usage: Some((4, 3)),
            fail_status: None,
        });
        let e = Engine::spawn(Arc::new(|| {}));
        e.send(Command::Load {
            backend: "fake".into(),
            config: BackendConfig::Openai { url: fake.url.clone(), api_key: None, label: Some("fake".into()) },
            model_id: "m".into(),
            display: "served-model".into(),
        });
        for _ in 0..2 {
            let ev = e.events.recv_timeout(Duration::from_secs(10)).unwrap();
            assert!(!matches!(ev, EngineEvent::LoadFailed(_)), "{ev:?}");
        }
        (e, fake)
    }

    fn start(e: &Engine, key: &str) -> Server {
        Server::start(e.handle(), 0, key, false, Arc::new(|| {})).unwrap()
    }

    #[test]
    fn the_bench_backend_can_load_and_run_against_the_local_server() {
        let (e, _fake) = loaded_engine();
        let server = start(&e, "");
        let mut b = OpenAi::new("local".into(), &server.url(), None);
        b.load("served-model").unwrap();
        let g = b.generate(&Request { prompt: "hi".into(), max_tokens: 3 }, &mut |_| {}).unwrap();
        assert_eq!(g.text, "a b c");
        assert_eq!(g.completion_tokens, 3);
        let log = server.log.lock().unwrap();
        assert!(log.iter().any(|l| l.path == "/v1/completions" && l.status == 200 && l.stats.is_some()));
        assert!(log.iter().any(|l| l.path == "/v1/models"));
    }

    #[test]
    fn chat_completions_are_non_streaming_json_by_default() {
        let (e, _fake) = loaded_engine();
        let server = start(&e, "");
        let mut r = ureq::post(format!("{}/v1/chat/completions", server.url()))
            .send_json(json!({"model": "served-model", "messages": [{"role": "user", "content": "hi"}]}))
            .unwrap();
        let v: Value = r.body_mut().read_json().unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "a b c");
        assert_eq!(v["usage"]["completion_tokens"], 3);
    }

    #[test]
    fn a_wrong_api_key_is_refused_and_logged() {
        let (e, _fake) = loaded_engine();
        let server = start(&e, "sekret");
        let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
        let r = agent.get(format!("{}/v1/models", server.url())).call().unwrap();
        assert_eq!(r.status().as_u16(), 401);
        let ok = agent.get(format!("{}/v1/models", server.url())).header("Authorization", "Bearer sekret").call().unwrap();
        assert_eq!(ok.status().as_u16(), 200);
        let log = server.log.lock().unwrap();
        assert_eq!(log.iter().map(|l| l.status).collect::<Vec<_>>(), vec![401, 200]);
    }

    #[test]
    fn with_no_model_loaded_the_server_says_so() {
        let e = Engine::spawn(Arc::new(|| {}));
        let server = start(&e, "");
        let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
        let mut r = agent
            .post(format!("{}/v1/completions", server.url()))
            .send_json(json!({"prompt": "x"}))
            .unwrap();
        assert_eq!(r.status().as_u16(), 400);
        let v: Value = r.body_mut().read_json().unwrap();
        assert!(v["error"]["message"].as_str().unwrap().contains("no model"));
    }
}
