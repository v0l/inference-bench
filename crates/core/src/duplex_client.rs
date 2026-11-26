use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::net::TcpStream;
use std::time::{Duration, Instant};
use tungstenite::client::IntoClientRequest;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

pub struct Options {
    pub url: String,
    pub key: Option<String>,
    pub start: Value,
    pub realtime: bool,
    pub tail_frames: usize,
    pub paced: bool,
}

#[derive(Debug, Default)]
pub struct Run {
    pub ready: Value,
    pub audio: Vec<f32>,
    pub frames: Vec<Vec<f32>>,
    pub events: Vec<Value>,
    pub text: String,
    pub latency_ms: Vec<f64>,
    pub late: usize,
}

impl Run {
    pub fn percentile(&self, p: f64) -> Option<f64> {
        let mut v = self.latency_ms.clone();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[((v.len() - 1) as f64 * p).round() as usize])
    }
}

type Socket = WebSocket<MaybeTlsStream<TcpStream>>;

fn connect(o: &Options) -> Result<Socket> {
    let mut request = o.url.as_str().into_client_request().with_context(|| format!("bad websocket url {}", o.url))?;
    if let Some(k) = &o.key {
        request.headers_mut().insert("authorization", format!("Bearer {k}").parse()?);
    }
    if o.realtime {
        request.headers_mut().insert("sec-websocket-protocol", "realtime".parse()?);
    }
    let (socket, _) = tungstenite::connect(request).with_context(|| format!("connect {}", o.url))?;
    if let MaybeTlsStream::Plain(s) = socket.get_ref() {
        s.set_nodelay(true)?;
    }
    Ok(socket)
}

fn set_timeout(s: &Socket, t: Option<Duration>) -> Result<()> {
    if let MaybeTlsStream::Plain(tcp) = s.get_ref() {
        tcp.set_read_timeout(t)?;
    }
    Ok(())
}

fn read(s: &mut Socket) -> Result<Option<Message>> {
    match s.read() {
        Ok(m) => Ok(Some(m)),
        Err(tungstenite::Error::Io(e)) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => Ok(None),
        Err(tungstenite::Error::ConnectionClosed) | Err(tungstenite::Error::AlreadyClosed) => bail!("the server closed the session"),
        Err(e) => Err(e.into()),
    }
}

fn pcm(b: &[u8]) -> Vec<f32> {
    specrt::duplex::wav::pcm16_to_f32(b)
}

struct Collector {
    run: Run,
    realtime: bool,
    sent: Vec<Instant>,
}

impl Collector {
    fn take(&mut self, m: Message) -> Result<bool> {
        let now = Instant::now();
        match m {
            Message::Binary(b) => self.frame(pcm(&b), now),
            Message::Text(t) => {
                let v: Value = serde_json::from_str(&t).with_context(|| format!("not JSON: {t}"))?;
                match v["type"].as_str().unwrap_or("") {
                    "error" => bail!("server error: {}", v.get("message").or(v["error"].get("message")).cloned().unwrap_or(v.clone())),
                    "late" => self.run.late += 1,
                    "text" => self.run.text.push_str(v["text"].as_str().unwrap_or("")),
                    "response.output_audio.delta" if self.realtime => {
                        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, v["delta"].as_str().unwrap_or(""))?;
                        self.frame(pcm(&bytes), now);
                    }
                    "response.output_audio_transcript.delta" => self.run.text.push_str(v["delta"].as_str().unwrap_or("")),
                    "end" => {
                        self.run.events.push(v);
                        return Ok(false);
                    }
                    _ => {}
                }
                if v["type"] != "response.output_audio.delta" {
                    self.run.events.push(v);
                }
            }
            Message::Close(_) => return Ok(false),
            _ => {}
        }
        Ok(true)
    }

    fn frame(&mut self, f: Vec<f32>, now: Instant) {
        let n = self.run.frames.len();
        if n >= 1
            && let Some(t) = self.sent.get(n - 1)
        {
            self.run.latency_ms.push(now.duration_since(*t).as_secs_f64() * 1e3);
        }
        self.run.audio.extend(&f);
        self.run.frames.push(f);
    }
}

pub fn stream(o: &Options, input: &[f32]) -> Result<Run> {
    let mut s = connect(o)?;
    let mut c = Collector { run: Run::default(), realtime: o.realtime, sent: Vec::new() };
    if o.realtime {
        s.send(Message::Text(json!({"type": "session.update", "session": {"instructions": o.start["card"].as_str().unwrap_or("")}}).to_string().into()))?;
    } else {
        s.send(Message::Text(o.start.to_string().into()))?;
    }
    let (rate, samples) = loop {
        let m = read(&mut s)?.context("no reply to the start message")?;
        let Message::Text(t) = m else { continue };
        let v: Value = serde_json::from_str(&t)?;
        match v["type"].as_str() {
            Some("ready") => {
                let r = (v["sample_rate"].as_u64().unwrap_or(24000) as usize, v["frame_samples"].as_u64().unwrap_or(3200) as usize);
                c.run.ready = v;
                break r;
            }
            Some("session.updated") if o.realtime => {
                c.run.ready = v;
                break (24000, 3200);
            }
            Some("error") => bail!("refused: {}", v.get("message").or(v["error"].get("message")).cloned().unwrap_or(v.clone())),
            _ => c.run.events.push(v),
        }
    };
    let frame = Duration::from_secs_f64(samples as f64 / rate as f64);
    let count = input.len().div_ceil(samples) + o.tail_frames;
    let mut padded = input.to_vec();
    padded.resize(count * samples, 0.0);
    let started = Instant::now();
    set_timeout(&s, Some(Duration::from_millis(2)))?;
    let mut open = true;
    for (i, chunk) in padded.chunks(samples).enumerate() {
        let due = started + frame * i as u32;
        while open && o.paced && Instant::now() < due {
            if let Some(m) = read(&mut s)? {
                open = c.take(m)?;
            }
        }
        if !open {
            break;
        }
        let bytes = specrt::duplex::wav::f32_to_pcm16(chunk);
        let msg = match o.realtime {
            true => Message::Text(json!({"type": "input_audio_buffer.append", "audio": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)}).to_string().into()),
            false => Message::Binary(bytes.into()),
        };
        c.sent.push(Instant::now());
        s.send(msg)?;
    }
    let want = count + 1;
    let deadline = Instant::now() + frame * 4 + Duration::from_secs(10);
    while open && c.run.frames.len() < want && Instant::now() < deadline {
        if let Some(m) = read(&mut s)? {
            open = c.take(m)?;
        }
    }
    if open {
        if !o.realtime {
            let _ = s.send(Message::Text(json!({"type": "stop"}).to_string().into()));
        }
        let _ = s.close(None);
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end {
            match read(&mut s) {
                Ok(Some(m)) => {
                    if !matches!(m, Message::Binary(_)) {
                        let _ = c.take(m);
                    }
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    }
    Ok(c.run)
}

pub fn ws_url(base: &str, path: &str) -> String {
    let b = base.trim_end_matches('/');
    let b = b.strip_suffix("/v1").unwrap_or(b);
    let b = b.replacen("https://", "wss://", 1).replacen("http://", "ws://", 1);
    let b = if b.starts_with("ws://") || b.starts_with("wss://") { b } else { format!("ws://{b}") };
    format!("{b}{path}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_urls_become_websocket_urls() {
        assert_eq!(ws_url("http://127.0.0.1:8080/v1", "/v1/duplex"), "ws://127.0.0.1:8080/v1/duplex");
        assert_eq!(ws_url("https://h/", "/v1/duplex"), "wss://h/v1/duplex");
        assert_eq!(ws_url("127.0.0.1:9", "/v1/realtime?model=x"), "ws://127.0.0.1:9/v1/realtime?model=x");
    }
}
