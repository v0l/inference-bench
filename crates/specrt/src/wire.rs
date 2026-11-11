use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: usize = 256 << 20;
const NONCE_BYTES: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Tokens(Vec<u32>),
    Hidden { rows: u32, data: Vec<u8> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    Hello { version: u32 },
    Ready { layer_start: u32, layer_end: u32, hidden: u32 },
    Forward { slot: u32, position: u32, input: Input, decode: bool },
    Output { rows: u32, data: Vec<u8>, compute_us: u32 },
    Token { token: u32, compute_us: u32 },
    Free { slot: u32 },
    Cancel { slot: u32 },
    Ack,
    Error { message: String },
    Assign { layer_start: u32, layer_end: u32 },
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}

struct Cursor<'a> {
    buf: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32> {
        ensure!(self.buf.len() >= 4, "truncated frame");
        let (head, rest) = self.buf.split_at(4);
        self.buf = rest;
        Ok(u32::from_le_bytes(head.try_into().unwrap()))
    }

    fn u8(&mut self) -> Result<u8> {
        ensure!(!self.buf.is_empty(), "truncated frame");
        let v = self.buf[0];
        self.buf = &self.buf[1..];
        Ok(v)
    }

    fn bytes(&mut self) -> Result<Vec<u8>> {
        let n = self.u32()? as usize;
        ensure!(self.buf.len() >= n, "truncated frame");
        let (head, rest) = self.buf.split_at(n);
        self.buf = rest;
        Ok(head.to_vec())
    }

    fn done(&self) -> Result<()> {
        ensure!(self.buf.is_empty(), "trailing bytes in frame");
        Ok(())
    }
}

impl Message {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Message::Hello { version } => {
                out.push(1);
                put_u32(&mut out, *version);
            }
            Message::Ready { layer_start, layer_end, hidden } => {
                out.push(2);
                put_u32(&mut out, *layer_start);
                put_u32(&mut out, *layer_end);
                put_u32(&mut out, *hidden);
            }
            Message::Forward { slot, position, input, decode } => {
                out.push(3);
                put_u32(&mut out, *slot);
                put_u32(&mut out, *position);
                out.push(*decode as u8);
                match input {
                    Input::Tokens(tokens) => {
                        out.push(0);
                        put_u32(&mut out, tokens.len() as u32);
                        for t in tokens {
                            put_u32(&mut out, *t);
                        }
                    }
                    Input::Hidden { rows, data } => {
                        out.push(1);
                        put_u32(&mut out, *rows);
                        put_bytes(&mut out, data);
                    }
                }
            }
            Message::Output { rows, data, compute_us } => {
                out.push(4);
                put_u32(&mut out, *rows);
                put_u32(&mut out, *compute_us);
                put_bytes(&mut out, data);
            }
            Message::Token { token, compute_us } => {
                out.push(5);
                put_u32(&mut out, *token);
                put_u32(&mut out, *compute_us);
            }
            Message::Free { slot } => {
                out.push(6);
                put_u32(&mut out, *slot);
            }
            Message::Cancel { slot } => {
                out.push(7);
                put_u32(&mut out, *slot);
            }
            Message::Ack => out.push(8),
            Message::Error { message } => {
                out.push(9);
                put_bytes(&mut out, message.as_bytes());
            }
            Message::Assign { layer_start, layer_end } => {
                out.push(10);
                put_u32(&mut out, *layer_start);
                put_u32(&mut out, *layer_end);
            }
        }
        out
    }

    pub fn decode(frame: &[u8]) -> Result<Message> {
        let mut c = Cursor { buf: frame };
        let tag = c.u8()?;
        let message = match tag {
            1 => Message::Hello { version: c.u32()? },
            2 => Message::Ready { layer_start: c.u32()?, layer_end: c.u32()?, hidden: c.u32()? },
            3 => {
                let slot = c.u32()?;
                let position = c.u32()?;
                let decode = c.u8()? != 0;
                let input = match c.u8()? {
                    0 => {
                        let n = c.u32()? as usize;
                        ensure!(n <= c.buf.len() / 4, "token count exceeds frame");
                        let mut tokens = Vec::with_capacity(n);
                        for _ in 0..n {
                            tokens.push(c.u32()?);
                        }
                        Input::Tokens(tokens)
                    }
                    1 => {
                        let rows = c.u32()?;
                        Input::Hidden { rows, data: c.bytes()? }
                    }
                    other => bail!("unknown input kind {other}"),
                };
                Message::Forward { slot, position, input, decode }
            }
            4 => {
                let rows = c.u32()?;
                let compute_us = c.u32()?;
                Message::Output { rows, data: c.bytes()?, compute_us }
            }
            5 => Message::Token { token: c.u32()?, compute_us: c.u32()? },
            6 => Message::Free { slot: c.u32()? },
            7 => Message::Cancel { slot: c.u32()? },
            8 => Message::Ack,
            9 => Message::Error { message: String::from_utf8(c.bytes()?).context("error text")? },
            10 => Message::Assign { layer_start: c.u32()?, layer_end: c.u32()? },
            other => bail!("unknown message tag {other}"),
        };
        c.done()?;
        Ok(message)
    }
}

pub fn write_frame(w: &mut impl Write, message: &Message) -> Result<()> {
    let body = message.encode();
    ensure!(body.len() <= MAX_FRAME_BYTES, "frame of {} bytes exceeds the {} byte cap", body.len(), MAX_FRAME_BYTES);
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    w.write_all(&out)?;
    w.flush()?;
    Ok(())
}

pub fn read_frame(r: &mut impl Read) -> Result<Message> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    ensure!(len <= MAX_FRAME_BYTES, "frame of {len} bytes exceeds the {MAX_FRAME_BYTES} byte cap");
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Message::decode(&body)
}

pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(data);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner);
    outer.finalize().into()
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn fresh_nonce() -> Result<[u8; NONCE_BYTES]> {
    let mut nonce = [0u8; NONCE_BYTES];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut nonce)).context("read /dev/urandom")?;
    Ok(nonce)
}

fn server_handshake(stream: &mut TcpStream, key: &[u8]) -> Result<()> {
    let nonce = fresh_nonce()?;
    stream.write_all(&nonce)?;
    let mut answer = [0u8; 32];
    stream.read_exact(&mut answer).context("read auth answer")?;
    let expected = hmac_sha256(key, &nonce);
    if !constant_time_eq(&answer, &expected) {
        let _ = write_frame(stream, &Message::Error { message: "authentication failed".into() });
        bail!("peer failed authentication");
    }
    match read_frame(stream)? {
        Message::Hello { version } if version == PROTOCOL_VERSION => {}
        Message::Hello { version } => {
            let _ = write_frame(stream, &Message::Error { message: format!("protocol version {version}, server speaks {PROTOCOL_VERSION}") });
            bail!("protocol version mismatch: {version}");
        }
        other => bail!("expected hello, got {other:?}"),
    }
    Ok(())
}

pub trait StageHandler: Send + Sync + 'static {
    fn ready(&self) -> Message;
    fn handle(&self, message: Message) -> Message;
}

pub struct StageServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StageServer {
    pub fn start(bind: &str, key: &[u8], handler: Arc<dyn StageHandler>) -> Result<StageServer> {
        ensure!(!key.is_empty(), "a stage server needs a non-empty pre-shared key");
        let listener = TcpListener::bind(bind).with_context(|| format!("bind {bind}"))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let key = key.to_vec();
        let flag = stop.clone();
        let thread = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        let key = key.clone();
                        let handler = handler.clone();
                        let flag = flag.clone();
                        std::thread::spawn(move || {
                            if let Err(e) = serve_connection(stream, &key, handler, &flag) {
                                tracing::warn!(%peer, error = %format!("{e:#}"), "stage connection closed");
                            }
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(5)),
                    Err(e) => {
                        tracing::error!(error = %e, "stage accept failed");
                        break;
                    }
                }
            }
        });
        Ok(StageServer { addr, stop, thread: Some(thread) })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn shutdown(mut self) {
        self.stop_now();
    }

    fn stop_now(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for StageServer {
    fn drop(&mut self) {
        self.stop_now();
    }
}

fn serve_connection(mut stream: TcpStream, key: &[u8], handler: Arc<dyn StageHandler>, stop: &AtomicBool) -> Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    server_handshake(&mut stream, key)?;
    write_frame(&mut stream, &handler.ready())?;
    stream.set_read_timeout(Some(Duration::from_millis(200)))?;
    while !stop.load(Ordering::Relaxed) {
        let mut len = [0u8; 4];
        match stream.read_exact(&mut len) {
            Ok(()) => {}
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len) as usize;
        ensure!(len <= MAX_FRAME_BYTES, "frame of {len} bytes exceeds the cap");
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        let mut body = vec![0u8; len];
        stream.read_exact(&mut body)?;
        stream.set_read_timeout(Some(Duration::from_millis(200)))?;
        let reply = match Message::decode(&body) {
            Ok(message) => handler.handle(message),
            Err(e) => Message::Error { message: format!("{e:#}") },
        };
        write_frame(&mut stream, &reply)?;
    }
    Ok(())
}

pub struct StageClient {
    stream: TcpStream,
    pub ready: Message,
    pub addr: String,
}

impl StageClient {
    pub fn connect(addr: &str, key: &[u8], timeout: Duration) -> Result<StageClient> {
        let target = std::net::ToSocketAddrs::to_socket_addrs(addr)
            .with_context(|| format!("resolve {addr}"))?
            .next()
            .with_context(|| format!("no address for {addr}"))?;
        let mut stream = TcpStream::connect_timeout(&target, timeout).with_context(|| format!("connect {addr}"))?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        let mut nonce = [0u8; NONCE_BYTES];
        stream.read_exact(&mut nonce).context("read auth nonce")?;
        stream.write_all(&hmac_sha256(key, &nonce))?;
        write_frame(&mut stream, &Message::Hello { version: PROTOCOL_VERSION })?;
        let ready = match read_frame(&mut stream).context("stage refused the connection")? {
            Message::Error { message } => bail!("stage {addr} refused: {message}"),
            ready @ Message::Ready { .. } => ready,
            other => bail!("stage {addr} sent {other:?} instead of ready"),
        };
        Ok(StageClient { stream, ready, addr: addr.to_string() })
    }

    pub fn hidden(&self) -> u32 {
        match self.ready {
            Message::Ready { hidden, .. } => hidden,
            _ => 0,
        }
    }

    pub fn call(&mut self, message: &Message) -> Result<Message> {
        write_frame(&mut self.stream, message)?;
        match read_frame(&mut self.stream)? {
            Message::Error { message } => bail!("stage {} error: {message}", self.addr),
            reply => Ok(reply),
        }
    }

    pub fn assign(&mut self, layer_start: u32, layer_end: u32) -> Result<()> {
        match self.call(&Message::Assign { layer_start, layer_end })? {
            ready @ Message::Ready { .. } => {
                self.ready = ready;
                Ok(())
            }
            other => bail!("stage {} answered an assignment with {other:?}", self.addr),
        }
    }

    pub fn ping(&mut self) -> Result<Duration> {
        let started = Instant::now();
        let reply = self.call(&Message::Cancel { slot: u32::MAX })?;
        ensure!(reply == Message::Ack, "unexpected ping reply {reply:?}");
        Ok(started.elapsed())
    }
}

pub struct Chain {
    stages: Vec<StageClient>,
}

impl Chain {
    pub fn connect(addrs: &[String], key: &[u8], timeout: Duration) -> Result<Chain> {
        ensure!(!addrs.is_empty(), "a chain needs at least one stage");
        let mut stages: Vec<StageClient> = Vec::new();
        for addr in addrs.iter().rev() {
            stages.push(StageClient::connect(addr, key, timeout).with_context(|| format!("stage {addr} is not ready, downstream first"))?);
        }
        stages.reverse();
        Chain::assemble(stages)
    }

    pub fn connect_assigned(addrs: &[String], ranges: &[(usize, usize)], key: &[u8], timeout: Duration) -> Result<Chain> {
        ensure!(!addrs.is_empty() && addrs.len() == ranges.len(), "a chain needs one layer range per stage");
        let mut stages: Vec<StageClient> = Vec::new();
        for (addr, &(start, end)) in addrs.iter().zip(ranges).rev() {
            let mut client = StageClient::connect(addr, key, timeout).with_context(|| format!("stage {addr} is not ready, downstream first"))?;
            if client.ready != (Message::Ready { layer_start: start as u32, layer_end: end as u32, hidden: client.hidden() }) {
                client.assign(start as u32, end as u32).with_context(|| format!("stage {addr} could not load layers {start}..{end}"))?;
            }
            stages.push(client);
        }
        stages.reverse();
        Chain::assemble(stages)
    }

    fn assemble(stages: Vec<StageClient>) -> Result<Chain> {
        let mut expected = 0;
        let mut hidden = None;
        for (i, stage) in stages.iter().enumerate() {
            let Message::Ready { layer_start, layer_end, hidden: h } = stage.ready else { unreachable!() };
            ensure!(layer_start == expected, "stage {i} starts at layer {layer_start}, expected {expected}");
            ensure!(layer_end > layer_start, "stage {i} serves no layers");
            ensure!(hidden.map_or(true, |x| x == h), "stage {i} hidden size {h} differs from the previous stage");
            hidden = Some(h);
            expected = layer_end;
        }
        Ok(Chain { stages })
    }

    pub fn layers(&self) -> u32 {
        match self.stages.last().map(|s| &s.ready) {
            Some(Message::Ready { layer_end, .. }) => *layer_end,
            _ => 0,
        }
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    pub fn forward(&mut self, slot: u32, position: u32, tokens: Vec<u32>, decode: bool) -> Result<(u32, ChainTimes)> {
        let mut times = ChainTimes::default();
        let last = self.stages.len() - 1;
        let mut input = Input::Tokens(tokens);
        for (i, stage) in self.stages.iter_mut().enumerate() {
            let started = Instant::now();
            let reply = stage.call(&Message::Forward { slot, position, input, decode }).with_context(|| format!("stage {i} at {} was lost", stage.addr))?;
            times.stage.push(started.elapsed());
            match reply {
                Message::Output { rows, data, compute_us } if i < last => {
                    times.compute.push(Duration::from_micros(compute_us as u64));
                    input = Input::Hidden { rows, data };
                }
                Message::Token { token, compute_us } if i == last => {
                    times.compute.push(Duration::from_micros(compute_us as u64));
                    times.total = times.stage.iter().sum();
                    return Ok((token, times));
                }
                other => bail!("stage {i} replied {other:?}"),
            }
        }
        unreachable!()
    }

    pub fn free(&mut self, slot: u32) -> Result<()> {
        for stage in self.stages.iter_mut() {
            ensure!(stage.call(&Message::Free { slot })? == Message::Ack, "free was not acknowledged");
        }
        Ok(())
    }

    pub fn cancel(&mut self, slot: u32) -> Result<()> {
        for stage in self.stages.iter_mut() {
            ensure!(stage.call(&Message::Cancel { slot })? == Message::Ack, "cancel was not acknowledged");
        }
        Ok(())
    }

    pub fn ping_all(&mut self) -> Result<Vec<Duration>> {
        self.stages.iter_mut().map(|s| s.ping()).collect()
    }
}

#[derive(Default, Debug, Clone)]
pub struct ChainTimes {
    pub stage: Vec<Duration>,
    pub compute: Vec<Duration>,
    pub total: Duration,
}
