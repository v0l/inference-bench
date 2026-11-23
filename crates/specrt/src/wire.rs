use crate::rdma::{ConnInfo, Endpoint, LinkOptions, PeerClosed, RdmaChannel, Transport, Verbs};
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const PROTOCOL_VERSION: u32 = 2;
pub const MAX_FRAME_BYTES: usize = 256 << 20;
const NONCE_BYTES: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Tokens(Vec<u32>),
    Hidden { rows: u32, data: Vec<u8> },
    Direct { rows: u32 },
}

pub enum HiddenView<'a> {
    Host { rows: u32, data: &'a [u8] },
    Device { rows: u32 },
}

impl Input {
    pub fn hidden_view(&self) -> Option<HiddenView<'_>> {
        match self {
            Input::Tokens(_) => None,
            Input::Hidden { rows, data } => Some(HiddenView::Host { rows: *rows, data }),
            Input::Direct { rows } => Some(HiddenView::Device { rows: *rows }),
        }
    }
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
    DecodeBatch { slots: Vec<u32>, positions: Vec<u32>, input: Input },
    Tokens { tokens: Vec<u32>, compute_us: u32 },
    Rdma { qpn: u32, psn: u32, gid: [u8; 16], addr: u64, rkey: u32 },
    DirectOpen { next: bool, ip: String },
    DirectInfo { qpn: u32, psn: u32, gid: [u8; 16], addr: u64, rkey: u32, capacity: u64 },
    DirectConnect { next: bool, qpn: u32, psn: u32, gid: [u8; 16], addr: u64, rkey: u32, capacity: u64 },
    DirectMode { enabled: bool },
    Pushed { rows: u32, compute_us: u32 },
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_conn(out: &mut Vec<u8>, qpn: u32, psn: u32, gid: &[u8; 16], addr: u64, rkey: u32, capacity: u64) {
    put_u32(out, qpn);
    put_u32(out, psn);
    out.extend_from_slice(gid);
    out.extend_from_slice(&addr.to_le_bytes());
    put_u32(out, rkey);
    out.extend_from_slice(&capacity.to_le_bytes());
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

    fn conn(&mut self) -> Result<(u32, u32, [u8; 16], u64, u32, u64)> {
        let qpn = self.u32()?;
        let psn = self.u32()?;
        ensure!(self.buf.len() >= 36, "truncated frame");
        let (g, rest) = self.buf.split_at(16);
        let gid: [u8; 16] = g.try_into().unwrap();
        let (a, rest) = rest.split_at(8);
        let addr = u64::from_le_bytes(a.try_into().unwrap());
        self.buf = rest;
        let rkey = self.u32()?;
        let (c, rest) = self.buf.split_at(8);
        let capacity = u64::from_le_bytes(c.try_into().unwrap());
        self.buf = rest;
        Ok((qpn, psn, gid, addr, rkey, capacity))
    }

    fn u32_list(&mut self) -> Result<Vec<u32>> {
        let n = self.u32()? as usize;
        ensure!(n <= self.buf.len() / 4, "token count exceeds frame");
        (0..n).map(|_| self.u32()).collect()
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
                    Input::Direct { rows } => {
                        out.push(2);
                        put_u32(&mut out, *rows);
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
            Message::DecodeBatch { slots, positions, input } => {
                out.push(11);
                put_u32(&mut out, slots.len() as u32);
                for v in slots.iter().chain(positions) {
                    put_u32(&mut out, *v);
                }
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
                    Input::Direct { rows } => {
                        out.push(2);
                        put_u32(&mut out, *rows);
                    }
                }
            }
            Message::Tokens { tokens, compute_us } => {
                out.push(12);
                put_u32(&mut out, *compute_us);
                put_u32(&mut out, tokens.len() as u32);
                for t in tokens {
                    put_u32(&mut out, *t);
                }
            }
            Message::Rdma { qpn, psn, gid, addr, rkey } => {
                out.push(13);
                put_u32(&mut out, *qpn);
                put_u32(&mut out, *psn);
                out.extend_from_slice(gid);
                out.extend_from_slice(&addr.to_le_bytes());
                put_u32(&mut out, *rkey);
            }
            Message::DirectOpen { next, ip } => {
                out.push(14);
                out.push(*next as u8);
                put_bytes(&mut out, ip.as_bytes());
            }
            Message::DirectInfo { qpn, psn, gid, addr, rkey, capacity } => {
                out.push(15);
                put_conn(&mut out, *qpn, *psn, gid, *addr, *rkey, *capacity);
            }
            Message::DirectConnect { next, qpn, psn, gid, addr, rkey, capacity } => {
                out.push(16);
                out.push(*next as u8);
                put_conn(&mut out, *qpn, *psn, gid, *addr, *rkey, *capacity);
            }
            Message::DirectMode { enabled } => {
                out.push(17);
                out.push(*enabled as u8);
            }
            Message::Pushed { rows, compute_us } => {
                out.push(18);
                put_u32(&mut out, *rows);
                put_u32(&mut out, *compute_us);
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
                    2 => Input::Direct { rows: c.u32()? },
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
            11 => {
                let n = c.u32()? as usize;
                ensure!(n <= c.buf.len() / 8, "row count exceeds frame");
                let mut slots = Vec::with_capacity(n);
                let mut positions = Vec::with_capacity(n);
                for _ in 0..n {
                    slots.push(c.u32()?);
                }
                for _ in 0..n {
                    positions.push(c.u32()?);
                }
                let input = match c.u8()? {
                    0 => Input::Tokens(c.u32_list()?),
                    1 => {
                        let rows = c.u32()?;
                        Input::Hidden { rows, data: c.bytes()? }
                    }
                    2 => Input::Direct { rows: c.u32()? },
                    other => bail!("unknown input kind {other}"),
                };
                Message::DecodeBatch { slots, positions, input }
            }
            12 => {
                let compute_us = c.u32()?;
                Message::Tokens { tokens: c.u32_list()?, compute_us }
            }
            13 => {
                let qpn = c.u32()?;
                let psn = c.u32()?;
                ensure!(c.buf.len() >= 28, "truncated frame");
                let (g, rest) = c.buf.split_at(16);
                let gid: [u8; 16] = g.try_into().unwrap();
                let (a, rest) = rest.split_at(8);
                let addr = u64::from_le_bytes(a.try_into().unwrap());
                c.buf = rest;
                Message::Rdma { qpn, psn, gid, addr, rkey: c.u32()? }
            }
            14 => Message::DirectOpen { next: c.u8()? != 0, ip: String::from_utf8(c.bytes()?).context("ip text")? },
            15 => {
                let (qpn, psn, gid, addr, rkey, capacity) = c.conn()?;
                Message::DirectInfo { qpn, psn, gid, addr, rkey, capacity }
            }
            16 => {
                let next = c.u8()? != 0;
                let (qpn, psn, gid, addr, rkey, capacity) = c.conn()?;
                Message::DirectConnect { next, qpn, psn, gid, addr, rkey, capacity }
            }
            17 => Message::DirectMode { enabled: c.u8()? != 0 },
            18 => Message::Pushed { rows: c.u32()?, compute_us: c.u32()? },
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
    nonce.copy_from_slice(&crate::pairing::random(NONCE_BYTES)?);
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
        StageServer::start_with(bind, key, handler, LinkOptions::default())
    }

    pub fn start_with(bind: &str, key: &[u8], handler: Arc<dyn StageHandler>, link: LinkOptions) -> Result<StageServer> {
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
                        let link = link.clone();
                        std::thread::spawn(move || {
                            if let Err(e) = serve_connection(stream, &key, handler, &flag, &link) {
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

fn offer_rdma(stream: &TcpStream, link: &LinkOptions, remote: ConnInfo) -> Result<(RdmaChannel, Message)> {
    ensure!(link.transport != Transport::Tcp, "rdma is not enabled on this stage");
    let verbs = Verbs::load(link.library())?;
    let mut ep = Endpoint::open(verbs, stream.local_addr()?.ip())?;
    let info = ep.info();
    ep.connect(&remote)?;
    let reply = Message::Rdma { qpn: info.qpn, psn: info.psn, gid: info.gid, addr: info.addr, rkey: info.rkey };
    Ok((RdmaChannel::new(ep, remote, stream.try_clone()?, MAX_FRAME_BYTES), reply))
}

fn serve_rdma(mut channel: RdmaChannel, handler: &Arc<dyn StageHandler>, stop: &AtomicBool) -> Result<()> {
    while !stop.load(Ordering::Relaxed) {
        let body = match channel.recv(Duration::from_millis(200)) {
            Ok(Some(body)) => body,
            Ok(None) => continue,
            Err(e) if e.downcast_ref::<PeerClosed>().is_some() => return Ok(()),
            Err(e) => return Err(e),
        };
        let reply = match Message::decode(&body) {
            Ok(message) => handler.handle(message),
            Err(e) => Message::Error { message: format!("{e:#}") },
        };
        channel.send(&reply.encode(), Duration::from_secs(30))?;
    }
    Ok(())
}

fn serve_connection(mut stream: TcpStream, key: &[u8], handler: Arc<dyn StageHandler>, stop: &AtomicBool, link: &LinkOptions) -> Result<()> {
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
        let decoded = Message::decode(&body);
        if let Ok(Message::Rdma { qpn, psn, gid, addr, rkey }) = decoded {
            match offer_rdma(&stream, link, ConnInfo { qpn, psn, gid, addr, rkey }) {
                Ok((channel, reply)) => {
                    write_frame(&mut stream, &reply)?;
                    stream.set_nonblocking(true)?;
                    return serve_rdma(channel, &handler, stop);
                }
                Err(e) => {
                    write_frame(&mut stream, &Message::Error { message: format!("rdma unavailable: {e:#}") })?;
                    continue;
                }
            }
        }
        let reply = match decoded {
            Ok(message) => handler.handle(message),
            Err(e) => Message::Error { message: format!("{e:#}") },
        };
        write_frame(&mut stream, &reply)?;
    }
    Ok(())
}

pub struct StageClient {
    stream: TcpStream,
    rdma: Option<RdmaChannel>,
    timeout: Duration,
    pub ready: Message,
    pub addr: String,
}

impl StageClient {
    pub fn connect(addr: &str, key: &[u8], timeout: Duration) -> Result<StageClient> {
        StageClient::connect_with(addr, key, timeout, &LinkOptions::default())
    }

    pub fn connect_with(addr: &str, key: &[u8], timeout: Duration, link: &LinkOptions) -> Result<StageClient> {
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
        let mut client = StageClient { stream, rdma: None, timeout, ready, addr: addr.to_string() };
        if link.transport != Transport::Tcp {
            match client.upgrade_rdma(link) {
                Ok(()) => tracing::info!(%addr, device = client.rdma.as_ref().map(|c| c.device()).unwrap_or(""), "stage link is rdma"),
                Err(e) if link.transport == Transport::Rdma => return Err(e.context(format!("rdma to stage {addr}"))),
                Err(e) => tracing::info!(%addr, error = %format!("{e:#}"), "rdma unavailable, using tcp"),
            }
        }
        Ok(client)
    }

    pub fn is_rdma(&self) -> bool {
        self.rdma.is_some()
    }

    pub fn peer_ip(&self) -> Result<String> {
        Ok(self.stream.peer_addr()?.ip().to_string())
    }

    fn upgrade_rdma(&mut self, link: &LinkOptions) -> Result<()> {
        let verbs = Verbs::load(link.library())?;
        let mut ep = Endpoint::open(verbs, self.stream.local_addr()?.ip())?;
        let info = ep.info();
        write_frame(&mut self.stream, &Message::Rdma { qpn: info.qpn, psn: info.psn, gid: info.gid, addr: info.addr, rkey: info.rkey })?;
        match read_frame(&mut self.stream)? {
            Message::Rdma { qpn, psn, gid, addr, rkey } => {
                let remote = ConnInfo { qpn, psn, gid, addr, rkey };
                ep.connect(&remote)?;
                self.rdma = Some(RdmaChannel::new(ep, remote, self.stream.try_clone()?, MAX_FRAME_BYTES));
                self.stream.set_nonblocking(true)?;
                Ok(())
            }
            Message::Error { message } => bail!("{message}"),
            other => bail!("stage answered the rdma offer with {other:?}"),
        }
    }

    pub fn hidden(&self) -> u32 {
        match self.ready {
            Message::Ready { hidden, .. } => hidden,
            _ => 0,
        }
    }

    pub fn call(&mut self, message: &Message) -> Result<Message> {
        if let Some(channel) = self.rdma.as_mut() {
            let reply = channel
                .send(&message.encode(), self.timeout)
                .and_then(|()| channel.recv(self.timeout)?.context("timed out waiting for the stage reply"))
                .with_context(|| format!("stage {}", self.addr))?;
            return match Message::decode(&reply)? {
                Message::Error { message } => bail!("stage {} error: {message}", self.addr),
                reply => Ok(reply),
            };
        }
        write_frame(&mut self.stream, message).with_context(|| format!("stage {}", self.addr))?;
        match read_frame(&mut self.stream).with_context(|| format!("stage {}", self.addr))? {
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
    direct: bool,
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
        Chain::connect_assigned_with(addrs, ranges, key, timeout, &LinkOptions::default())
    }

    pub fn connect_assigned_with(addrs: &[String], ranges: &[(usize, usize)], key: &[u8], timeout: Duration, link: &LinkOptions) -> Result<Chain> {
        ensure!(!addrs.is_empty() && addrs.len() == ranges.len(), "a chain needs one layer range per stage");
        let mut stages: Vec<StageClient> = Vec::new();
        for (addr, &(start, end)) in addrs.iter().zip(ranges).rev() {
            let mut client = StageClient::connect_with(addr, key, timeout, link).with_context(|| format!("stage {addr} is not ready, downstream first"))?;
            if client.ready != (Message::Ready { layer_start: start as u32, layer_end: end as u32, hidden: client.hidden() }) {
                client.assign(start as u32, end as u32).with_context(|| format!("stage {addr} could not load layers {start}..{end}"))?;
            }
            stages.push(client);
        }
        stages.reverse();
        let mut chain = Chain::assemble(stages)?;
        chain.try_direct(link);
        Ok(chain)
    }

    fn assemble(stages: Vec<StageClient>) -> Result<Chain> {
        let mut expected = 0;
        let mut hidden = None;
        for (i, stage) in stages.iter().enumerate() {
            let Message::Ready { layer_start, layer_end, hidden: h } = stage.ready else { unreachable!() };
            ensure!(layer_start == expected, "stage {i} starts at layer {layer_start}, expected {expected}");
            ensure!(layer_end > layer_start, "stage {i} serves no layers");
            ensure!(hidden.is_none_or(|x| x == h), "stage {i} hidden size {h} differs from the previous stage");
            hidden = Some(h);
            expected = layer_end;
        }
        Ok(Chain { stages, direct: false })
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    pub fn set_direct(&mut self, on: bool) -> Result<()> {
        for stage in self.stages.iter_mut() {
            ensure!(stage.call(&Message::DirectMode { enabled: on })? == Message::Ack, "direct mode was not acknowledged");
        }
        self.direct = on;
        Ok(())
    }

    fn open_direct(&mut self) -> Result<()> {
        ensure!(self.stages.len() >= 2, "a single stage has no hop to open");
        for i in 0..self.stages.len() - 1 {
            let (left, right) = self.stages.split_at_mut(i + 1);
            let (a, b) = (&mut left[i], &mut right[0]);
            let (ip_a, ip_b) = (a.peer_ip()?, b.peer_ip()?);
            let Message::DirectInfo { qpn: aq, psn: ap, gid: ag, addr: aa, rkey: ar, capacity: ac } = a.call(&Message::DirectOpen { next: true, ip: ip_a })? else { bail!("stage {} answered a direct offer wrongly", a.addr) };
            let Message::DirectInfo { qpn: bq, psn: bp, gid: bg, addr: ba, rkey: br, capacity: bc } = b.call(&Message::DirectOpen { next: false, ip: ip_b })? else { bail!("stage {} answered a direct offer wrongly", b.addr) };
            ensure!(a.call(&Message::DirectConnect { next: true, qpn: bq, psn: bp, gid: bg, addr: ba, rkey: br, capacity: bc })? == Message::Ack, "direct connect refused");
            ensure!(b.call(&Message::DirectConnect { next: false, qpn: aq, psn: ap, gid: ag, addr: aa, rkey: ar, capacity: ac })? == Message::Ack, "direct connect refused");
        }
        self.set_direct(true)
    }

    pub fn try_direct(&mut self, link: &LinkOptions) {
        if !link.gpu_direct || link.transport == Transport::Tcp || self.stages.len() < 2 {
            return;
        }
        match self.open_direct() {
            Ok(()) => tracing::info!(stages = self.stages.len(), "stage hops are gpu-direct"),
            Err(e) => {
                tracing::info!(error = %format!("{e:#}"), "gpu-direct unavailable, hidden states use byte frames");
                let _ = self.set_direct(false);
            }
        }
    }

    pub fn is_rdma(&self) -> Vec<bool> {
        self.stages.iter().map(|s| s.is_rdma()).collect()
    }

    pub fn layers(&self) -> u32 {
        match self.stages.last().map(|s| &s.ready) {
            Some(Message::Ready { layer_end, .. }) => *layer_end,
            _ => 0,
        }
    }

    pub fn ranges(&self) -> Vec<(u32, u32)> {
        self.stages
            .iter()
            .map(|s| match s.ready {
                Message::Ready { layer_start, layer_end, .. } => (layer_start, layer_end),
                _ => (0, 0),
            })
            .collect()
    }

    pub fn hidden(&self) -> u32 {
        self.stages.first().map_or(0, |s| s.hidden())
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
                Message::Pushed { rows, compute_us } if i < last => {
                    times.compute.push(Duration::from_micros(compute_us as u64));
                    input = Input::Direct { rows };
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

    pub fn forward_batch(&mut self, slots: &[u32], positions: &[u32], tokens: &[u32]) -> Result<(Vec<u32>, ChainTimes)> {
        let mut times = ChainTimes::default();
        let last = self.stages.len() - 1;
        let mut input = Input::Tokens(tokens.to_vec());
        for (i, stage) in self.stages.iter_mut().enumerate() {
            let started = Instant::now();
            let message = Message::DecodeBatch { slots: slots.to_vec(), positions: positions.to_vec(), input };
            let reply = stage.call(&message).with_context(|| format!("stage {i} at {} was lost", stage.addr))?;
            times.stage.push(started.elapsed());
            match reply {
                Message::Output { rows, data, compute_us } if i < last => {
                    times.compute.push(Duration::from_micros(compute_us as u64));
                    input = Input::Hidden { rows, data };
                }
                Message::Pushed { rows, compute_us } if i < last => {
                    times.compute.push(Duration::from_micros(compute_us as u64));
                    input = Input::Direct { rows };
                }
                Message::Tokens { tokens, compute_us } if i == last => {
                    ensure!(tokens.len() == slots.len(), "stage {i} returned {} tokens for {} rows", tokens.len(), slots.len());
                    times.compute.push(Duration::from_micros(compute_us as u64));
                    times.total = times.stage.iter().sum();
                    return Ok((tokens, times));
                }
                other => bail!("stage {i} replied {other:?}"),
            }
        }
        unreachable!()
    }

    pub fn forward_pipelined(&mut self, slot: u32, position: u32, tokens: &[u32], micro: usize) -> Result<(u32, Vec<ChainTimes>)> {
        ensure!(micro >= 1 && !tokens.is_empty(), "pipelined prefill needs tokens and a micro-batch size");
        let was_direct = self.direct;
        if was_direct {
            self.set_direct(false)?;
        }
        let result = self.pipelined_inner(slot, position, tokens, micro);
        if was_direct {
            self.set_direct(true)?;
        }
        result
    }

    fn pipelined_inner(&mut self, slot: u32, position: u32, tokens: &[u32], micro: usize) -> Result<(u32, Vec<ChainTimes>)> {
        let last = self.stages.len() - 1;
        let pieces: Vec<(u32, Vec<u32>)> = tokens
            .chunks(micro)
            .enumerate()
            .map(|(i, c)| (position + (i * micro) as u32, c.to_vec()))
            .collect();
        let count = pieces.len();
        let (first_tx, first_rx) = std::sync::mpsc::channel::<(usize, u32, Input)>();
        for (i, (at, piece)) in pieces.into_iter().enumerate() {
            let _ = first_tx.send((i, at, Input::Tokens(piece)));
        }
        drop(first_tx);
        let mut rx = Some(first_rx);
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut final_rx = None;
            for (i, stage) in self.stages.iter_mut().enumerate() {
                let input_rx = rx.take().expect("a receiver per stage");
                let (tx, next_rx) = std::sync::mpsc::channel::<(usize, u32, Input)>();
                if i < last {
                    rx = Some(next_rx);
                } else {
                    final_rx = Some(next_rx);
                }
                handles.push(scope.spawn(move || -> Result<Vec<(usize, Duration, Duration)>> {
                    let mut seen = Vec::new();
                    for (index, at, input) in input_rx {
                        let started = Instant::now();
                        let reply = stage.call(&Message::Forward { slot, position: at, input, decode: false }).with_context(|| format!("stage {i} at {} was lost", stage.addr))?;
                        let round = started.elapsed();
                        match reply {
                            Message::Output { rows, data, compute_us } if i < last => {
                                seen.push((index, round, Duration::from_micros(compute_us as u64)));
                                if tx.send((index, at, Input::Hidden { rows, data })).is_err() {
                                    break;
                                }
                            }
                            Message::Token { token, compute_us } if i == last => {
                                seen.push((index, round, Duration::from_micros(compute_us as u64)));
                                let _ = tx.send((index, at, Input::Tokens(vec![token])));
                            }
                            other => bail!("stage {i} replied {other:?}"),
                        }
                    }
                    Ok(seen)
                }));
            }
            let final_rx = final_rx.expect("a last stage");
            let mut token = None;
            for (index, _, input) in final_rx {
                if index + 1 == count
                    && let Input::Tokens(t) = input {
                        token = t.first().copied();
                    }
            }
            let mut per_stage = Vec::new();
            for h in handles {
                per_stage.push(h.join().map_err(|_| anyhow::anyhow!("a stage thread panicked"))?);
            }
            Ok::<_, anyhow::Error>((token, per_stage))
        });
        let (token, per_stage) = results?;
        let per_stage = per_stage.into_iter().collect::<Result<Vec<_>>>()?;
        let token = token.context("the last micro-batch produced no token")?;
        let mut times = vec![ChainTimes::default(); count];
        for stage in &per_stage {
            for &(index, round, compute) in stage {
                times[index].stage.push(round);
                times[index].compute.push(compute);
            }
        }
        Ok((token, times))
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
