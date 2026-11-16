use crate::cluster::{ANNOUNCEMENT_VERSION, Peer, PeerGpu};
use crate::discovery::PeerAnnouncer;
use crate::hardware::{self, Machine};
use crate::pairing::{self, Offer};
use crate::stage::{MemoryGuard, RuntimeStage};
use crate::wire::{StageServer, hmac_sha256};
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct HostedStage {
    pub server: StageServer,
    pub stage: Arc<RuntimeStage>,
}

pub fn spawn_stage(model: &str, ctx: usize, slots: usize, gpu: usize, key: &[u8], bind: &str) -> Result<HostedStage> {
    spawn_stage_guarded(model, ctx, slots, gpu, key, bind, None)
}

pub fn spawn_stage_guarded(model: &str, ctx: usize, slots: usize, gpu: usize, key: &[u8], bind: &str, guard: Option<MemoryGuard>) -> Result<HostedStage> {
    let mut spec = crate::engine::resolve_spec(model).with_context(|| model.to_string())?;
    spec.mtp = None;
    let stage = Arc::new(RuntimeStage::unloaded_on(&spec, ctx, slots, gpu));
    stage.guard_memory(guard);
    let server = StageServer::start(bind, key, stage.clone())?;
    Ok(HostedStage { server, stage })
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostConfig {
    pub key_file: PathBuf,
    pub mdns_port: u16,
    pub interface: Option<String>,
    pub bind: String,
    pub app_version: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    PairStart { msg: String },
    PairFinish { confirm: String, nonce: String, sealed: String },
    Stage { model: String, ctx: usize, slots: usize, gpus: Vec<usize> },
    Release,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Default)]
pub struct Reply {
    pub ok: bool,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub msg: String,
    #[serde(default)]
    pub confirm: String,
    #[serde(default)]
    pub addrs: Vec<String>,
}

struct State {
    machine: Machine,
    shared: Vec<usize>,
    busy: bool,
    serving: Option<String>,
    offer: Option<Offer>,
    stages: Vec<HostedStage>,
    paired: bool,
    guard: Option<MemoryGuard>,
}

pub struct Host {
    config: HostConfig,
    state: Arc<Mutex<State>>,
    announcer: PeerAnnouncer,
    control: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn peer_of(config: &HostConfig, s: &State) -> Peer {
    let hidden: Vec<usize> = s.machine.gpus.iter().map(|g| g.ordinal).filter(|o| !s.shared.contains(o)).collect();
    let devices = s.machine.devices(&hidden);
    let gpus = s
        .machine
        .gpus
        .iter()
        .filter(|g| s.shared.contains(&g.ordinal))
        .zip(devices)
        .map(|(g, device)| PeerGpu { device, gpu_name: g.name.clone() })
        .collect();
    Peer { version: ANNOUNCEMENT_VERSION, host: s.machine.host.clone(), app_version: Some(config.app_version.clone()), busy: s.busy, serving: s.serving.clone(), control: None, gpus }
}

fn write_json(stream: &mut TcpStream, value: &impl Serialize) -> Result<()> {
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    Ok(())
}

fn auth_tag(key: &[u8], nonce: &str, body: &str) -> String {
    pairing::hex(&hmac_sha256(key, format!("{nonce}\n{body}").as_bytes()))
}

fn read_line(reader: &mut BufReader<TcpStream>) -> Result<String> {
    let mut line = String::new();
    std::io::Read::take(&mut *reader, 1 << 16).read_line(&mut line)?;
    ensure!(line.ends_with('\n'), "request line is incomplete or too long");
    Ok(line)
}

fn handle(config: &HostConfig, state: &Arc<Mutex<State>>, stream: TcpStream) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let local_ip = stream.local_addr()?.ip();
    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let nonce = pairing::hex(&pairing::fresh_nonce()?);
    write_json(&mut writer, &serde_json::json!({ "nonce": nonce }))?;
    let line = read_line(&mut reader)?;
    let (tag, body) = line.trim_end().split_once(' ').ok_or_else(|| anyhow!("request has no auth tag"))?;
    let request: Request = serde_json::from_str(body).context("request")?;
    let reply = match request {
        Request::PairStart { msg } => pair(config, state, &msg, &mut reader, &mut writer),
        Request::PairFinish { .. } => Err(anyhow!("pairing must start with a first message")),
        other => {
            let key = pairing::read_key(&config.key_file).ok().and_then(|k| pairing::unhex(&k).ok());
            match key {
                Some(key) if constant_eq(tag, &auth_tag(&key, &nonce, body)) => run(state, other, pairing::hex(&key).as_bytes(), local_ip),
                Some(_) => Err(anyhow!("authentication failed")),
                None => Err(anyhow!("this machine is not paired")),
            }
        }
    };
    let reply = match reply {
        Ok(r) => r,
        Err(e) => Reply { ok: false, error: format!("{e:#}"), ..Reply::default() },
    };
    write_json(&mut writer, &reply)
}

fn constant_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn pair(config: &HostConfig, state: &Arc<Mutex<State>>, msg: &str, reader: &mut BufReader<TcpStream>, writer: &mut TcpStream) -> Result<Reply> {
    let (session, msg_b, confirm_b) = {
        let mut s = state.lock().unwrap();
        let offer = s.offer.as_mut().ok_or_else(|| anyhow!("no pairing code is showing on this machine"))?;
        let (msg_b, confirm_b, session) = offer.start(&pairing::unhex(msg)?)?;
        (session, msg_b, confirm_b)
    };
    write_json(writer, &Reply { ok: true, msg: pairing::hex(&msg_b), confirm: pairing::hex(&confirm_b), ..Reply::default() })?;
    let line = read_line(reader)?;
    let Request::PairFinish { confirm, nonce, sealed } = serde_json::from_str(line.trim_end().split_once(' ').map_or("", |(_, b)| b)).context("second pairing message")? else {
        bail!("expected the second pairing message");
    };
    let key = session.open(&pairing::unhex(&confirm)?, &pairing::unhex(&nonce)?, &pairing::unhex(&sealed)?)?;
    pairing::write_key(&config.key_file, &key)?;
    let mut s = state.lock().unwrap();
    if let Some(o) = s.offer.as_mut() {
        o.consume();
    }
    s.offer = None;
    s.paired = true;
    Ok(Reply { ok: true, ..Reply::default() })
}

fn run(state: &Arc<Mutex<State>>, request: Request, key: &[u8], local_ip: std::net::IpAddr) -> Result<Reply> {
    match request {
        Request::Stage { model, ctx, slots, gpus } => {
            let mut s = state.lock().unwrap();
            s.stages.clear();
            let guard = s.guard.clone();
            let mut addrs = Vec::new();
            for gpu in gpus {
                ensure!(s.shared.contains(&gpu), "GPU {gpu} is not shared by this machine");
                let hosted = spawn_stage_guarded(&model, ctx, slots, gpu, key, &format!("{local_ip}:0"), guard.clone())?;
                addrs.push(hosted.server.addr().to_string());
                s.stages.push(hosted);
            }
            s.busy = true;
            s.serving = Some(model);
            Ok(Reply { ok: true, addrs, ..Reply::default() })
        }
        Request::Release => {
            let mut s = state.lock().unwrap();
            s.stages.clear();
            s.busy = false;
            s.serving = None;
            Ok(Reply { ok: true, ..Reply::default() })
        }
        Request::PairStart { .. } | Request::PairFinish { .. } => bail!("unreachable"),
    }
}

impl Host {
    pub fn start(config: HostConfig, machine: Machine, shared: Vec<usize>) -> Result<Host> {
        let listener = TcpListener::bind(&config.bind).with_context(|| format!("bind {}", config.bind))?;
        listener.set_nonblocking(true)?;
        let control = listener.local_addr()?;
        let paired = pairing::read_key(&config.key_file).is_ok();
        let state = Arc::new(Mutex::new(State { machine, shared, busy: false, serving: None, offer: None, stages: Vec::new(), paired, guard: None }));
        let peer = peer_of(&config, &state.lock().unwrap());
        let announcer = PeerAnnouncer::start(&peer, control, config.mdns_port, config.interface.as_deref())?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (state, config, stop) = (state.clone(), config.clone(), stop.clone());
            std::thread::Builder::new().name("host-control".into()).spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let (state, config) = (state.clone(), config.clone());
                            std::thread::spawn(move || {
                                let _ = stream.set_nonblocking(false);
                                if let Err(e) = handle(&config, &state, stream) {
                                    tracing::warn!(error = %format!("{e:#}"), "cluster control request failed");
                                }
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
                        Err(_) => break,
                    }
                }
            })?
        };
        Ok(Host { config, state, announcer, control, stop, thread: Some(thread) })
    }

    pub fn guard_memory(&self, guard: MemoryGuard) {
        self.state.lock().unwrap().guard = Some(guard);
    }

    pub fn control_addr(&self) -> SocketAddr {
        self.control
    }

    pub fn paired(&self) -> bool {
        self.state.lock().unwrap().paired
    }

    pub fn show_code(&self) -> Result<String> {
        let offer = Offer::new(&self.state.lock().unwrap().machine.host)?;
        let code = offer.code();
        self.state.lock().unwrap().offer = Some(offer);
        Ok(code)
    }

    pub fn code(&self) -> Option<(String, u64)> {
        self.state.lock().unwrap().offer.as_ref().filter(|o| o.live()).map(|o| (o.code(), o.seconds_left()))
    }

    pub fn staged(&self) -> usize {
        self.state.lock().unwrap().stages.len()
    }

    pub fn refresh(&self, machine: Machine) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        s.machine = machine;
        let peer = peer_of(&self.config, &s);
        drop(s);
        self.announcer.update(&peer, self.control)
    }

    pub fn set_serving(&self, serving: Option<String>) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        s.busy = serving.is_some() || !s.stages.is_empty();
        s.serving = serving;
        let peer = peer_of(&self.config, &s);
        drop(s);
        self.announcer.update(&peer, self.control)
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Conn {
    writer: TcpStream,
    reader: BufReader<TcpStream>,
    nonce: String,
    control: String,
}

fn connect(control: &str, timeout: Duration) -> Result<Conn> {
    let addr = std::net::ToSocketAddrs::to_socket_addrs(control).with_context(|| control.to_string())?.next().context("no address")?;
    let stream = TcpStream::connect_timeout(&addr, timeout).with_context(|| format!("connect {control}"))?;
    stream.set_read_timeout(Some(timeout))?;
    let writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let nonce = serde_json::from_str::<serde_json::Value>(&line)?["nonce"].as_str().context("no nonce")?.to_string();
    Ok(Conn { writer, reader, nonce, control: control.to_string() })
}

impl Conn {
    fn send(&mut self, tag: &str, request: &Request) -> Result<Reply> {
        let body = serde_json::to_string(request)?;
        self.writer.write_all(format!("{tag} {body}\n").as_bytes())?;
        let mut answer = String::new();
        self.reader.read_line(&mut answer)?;
        let reply: Reply = serde_json::from_str(&answer).context("reply")?;
        if !reply.ok {
            bail!("{}: {}", self.control, reply.error);
        }
        Ok(reply)
    }
}

pub fn pair_with(control: &str, code: &str, key: &str) -> Result<()> {
    let mut conn = connect(control, Duration::from_secs(10))?;
    let (client, msg_a) = pairing::ClientSession::start(code);
    let first = conn.send("-", &Request::PairStart { msg: pairing::hex(&msg_a) })?;
    let sealed = client.finish(&pairing::unhex(&first.msg)?, &pairing::unhex(&first.confirm)?, key).map_err(|e| if e.to_string().contains("did not prove") { anyhow!("wrong pairing code") } else { e })?;
    conn.send("-", &Request::PairFinish { confirm: pairing::hex(&sealed.confirm), nonce: pairing::hex(&sealed.nonce), sealed: pairing::hex(&sealed.ciphertext) })?;
    Ok(())
}

fn authed(control: &str, key: &str, request: &Request, timeout: Duration) -> Result<Reply> {
    let raw = pairing::unhex(key)?;
    let mut conn = connect(control, timeout)?;
    let body = serde_json::to_string(request)?;
    let tag = auth_tag(&raw, &conn.nonce, &body);
    conn.send(&tag, request)
}

pub fn request_stages(control: &str, key: &str, model: &str, ctx: usize, slots: usize, gpus: &[usize]) -> Result<Vec<String>> {
    Ok(authed(control, key, &Request::Stage { model: model.to_string(), ctx, slots, gpus: gpus.to_vec() }, Duration::from_secs(120))?.addrs)
}

pub fn release_stages(control: &str, key: &str) -> Result<()> {
    authed(control, key, &Request::Release, Duration::from_secs(10)).map(|_| ())
}

pub fn local_machine_summary(m: &Machine) -> String {
    hardware::summarize(&m.host, &m.gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>())
}
