use specrt::rdma::{LinkOptions, Transport, Verbs};
use specrt::wire::{Input, Message, StageClient, StageHandler, StageServer};
use std::sync::Arc;
use std::time::{Duration, Instant};

const KEY: &[u8] = b"rdma-key";

struct Echo;

impl StageHandler for Echo {
    fn ready(&self) -> Message {
        Message::Ready { layer_start: 0, layer_end: 2, hidden: 4 }
    }

    fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward { input: Input::Hidden { rows, data }, .. } => Message::Output { rows, data, compute_us: 1 },
            Message::Forward { position, .. } => Message::Token { token: position, compute_us: 1 },
            Message::Cancel { .. } => Message::Ack,
            _ => Message::Error { message: "unexpected".into() },
        }
    }
}

fn rdma_ip() -> Option<String> {
    if let Ok(ip) = std::env::var("SPECRT_RDMA_TEST_IP") {
        return Some(ip);
    }
    let out = std::fs::read_dir("/sys/class/infiniband").ok()?;
    if out.count() == 0 {
        return None;
    }
    let text = std::process::Command::new("ip").args(["-4", "-o", "addr"]).output().ok()?;
    String::from_utf8_lossy(&text.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().nth(3).map(|a| a.split('/').next().unwrap().to_string()))
        .find(|a| a.starts_with("10.100.50."))
}

fn link(transport: Transport) -> LinkOptions {
    LinkOptions { transport, library: None, gpu_direct: false }
}

fn skip(reason: &str) -> bool {
    eprintln!("SKIPPED: {reason}");
    true
}

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Held {
    _server: StageServer,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Held {
    fn drop(&mut self) {
        std::thread::sleep(Duration::from_millis(450));
    }
}

fn setup() -> Option<(String, Held)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if Verbs::load(specrt::rdma::DEFAULT_LIBRARY).is_err() {
        skip("libibverbs is not installed");
        return None;
    }
    let Some(ip) = rdma_ip() else {
        skip("no RoCE device with a routable IPv4 address (set SPECRT_RDMA_TEST_IP)");
        return None;
    };
    let server = StageServer::start_with(&format!("{ip}:0"), KEY, Arc::new(Echo), link(Transport::Rdma)).unwrap();
    let addr = format!("{ip}:{}", server.addr().port());
    Some((addr, Held { _server: server, _guard: guard }))
}

fn payload(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 + 7) as u8).collect()
}

#[test]
fn rdma_frames_round_trip_like_tcp() {
    let Some((addr, _server)) = setup() else { return };
    let mut rdma = StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &link(Transport::Rdma)).unwrap();
    let mut tcp = StageClient::connect(&addr, KEY, Duration::from_secs(5)).unwrap();
    assert!(rdma.is_rdma() && !tcp.is_rdma());
    assert_eq!(rdma.ready, tcp.ready);
    for bytes in [0usize, 4, 20 * 1024, (1 << 20) - 3, 1 << 20, (1 << 20) + 1, 5 << 20] {
        let data = payload(bytes);
        let message = Message::Forward { slot: 1, position: 3, input: Input::Hidden { rows: (bytes / 4) as u32, data: data.clone() }, decode: false };
        let a = rdma.call(&message).unwrap();
        let b = tcp.call(&message).unwrap();
        assert_eq!(a, b, "{bytes} bytes");
        assert_eq!(a, Message::Output { rows: (bytes / 4) as u32, data, compute_us: 1 });
    }
    assert!(rdma.ping().is_ok());
}

#[test]
fn auto_uses_rdma_when_available() {
    let Some((addr, _server)) = setup() else { return };
    let client = StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &link(Transport::Auto)).unwrap();
    assert!(client.is_rdma());
}

#[test]
fn wrong_key_is_refused_before_rdma() {
    let Some((addr, _server)) = setup() else { return };
    assert!(StageClient::connect_with(&addr, b"other", Duration::from_secs(2), &link(Transport::Rdma)).is_err());
}

#[test]
fn dead_peer_fails_cleanly() {
    let Some((addr, server)) = setup() else { return };
    let mut client = StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &link(Transport::Rdma)).unwrap();
    assert!(client.ping().is_ok());
    drop(server);
    std::thread::sleep(Duration::from_millis(600));
    let started = Instant::now();
    let err = client.ping().unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(5), "took {:?}", started.elapsed());
    eprintln!("dead peer: {err:#}");
}

#[test]
fn auto_falls_back_to_tcp_without_the_library() {
    let server = StageServer::start("127.0.0.1:0", KEY, Arc::new(Echo)).unwrap();
    let addr = server.addr().to_string();
    let missing = LinkOptions { transport: Transport::Auto, library: Some("libno-such-ibverbs.so.9".into()) };
    let mut client = StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &missing).unwrap();
    assert!(!client.is_rdma());
    assert!(client.ping().is_ok());
}

#[test]
fn forced_rdma_without_the_library_is_an_error() {
    let server = StageServer::start("127.0.0.1:0", KEY, Arc::new(Echo)).unwrap();
    let addr = server.addr().to_string();
    let missing = LinkOptions { transport: Transport::Rdma, library: Some("libno-such-ibverbs.so.9".into()) };
    assert!(StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &missing).is_err());
}

#[test]
fn auto_falls_back_when_the_server_is_tcp_only() {
    let Some(ip) = rdma_ip() else {
        skip("no RoCE device");
        return;
    };
    let server = StageServer::start(&format!("{ip}:0"), KEY, Arc::new(Echo)).unwrap();
    let addr = format!("{ip}:{}", server.addr().port());
    let mut client = StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &link(Transport::Auto)).unwrap();
    assert!(!client.is_rdma());
    assert!(client.ping().is_ok());
}

#[test]
#[ignore]
fn hop_latency_rdma_vs_tcp() {
    let Some((addr, _server)) = setup() else { return };
    for (name, transport) in [("tcp", Transport::Tcp), ("rdma", Transport::Rdma)] {
        let mut c = StageClient::connect_with(&addr, KEY, Duration::from_secs(5), &link(transport)).unwrap();
        for (label, bytes, iters) in [("decode 20KB", 20 * 1024, 2000usize), ("prefill 4MB", 4 << 20, 100), ("prefill 64MB", 64 << 20, 10)] {
            let message = Message::Forward { slot: 0, position: 0, input: Input::Hidden { rows: 1, data: payload(bytes) }, decode: false };
            c.call(&message).unwrap();
            let started = Instant::now();
            for _ in 0..iters {
                c.call(&message).unwrap();
            }
            let per = started.elapsed() / iters as u32;
            let gbs = 2.0 * bytes as f64 / per.as_secs_f64() / 1e9;
            eprintln!("{name} {label}: {per:?} per round trip, {gbs:.2} GB/s both ways");
        }
    }
}
