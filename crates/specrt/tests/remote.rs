use specrt::coordinator::Coordinator;
use specrt::placement::{DeviceConfig, Discovery, LinkConfig, PlacementConfig};
use specrt::remote::ChainExecutor;
use specrt::runtime::Sampling;
use specrt::sched::{Config, Event, Executor, Request, Scheduler};
use specrt::spec::Spec;
use specrt::wire::{Message, StageHandler, StageServer};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const KEY: &[u8] = b"remote-key";

struct Echo {
    range: Mutex<(u32, u32)>,
    seen: Mutex<Vec<(u32, u32, bool)>>,
}

impl StageHandler for Echo {
    fn ready(&self) -> Message {
        let (a, b) = *self.range.lock().unwrap();
        Message::Ready { layer_start: a, layer_end: b, hidden: 4 }
    }

    fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward { slot, position, input, decode } => {
                self.seen.lock().unwrap().push((slot, position, decode));
                let n = match input {
                    specrt::wire::Input::Tokens(t) => t.len() as u32,
                    _ => 0,
                };
                Message::Token { token: position + n, compute_us: 1 }
            }
            Message::DecodeBatch { positions, .. } => {
                self.seen.lock().unwrap().push((positions[0], positions[0], true));
                Message::Tokens { tokens: positions.iter().map(|p| p + 1).collect(), compute_us: 1 }
            }
            Message::Assign { layer_start, layer_end } => {
                *self.range.lock().unwrap() = (layer_start, layer_end);
                self.ready()
            }
            Message::Free { .. } | Message::Cancel { .. } => Message::Ack,
            _ => Message::Error { message: "unexpected".into() },
        }
    }
}

fn device(name: &str, address: String) -> DeviceConfig {
    DeviceConfig {
        name: name.into(),
        memory_gib: 96.0,
        bandwidth_gbs: 1800.0,
        tflops: 1000.0,
        ordinal: None,
        address: Some(address),
        reserve_gib: Some(0.0),
        link: LinkConfig::default(),
        offload: None,
    }
}

fn executor(addr: &str) -> ChainExecutor {
    let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    let base = PlacementConfig { context: 1024, slots: 2, devices: vec![device("a", addr.into())], discovery: Discovery::Static, ..PlacementConfig::default() };
    let c = Coordinator::new(&spec, base, KEY.to_vec(), Duration::from_secs(3), Duration::from_secs(60)).unwrap();
    ChainExecutor::new(c, 256, 2, 64)
}

fn request(n: usize) -> Request {
    Request { prompt: (1..=n as u32).collect(), max_tokens: 4, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: vec![] }
}

fn drain(s: &mut Scheduler<ChainExecutor>) -> anyhow::Result<Vec<Event>> {
    let mut events = Vec::new();
    while !s.idle() {
        s.step(&mut events)?;
    }
    Ok(events)
}

#[test]
fn the_scheduler_serves_over_a_chain_and_fails_in_flight_when_the_stage_dies() {
    let stage = Arc::new(Echo { range: Mutex::new((0, 0)), seen: Mutex::new(Vec::new()) });
    let server = StageServer::start("127.0.0.1:0", KEY, stage.clone()).unwrap();
    let addr = server.addr().to_string();
    let mut exec = executor(&addr);
    exec.connect().unwrap();
    assert_eq!(exec.kv_blocks(), 2 * 16);
    let mut sched = Scheduler::new(exec, Config::default());
    sched.submit(request(5)).unwrap();
    let events = drain(&mut sched).unwrap();
    let tokens = events.iter().filter(|e| matches!(e, Event::Token { .. })).count();
    assert_eq!(tokens, 4);
    assert!(stage.seen.lock().unwrap().iter().any(|(_, pos, decode)| *pos == 0 && !*decode));
    assert!(sched.exec.refresh().is_none());

    sched.submit(request(5)).unwrap();
    server.shutdown();
    std::thread::sleep(Duration::from_millis(600));
    let err = drain(&mut sched).unwrap_err();
    assert!(format!("{err:#}").contains("stage"), "{err:#}");
    let reason = sched.exec.refresh().expect("the loss is reported");
    assert!(reason.contains("lost") || !reason.is_empty());
    assert!(!sched.exec.connected());

    let again = StageServer::start(&addr, KEY, stage.clone()).unwrap();
    let mut ids = Vec::new();
    let mut events = Vec::new();
    for _ in 0..2 {
        sched.cancel(1, &mut events);
        sched.cancel(2, &mut events);
        ids.push(());
    }
    sched.submit(request(5)).unwrap();
    let events = drain(&mut sched).unwrap();
    assert_eq!(events.iter().filter(|e| matches!(e, Event::Token { .. })).count(), 4);
    drop(again);
}

fn tokens_of(events: &[Event]) -> Vec<u32> {
    events.iter().filter_map(|e| if let Event::Token { token, .. } = e { Some(*token) } else { None }).collect()
}

#[test]
fn out_of_order_scheduler_blocks_decode_the_same_over_the_chain() {
    let stage = Arc::new(Echo { range: Mutex::new((0, 0)), seen: Mutex::new(Vec::new()) });
    let server = StageServer::start("127.0.0.1:0", KEY, stage.clone()).unwrap();
    let mut exec = executor(&server.addr().to_string());
    exec.connect().unwrap();
    let mut sched = Scheduler::new(exec, Config::default());
    sched.submit(request(40)).unwrap();
    let first = tokens_of(&drain(&mut sched).unwrap());
    assert_eq!(first.len(), 4);

    sched.submit(request(40)).unwrap();
    let mut events = Vec::new();
    sched.step(&mut events).unwrap();
    let table: Vec<u32> = (0..2).map(|s| sched.exec.table(s).to_vec()).max_by_key(|t| t.len()).unwrap();
    assert!(table.len() >= 3);
    assert!(table.windows(2).any(|w| w[0] > w[1]), "blocks came back in order: {table:?}");
    while !sched.idle() {
        sched.step(&mut events).unwrap();
    }
    assert_eq!(tokens_of(&events), first);
}
