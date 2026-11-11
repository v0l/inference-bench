use specrt::cluster::{Event, Replan};
use specrt::coordinator::Coordinator;
use specrt::discovery::{Announcer, Browser};
use specrt::placement::{DeviceConfig, Discovery, LinkConfig, PlacementConfig, StageConfig};
use specrt::spec::Spec;
use specrt::wire::{Input, Message, StageHandler, StageServer};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const KEY: &[u8] = b"discovery-key";

struct Fake {
    range: Mutex<(u32, u32)>,
    last: bool,
    assigns: Mutex<Vec<(u32, u32)>>,
}

impl Fake {
    fn new(start: u32, end: u32, last: bool) -> Arc<Fake> {
        Arc::new(Fake { range: Mutex::new((start, end)), last, assigns: Mutex::new(Vec::new()) })
    }
}

impl StageHandler for Fake {
    fn ready(&self) -> Message {
        let (a, b) = *self.range.lock().unwrap();
        Message::Ready { layer_start: a, layer_end: b, hidden: 4 }
    }

    fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward { input, .. } => {
                let sum: u32 = match &input {
                    Input::Tokens(t) => t.iter().sum(),
                    Input::Hidden { data, .. } => data.len() as u32,
                };
                if self.last {
                    Message::Token { token: sum, compute_us: 1 }
                } else {
                    Message::Output { rows: 1, data: vec![1; 16], compute_us: 1 }
                }
            }
            Message::Assign { layer_start, layer_end } => {
                *self.range.lock().unwrap() = (layer_start, layer_end);
                self.assigns.lock().unwrap().push((layer_start, layer_end));
                self.ready()
            }
            Message::Free { .. } | Message::Cancel { .. } => Message::Ack,
            _ => Message::Error { message: "unexpected".into() },
        }
    }
}

fn device(name: &str, address: Option<String>) -> DeviceConfig {
    DeviceConfig {
        name: name.into(),
        memory_gib: 96.0,
        bandwidth_gbs: 1800.0,
        tflops: 1000.0,
        ordinal: None,
        address,
        reserve_gib: Some(0.0),
        link: LinkConfig::default(),
        offload: None,
    }
}

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap()
}

fn config(devices: Vec<DeviceConfig>, stages: Vec<(&str, [usize; 2])>, port: u16, discovery: Discovery) -> PlacementConfig {
    PlacementConfig {
        context: 1024,
        slots: 1,
        devices,
        stages: stages.into_iter().map(|(d, layers)| StageConfig { device: d.into(), layers }).collect(),
        discovery,
        mdns_port: port,
        ..PlacementConfig::default()
    }
}

fn until<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn multicast_works(port: u16) -> bool {
    let d = device("probe", None);
    let Ok(browser) = Browser::start(port) else { return false };
    let Ok(_announce) = Announcer::start(&d, "127.0.0.1:9".parse().unwrap(), port) else { return false };
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if let Ok(Event::Join { device, .. }) = browser.events().recv_timeout(Duration::from_millis(200)) {
            if device.name == "probe" {
                return true;
            }
        }
    }
    false
}

#[test]
fn a_stage_announcement_is_browsed_and_withdrawn_over_mdns() {
    let port = 15454;
    if !multicast_works(port) {
        eprintln!("multicast is not available here, skipping");
        return;
    }
    let browser = Browser::start(port + 1).unwrap();
    let announce = Announcer::start(&device("loop-a", None), "127.0.0.1:7777".parse().unwrap(), port + 1).unwrap();
    let joined = until("the join", || match browser.events().recv_timeout(Duration::from_millis(200)) {
        Ok(Event::Join { device, address, .. }) if device.name == "loop-a" => Some((device, address)),
        _ => None,
    });
    assert_eq!(joined.1, "127.0.0.1:7777");
    assert_eq!(joined.0.memory_gib, 96.0);
    drop(announce);
    until("the leave", || match browser.events().recv_timeout(Duration::from_millis(200)) {
        Ok(Event::Leave { name }) if name == "loop-a" => Some(()),
        _ => None,
    });
}

#[test]
fn the_coordinator_replans_on_join_and_leave_and_fails_a_lost_stage_cleanly() {
    let first = Fake::new(0, 0, false);
    let second = Fake::new(0, 0, true);
    let s1 = StageServer::start("127.0.0.1:0", KEY, first.clone()).unwrap();
    let s2 = StageServer::start("127.0.0.1:0", KEY, second.clone()).unwrap();
    let a1 = s1.addr().to_string();
    let a2 = s2.addr().to_string();
    let base = config(vec![device("a", Some(a1.clone()))], Vec::new(), 0, Discovery::Static);
    let mut c = Coordinator::new(&spec(), base, KEY.to_vec(), Duration::from_secs(5), Duration::from_secs(60)).unwrap();
    assert_eq!(c.placement().unwrap().stages.len(), 1);
    let replan = c.apply(Event::Join { device: device("b", None), address: a2.clone(), now_ms: 0 }).unwrap();
    assert_eq!(replan, Replan { moved_stages: 0, unchanged: true });
    let a_member = c.members().members().iter().find(|m| m.device.name == "a").unwrap();
    assert_eq!(a_member.address.as_deref(), Some(a1.as_str()));
    assert_eq!(c.replans(), 2);
    assert!(c.last_error().is_none());

    c.apply(Event::Leave { name: "b".into() }).unwrap();
    assert_eq!(c.members().names(), ["a"]);
    assert_eq!(c.placement().unwrap().stages.len(), 1);
    assert_eq!(c.replans(), 3);
    assert!(!c.connected());
    drop(s1);
    drop(s2);
}

#[test]
fn a_two_stage_chain_loads_assigned_layers_fails_when_a_stage_dies_and_recovers_on_rejoin() {
    let tail = Fake::new(0, 0, true);
    let head = Fake::new(0, 0, false);
    let sh = StageServer::start("127.0.0.1:0", KEY, head.clone()).unwrap();
    let mut st = Some(StageServer::start("127.0.0.1:0", KEY, tail.clone()).unwrap());
    let ah = sh.addr().to_string();
    let at = st.as_ref().unwrap().addr().to_string();
    let base = config(
        vec![device("head", Some(ah.clone())), device("tail", Some(at.clone()))],
        vec![("head", [0, 40]), ("tail", [40, 64])],
        0,
        Discovery::Static,
    );
    let mut c = Coordinator::new(&spec(), base, KEY.to_vec(), Duration::from_secs(3), Duration::from_secs(60)).unwrap();
    let prompt = [1u32, 2, 3];
    let got = c.generate(0, &prompt, 4, 2, Duration::from_secs(5)).unwrap();
    assert_eq!(got.tokens.len(), 4);
    assert_eq!(*head.assigns.lock().unwrap(), vec![(0, 40)]);
    assert_eq!(*tail.assigns.lock().unwrap(), vec![(40, 64)]);

    st.take().unwrap().shutdown();
    std::thread::sleep(Duration::from_millis(600));
    let err = c.generate(0, &prompt, 4, 2, Duration::from_millis(500)).err().unwrap();
    let text = format!("{err:#}");
    assert!(text.contains("lost") || text.contains("not ready"), "{text}");
    assert!(!c.connected());

    let again = StageServer::start(&at, KEY, tail.clone()).unwrap();
    let ok = c.generate(0, &prompt, 4, 2, Duration::from_secs(5)).unwrap();
    assert_eq!(ok.tokens, got.tokens);
    drop(again);
}

#[test]
fn a_coordinator_discovers_a_stage_over_mdns_and_drops_it_when_it_leaves() {
    let port = 15464;
    if !multicast_works(port + 10) {
        eprintln!("multicast is not available here, skipping");
        return;
    }
    let head = Fake::new(0, 0, true);
    let server = StageServer::start("127.0.0.1:0", KEY, head).unwrap();
    let addr = server.addr();
    let base = config(Vec::new(), Vec::new(), port, Discovery::Mdns);
    let mut c = Coordinator::new(&spec(), base, KEY.to_vec(), Duration::from_secs(3), Duration::from_secs(600)).unwrap();
    assert!(c.placement().is_none());
    let announce = Announcer::start(&device("mdns-a", None), addr, port).unwrap();
    until("the placement", || {
        c.poll();
        c.placement().map(|_| ())
    });
    assert_eq!(c.members().names(), ["mdns-a"]);
    assert_eq!(c.members().members()[0].address.as_deref(), Some(addr.to_string().as_str()));
    drop(announce);
    until("the leave", || {
        c.poll();
        c.members().names().is_empty().then_some(())
    });
    assert!(c.placement().is_none());
    assert!(c.last_error().is_some());
}

fn tiny_spec() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.name = "tiny-qwen3.5".into();
    spec.repo = dir.display().to_string();
    spec
}

struct Node {
    server: StageServer,
    announcer: Option<Announcer>,
}

fn node(spec: &Spec, name: &str, bind: &str, announce_port: Option<u16>) -> Node {
    let stage = Arc::new(specrt::stage::RuntimeStage::unloaded(spec, 512, 1));
    let server = StageServer::start(bind, KEY, stage).unwrap();
    let announcer = announce_port.map(|p| Announcer::start(&device(name, None), server.addr(), p).unwrap());
    Node { server, announcer }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn killing_and_restarting_a_stage_process_fails_the_request_then_serves_identical_tokens() {
    use specrt::runtime::{Layout, Runtime, Sampling};
    let spec = tiny_spec();
    let prompt: Vec<u32> = (0..40u32).map(|i| (i * 37 + 11) % 1024).collect();
    let want = {
        let mut rt = Runtime::load_layout(&spec, Layout { kv_planes: 1, ..Layout::contiguous(512, 1) }).unwrap();
        let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
        rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
        rt.generate(&prompt, 24, Sampling::GREEDY).unwrap().tokens
    };
    let port = 15480;
    let mdns = multicast_works(port + 10);
    eprintln!("discovery over {}", if mdns { "mdns" } else { "the static list" });
    let announce = mdns.then_some(port);
    let head = node(&spec, "head", "127.0.0.1:0", announce);
    let mut tail = Some(node(&spec, "tail", "127.0.0.1:0", announce));
    let tail_addr = tail.as_ref().unwrap().server.addr();
    let stages = vec![("head", [0, 1]), ("tail", [1, 3])];
    let devices = if mdns {
        Vec::new()
    } else {
        vec![device("head", Some(head.server.addr().to_string())), device("tail", Some(tail_addr.to_string()))]
    };
    let discovery = if mdns { Discovery::Mdns } else { Discovery::Static };
    let mut c = Coordinator::new(&tiny_spec(), config(devices, stages, port, discovery), KEY.to_vec(), Duration::from_secs(30), Duration::from_secs(600)).unwrap();

    let got = c.generate(0, &prompt, 24, 16, Duration::from_secs(60)).unwrap();
    assert_eq!(got.tokens, want, "first request over the discovered chain");

    let victim = tail.take().unwrap();
    let in_flight = std::thread::scope(|s| {
        let killer = s.spawn(|| {
            std::thread::sleep(Duration::from_millis(5));
            drop(victim);
        });
        let long: Vec<u32> = (0..2000u32).map(|i| i % 1024).collect();
        let r = c.generate(0, &long[..400], 200, 8, Duration::from_secs(2));
        killer.join().unwrap();
        r
    });
    let err = format!("{:#}", in_flight.err().expect("the request must fail"));
    assert!(err.contains("lost") || err.contains("not ready") || err.contains("refused") || err.contains("connect"), "{err}");
    assert!(!c.connected());

    let tail_again = node(&spec, "tail", &tail_addr.to_string(), announce);
    let again = c.generate(0, &prompt, 24, 16, Duration::from_secs(60)).unwrap();
    assert_eq!(again.tokens, want, "request after the stage came back");
    drop(tail_again);
    drop(head);
}
