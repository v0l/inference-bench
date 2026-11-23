use specrt::rdma::{LinkOptions, Transport};
use specrt::spec::Spec;
use specrt::stage::RuntimeStage;
use specrt::wire::{Chain, Input, Message, StageHandler, StageServer};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const KEY: &[u8] = b"direct-fallback";

struct Fake {
    start: u32,
    end: u32,
    last: bool,
    seen: Mutex<Vec<Message>>,
}

impl Fake {
    fn new(start: u32, end: u32, last: bool) -> Arc<Fake> {
        Arc::new(Fake { start, end, last, seen: Mutex::new(Vec::new()) })
    }

    fn direct_messages(&self) -> usize {
        self.seen.lock().unwrap().iter().filter(|m| matches!(m, Message::DirectOpen { .. } | Message::DirectConnect { .. } | Message::DirectMode { .. })).count()
    }
}

impl StageHandler for Fake {
    fn ready(&self) -> Message {
        Message::Ready { layer_start: self.start, layer_end: self.end, hidden: 4 }
    }

    fn handle(&self, message: Message) -> Message {
        self.seen.lock().unwrap().push(message.clone());
        match message {
            Message::Forward { input, .. } => {
                let sum: u32 = match &input {
                    Input::Tokens(t) => t.iter().sum(),
                    Input::Hidden { data, .. } => data.iter().map(|b| *b as u32).sum(),
                    Input::Direct { .. } => 9999,
                };
                if self.last {
                    Message::Token { token: sum + 1, compute_us: 7 }
                } else {
                    Message::Output { rows: 1, data: vec![sum as u8; 16], compute_us: 7 }
                }
            }
            Message::DirectMode { .. } => Message::Ack,
            Message::Free { .. } | Message::Cancel { .. } => Message::Ack,
            _ => Message::Error { message: "direct is not available".into() },
        }
    }
}

fn run(link: LinkOptions) -> (u32, bool, usize) {
    let a = Fake::new(0, 10, false);
    let b = Fake::new(10, 20, true);
    let servers: Vec<_> = [&a, &b].iter().map(|f| StageServer::start("127.0.0.1:0", KEY, (*f).clone()).unwrap()).collect();
    let addrs: Vec<String> = servers.iter().map(|s| s.addr().to_string()).collect();
    let mut chain = Chain::connect_assigned_with(&addrs, &[(0, 10), (10, 20)], KEY, Duration::from_secs(5), &link).unwrap();
    let (token, _) = chain.forward(0, 0, vec![1, 2, 3], false).unwrap();
    (token, chain.is_direct(), a.direct_messages() + b.direct_messages())
}

fn link(transport: Transport, gpu_direct: bool) -> LinkOptions {
    LinkOptions { transport, library: None, gpu_direct }
}

#[test]
fn direct_off_sends_no_direct_messages_and_keeps_byte_frames() {
    let (token, direct, messages) = run(link(Transport::Auto, false));
    assert_eq!((direct, messages), (false, 0));
    assert_eq!(token, 97);
}

#[test]
fn tcp_transport_ignores_gpu_direct() {
    let (token, direct, messages) = run(link(Transport::Tcp, true));
    assert_eq!((direct, messages), (false, 0));
    assert_eq!(token, 97);
}

#[test]
fn a_stage_that_refuses_the_direct_offer_falls_back_with_identical_results() {
    let (token, direct, messages) = run(link(Transport::Auto, true));
    assert!(!direct, "the chain must not claim direct hops");
    assert!(messages >= 2, "the offer was tried and the mode reset: {messages}");
    assert_eq!(token, 97);
}

fn unloaded(dir: &str) -> RuntimeStage {
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir);
    let json = std::fs::read_to_string(root.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = root.display().to_string();
    RuntimeStage::unloaded_on(&spec, 512, 1, 0)
}

fn open_error(stage: &RuntimeStage) -> String {
    match stage.handle(Message::DirectOpen { next: true, ip: "127.0.0.1".into() }) {
        Message::Error { message } => message,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_stage_without_gpu_direct_enabled_refuses_the_offer() {
    let stage = unloaded("tiny-qwen3.5");
    stage.set_link(link(Transport::Rdma, false));
    assert!(open_error(&stage).contains("not enabled"));
    stage.set_link(link(Transport::Tcp, true));
    assert!(open_error(&stage).contains("not enabled"));
    assert!(!stage.direct_active());
}

#[test]
fn a_stage_without_libibverbs_refuses_the_offer() {
    let stage = unloaded("tiny-qwen3.5");
    stage.set_link(LinkOptions { transport: Transport::Rdma, library: Some("libno-such-ibverbs.so.9".into()), gpu_direct: true });
    assert!(open_error(&stage).contains("libno-such-ibverbs"));
    assert!(!stage.direct_active());
}

#[test]
fn an_unloaded_stage_refuses_direct_input() {
    let stage = unloaded("tiny-qwen3.5");
    let reply = stage.handle(Message::Forward { slot: 0, position: 0, input: Input::Direct { rows: 1 }, decode: false });
    assert!(matches!(reply, Message::Error { .. }));
}
