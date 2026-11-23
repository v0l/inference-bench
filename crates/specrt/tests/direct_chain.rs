use specrt::direct::{DirectLink, DirectRegion};
use specrt::rdma::{LinkOptions, Transport, Verbs};
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::spec::Spec;
use specrt::stage::{RuntimeStage, generate};
use specrt::wire::{Chain, Input, Message, StageClient, StageHandler, StageServer};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const KEY: &[u8] = b"direct-chain";
const CTX: usize = 512;

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn gpu_allowed() -> bool {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    let first = visible.split(',').next().unwrap_or("").trim().to_string();
    if first.is_empty() || first == "0" {
        eprintln!("SKIPPED: CUDA_VISIBLE_DEVICES={visible:?} would use GPU0. Run under `~/.agents/bin/gpu-run 1 --`.");
        return false;
    }
    true
}

fn rdma_ip() -> Option<String> {
    if let Ok(ip) = std::env::var("SPECRT_RDMA_TEST_IP") {
        return Some(ip);
    }
    let text = std::process::Command::new("ip").args(["-4", "-o", "addr"]).output().ok()?;
    String::from_utf8_lossy(&text.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().nth(3).map(|a| a.split('/').next().unwrap().to_string()))
        .find(|a| a.starts_with("10.100.50."))
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 1024).collect()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

struct Cluster {
    _servers: Vec<StageServer>,
    stages: Vec<Arc<RuntimeStage>>,
    chain: Chain,
}

fn cluster(spec: &Spec, ip: &str, gpu_direct: bool) -> Cluster {
    let link = LinkOptions { transport: Transport::Rdma, library: None, gpu_direct };
    let bounds = [0, 1, spec.layers.count];
    let (mut servers, mut stages, mut addrs) = (Vec::new(), Vec::new(), Vec::new());
    for pair in bounds.windows(2) {
        let stage = Arc::new(RuntimeStage::load(spec, [pair[0], pair[1]], CTX, 1, 0).unwrap());
        stage.set_link(link.clone());
        let server = StageServer::start_with(&format!("{ip}:0"), KEY, stage.clone(), link.clone()).unwrap();
        addrs.push(format!("{ip}:{}", server.addr().port()));
        servers.push(server);
        stages.push(stage);
    }
    let chain = Chain::connect_assigned_with(&addrs, &[(0, 1), (1, spec.layers.count)], KEY, Duration::from_secs(60), &link).unwrap();
    Cluster { _servers: servers, stages, chain }
}

struct Sink;

impl StageHandler for Sink {
    fn ready(&self) -> Message {
        Message::Ready { layer_start: 0, layer_end: 2, hidden: 4 }
    }

    fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward { input: Input::Hidden { .. }, .. } => Message::Ack,
            _ => Message::Error { message: "unexpected".into() },
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU and a RoCE NIC, run under gpu-run 1"]
fn a_served_two_stage_chain_matches_one_stage_with_direct_hops_on_and_off() {
    if !gpu_allowed() {
        return;
    }
    if Verbs::load(specrt::rdma::DEFAULT_LIBRARY).is_err() {
        eprintln!("SKIPPED: no libibverbs");
        return;
    }
    let Some(ip) = rdma_ip() else {
        eprintln!("SKIPPED: no RoCE device with a 10.100.50.x address (set SPECRT_RDMA_TEST_IP)");
        return;
    };
    let spec = spec_for("tiny-qwen3.5");
    let p = prompt(300);
    let want = {
        let mut rt = Runtime::load_layout(&spec, Layout { kv_planes: 1, ..Layout::contiguous(CTX, 1) }).unwrap();
        let pages = CTX.div_ceil(specrt::codegen::PAGE) as u32;
        rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
        rt.generate(&p, 48, Sampling::GREEDY).unwrap().tokens
    };

    let mut ring = cluster(&spec, &ip, false);
    assert!(!ring.chain.is_direct());
    assert!(ring.stages.iter().all(|s| !s.direct_active()));
    let ring_run = generate(&mut ring.chain, 0, &p, 48, 128).unwrap();
    assert_eq!(ring_run.tokens, want, "byte frames over the rdma ring");
    drop(ring);

    let mut direct = cluster(&spec, &ip, true);
    assert!(direct.chain.is_direct(), "both ends registered, the chain must take the direct path");
    assert!(direct.stages.iter().all(|s| s.direct_active()));
    let direct_run = generate(&mut direct.chain, 0, &p, 48, 128).unwrap();
    assert_eq!(direct_run.tokens, want, "gpu-direct hops");

    direct.chain.set_direct(false).unwrap();
    let back = generate(&mut direct.chain, 0, &p, 48, 100).unwrap();
    assert_eq!(back.tokens, want, "direct switched off on the same chain");
    direct.chain.set_direct(true).unwrap();
    let again = generate(&mut direct.chain, 0, &p, 48, 100).unwrap();
    assert_eq!(again.tokens, want, "direct switched back on");

    let step = |g: &specrt::stage::Generated| median(g.decode_steps.iter().map(|t| t.stage[0]).collect());
    let prefill_step = |g: &specrt::stage::Generated| median(g.prefill_steps.iter().map(|t| t.stage[0] + t.stage[1]).collect());
    eprintln!("CHAIN decode step, first stage round trip (1 KB hidden): ring {:?}, gpu-direct {:?}", step(&ring_run), step(&direct_run));
    eprintln!("CHAIN prefill chunk (128 rows, 128 KB), both stages: ring {:?}, gpu-direct {:?}", prefill_step(&ring_run), prefill_step(&direct_run));
}

#[test]
#[ignore = "needs a CUDA GPU and a RoCE NIC, run under gpu-run 1"]
fn pinned_ring_and_gpu_direct_hops_timed_in_one_run_at_20kb_and_4mb() {
    if !gpu_allowed() {
        return;
    }
    let Some(ip) = rdma_ip() else {
        eprintln!("SKIPPED: no RoCE device");
        return;
    };
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let link = LinkOptions { transport: Transport::Rdma, library: None, gpu_direct: true };
    let parsed = ip.parse().unwrap();
    let region = || Arc::new(DirectRegion::new(&ctx, 4 << 20).unwrap());
    let (out_a, land_a, out_b, land_b) = (region(), region(), region(), region());
    let (pa, ia) = DirectLink::offer(&out_a, &land_a, &link, parsed).unwrap();
    let (pb, ib) = DirectLink::offer(&out_b, &land_b, &link, parsed).unwrap();
    let mut a = pa.connect(ib, land_b.len()).unwrap();
    let mut b = pb.connect(ia, land_a.len()).unwrap();

    let server = StageServer::start_with(&format!("{ip}:0"), KEY, Arc::new(Sink), link.clone()).unwrap();
    let addr = format!("{ip}:{}", server.addr().port());
    let mut ring = StageClient::connect_with(&addr, KEY, Duration::from_secs(10), &link).unwrap();
    assert!(ring.is_rdma());

    for (label, n, iters) in [("20KB", 20 * 1024usize, 300u32), ("4MB", 4 << 20, 40)] {
        let pattern: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        a.outbox.write_host(&pattern).unwrap();
        a.push(n, Duration::from_secs(5)).unwrap();
        b.wait(n, Duration::from_secs(5)).unwrap();
        assert_eq!(b.landing.read_host(n).unwrap(), pattern, "{label} arrived intact in GPU memory");

        let t = Instant::now();
        for _ in 0..iters {
            a.push(n, Duration::from_secs(5)).unwrap();
            b.wait(n, Duration::from_secs(5)).unwrap();
        }
        let direct = t.elapsed() / iters;

        let t = Instant::now();
        for _ in 0..iters {
            let host = a.outbox.read_host(n).unwrap();
            let message = Message::Forward { slot: 0, position: 0, input: Input::Hidden { rows: (n / 4) as u32, data: host }, decode: false };
            assert_eq!(ring.call(&message).unwrap(), Message::Ack);
            b.landing.write_host(&pattern).unwrap();
        }
        let pinned = t.elapsed() / iters;
        eprintln!("HOP {label}: gpu-direct {direct:?} ({:.2} GB/s), pinned ring with device-host copies {pinned:?} ({:.2} GB/s)", n as f64 / direct.as_secs_f64() / 1e9, n as f64 / pinned.as_secs_f64() / 1e9);
    }
}
