use specrt::codegen::PAGE;
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::sched::{Config, Event, Executor, GpuExecutor, Request, Scheduler};
use specrt::sessions::{SessionOptions, SessionStore, Shared};
use specrt::spec::{PrefixCacheOptions, Spec};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec
}

fn scheduler(s: &Spec) -> Scheduler<GpuExecutor> {
    let layout = Layout { max_ctx: 1024, slots: 4, kv_blocks: 200, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: true };
    let exec = GpuExecutor::new(Runtime::load_layout(s, layout).unwrap()).unwrap();
    let cfg = Config { max_num_seqs: 4, prefix_cache: PrefixCacheOptions { enabled: false, ..Default::default() }, ..Config::default() };
    Scheduler::new(exec, cfg)
}

fn with_sessions(s: &Spec, dir: &Path, clock: Arc<AtomicU64>) -> (Scheduler<GpuExecutor>, Arc<Shared>) {
    let mut sched = scheduler(s);
    let compat = sched.exec.session_compat().unwrap();
    let options = SessionOptions { enabled: true, min_tokens: 64, ..Default::default() };
    let shared = Arc::new(Shared::new(SessionStore::open(dir, compat).unwrap(), options));
    sched.enable_sessions(shared.clone());
    sched.set_session_clock(Box::new(move || clock.load(Ordering::SeqCst)));
    (sched, shared)
}

struct Reply {
    tokens: Vec<u32>,
    logprobs: Vec<u32>,
    cached: usize,
}

fn run(sched: &mut Scheduler<GpuExecutor>, prompt: Vec<u32>, n: usize) -> Reply {
    let sampling = Sampling::GREEDY.with_logprobs(Some(1));
    let id = sched.submit(Request { prompt, max_tokens: n, sampling, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None }).unwrap();
    let mut reply = Reply { tokens: Vec::new(), logprobs: Vec::new(), cached: 0 };
    while !sched.idle() {
        let mut events = Vec::new();
        sched.step(&mut events).unwrap();
        for e in events {
            match e {
                Event::Token { id: i, token } if i == id => reply.tokens.push(token),
                Event::Logprob { id: i, logprob } if i == id => reply.logprobs.push(logprob.logprob.to_bits()),
                Event::Finished { id: i, cached_tokens, .. } if i == id => reply.cached = cached_tokens,
                _ => {}
            }
        }
    }
    reply
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 5) % 1024).collect()
}

fn roundtrip(name: &str) {
    let s = spec_for(name);
    let dir = std::env::temp_dir().join(format!("specrt-sessions-gpu-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let clock = Arc::new(AtomicU64::new(1000));
    let first_prompt = prompt(150);
    let (mut a, shared) = with_sessions(&s, &dir, clock.clone());
    let turn1 = run(&mut a, first_prompt.clone(), 40);
    clock.fetch_add(301, Ordering::SeqCst);
    a.tick_sessions();
    let saved = shared.store.list();
    assert_eq!(saved.len(), 1);
    assert_eq!(saved[0].tokens, 150 + 40 - 1);
    assert_eq!(a.blocks_used(), 0);
    drop(a);
    drop(shared);

    let mut turn2 = first_prompt.clone();
    turn2.extend(&turn1.tokens);
    turn2.extend(prompt(30).iter().map(|t| (t + 3) % 1024));

    let mut fresh = scheduler(&s);
    let want = run(&mut fresh, turn2.clone(), 64);
    drop(fresh);

    let (mut b, shared) = with_sessions(&s, &dir, clock);
    assert_eq!(shared.store.list().len(), 1, "the snapshot survives a new process and engine");
    let got = run(&mut b, turn2.clone(), 64);
    assert_eq!(got.cached, saved[0].tokens, "restored without prefill");
    assert_eq!(got.tokens, want.tokens, "64 greedy tokens identical");
    assert_eq!(got.logprobs, want.logprobs, "logits of every step bit identical");
    assert_eq!(shared.stats().restores_ok, 1);
    assert_eq!(shared.stats().prefill_tokens_saved, saved[0].tokens as u64);
    assert!(PAGE > 0);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_hybrid_session_restored_into_a_fresh_engine_matches_the_uninterrupted_run() {
    roundtrip("tiny-qwen3.5");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_dense_session_restored_into_a_fresh_engine_matches_the_uninterrupted_run() {
    roundtrip("tiny-granite");
}
