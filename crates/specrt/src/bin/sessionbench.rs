use anyhow::{Context, Result};
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::sched::{Config, Event, Executor, GpuExecutor, Request, Scheduler};
use specrt::sessions::{SessionOptions, SessionStore, Shared};
use specrt::spec::{PrefixCacheOptions, Spec};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

fn load_spec(path: &Path) -> Result<Spec> {
    if path.is_dir() {
        let config = std::fs::read_to_string(path.join("config.json")).with_context(|| path.display().to_string())?;
        let mut spec = Spec::from_hf_config(&config)?;
        spec.name = path.file_name().and_then(|n| n.to_str()).unwrap_or("checkpoint").to_string();
        spec.repo = path.display().to_string();
        return Ok(spec);
    }
    Spec::load(path)
}

fn tokens(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7 + salt * 131) % 5000).collect()
}

fn run(sched: &mut Scheduler<GpuExecutor>, prompt: Vec<u32>, new: usize) -> Result<(f64, usize, Vec<u32>)> {
    let started = Instant::now();
    let id = sched.submit(Request { prompt, max_tokens: new, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None })?;
    let (mut ttft, mut cached, mut out) = (0.0, 0, Vec::new());
    while !sched.idle() {
        let mut events = Vec::new();
        sched.step(&mut events)?;
        for e in events {
            match e {
                Event::Token { id: i, token } if i == id => {
                    if out.is_empty() {
                        ttft = started.elapsed().as_secs_f64();
                    }
                    out.push(token);
                }
                Event::Finished { id: i, cached_tokens, .. } if i == id => cached = cached_tokens,
                _ => {}
            }
        }
    }
    Ok((ttft, cached, out))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: sessionbench <spec.toml | checkpoint dir> <context tokens> <store dir> [kv_gb]";
    let spec = load_spec(&PathBuf::from(args.first().context(usage)?))?;
    let context: usize = args.get(1).context(usage)?.parse()?;
    let dir = PathBuf::from(args.get(2).context(usage)?);
    let kv_gb: f64 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4.0);
    let _ = std::fs::remove_dir_all(&dir);
    let new = 32;
    let layout = Layout::serving(&spec, context + 2048 + new, 2, Some((kv_gb * (1u64 << 30) as f64) as usize));
    let rt = Runtime::load_layout(&spec, layout)?;
    let cfg = Config { max_num_seqs: 2, prefix_cache: PrefixCacheOptions { enabled: false, ..Default::default() }, ..Config::default() };
    let mut sched = Scheduler::new(GpuExecutor::new(rt)?, cfg);
    let compat = sched.exec.session_compat().context("session compat")?;
    let shared = Arc::new(Shared::new(SessionStore::open(&dir, compat)?, SessionOptions { enabled: true, min_tokens: 64, ..Default::default() }));
    sched.enable_sessions(shared.clone());
    let clock = Arc::new(AtomicU64::new(1000));
    let c = clock.clone();
    sched.set_session_clock(Box::new(move || c.load(Ordering::SeqCst)));

    run(&mut sched, tokens(64, 99), 4)?;
    let first = tokens(context, 1);
    let (cold_ttft, _, out) = run(&mut sched, first.clone(), new)?;
    clock.fetch_add(301, Ordering::SeqCst);
    let write_started = Instant::now();
    sched.tick_sessions();
    let write_secs = write_started.elapsed().as_secs_f64();
    let saved = shared.store.list();
    let entry = saved.first().context("no snapshot was written")?;

    let mut next = first.clone();
    next.extend(&out);
    next.extend(tokens(64, 7));
    let (restored_ttft, cached, restored_out) = run(&mut sched, next.clone(), new)?;
    let stats = shared.stats();

    shared.store.purge()?;
    let (full_ttft, _, full_out) = run(&mut sched, next.clone(), new)?;
    anyhow::ensure!(full_out == restored_out, "restored output differs from a full prefill");

    let gb = entry.bytes as f64 / (1u64 << 30) as f64;
    println!("| model | context | snapshot bytes | write s | restore s | prefill s (cold turn) | ttft restored s | ttft full prefill s | tokens not prefilled |");
    println!("|---|---|---|---|---|---|---|---|---|");
    println!(
        "| {} | {} | {:.2} GiB | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |",
        spec.name, context, gb, write_secs, stats.restore_seconds, cold_ttft, restored_ttft, full_ttft, cached
    );
    Ok(())
}
