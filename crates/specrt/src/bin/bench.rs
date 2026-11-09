use anyhow::{Context, Result};
use specrt::runtime::{Generation, Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn runs() -> usize {
    std::env::var("SPECRT_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(6)
}

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let speculative = args.first().is_some_and(|a| a == "--speculative");
    if speculative {
        args.remove(0);
    }
    let mut args = args.into_iter();
    let usage = "usage: bench [--speculative] <spec.toml | checkpoint dir> [prompt_len[,prompt_len..]] [new_tokens] [batch]";
    let spec_path = PathBuf::from(args.next().context(usage)?);
    let prompts: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "128".into())
        .split(',')
        .map(|s| s.parse().context(usage))
        .collect::<Result<_>>()?;
    let new_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
    let batch: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(0);
    let spec = load_spec(&spec_path)?;
    let longest = *prompts.iter().max().context(usage)?;
    if speculative {
        return bench_spec(&spec, new_tokens, batch.max(1), longest);
    }
    if std::env::var_os("SPECRT_BENCH_SERVE").is_some() {
        return bench_serve(&spec, longest, new_tokens, batch.max(1));
    }
    if batch > 0 {
        return bench_batch(&spec, longest, new_tokens, batch);
    }
    let mut rt = Runtime::load(&spec, longest + new_tokens)?;
    println!(
        "{} quant={:?} arch={} jit_cache_hit={} load={:.2}s weights={:.2} GB in {:.2}s device={:.2} GB peak_host_rss={:.2} GB",
        spec.name,
        spec.quant,
        rt.arch,
        rt.cache_hit,
        rt.load_secs,
        rt.weight_bytes as f64 / 1e9,
        rt.weight_secs,
        rt.device_bytes as f64 / 1e9,
        rt.peak_host_rss_bytes as f64 / 1e9,
    );
    for &prompt_len in &prompts {
        let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7 % 5000).collect();
        rt.generate(&prompt, 8.min(new_tokens), Sampling::GREEDY)?;
        let runs: Vec<Generation> =
            (0..runs()).map(|_| rt.generate(&prompt, new_tokens, Sampling::GREEDY)).collect::<Result<_>>()?;
        let prefill = median(runs.iter().map(|g| prompt_len as f64 / g.prefill_secs).collect());
        let decode = median(runs.iter().map(|g| (new_tokens - 1) as f64 / g.decode_secs).collect());
        let peak = runs.iter().map(|g| g.peak_device_bytes).max().unwrap_or(0);
        let all = |f: &dyn Fn(&Generation) -> f64| runs.iter().map(|g| format!("{:.1}", f(g))).collect::<Vec<_>>().join(",");
        println!(
            "  prompt={prompt_len} prefill={prefill:.1} tok/s decode={new_tokens} tokens {decode:.1} tok/s peak_device={:.2} GB (median of {}) prefill_runs=[{}] decode_runs=[{}]",
            peak as f64 / 1e9,
            runs.len(),
            all(&|g| prompt_len as f64 / g.prefill_secs),
            all(&|g| (new_tokens - 1) as f64 / g.decode_secs),
        );
    }
    Ok(())
}

fn spec_prompts(spec: &Spec) -> Result<Vec<Vec<u32>>> {
    let path = specrt::weights::find_in_snapshot(&spec.repo, "tokenizer.json")?;
    let tok = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let text = include_str!("../../scripts/spec_prompts.txt");
    text.split("\n---\n")
        .map(|p| Ok(tok.encode(p.trim_end(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()))
        .collect()
}

fn bench_spec(spec: &Spec, new_tokens: usize, batch: usize, max_ctx: usize) -> Result<()> {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let prompts = spec_prompts(spec)?;
    for speculate in [true, false] {
        let mut s = spec.clone();
        if let Some(m) = s.mtp.as_mut() {
            m.speculate = speculate;
        }
        let layout = specrt::runtime::Layout::serving(&s, max_ctx, batch, Some(1 << 30));
        let rt = Runtime::load_layout(&s, layout)?;
        let mut sched = Scheduler::new(GpuExecutor::new(rt)?, Config { max_num_seqs: batch, ..Config::default() });
        let mut warm = Vec::new();
        sched.submit(Request { prompt: prompts[0].clone(), max_tokens: 16, sampling: Sampling::GREEDY, stop: vec![] })?;
        while !sched.idle() {
            sched.step(&mut warm)?;
        }
        let before = sched.exec.rt.spec_stats();
        let (mut decode_secs, mut tokens, mut drafted, mut accepted) = (0.0, 0usize, 0usize, 0usize);
        let mut outs: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        let mut ids = Vec::new();
        for p in &prompts {
            ids.push(sched.submit(Request { prompt: p.clone(), max_tokens: new_tokens, sampling: Sampling::GREEDY, stop: vec![] })?);
        }
        while !sched.idle() {
            let mut events = Vec::new();
            let r = sched.step(&mut events)?;
            if r.prefill_tokens == 0 {
                decode_secs += r.decode_secs;
                tokens += r.decode_rows + r.accepted;
            }
            drafted += r.drafted;
            accepted += r.accepted;
            for e in events {
                if let Event::Token { id, token } = e {
                    outs.entry(id).or_default().push(token);
                }
            }
        }
        let after = sched.exec.rt.spec_stats();
        let rows = (after.rows - before.rows).max(1) as f64;
        println!(
            "{} speculate={speculate} k={} batch={batch} prompts={} new={new_tokens} decode={:.1} tok/s aggregate ({:.1} per sequence) mean_acceptance_length={:.3} draft_acceptance={:.3} drafted={drafted} accepted={accepted}",
            spec.name,
            spec.mtp.as_ref().map_or(0, |m| m.draft_tokens),
            prompts.len(),
            tokens as f64 / decode_secs,
            tokens as f64 / decode_secs / batch as f64,
            1.0 + (after.accepted - before.accepted) as f64 / rows,
            accepted as f64 / drafted.max(1) as f64,
        );
        let first: Vec<u32> = outs[&ids[0]].iter().take(24).copied().collect();
        println!("  first prompt tokens {first:?}");
    }
    Ok(())
}

fn bench_serve(spec: &Spec, prompt_len: usize, new_tokens: usize, batch: usize) -> Result<()> {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let max_ctx = prompt_len + new_tokens + 16;
    let layout = specrt::runtime::Layout::serving(spec, max_ctx, batch, None);
    let rt = Runtime::load_layout(spec, layout)?;
    let mut sched = Scheduler::new(GpuExecutor::new(rt)?, Config { max_num_seqs: batch, ..Config::default() });
    for round in 0..runs() {
        for s in 0..batch as u32 {
            let prompt = (0..prompt_len as u32).map(|i| 1000 + (i * 7 + s * 131) % 5000).collect();
            sched.submit(Request { prompt, max_tokens: new_tokens, sampling: Sampling::GREEDY, stop: vec![] })?;
        }
        let (mut decode_secs, mut decode_tokens, mut prefill_secs, mut steps) = (0.0, 0, 0.0, 0);
        let started = std::time::Instant::now();
        while !sched.idle() {
            let mut events = Vec::new();
            let r = sched.step(&mut events)?;
            decode_secs += r.decode_secs;
            decode_tokens += r.decode_rows;
            prefill_secs += r.prefill_secs;
            steps += 1;
            let _ = events.iter().filter(|e| matches!(e, Event::Token { .. })).count();
        }
        println!(
            "{} serve round={round} batch={batch} prompt={prompt_len} new={new_tokens} steps={steps} decode={:.1} tok/s aggregate (decode step {:.2} ms) prefill {:.2} s wall {:.2} s",
            spec.name,
            decode_tokens as f64 / decode_secs,
            decode_secs * 1e3 / (steps as f64),
            prefill_secs,
            started.elapsed().as_secs_f64(),
        );
    }
    Ok(())
}

fn bench_batch(spec: &Spec, prompt_len: usize, new_tokens: usize, batch: usize) -> Result<()> {
    let mut rt = Runtime::load_batched(spec, prompt_len + new_tokens, batch)?;
    let prompts: Vec<Vec<u32>> = (0..batch as u32)
        .map(|s| (0..prompt_len as u32).map(|i| 1000 + (i * 7 + s * 131) % 5000).collect())
        .collect();
    rt.generate_batch(&prompts, 8.min(new_tokens))?;
    let rates: Vec<f64> = (0..runs())
        .map(|_| {
            rt.generate_batch(&prompts, new_tokens)
                .map(|g| (batch * (new_tokens - 1)) as f64 / g.decode_secs)
        })
        .collect::<Result<_>>()?;
    let rate = median(rates);
    println!(
        "{} batch={} prompt={} new={} decode={:.1} tok/s aggregate, {:.1} per sequence (median of {})",
        spec.name,
        batch,
        prompt_len,
        new_tokens,
        rate,
        rate / batch as f64,
        runs()
    );
    Ok(())
}
