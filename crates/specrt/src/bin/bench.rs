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
    let mut args = std::env::args().skip(1);
    let usage = "usage: bench <spec.toml | checkpoint dir> [prompt_len[,prompt_len..]] [new_tokens] [batch]";
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
