use anyhow::{Context, Result};
use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let spec_path =
        PathBuf::from(args.next().context("usage: bench <spec.toml> [prompt_len] [new_tokens]")?);
    let prompt_len: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
    let new_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
    let spec = Spec::load(&spec_path)?;
    let started = std::time::Instant::now();
    let mut rt = Runtime::load(&spec, prompt_len + new_tokens)?;
    let load_secs = started.elapsed().as_secs_f64();
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7).collect();
    rt.generate(&prompt, 8.min(new_tokens))?;
    let mut runs = Vec::new();
    for _ in 0..5 {
        runs.push(rt.generate(&prompt, new_tokens)?);
    }
    let median = |f: &dyn Fn(&specrt::runtime::Generation) -> f64| {
        let mut v: Vec<f64> = runs.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let prefill = median(&|g| prompt_len as f64 / g.prefill_secs);
    let decode = median(&|g| (new_tokens - 1) as f64 / g.decode_secs);
    println!(
        "{} quant={:?} arch={} jit_cache_hit={} load={:.2}s prompt={} prefill={:.1} tok/s decode={} tokens {:.1} tok/s (median of 5)",
        spec.name,
        spec.quant,
        rt.arch,
        rt.cache_hit,
        load_secs,
        prompt_len,
        prefill,
        new_tokens,
        decode,
    );
    Ok(())
}
