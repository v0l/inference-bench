use anyhow::{Context, Result};
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn load_spec(path: &Path) -> Result<Spec> {
    if path.extension().is_some_and(|e| e == "gguf") {
        return Spec::from_gguf_file(path);
    }
    if path.is_dir() {
        let config = std::fs::read_to_string(path.join("config.json")).with_context(|| path.display().to_string())?;
        let mut spec = Spec::from_hf_config(&config)?;
        spec.name = path.file_name().and_then(|n| n.to_str()).unwrap_or("checkpoint").to_string();
        spec.repo = path.display().to_string();
        return Ok(spec);
    }
    Spec::load(path)
}

fn hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |h, f| (h ^ f.to_bits() as u64).wrapping_mul(0x100000001b3))
}

fn hash_u32(v: &[u32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |h, t| (h ^ *t as u64).wrapping_mul(0x100000001b3))
}

fn prompt(len: usize, seed: u32, vocab: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (1000 + (i * 7 + seed * 131) % 5000) % vocab).collect()
}

fn main() -> Result<()> {
    let usage = "usage: logitdump <spec.toml | checkpoint dir | model.gguf> [len,len..] [batch]";
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().context(usage)?);
    let lens: Vec<usize> = args.next().unwrap_or_else(|| "17,300,1100".into()).split(',').map(|s| s.parse().context(usage)).collect::<Result<_>>()?;
    let batch: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(4);
    let spec = load_spec(&path)?.with_env_overrides();
    let vocab = spec.dims.vocab as u32;
    let longest = *lens.iter().max().context(usage)?;
    let mut rt = Runtime::load(&spec, longest + 32)?;
    for &len in &lens {
        let p = prompt(len, 0, vocab);
        let all = rt.prompt_logits(&p)?;
        let decode: Vec<u32> = prompt(8, 1, vocab);
        let stepped = rt.prefill_then_decode(&p, &decode)?;
        let greedy = rt.generate(&p, 16, Sampling::GREEDY)?.tokens;
        println!("{} len={len} prompt={:016x} decode={:016x} greedy={:016x}", spec.name, hash(&all), hash(&stepped), hash_u32(&greedy));
    }
    drop(rt);
    if batch > 1 {
        let mut rt = Runtime::load_batched(&spec, 64 + 16, batch)?;
        let prompts: Vec<Vec<u32>> = (0..batch as u32).map(|s| prompt(64, s, vocab)).collect();
        let g = rt.generate_batch(&prompts, 16)?;
        println!("{} batch={batch} greedy={:016x}", spec.name, hash_u32(&g.tokens.concat()));
    }
    Ok(())
}
