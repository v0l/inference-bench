use anyhow::{Context, Result};
use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

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

fn main() -> Result<()> {
    let usage = "usage: envelope <checkpoint dir | spec.toml> <prompts.json> <out dir> [split|unsplit|unsplit4 ...]";
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().context(usage)?);
    let prompts: Vec<Vec<u32>> = serde_json::from_str(&std::fs::read_to_string(args.next().context(usage)?)?)?;
    let out = PathBuf::from(args.next().context(usage)?);
    let modes: Vec<String> = args.collect();
    std::fs::create_dir_all(&out)?;
    let longest = prompts.iter().map(Vec::len).max().context(usage)?;
    for mode in modes {
        let mut spec = load_spec(&path)?;
        spec.runtime.unsplit = match mode.as_str() {
            "split" => Vec::new(),
            "unsplit4" => ["qkv", "o", "gate_up", "down"].map(String::from).to_vec(),
            "unsplit" => ["qkv", "o", "gate_up", "down", "lm_head"].map(String::from).to_vec(),
            other => anyhow::bail!("mode {other}"),
        };
        let mut rt = Runtime::load(&spec, longest + 8)?;
        let mut bytes = Vec::new();
        for p in &prompts {
            for v in rt.prompt_logits(p)? {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(out.join(format!("specrt-{mode}.f32")), bytes)?;
        println!("{mode} done");
    }
    Ok(())
}
