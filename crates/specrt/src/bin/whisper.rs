use anyhow::{Context, Result, ensure};
use specrt::speech::decode::Options;
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 3, "whisper <model dir> <manifest.jsonl> <out.jsonl>");
    let (dir, manifest, out) = (PathBuf::from(&args[0]), &args[1], &args[2]);
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let jit = specrt::jit::Jit::new(ctx.clone(), specrt::jit::default_cache_dir())?;
    let start = std::time::Instant::now();
    let t = specrt::speech::gpu::GpuTranscriber::open(&ctx.default_stream(), &jit, &dir)?;
    eprintln!("{} loaded in {:.1}s, {:.1} MB of f32 weights", dir.display(), start.elapsed().as_secs_f64(), t.engine.weight_bytes as f64 / 1e6);
    let language = t.rules.multilingual.then(|| "en".to_string());
    let task = language.as_ref().map(|_| "transcribe".to_string());
    let short = Options { language: language.clone(), task: task.clone(), ..Options::greedy() };
    let long = Options { timestamps: true, ..short.clone() };
    let mut f = std::fs::File::create(out)?;
    let (mut audio, mut spent) = (0f64, 0f64);
    for line in std::fs::read_to_string(manifest)?.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let id = row["id"].as_str().context("a manifest row has no id")?;
        let (pcm, rate) = specrt::duplex::wav::read_wav(&std::fs::read(row["wav"].as_str().context("a manifest row has no wav")?)?)?;
        ensure!(rate as f64 == t.plan.mel.rate, "{id}: {rate} Hz, the model reads {} Hz", t.plan.mel.rate);
        let o = if pcm.len() > t.plan.mel.samples { &long } else { &short };
        let clock = std::time::Instant::now();
        let got = t.transcribe(&pcm, o)?;
        let secs = clock.elapsed().as_secs_f64();
        audio += pcm.len() as f64 / t.plan.mel.rate;
        spent += secs;
        writeln!(f, "{}", serde_json::json!({ "id": id, "tokens": got.sequence, "windows": got.windows.len(), "seconds": secs }))?;
        eprintln!("{id} {secs:.2}s {} tokens", got.sequence.len());
    }
    eprintln!("{audio:.1}s of audio in {spent:.1}s, {:.1}x real time", audio / spent.max(1e-9));
    Ok(())
}
