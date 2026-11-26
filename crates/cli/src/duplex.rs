use anyhow::{Context, Result, ensure};
use clap::Args;
use inference_bench_core::duplex_client::{self, Options};
use inference_bench_core::specrt_duplex::wav;
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Args)]
pub struct DuplexArgs {
    #[arg(long, help = "The user's side of the conversation, a WAV file at the model's rate")]
    wav: PathBuf,
    #[arg(long, help = "Where to write the model's reply as a PCM16 WAV")]
    out: PathBuf,
    #[arg(long, default_value = "http://127.0.0.1:8080", help = "The server's base URL")]
    url: String,
    #[arg(long, help = "The model id; defaults to the server's only model")]
    model: Option<String>,
    #[arg(long, env = "INFERENCE_BENCH_API_KEY", help = "Bearer API key")]
    api_key: Option<String>,
    #[arg(long, help = "A text file with the character card")]
    card: Option<PathBuf>,
    #[arg(long, help = "A WAV voice reference at the model's rate")]
    voice: Option<PathBuf>,
    #[arg(long)]
    speak_first: bool,
    #[arg(long)]
    guidance: Option<f64>,
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long, help = "free or relay")]
    mode: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 4, help = "Frames of silence sent after the clip so the reply can finish")]
    tail_frames: usize,
    #[arg(long, help = "Send frames as fast as the socket takes them instead of in real time")]
    fast: bool,
    #[arg(long, help = "Speak the OpenAI Realtime subset on /v1/realtime instead of /v1/duplex")]
    realtime: bool,
    #[arg(long, help = "Also write the events and the per frame latency as JSON here")]
    events: Option<PathBuf>,
}

pub fn run(a: &DuplexArgs) -> Result<()> {
    let (input, rate) = wav::read_wav(&std::fs::read(&a.wav).with_context(|| a.wav.display().to_string())?).with_context(|| a.wav.display().to_string())?;
    ensure!(!input.is_empty(), "{} has no samples", a.wav.display());
    let mut start = json!({"type": "start"});
    if let Some(m) = &a.model {
        start["model"] = json!(m);
    }
    if let Some(c) = &a.card {
        start["card"] = json!(std::fs::read_to_string(c).with_context(|| c.display().to_string())?.trim());
    }
    if let Some(v) = &a.voice {
        start["voice"] = json!({"wav": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, std::fs::read(v).with_context(|| v.display().to_string())?)});
    }
    if a.speak_first {
        start["speak_first"] = json!(true);
    }
    for (k, v) in [("guidance", a.guidance), ("temperature", a.temperature)] {
        if let Some(v) = v {
            start[k] = json!(v);
        }
    }
    if let Some(m) = &a.mode {
        start["mode"] = json!(m);
    }
    if let Some(s) = a.seed {
        start["seed"] = json!(s);
    }
    let path = match (a.realtime, &a.model) {
        (true, Some(m)) => format!("/v1/realtime?model={m}"),
        (true, None) => "/v1/realtime".to_string(),
        (false, _) => "/v1/duplex".to_string(),
    };
    let o = Options { url: duplex_client::ws_url(&a.url, &path), key: a.api_key.clone(), start, realtime: a.realtime, tail_frames: a.tail_frames, paced: !a.fast };
    let run = duplex_client::stream(&o, &input)?;
    let out_rate = run.ready["sample_rate"].as_u64().map(|r| r as u32).unwrap_or(rate);
    ensure!(out_rate == rate, "{} is {rate} Hz and the model runs at {out_rate} Hz; resample it first", a.wav.display());
    std::fs::write(&a.out, wav::wav_bytes(&run.audio, out_rate)).with_context(|| a.out.display().to_string())?;
    let (p50, p95) = (run.percentile(0.5), run.percentile(0.95));
    if let Some(e) = &a.events {
        let report: Value = json!({"ready": run.ready, "frames": run.frames.len(), "late": run.late, "latency_ms": run.latency_ms, "p50_ms": p50, "p95_ms": p95, "text": run.text, "events": run.events});
        std::fs::write(e, serde_json::to_string_pretty(&report)?)?;
    }
    println!("{} frames in, {} frames out, {} late -> {}", input.len().div_ceil(run.ready["frame_samples"].as_u64().unwrap_or(3200) as usize), run.frames.len(), run.late, a.out.display());
    if let (Some(p50), Some(p95)) = (p50, p95) {
        println!("frame latency p50 {p50:.1} ms, p95 {p95:.1} ms (user frame sent to the reply frame received)");
    }
    println!("{}", run.text.trim());
    Ok(())
}
