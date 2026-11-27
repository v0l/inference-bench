use anyhow::{Context, Result, ensure};
use specrt::speech::tts::service::{Job, Speak, Speaker};
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 4, "phonon <model dir> <sentences.jsonl with id, text, voice> <out dir> <seed>");
    let (dir, manifest, out, seed) = (PathBuf::from(&args[0]), &args[1], PathBuf::from(&args[2]), args[3].parse::<u64>()?);
    std::fs::create_dir_all(out.join("wav"))?;
    let start = std::time::Instant::now();
    let mut speaker = Speaker::open_gpu(&dir, 0)?;
    let rate = speaker.info().sample_rate;
    eprintln!("{} loaded in {:.1}s, {rate} Hz", dir.display(), start.elapsed().as_secs_f64());
    let mut f = std::fs::File::create(out.join("engine.jsonl"))?;
    for line in std::fs::read_to_string(manifest)?.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let id = row["id"].as_str().context("a row has no id")?;
        let voice = PathBuf::from(row["voice"].as_str().context("a row has no voice")?);
        let clock = std::time::Instant::now();
        let job = Job { text: row["text"].as_str().context("a row has no text")?.to_string(), voice: Some(voice), seed };
        let mut pcm = Vec::new();
        let spoken = speaker.speak(&job, &mut |frame| {
            pcm.extend_from_slice(frame);
            true
        })?;
        let secs = clock.elapsed().as_secs_f64();
        let wav = out.join("wav").join(format!("{id}.wav"));
        std::fs::write(&wav, specrt::duplex::wav::wav_bytes(&pcm, rate))?;
        let audio = pcm.len() as f64 / rate as f64;
        writeln!(f, "{}", serde_json::json!({ "id": id, "wav": wav, "samples": pcm.len(), "audio_seconds": audio, "frames": spoken.frames, "chunks": spoken.chunks, "seconds": secs }))?;
        eprintln!("{id} {audio:.2}s of audio in {secs:.2}s");
    }
    Ok(())
}
