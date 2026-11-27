use anyhow::{Context, Result, ensure};
use specrt::codec::cuda::CudaCodec;
use specrt::flow_head::{FlowHead, frame_noise};
use specrt::flow_head_cuda::CudaHead;
use specrt::speech::tts::{HostTts, Voice, plan, text};
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 4, "phonon <model dir> <sentences.jsonl with id, text, voice> <out dir> <seed>");
    let (dir, manifest, out, seed) = (PathBuf::from(&args[0]), &args[1], PathBuf::from(&args[2]), args[3].parse::<u64>()?);
    std::fs::create_dir_all(out.join("wav"))?;
    let start = std::time::Instant::now();
    let t = HostTts::open(&dir)?;
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let norm = text::Normalize::for_lang(t.tts.language.parse::<text::Lang>().map_err(anyhow::Error::msg)?);
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let jit = specrt::jit::Jit::new(ctx.clone(), specrt::jit::default_cache_dir())?;
    let head = CudaHead::load(&stream, &jit, FlowHead::from_dir(&dir, None)?, &dir, true)?;
    let host_codec = specrt::codec::HostCodec::open(&dir)?;
    let codec = CudaCodec::load(&ctx, &jit, &host_codec, &[(1, 1)])?;
    let rate = (host_codec.plan.frame_samples as f64 * t.rules.frame_rate).round() as u32;
    let scale = t.noise_scale(t.tts.param("temperature")? as f32);
    let width = t.spec.inputs.as_ref().context("no [inputs]")?.width_of(&t.tts.voice).context("no voice stream")?;
    eprintln!("{} loaded in {:.1}s, {rate} Hz", dir.display(), start.elapsed().as_secs_f64());
    let mut f = std::fs::File::create(out.join("engine.jsonl"))?;
    let mut voices: std::collections::BTreeMap<String, Voice> = Default::default();
    for line in std::fs::read_to_string(manifest)?.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let id = row["id"].as_str().context("a row has no id")?;
        let voice_path = row["voice"].as_str().context("a row has no voice")?;
        if !voices.contains_key(voice_path) {
            voices.insert(voice_path.to_string(), Voice::open(Path::new(voice_path), &t.tts, width)?);
        }
        let voice = &voices[voice_path];
        let clock = std::time::Instant::now();
        let normalised = norm.apply(row["text"].as_str().context("a row has no text")?).into_owned();
        let (mut pcm, mut frame, mut chunks) = (Vec::new(), 0u64, 0usize);
        for chunk in plan::chunks(&tok, &normalised, &t.rules)?.into_iter().filter(|c| !c.tokens.is_empty()) {
            chunks += 1;
            let mut hs = head.session(&[1.0])?;
            let mut cs = codec.sessions(&codec.decoder, 1)?;
            let (mut s, mut hidden, mut eos) = t.start(Some(voice), &chunk.tokens)?;
            let mut rule = t.rules.eos(chunk.frames_after_eos);
            for _ in 0..chunk.frame_budget {
                let noise: Vec<f32> = frame_noise(seed, frame, t.latent_dim()).iter().map(|v| v * scale).collect();
                frame += 1;
                let latent = head.frame(&mut hs, &hidden, &noise)?;
                let z = head.read(&hs, "codec")?;
                pcm.extend(codec.run(&codec.decoder, &z, &mut cs)?);
                if rule.should_stop(eos) {
                    break;
                }
                (hidden, eos) = t.advance(&mut s, &latent)?;
            }
        }
        let secs = clock.elapsed().as_secs_f64();
        let wav = out.join("wav").join(format!("{id}.wav"));
        std::fs::write(&wav, specrt::duplex::wav::wav_bytes(&pcm, rate))?;
        let audio = pcm.len() as f64 / rate as f64;
        writeln!(f, "{}", serde_json::json!({ "id": id, "wav": wav, "samples": pcm.len(), "audio_seconds": audio, "frames": frame, "chunks": chunks, "seconds": secs }))?;
        eprintln!("{id} {audio:.2}s of audio in {secs:.2}s");
    }
    Ok(())
}
