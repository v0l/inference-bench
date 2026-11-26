use anyhow::{Context, Result, bail, ensure};
use specrt::duplex::{Engine, EngineOptions, Mode, StartOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    ensure!(b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE", "{} is not a WAV file", path.display());
    let mut at = 12;
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes(b[at + 4..at + 8].try_into()?) as usize;
        let body = &b[at + 8..(at + 8 + len).min(b.len())];
        if id == b"fmt " {
            format = u16::from_le_bytes(body[0..2].try_into()?);
            channels = u16::from_le_bytes(body[2..4].try_into()?);
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes(body[14..16].try_into()?);
        } else if id == b"data" {
            let samples: Vec<f32> = match (format, bits) {
                (1, 16) => body.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
                (3, 32) | (0xfffe, 32) => body.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                _ => bail!("{}: WAV format {format} with {bits} bits is not read; use PCM16 or float32", path.display()),
            };
            let ch = channels.max(1) as usize;
            return Ok((samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect(), rate));
        }
        at += 8 + len + (len & 1);
    }
    bail!("{} has no data chunk", path.display())
}

fn write_wav(path: &Path, samples: &[f32], rate: u32) -> Result<()> {
    let data: Vec<u8> = samples.iter().flat_map(|v| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect();
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend(b"RIFF");
    out.extend(((36 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    Ok(std::fs::write(path, out)?)
}

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn value(&mut self, flag: &str) -> Option<String> {
        let i = self.rest.iter().position(|a| a == flag)?;
        self.rest.remove(i);
        (i < self.rest.len()).then(|| self.rest.remove(i))
    }

    fn flag(&mut self, flag: &str) -> bool {
        match self.rest.iter().position(|a| a == flag) {
            Some(i) => {
                self.rest.remove(i);
                true
            }
            None => false,
        }
    }

    fn parse<T: std::str::FromStr>(&mut self, flag: &str, default: T) -> Result<T> {
        match self.value(flag) {
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{flag} {v} is not valid")),
            None => Ok(default),
        }
    }

    fn done(&self) -> Result<()> {
        ensure!(self.rest.is_empty(), "unknown arguments {:?}", self.rest);
        Ok(())
    }
}

fn start_options(a: &mut Args, rate: usize) -> Result<StartOptions> {
    let card = match a.value("--card") {
        Some(p) => std::fs::read_to_string(&p).with_context(|| p.clone())?.trim().to_string(),
        None => String::new(),
    };
    let reference = match a.value("--voice") {
        Some(p) => {
            let (v, r) = read_wav(Path::new(&p))?;
            ensure!(r as usize == rate, "{p} is {r} Hz; give the voice at {rate} Hz");
            Some(v)
        }
        None => None,
    };
    Ok(StartOptions { card, reference, speak_first: a.flag("--speak-first"), greedy: a.flag("--greedy"), seed: a.parse("--seed", 0u64)?, ..StartOptions::default() })
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn bench(dir: &Path, a: &mut Args) -> Result<()> {
    let batches: Vec<usize> = a.value("--batches").unwrap_or_else(|| "1,4,16".into()).split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
    let frames: usize = a.parse("--frames", 60)?;
    let max_ctx: usize = a.parse("--max-ctx", 1024)?;
    a.done()?;
    let slots = *batches.iter().max().context("no batch sizes")?;
    let loaded = Instant::now();
    let mut e = Engine::load(dir, EngineOptions { slots, max_ctx, ..EngineOptions::default() })?;
    let budget = e.duplex.frame_seconds() * 1e3;
    eprintln!("loaded in {:.1} s", loaded.elapsed().as_secs_f64());
    println!("| sessions | median ms | p95 ms | max ms | budget ms | frames | graph capture ms |");
    println!("|---|---|---|---|---|---|---|");
    for &n in &batches {
        let ids: Vec<u64> = (0..n).map(|i| e.start(StartOptions { card: "You are a calm friend.".into(), seed: i as u64, ..StartOptions::default() }).map(|(id, _)| id)).collect::<Result<_>>()?;
        let silence = vec![0.0f32; e.duplex.frame_samples];
        let batch: Vec<(u64, Vec<f32>)> = ids.iter().map(|&id| (id, silence.clone())).collect();
        for _ in 0..5 {
            e.step(&batch)?;
        }
        let mut ms = Vec::with_capacity(frames);
        for _ in 0..frames {
            let t = Instant::now();
            e.step(&batch)?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let capture = e.capture_secs.iter().filter(|c| c.0 == specrt::duplex::engine::padded(n)).map(|c| c.1).sum::<f64>();
        let max = ms.iter().copied().fold(0.0, f64::max);
        println!("| {n} | {:.2} | {:.2} | {max:.2} | {budget:.1} | {frames} | {:.1} |", percentile(&mut ms.clone(), 0.5), percentile(&mut ms, 0.95), capture * 1e3);
        for id in ids {
            e.end(id)?;
        }
    }
    Ok(())
}

fn run(dir: &Path, a: &mut Args) -> Result<()> {
    let user_path = PathBuf::from(a.value("--user").context("--user <wav>")?);
    let out = PathBuf::from(a.value("--out").context("--out <wav>")?);
    let transcript = a.value("--transcript").map(PathBuf::from);
    let mut e = Engine::load(dir, EngineOptions { slots: 1, max_ctx: a.parse("--max-ctx", 8192)?, ..EngineOptions::default() })?;
    let o = start_options(a, e.duplex.sample_rate)?;
    a.done()?;
    let (mut user, rate) = read_wav(&user_path)?;
    ensure!(rate as usize == e.duplex.sample_rate, "{} is {rate} Hz; give the user clip at {} Hz", user_path.display(), e.duplex.sample_rate);
    let f = e.duplex.frame_samples;
    let count = user.len().div_ceil(f).max(1);
    user.resize(count * f, 0.0);
    let (id, first) = e.start(o)?;
    let mut frames = vec![first];
    let mut ms = Vec::new();
    for i in 0..count - 1 {
        let t = Instant::now();
        frames.push(e.step(&[(id, user[i * f..(i + 1) * f].to_vec())])?.remove(0));
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let reply: Vec<f32> = frames.iter().flat_map(|fr| fr.audio.iter().copied()).collect();
    write_wav(&out, &reply, e.duplex.sample_rate as u32)?;
    if let Some(t) = transcript {
        let lines: Vec<String> = frames.iter().map(|fr| serde_json::json!({"frame": fr.index, "token": fr.token, "text": fr.text}).to_string()).collect();
        std::fs::write(t, lines.join("\n") + "\n")?;
    }
    let text: String = frames.iter().map(|fr| fr.text.as_str()).collect();
    println!("{count} frames, {:.2} ms median per frame -> {}", if ms.is_empty() { 0.0 } else { percentile(&mut ms, 0.5) }, out.display());
    println!("{}", text.trim());
    Ok(())
}

fn say(dir: &Path, a: &mut Args) -> Result<()> {
    let sentences = a.value("--sentences").context("--sentences <file, one per line>")?;
    let out_dir = PathBuf::from(a.value("--out-dir").context("--out-dir <dir>")?);
    let lead: usize = a.parse("--lead-frames", 8)?;
    let tail: usize = a.parse("--tail-frames", 12)?;
    let limit: usize = a.parse("--max-frames", 600)?;
    let mut e = Engine::load(dir, EngineOptions { slots: 1, max_ctx: a.parse("--max-ctx", 8192)?, ..EngineOptions::default() })?;
    let base = start_options(a, e.duplex.sample_rate)?;
    a.done()?;
    std::fs::create_dir_all(&out_dir)?;
    let f = e.duplex.frame_samples;
    let silence = vec![0.0f32; f];
    let mut report = Vec::new();
    for (i, line) in std::fs::read_to_string(&sentences)?.lines().map(str::trim).filter(|l| !l.is_empty()).enumerate() {
        let (id, first) = e.start(StartOptions { mode: Mode::Relay, seed: base.seed + i as u64, ..base.clone() })?;
        let mut audio = first.audio;
        let mut tokens = Vec::new();
        for _ in 0..lead {
            audio.extend(e.step(&[(id, silence.clone())])?.remove(0).audio);
        }
        let queued = e.say(id, &format!(" {line}"))?;
        let (mut steps, mut after) = (0, 0);
        while steps < limit && after < tail {
            let fr = e.step(&[(id, silence.clone())])?.remove(0);
            if !e.duplex.sampling.silent.contains(&fr.token) {
                tokens.push(fr.token);
            }
            audio.extend(fr.audio);
            steps += 1;
            if e.live(id)?.text.queued() == 0 {
                after += 1;
            }
        }
        let left = e.live(id)?.text.queued();
        e.end(id)?;
        let wav = out_dir.join(format!("{i:02}.wav"));
        write_wav(&wav, &audio, e.duplex.sample_rate as u32)?;
        report.push(serde_json::json!({"index": i, "text": line, "wav": wav.display().to_string(), "queued": queued, "spoken_tokens": tokens.len(), "unsaid": left, "said": e.detokenize(&tokens), "frames": steps + lead + 1}));
        eprintln!("{i:02}: {queued} tokens queued, {left} unsaid after {steps} frames");
    }
    std::fs::write(out_dir.join("say.json"), serde_json::to_string_pretty(&report)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    ensure!(argv.len() >= 2, "usage: haru <bench|run|say> <model dir> [options]");
    let cmd = argv.remove(0);
    let dir = PathBuf::from(argv.remove(0));
    let mut a = Args { rest: argv };
    match cmd.as_str() {
        "bench" => bench(&dir, &mut a),
        "run" => run(&dir, &mut a),
        "say" => say(&dir, &mut a),
        other => bail!("unknown command {other}; use bench, run or say"),
    }
}
