use anyhow::{Context, Result};
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
use specrt::spec::{PrefixCacheOptions, Spec};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

fn tokens(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7 + salt * 131) % 5000).collect()
}

static ASYNC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

struct Bench {
    sched: Scheduler<GpuExecutor>,
}

struct Reply {
    ttft: f64,
    total: f64,
    cached: usize,
    out: Vec<u32>,
}

impl Bench {
    fn new(spec: &Spec, cache: bool, interval: usize, max_ctx: usize, slots: usize, kv_gb: f64) -> Result<Self> {
        let mut spec = spec.clone();
        spec.runtime.prefix_cache = PrefixCacheOptions { enabled: cache, snapshot_interval: interval, max_snapshots: 8, max_blocks: 0, async_snapshots: ASYNC.load(std::sync::atomic::Ordering::Relaxed) };
        let layout = Layout::serving(&spec, max_ctx, slots, Some((kv_gb * (1u64 << 30) as f64) as usize));
        let rt = Runtime::load_layout(&spec, layout)?;
        let cfg = Config { max_num_seqs: slots, prefix_cache: spec.runtime.prefix_cache, ..Config::default() };
        Ok(Bench { sched: Scheduler::new(GpuExecutor::new(rt)?, cfg) })
    }

    fn run(&mut self, prompts: Vec<Vec<u32>>, new: usize) -> Result<Vec<Reply>> {
        self.run_media(prompts.into_iter().map(|p| (p, Vec::new())).collect(), new)
    }

    fn run_media(&mut self, prompts: Vec<(Vec<u32>, Vec<specrt::sched::MediaInput>)>, new: usize) -> Result<Vec<Reply>> {
        let started = Instant::now();
        let ids: Vec<u64> = prompts
            .into_iter()
            .map(|(prompt, media)| self.sched.submit(Request { prompt, max_tokens: new, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media }))
            .collect::<Result<_>>()?;
        let mut replies: HashMap<u64, Reply> = ids.iter().map(|&id| (id, Reply { ttft: 0.0, total: 0.0, cached: 0, out: Vec::new() })).collect();
        while !self.sched.idle() {
            let mut events = Vec::new();
            self.sched.step(&mut events)?;
            let now = started.elapsed().as_secs_f64();
            for e in events {
                match e {
                    Event::Token { id, token } => {
                        let r = replies.get_mut(&id).expect("known id");
                        if r.out.is_empty() {
                            r.ttft = now;
                        }
                        r.out.push(token);
                    }
                    Event::Finished { id, cached_tokens, .. } => {
                        let r = replies.get_mut(&id).expect("known id");
                        r.cached = cached_tokens;
                        r.total = now;
                    }
                    _ => {}
                }
            }
        }
        Ok(ids.iter().map(|id| replies.remove(id).expect("reply")).collect())
    }
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn turns(spec: &Spec, cache: bool, interval: usize, kv_gb: f64) -> Result<()> {
    let (system, step, new, turns) = (1000usize, 500usize, 32usize, 20usize);
    let max_ctx = system + turns * (step + new) + 64;
    let mut b = Bench::new(spec, cache, interval, max_ctx, 2, kv_gb)?;
    b.run(vec![tokens(64, 99)], 4)?;
    let mut prompt = tokens(system, 1);
    let mut series = Vec::new();
    for t in 0..turns {
        let r = b.run(vec![prompt.clone()], new)?.remove(0);
        println!("turns cache={cache} turn={} prompt={} cached={} ttft_ms={:.1}", t + 1, prompt.len(), r.cached, r.ttft * 1e3);
        series.push((r.ttft, r.cached, prompt.len()));
        prompt.extend(&r.out);
        prompt.extend(tokens(step, t as u32 + 2));
    }
    let ttfts: Vec<f64> = series.iter().map(|s| s.0 * 1e3).collect();
    let hit: usize = series.iter().map(|s| s.1).sum();
    let all: usize = series.iter().map(|s| s.2).sum();
    let p = b.sched.prefix_stats();
    println!(
        "SUMMARY turns cache={cache} model={} first_ms={:.1} last_ms={:.1} median_ms={:.1} hit_rate={:.3} cache_blocks={} snapshots={} snapshot_mb={:.1}",
        spec.name,
        ttfts[0],
        ttfts[turns - 1],
        percentile(ttfts.clone(), 0.5),
        hit as f64 / all as f64,
        p.blocks,
        p.snapshots,
        p.snapshot_bytes as f64 / 1e6
    );
    Ok(())
}

fn clients(spec: &Spec, cache: bool, interval: usize, kv_gb: f64) -> Result<()> {
    let (system, unique, new, clients) = (2000usize, 200usize, 16usize, 16usize);
    let max_ctx = system + unique + new + 64;
    let mut b = Bench::new(spec, cache, interval, max_ctx, clients, kv_gb)?;
    b.run(vec![tokens(64, 99)], 4)?;
    let shared = tokens(system, 7);
    let batch = |salt: u32| -> Vec<Vec<u32>> { (0..clients as u32).map(|c| [shared.clone(), tokens(unique, salt * 100 + c + 10)].concat()).collect() };
    for phase in ["cold_first_wave", "warm_second_wave"] {
        let salt = if phase == "cold_first_wave" { 1 } else { 2 };
        let started = Instant::now();
        let replies = b.run(batch(salt), new)?;
        let wall = started.elapsed().as_secs_f64();
        let ttfts: Vec<f64> = replies.iter().map(|r| r.ttft * 1e3).collect();
        let cached: usize = replies.iter().map(|r| r.cached).sum();
        println!(
            "SUMMARY clients cache={cache} model={} phase={phase} ttft_mean_ms={:.1} ttft_p50_ms={:.1} ttft_p95_ms={:.1} wall_s={wall:.2} hit_rate={:.3}",
            spec.name,
            ttfts.iter().sum::<f64>() / ttfts.len() as f64,
            percentile(ttfts.clone(), 0.5),
            percentile(ttfts, 0.95),
            cached as f64 / (clients * (system + unique)) as f64
        );
    }
    Ok(())
}

fn noise_png(salt: u32, side: u32) -> Vec<u8> {
    let mut img = image::RgbImage::new(side, side);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let n = ((x * 7 + y * 13 + salt * 31) % 40) as u8;
        *p = image::Rgb([((x * 255 / side) as u8).saturating_add(n), ((y * 255 / side) as u8).saturating_add(n), 128u8.saturating_add(n)]);
    }
    let mut out = Vec::new();
    let _ = image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png);
    out
}

fn media(spec: &Spec, cache: bool, interval: usize, kv_gb: f64, video: Option<&Path>) -> Result<()> {
    use specrt::vision::{MediaItem, MediaKind};
    let new = 32usize;
    let mut spec = spec.clone();
    specrt::engine::complete_spec(&mut spec)?;
    let mut b = Bench::new(&spec, cache, interval, 16384, 2, kv_gb)?;
    let assets = specrt::engine::media_prompt_assets(&spec)?;
    b.run(vec![tokens(64, 99)], 4)?;
    let mut cases: Vec<(String, Vec<MediaItem>)> = vec![
        ("image_1".into(), vec![MediaItem::new(MediaKind::Image, noise_png(1, 448))]),
        ("images_3".into(), (0..3).map(|i| MediaItem::new(MediaKind::Image, noise_png(10 + i, 448))).collect()),
    ];
    if let Some(v) = video {
        cases.push(("video".into(), vec![MediaItem::new(MediaKind::Video, std::fs::read(v)?)]));
    }
    for (name, items) in cases {
        let build = || -> Result<(Vec<u32>, Vec<specrt::sched::MediaInput>, usize)> {
            let part = |k: MediaKind| if k == MediaKind::Image { "<|vision_start|><|image_pad|><|vision_end|>" } else { "<|vision_start|><|video_pad|><|vision_end|>" };
            let slots: String = items.iter().map(|i| part(i.kind)).collect();
            assets.encode(&format!("<|im_start|>user\n{slots}Describe this in detail.<|im_end|>\n<|im_start|>assistant\n"), &items)
        };
        for pass in ["cold", "warm"] {
            let (ids, media, vision) = build()?;
            let encode_started = Instant::now();
            let r = b.run_media(vec![(ids.clone(), media)], new)?.remove(0);
            println!(
                "SUMMARY media cache={cache} model={} case={name} pass={pass} prompt={} vision={vision} cached={} ttft_ms={:.1} total_ms={:.1} decode_tok_s={:.1} wall_ms={:.1}",
                spec.name,
                ids.len(),
                r.cached,
                r.ttft * 1e3,
                r.total * 1e3,
                (r.out.len().saturating_sub(1)) as f64 / (r.total - r.ttft).max(1e-9),
                encode_started.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}

fn itl(spec: &Spec, cache: bool, interval: usize, kv_gb: f64) -> Result<()> {
    let (long, newcomers, new) = (3072usize, 6usize, 400usize);
    let max_ctx = long + new + 64;
    let mut b = Bench::new(spec, cache, interval, max_ctx, newcomers + 1, kv_gb)?;
    b.run(vec![tokens(64, 99)], 4)?;
    let steady = b.sched.submit(Request { prompt: tokens(64, 5), max_tokens: new, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new() })?;
    let mut waiting: Vec<Vec<u32>> = (0..newcomers as u32).map(|c| tokens(long, 40 + c)).collect();
    let mut last: Option<Instant> = None;
    let mut gaps: Vec<f64> = Vec::new();
    let mut step = 0usize;
    while !b.sched.idle() {
        if step % 40 == 20 && !waiting.is_empty() {
            let prompt = waiting.remove(0);
            b.sched.submit(Request { prompt, max_tokens: 4, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new() })?;
        }
        let mut events = Vec::new();
        b.sched.step(&mut events)?;
        let now = Instant::now();
        let got = events.iter().filter(|e| matches!(e, Event::Token { id, .. } if *id == steady)).count();
        if got > 0 {
            if let Some(prev) = last {
                gaps.push(now.duration_since(prev).as_secs_f64() * 1e3);
            }
            last = Some(now);
        }
        step += 1;
    }
    gaps.sort_by(f64::total_cmp);
    let p = b.sched.prefix_stats();
    println!(
        "SUMMARY itl cache={cache} async={} model={} gaps={} p50_ms={:.1} p99_ms={:.1} max_ms={:.1} snapshots={} snapshot_mb={:.0}",
        ASYNC.load(std::sync::atomic::Ordering::Relaxed),
        spec.name,
        gaps.len(),
        gaps[gaps.len() / 2],
        gaps[(gaps.len() - 1) * 99 / 100],
        gaps[gaps.len() - 1],
        p.snapshots,
        p.snapshot_bytes as f64 / 1e6
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: prefixbench <spec.toml | checkpoint dir> <turns|clients|itl|media> <on|off> [snapshot_interval] [kv_gb] [sync|video.mp4]";
    let spec = load_spec(&PathBuf::from(args.first().context(usage)?))?;
    let mode = args.get(1).context(usage)?.as_str();
    let cache = args.get(2).context(usage)? == "on";
    let interval: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(2048);
    let kv_gb: f64 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4.0);
    ASYNC.store(args.get(5).is_none_or(|s| s != "sync"), std::sync::atomic::Ordering::Relaxed);
    match mode {
        "itl" => itl(&spec, cache, interval, kv_gb),
        "turns" => turns(&spec, cache, interval, kv_gb),
        "media" => media(&spec, cache, interval, kv_gb, args.get(5).map(PathBuf::from).as_deref()),
        "clients" => clients(&spec, cache, interval, kv_gb),
        _ => anyhow::bail!(usage),
    }
}
