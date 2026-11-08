use crate::metrics_text::{Sample, scrape};
use serde::Serialize;
use serde_json::{Value, json};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct LengthMix {
    pub buckets: Vec<(f64, usize, usize)>,
}

impl LengthMix {
    pub fn journal() -> Self {
        LengthMix { buckets: vec![(0.6, 1024, 8192), (0.3, 8192, 32768), (0.1, 32768, 131072)] }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let mut buckets = Vec::new();
        for part in s.split(',').filter(|p| !p.trim().is_empty()) {
            let (w, range) = part.split_once(':').ok_or_else(|| format!("{part}: want weight:min-max"))?;
            let (lo, hi) = range.split_once('-').ok_or_else(|| format!("{part}: want weight:min-max"))?;
            let num = |v: &str| -> Result<usize, String> {
                let v = v.trim().to_lowercase();
                let (digits, scale) = match v.strip_suffix('k') {
                    Some(d) => (d.to_string(), 1024),
                    None => (v.clone(), 1),
                };
                digits.parse::<usize>().map(|n| n * scale).map_err(|e| format!("{v}: {e}"))
            };
            let weight: f64 = w.trim().parse().map_err(|e| format!("{w}: {e}"))?;
            let (lo, hi) = (num(lo)?, num(hi)?);
            if weight <= 0.0 || lo == 0 || hi < lo {
                return Err(format!("{part}: weight must be positive and min <= max"));
            }
            buckets.push((weight, lo, hi));
        }
        if buckets.is_empty() {
            return Err("an empty length mix".into());
        }
        Ok(LengthMix { buckets })
    }

    pub fn capped(&self, max: usize) -> Self {
        let buckets = self
            .buckets
            .iter()
            .filter(|b| b.1 <= max)
            .map(|&(w, lo, hi)| (w, lo, hi.min(max)))
            .collect();
        LengthMix { buckets }
    }

    fn draw(&self, rng: &mut Rng) -> usize {
        let total: f64 = self.buckets.iter().map(|b| b.0).sum();
        let mut x = rng.unit() * total;
        for &(w, lo, hi) in &self.buckets {
            if x < w {
                return lo + (rng.unit() * (hi - lo + 1) as f64) as usize;
            }
            x -= w;
        }
        let last = self.buckets.last().expect("non-empty mix");
        last.2
    }
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub requests: usize,
    pub rate: f64,
    pub clients: usize,
    pub mix: LengthMix,
    pub output: (usize, usize),
    pub seed: u64,
    pub sample_every: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Planned {
    pub at: Duration,
    pub prompt_words: usize,
    pub max_tokens: usize,
}

pub fn plan(cfg: &LoadConfig) -> Vec<Planned> {
    let mut rng = Rng::new(cfg.seed);
    let mut t = 0.0;
    (0..cfg.requests)
        .map(|_| {
            if cfg.rate > 0.0 {
                t += -(1.0 - rng.unit()).ln() / cfg.rate;
            }
            let (lo, hi) = cfg.output;
            Planned {
                at: Duration::from_secs_f64(t),
                prompt_words: cfg.mix.draw(&mut rng),
                max_tokens: lo + (rng.unit() * (hi - lo + 1) as f64) as usize,
            }
        })
        .collect()
}

const WORDS: [&str; 24] = [
    "river", "stone", "light", "garden", "window", "music", "paper", "silver", "morning", "letter", "forest", "bridge",
    "engine", "harbor", "candle", "planet", "valley", "market", "winter", "island", "signal", "cotton", "ladder", "meadow",
];

pub fn prompt(words: usize, salt: u64) -> String {
    let mut rng = Rng::new(salt.wrapping_add(17));
    let mut out = format!("Request {salt}. Continue this list of words without stopping:\n");
    for _ in 0..words.saturating_sub(12) {
        out.push_str(WORDS[(rng.next() % WORDS.len() as u64) as usize]);
        out.push(' ');
    }
    out
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Outcome {
    pub planned_prompt: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub queued_secs: f64,
    pub ttft_secs: f64,
    pub total_secs: f64,
    pub gaps: Vec<f64>,
    pub error: Option<String>,
}

fn one(agent: &ureq::Agent, cfg: &LoadConfig, p: &Planned, salt: u64) -> Outcome {
    let mut out = Outcome { planned_prompt: p.prompt_words, ..Outcome::default() };
    let body = json!({
        "model": cfg.model,
        "prompt": prompt(p.prompt_words, salt),
        "max_tokens": p.max_tokens,
        "temperature": 0,
        "stream": true,
        "ignore_eos": true,
        "stream_options": {"include_usage": true},
    });
    let started = Instant::now();
    let mut req = agent.post(format!("{}/v1/completions", cfg.url.trim_end_matches('/')));
    if let Some(k) = &cfg.api_key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    let resp = match req.send_json(&body) {
        Ok(r) => r,
        Err(e) => {
            out.error = Some(e.to_string());
            return out;
        }
    };
    if resp.status().as_u16() >= 400 {
        let status = resp.status().as_u16();
        let text = resp.into_body().read_to_string().unwrap_or_default();
        out.error = Some(format!("{status}: {}", text.chars().take(300).collect::<String>()));
        return out;
    }
    let mut reader = std::io::BufReader::new(resp.into_body().into_reader());
    let mut last: Option<Instant> = None;
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                out.error = Some(format!("stream broke: {e}"));
                break;
            }
        }
        let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else { continue };
        if data == "[DONE]" {
            break;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(m) = event["error"]["message"].as_str() {
            out.error = Some(m.to_string());
            break;
        }
        if let Some(u) = event.get("usage").filter(|u| u.is_object()) {
            out.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0);
            out.completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
        }
        let text = event["choices"][0]["text"].as_str().unwrap_or("");
        if text.is_empty() {
            continue;
        }
        let now = Instant::now();
        match last {
            None => out.ttft_secs = now.duration_since(started).as_secs_f64(),
            Some(prev) => out.gaps.push(now.duration_since(prev).as_secs_f64()),
        }
        last = Some(now);
    }
    out.total_secs = started.elapsed().as_secs_f64();
    out
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Report {
    pub requests: usize,
    pub failed: usize,
    pub wall_secs: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub ttft_p50: f64,
    pub ttft_p95: f64,
    pub itl_p50: f64,
    pub itl_p95: f64,
    pub itl_p99: f64,
    pub itl_max: f64,
    pub client_queue_p95: f64,
    pub preemptions: Option<f64>,
    pub kv_use_max: Option<f64>,
    pub kv_use_mean: Option<f64>,
    pub running_max: Option<f64>,
    pub samples: Vec<Sample>,
    pub errors: Vec<String>,
}

pub fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
}

pub fn summarise(outcomes: &[Outcome], wall: f64, samples: Vec<Sample>) -> Report {
    let ok: Vec<&Outcome> = outcomes.iter().filter(|o| o.error.is_none()).collect();
    let mut ttft: Vec<f64> = ok.iter().map(|o| o.ttft_secs).collect();
    let mut gaps: Vec<f64> = ok.iter().flat_map(|o| o.gaps.iter().copied()).collect();
    let mut queued: Vec<f64> = outcomes.iter().map(|o| o.queued_secs).collect();
    let completion: u64 = ok.iter().map(|o| o.completion_tokens).sum();
    let prompt: u64 = ok.iter().map(|o| o.prompt_tokens).sum();
    let kv: Vec<f64> = samples.iter().filter_map(|s| s.kv_use).collect();
    let pre = match (samples.first().and_then(|s| s.preemptions), samples.last().and_then(|s| s.preemptions)) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };
    let mut errors: Vec<String> = outcomes.iter().filter_map(|o| o.error.clone()).collect();
    errors.truncate(10);
    Report {
        requests: outcomes.len(),
        failed: outcomes.len() - ok.len(),
        wall_secs: wall,
        prompt_tokens: prompt,
        completion_tokens: completion,
        decode_tok_s: if wall > 0.0 { completion as f64 / wall } else { 0.0 },
        prefill_tok_s: if wall > 0.0 { prompt as f64 / wall } else { 0.0 },
        ttft_p50: percentile(&mut ttft, 0.5),
        ttft_p95: percentile(&mut ttft, 0.95),
        itl_p50: percentile(&mut gaps, 0.5),
        itl_p95: percentile(&mut gaps, 0.95),
        itl_p99: percentile(&mut gaps, 0.99),
        itl_max: gaps.iter().copied().fold(0.0, f64::max),
        client_queue_p95: percentile(&mut queued, 0.95),
        preemptions: pre,
        kv_use_max: kv.iter().copied().reduce(f64::max),
        kv_use_mean: (!kv.is_empty()).then(|| kv.iter().sum::<f64>() / kv.len() as f64),
        running_max: samples.iter().filter_map(|s| s.running).reduce(f64::max),
        samples,
        errors,
    }
}

fn sampler(agent: ureq::Agent, cfg: LoadConfig, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<Vec<Sample>> {
    std::thread::spawn(move || {
        let started = Instant::now();
        let mut out = Vec::new();
        loop {
            let mut req = agent.get(format!("{}/metrics", cfg.url.trim_end_matches('/')));
            if let Some(k) = &cfg.api_key {
                req = req.header("Authorization", &format!("Bearer {k}"));
            }
            if let Ok(r) = req.call()
                && let Ok(text) = r.into_body().read_to_string()
            {
                out.push(scrape(&text, started.elapsed().as_secs_f64()));
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let until = Instant::now() + cfg.sample_every;
            while Instant::now() < until && !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        out
    })
}

pub fn run(cfg: &LoadConfig, on_done: &mut dyn FnMut(usize, &Outcome)) -> Report {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(3600)))
        .build()
        .into();
    let planned = plan(cfg);
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = sampler(agent.clone(), cfg.clone(), stop.clone());
    let (tx, rx) = channel::<(usize, Planned, Instant)>();
    let rx: Arc<Mutex<Receiver<(usize, Planned, Instant)>>> = Arc::new(Mutex::new(rx));
    let (done_tx, done_rx) = channel::<(usize, Outcome)>();
    let workers: Vec<_> = (0..cfg.clients.max(1))
        .map(|_| {
            let (rx, done, agent, cfg) = (rx.clone(), done_tx.clone(), agent.clone(), cfg.clone());
            std::thread::spawn(move || {
                loop {
                    let job = rx.lock().unwrap().recv();
                    let Ok((i, p, due)) = job else { break };
                    let queued = due.elapsed().as_secs_f64();
                    let mut o = one(&agent, &cfg, &p, cfg.seed.wrapping_mul(1_000_003).wrapping_add(i as u64));
                    o.queued_secs = queued;
                    if done.send((i, o)).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(done_tx);
    let started = Instant::now();
    let n = planned.len();
    let feeder = std::thread::spawn(move || {
        for (i, p) in planned.into_iter().enumerate() {
            let due = started + p.at;
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
            if tx.send((i, p, due)).is_err() {
                break;
            }
        }
    });
    let mut outcomes: Vec<Option<Outcome>> = vec![None; n];
    for (i, o) in done_rx.iter() {
        on_done(i, &o);
        outcomes[i] = Some(o);
    }
    let wall = started.elapsed().as_secs_f64();
    let _ = feeder.join();
    for w in workers {
        let _ = w.join();
    }
    stop.store(true, Ordering::Relaxed);
    let samples = sampler.join().unwrap_or_default();
    let outcomes: Vec<Outcome> = outcomes.into_iter().flatten().collect();
    summarise(&outcomes, wall, samples)
}

impl Report {
    pub fn describe(&self) -> String {
        let opt = |v: Option<f64>, f: &dyn Fn(f64) -> String| v.map(f).unwrap_or_else(|| "-".into());
        format!(
            "requests {} failed {} wall {:.1} s\n\
             decode {:.1} tok/s aggregate, prompt {:.1} tok/s ({} prompt + {} completion tokens)\n\
             ttft p50 {:.0} ms p95 {:.0} ms, client queue p95 {:.0} ms\n\
             inter-token p50 {:.1} ms p95 {:.1} ms p99 {:.1} ms max {:.0} ms\n\
             preemptions {}, kv use max {} mean {}, running max {}",
            self.requests,
            self.failed,
            self.wall_secs,
            self.decode_tok_s,
            self.prefill_tok_s,
            self.prompt_tokens,
            self.completion_tokens,
            self.ttft_p50 * 1e3,
            self.ttft_p95 * 1e3,
            self.client_queue_p95 * 1e3,
            self.itl_p50 * 1e3,
            self.itl_p95 * 1e3,
            self.itl_p99 * 1e3,
            self.itl_max * 1e3,
            opt(self.preemptions, &|v| format!("{v:.0}")),
            opt(self.kv_use_max, &|v| format!("{:.1}%", v * 100.0)),
            opt(self.kv_use_mean, &|v| format!("{:.1}%", v * 100.0)),
            opt(self.running_max, &|v| format!("{v:.0}")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_journal_mix_draws_its_shares() {
        let mix = LengthMix::journal();
        let mut rng = Rng::new(3);
        let draws: Vec<usize> = (0..20_000).map(|_| mix.draw(&mut rng)).collect();
        let share = |lo: usize, hi: usize| draws.iter().filter(|&&d| d >= lo && d <= hi).count() as f64 / 20_000.0;
        assert!((share(1024, 8191) - 0.6).abs() < 0.02, "{}", share(1024, 8191));
        assert!((share(32769, 131072) - 0.1).abs() < 0.02);
        assert!(draws.iter().all(|&d| (1024..=131072).contains(&d)));
    }

    #[test]
    fn mixes_parse_and_cap() {
        let m = LengthMix::parse("0.6:1k-8k,0.3:8k-32k,0.1:32k-128k").unwrap();
        assert_eq!(m, LengthMix::journal());
        assert_eq!(m.capped(16384).buckets, vec![(0.6, 1024, 8192), (0.3, 8192, 16384)]);
        assert!(LengthMix::parse("1:5-2").is_err());
        assert!(LengthMix::parse("").is_err());
    }

    #[test]
    fn open_loop_arrivals_follow_the_rate() {
        let cfg = LoadConfig {
            url: String::new(),
            model: String::new(),
            api_key: None,
            requests: 4000,
            rate: 8.0,
            clients: 1,
            mix: LengthMix::journal(),
            output: (128, 1024),
            seed: 1,
            sample_every: Duration::from_secs(1),
        };
        let p = plan(&cfg);
        let span = p.last().unwrap().at.as_secs_f64();
        assert!((span - 500.0).abs() < 30.0, "{span}");
        assert!(p.iter().all(|r| (128..=1024).contains(&r.max_tokens)));
        assert_eq!(plan(&cfg), p, "the plan is reproducible from the seed");
    }

    #[test]
    fn percentiles_and_preemptions_summarise() {
        let o = |ttft: f64, gaps: Vec<f64>| Outcome { ttft_secs: ttft, gaps, completion_tokens: 10, prompt_tokens: 5, ..Outcome::default() };
        let samples = vec![
            Sample { at: 0.0, preemptions: Some(2.0), kv_use: Some(0.1), running: Some(1.0), waiting: None },
            Sample { at: 1.0, preemptions: Some(5.0), kv_use: Some(0.5), running: Some(4.0), waiting: None },
        ];
        let r = summarise(&[o(0.1, vec![0.01, 0.02]), o(0.3, vec![0.5]), Outcome { error: Some("x".into()), ..Outcome::default() }], 2.0, samples);
        assert_eq!((r.requests, r.failed, r.completion_tokens), (3, 1, 20));
        assert_eq!(r.decode_tok_s, 10.0);
        assert_eq!(r.itl_max, 0.5);
        assert_eq!(r.preemptions, Some(3.0));
        assert_eq!(r.kv_use_max, Some(0.5));
        assert_eq!(r.running_max, Some(4.0));
    }
}
