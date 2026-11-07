use crate::target::{TargetConfig, Request};
use crate::result::BackendError;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Barrier};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResult {
    pub backend: String,
    pub model: String,
    pub batch: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub wall_secs: f64,
    pub throughput_tok_s: f64,
    pub decode_tok_s: f64,
    pub median_ttft_secs: f64,
}

pub fn run_concurrent(
    config: &TargetConfig,
    label: &str,
    model_id: &str,
    request: &Request,
    batch: usize,
) -> Result<BatchResult, BackendError> {
    let mut backends = Vec::with_capacity(batch);
    for _ in 0..batch.max(1) {
        let mut b = config.build();
        b.load(model_id)?;
        backends.push(b);
    }
    let gate = Arc::new(Barrier::new(backends.len() + 1));
    let workers: Vec<_> = backends
        .into_iter()
        .map(|mut b| {
            let gate = gate.clone();
            let request = request.clone();
            std::thread::spawn(move || {
                gate.wait();
                b.generate(&request, &mut |_| {})
            })
        })
        .collect();
    gate.wait();
    let started = std::time::Instant::now();
    let mut generations = Vec::new();
    for w in workers {
        generations.push(w.join().map_err(|_| BackendError::failed("a request thread panicked"))??);
    }
    let wall_secs = started.elapsed().as_secs_f64();
    let completion_tokens: u64 = generations.iter().map(|g| g.completion_tokens).sum();
    let decode_tok_s = generations
        .iter()
        .filter(|g| g.decode_secs > 0.0)
        .map(|g| g.completion_tokens.saturating_sub(1) as f64 / g.decode_secs)
        .sum();
    let mut ttfts: Vec<f64> = generations.iter().map(|g| g.ttft_secs).collect();
    ttfts.sort_by(f64::total_cmp);
    Ok(BatchResult {
        backend: label.into(),
        model: model_id.into(),
        batch: generations.len(),
        prompt_tokens: generations.iter().map(|g| g.prompt_tokens).sum(),
        completion_tokens,
        wall_secs,
        throughput_tok_s: completion_tokens as f64 / wall_secs,
        decode_tok_s,
        median_ttft_secs: ttfts[ttfts.len() / 2],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeServer, Script};

    fn script() -> Script {
        Script {
            models: vec!["m".into()],
            chunks: (0..5).map(|i| format!("t{i}")).collect(),
            chunk_delay_ms: 40,
            first_delay_ms: 50,
            usage: Some((128, 5)),
            fail_status: None,
            files: Default::default(),
        }
    }

    fn config(url: &str) -> TargetConfig {
        TargetConfig::Openai { url: url.into(), api_key: None, label: Some("fake".into()) }
    }

    fn request() -> Request {
        Request { prompt: "p".into(), max_tokens: 5 }
    }

    #[test]
    fn concurrent_requests_overlap_and_their_decode_rates_add_up() {
        let server = FakeServer::start(script());
        let r = run_concurrent(&config(&server.url), "fake", "m", &request(), 4).unwrap();
        assert_eq!((r.batch, r.completion_tokens, r.prompt_tokens), (4, 20, 512));
        assert!(r.wall_secs < 0.5, "requests ran one after another: {}", r.wall_secs);
        let one = 4.0 / 0.2;
        assert!(r.decode_tok_s > 3.0 * one && r.decode_tok_s < 4.5 * one, "{}", r.decode_tok_s);
        assert!(r.median_ttft_secs >= 0.05 && r.median_ttft_secs < 0.2, "{}", r.median_ttft_secs);
        let sent = server.requests.lock().unwrap();
        assert_eq!(sent.iter().filter(|s| s.starts_with("/v1/completions")).count(), 8);
    }

    #[test]
    fn a_failing_request_fails_the_batch() {
        let server = FakeServer::start(Script { fail_status: Some((500, "boom".into())), ..script() });
        let e = run_concurrent(&config(&server.url), "fake", "m", &request(), 2).unwrap_err();
        assert!(e.message.contains("boom"), "{}", e.message);
    }
}
