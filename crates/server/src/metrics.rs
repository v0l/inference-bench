use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;

const TTFT_BUCKETS: [f64; 11] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

#[derive(Default)]
struct Histogram {
    counts: [u64; TTFT_BUCKETS.len()],
    sum: f64,
    total: u64,
}

impl Histogram {
    fn observe(&mut self, v: f64) {
        for (i, bound) in TTFT_BUCKETS.iter().enumerate() {
            if v <= *bound {
                self.counts[i] += 1;
            }
        }
        self.sum += v;
        self.total += 1;
    }
}

#[derive(Default)]
struct Inner {
    requests: BTreeMap<(String, u16), u64>,
    prompt_tokens: BTreeMap<String, u64>,
    completion_tokens: BTreeMap<String, u64>,
    ttft: Histogram,
    decode_tokens: u64,
    decode_secs: f64,
    last_decode_tok_s: f64,
    cancelled: u64,
    model_loads: BTreeMap<String, u64>,
    model_load_failures: u64,
    model_unloads: u64,
    spec_checks: BTreeMap<String, u64>,
    spec_last_success: Option<i64>,
}

#[derive(Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

#[derive(Default, Clone)]
pub struct Gauges {
    pub queue_depth: usize,
    pub in_flight: usize,
    pub loaded_models: Vec<(String, u64)>,
    pub gpu_used_bytes: Option<u64>,
}

impl Metrics {
    pub fn request(&self, endpoint: &str, status: u16) {
        *self.inner.lock().unwrap().requests.entry((endpoint.to_string(), status)).or_default() += 1;
    }

    pub fn generation(&self, model: &str, prompt: u64, completion: u64, ttft: f64, decode_tokens: u64, decode_secs: f64) {
        let mut m = self.inner.lock().unwrap();
        *m.prompt_tokens.entry(model.to_string()).or_default() += prompt;
        *m.completion_tokens.entry(model.to_string()).or_default() += completion;
        if ttft > 0.0 {
            m.ttft.observe(ttft);
        }
        if decode_tokens > 0 && decode_secs > 0.0 {
            m.decode_tokens += decode_tokens;
            m.decode_secs += decode_secs;
            m.last_decode_tok_s = decode_tokens as f64 / decode_secs;
        }
    }

    pub fn cancelled(&self) {
        self.inner.lock().unwrap().cancelled += 1;
    }

    pub fn model_loaded(&self, model: &str) {
        *self.inner.lock().unwrap().model_loads.entry(model.to_string()).or_default() += 1;
    }

    pub fn model_load_failed(&self) {
        self.inner.lock().unwrap().model_load_failures += 1;
    }

    pub fn model_unloaded(&self) {
        self.inner.lock().unwrap().model_unloads += 1;
    }

    pub fn spec_check(&self, result: &str, success_at: Option<i64>) {
        let mut m = self.inner.lock().unwrap();
        *m.spec_checks.entry(result.to_string()).or_default() += 1;
        if success_at.is_some() {
            m.spec_last_success = success_at;
        }
    }

    pub fn spec_last_success(&self, at: Option<i64>) {
        self.inner.lock().unwrap().spec_last_success = at;
    }

    pub fn render(&self, g: &Gauges) -> String {
        let m = self.inner.lock().unwrap();
        let mut o = String::new();
        let _ = writeln!(o, "# HELP inference_requests_total Requests by endpoint and HTTP status.\n# TYPE inference_requests_total counter");
        for ((endpoint, status), n) in &m.requests {
            let _ = writeln!(o, "inference_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_prompt_tokens_total counter");
        for (model, n) in &m.prompt_tokens {
            let _ = writeln!(o, "inference_prompt_tokens_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_completion_tokens_total counter");
        for (model, n) in &m.completion_tokens {
            let _ = writeln!(o, "inference_completion_tokens_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# HELP inference_ttft_seconds Time to first token.\n# TYPE inference_ttft_seconds histogram");
        for (i, bound) in TTFT_BUCKETS.iter().enumerate() {
            let _ = writeln!(o, "inference_ttft_seconds_bucket{{le=\"{bound}\"}} {}", m.ttft.counts[i]);
        }
        let _ = writeln!(o, "inference_ttft_seconds_bucket{{le=\"+Inf\"}} {}", m.ttft.total);
        let _ = writeln!(o, "inference_ttft_seconds_sum {}\ninference_ttft_seconds_count {}", m.ttft.sum, m.ttft.total);
        let mean = if m.decode_secs > 0.0 { m.decode_tokens as f64 / m.decode_secs } else { 0.0 };
        let _ = writeln!(o, "# HELP inference_decode_tokens_per_second Decode speed.\n# TYPE inference_decode_tokens_per_second gauge");
        let _ = writeln!(o, "inference_decode_tokens_per_second{{window=\"last\"}} {}", m.last_decode_tok_s);
        let _ = writeln!(o, "inference_decode_tokens_per_second{{window=\"mean\"}} {mean}");
        let _ = writeln!(o, "# TYPE inference_decode_tokens_total counter\ninference_decode_tokens_total {}", m.decode_tokens);
        let _ = writeln!(o, "# TYPE inference_requests_cancelled_total counter\ninference_requests_cancelled_total {}", m.cancelled);
        let _ = writeln!(o, "# TYPE inference_model_loads_total counter");
        for (model, n) in &m.model_loads {
            let _ = writeln!(o, "inference_model_loads_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_model_load_failures_total counter\ninference_model_load_failures_total {}", m.model_load_failures);
        let _ = writeln!(o, "# TYPE inference_model_unloads_total counter\ninference_model_unloads_total {}", m.model_unloads);
        let _ = writeln!(o, "# HELP inference_specs_checks_total Spec update checks by result.\n# TYPE inference_specs_checks_total counter");
        for (result, n) in &m.spec_checks {
            let _ = writeln!(o, "inference_specs_checks_total{{result=\"{result}\"}} {n}");
        }
        let _ = writeln!(
            o,
            "# HELP inference_specs_last_success_timestamp_seconds Unix time of the last successful spec update check, 0 if none.\n# TYPE inference_specs_last_success_timestamp_seconds gauge\ninference_specs_last_success_timestamp_seconds {}",
            m.spec_last_success.unwrap_or(0)
        );
        let _ = writeln!(o, "# HELP inference_queue_depth Requests waiting for a slot.\n# TYPE inference_queue_depth gauge\ninference_queue_depth {}", g.queue_depth);
        let _ = writeln!(o, "# TYPE inference_in_flight gauge\ninference_in_flight {}", g.in_flight);
        let _ = writeln!(o, "# TYPE inference_models_loaded gauge\ninference_models_loaded {}", g.loaded_models.len());
        let _ = writeln!(o, "# HELP inference_model_vram_bytes Device memory held by a loaded model.\n# TYPE inference_model_vram_bytes gauge");
        for (model, bytes) in &g.loaded_models {
            let _ = writeln!(o, "inference_model_vram_bytes{{model=\"{model}\"}} {bytes}");
        }
        if let Some(used) = g.gpu_used_bytes {
            let _ = writeln!(o, "# HELP inference_gpu_memory_used_bytes GPU memory in use on device 0.\n# TYPE inference_gpu_memory_used_bytes gauge\ninference_gpu_memory_used_bytes {used}");
        }
        o
    }
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative() {
        let m = Metrics::default();
        m.generation("m", 3, 5, 0.03, 4, 0.2);
        m.generation("m", 3, 5, 3.0, 4, 0.2);
        let text = m.render(&Gauges::default());
        assert!(text.contains("inference_ttft_seconds_bucket{le=\"0.05\"} 1"), "{text}");
        assert!(text.contains("inference_ttft_seconds_bucket{le=\"5\"} 2"), "{text}");
        assert!(text.contains("inference_ttft_seconds_count 2"));
        assert!(text.contains("inference_completion_tokens_total{model=\"m\"} 10"));
        assert!(text.contains("inference_decode_tokens_per_second{window=\"mean\"} 20"));
    }

    #[test]
    fn percentiles() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 0.95), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
