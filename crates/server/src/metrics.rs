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
    finished: BTreeMap<(String, String), u64>,
    thinking_wrapups: u64,
    model_loads: BTreeMap<String, u64>,
    model_load_failures: u64,
    model_unloads: u64,
    spec_checks: BTreeMap<String, u64>,
    spec_last_success: Option<i64>,
    image_jobs: BTreeMap<String, u64>,
    images: BTreeMap<String, u64>,
    image_steps: u64,
    image_step_secs: f64,
    image_seconds: f64,
    last_image_steps_per_sec: f64,
    duplex_sessions: BTreeMap<String, u64>,
    duplex_refused: BTreeMap<String, u64>,
    duplex_frames_in: u64,
    duplex_frames_out: u64,
    duplex_late: u64,
    transcriptions: BTreeMap<String, (u64, f64, f64)>,
    speeches: BTreeMap<String, (u64, f64, f64)>,
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
    pub engine: String,
    pub image_queue: usize,
    pub image_running: usize,
    pub duplex_sessions: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Summary {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_p50_secs: Option<f64>,
}

impl Metrics {
    pub fn summary(&self) -> Summary {
        let m = self.inner.lock().unwrap();
        let ttft_p50_secs = (m.ttft.total > 0)
            .then(|| {
                let half = m.ttft.total.div_ceil(2);
                TTFT_BUCKETS.iter().zip(m.ttft.counts.iter()).find(|(_, c)| **c >= half).map(|(b, _)| *b)
            })
            .flatten()
            .or((m.ttft.total > 0).then(|| m.ttft.sum / m.ttft.total as f64));
        Summary {
            requests: m.requests.values().sum(),
            prompt_tokens: m.prompt_tokens.values().sum(),
            completion_tokens: m.completion_tokens.values().sum(),
            ttft_p50_secs,
        }
    }

    pub fn request(&self, endpoint: &str, status: u16) {
        *self.inner.lock().unwrap().requests.entry((endpoint.to_string(), status)).or_default() += 1;
    }

    pub fn finished(&self, model: &str, reason: &str) {
        *self.inner.lock().unwrap().finished.entry((model.to_string(), reason.to_string())).or_default() += 1;
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

    pub fn thinking_wrapup(&self) {
        self.inner.lock().unwrap().thinking_wrapups += 1;
    }

    pub fn cancelled(&self) {
        self.inner.lock().unwrap().cancelled += 1;
    }

    pub fn transcription(&self, model: &str, audio_secs: f64, secs: f64) {
        let mut m = self.inner.lock().unwrap();
        let e = m.transcriptions.entry(model.to_string()).or_default();
        e.0 += 1;
        e.1 += audio_secs;
        e.2 += secs;
    }

    pub fn speech(&self, model: &str, audio_secs: f64, secs: f64) {
        let mut m = self.inner.lock().unwrap();
        let e = m.speeches.entry(model.to_string()).or_default();
        e.0 += 1;
        e.1 += audio_secs;
        e.2 += secs;
    }

    pub fn image_job(&self, model: &str, images: u64, steps: u64, step_secs: f64, total_secs: f64) {
        let mut m = self.inner.lock().unwrap();
        *m.image_jobs.entry(model.to_string()).or_default() += 1;
        *m.images.entry(model.to_string()).or_default() += images;
        m.image_steps += steps;
        m.image_step_secs += step_secs;
        m.image_seconds += total_secs;
        if step_secs > 0.0 {
            m.last_image_steps_per_sec = steps as f64 / step_secs;
        }
    }

    pub fn duplex_session(&self, model: &str) {
        *self.inner.lock().unwrap().duplex_sessions.entry(model.to_string()).or_default() += 1;
    }

    pub fn duplex_refused(&self, reason: &str) {
        *self.inner.lock().unwrap().duplex_refused.entry(reason.to_string()).or_default() += 1;
    }

    pub fn duplex_frames(&self, frames_in: u64, frames_out: u64, late: u64) {
        let mut m = self.inner.lock().unwrap();
        m.duplex_frames_in += frames_in;
        m.duplex_frames_out += frames_out;
        m.duplex_late += late;
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
        let _ = writeln!(o, "# HELP inference_tool_call_repairs_total Tool calls and arguments the parser recovered, by kind.\n# TYPE inference_tool_call_repairs_total counter");
        for (kind, n) in crate::tools::repair_counts() {
            let _ = writeln!(o, "inference_tool_call_repairs_total{{kind=\"{kind}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_thinking_wrapups_total counter\ninference_thinking_wrapups_total {}", m.thinking_wrapups);
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
        o.push_str(&g.engine);
        let _ = writeln!(o, "# HELP inference_transcriptions_total Transcriptions finished, by model.\n# TYPE inference_transcriptions_total counter");
        for (model, (n, _, _)) in &m.transcriptions {
            let _ = writeln!(o, "inference_transcriptions_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# HELP inference_transcription_audio_seconds_total Seconds of audio transcribed, by model.\n# TYPE inference_transcription_audio_seconds_total counter");
        for (model, (_, audio, _)) in &m.transcriptions {
            let _ = writeln!(o, "inference_transcription_audio_seconds_total{{model=\"{model}\"}} {audio}");
        }
        let _ = writeln!(o, "# HELP inference_transcription_seconds_total Seconds spent transcribing, by model.\n# TYPE inference_transcription_seconds_total counter");
        for (model, (_, _, spent)) in &m.transcriptions {
            let _ = writeln!(o, "inference_transcription_seconds_total{{model=\"{model}\"}} {spent}");
        }
        let _ = writeln!(o, "# HELP inference_speech_total Speech requests finished, by model.\n# TYPE inference_speech_total counter");
        for (model, (n, _, _)) in &m.speeches {
            let _ = writeln!(o, "inference_speech_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# HELP inference_speech_audio_seconds_total Seconds of audio spoken, by model.\n# TYPE inference_speech_audio_seconds_total counter");
        for (model, (_, audio, _)) in &m.speeches {
            let _ = writeln!(o, "inference_speech_audio_seconds_total{{model=\"{model}\"}} {audio}");
        }
        let _ = writeln!(o, "# HELP inference_speech_seconds_total Seconds spent speaking, by model.\n# TYPE inference_speech_seconds_total counter");
        for (model, (_, _, spent)) in &m.speeches {
            let _ = writeln!(o, "inference_speech_seconds_total{{model=\"{model}\"}} {spent}");
        }
        let _ = writeln!(o, "# HELP inference_image_jobs_total Image generation jobs finished, by model.\n# TYPE inference_image_jobs_total counter");
        for (model, n) in &m.image_jobs {
            let _ = writeln!(o, "inference_image_jobs_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_images_total counter");
        for (model, n) in &m.images {
            let _ = writeln!(o, "inference_images_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_image_steps_total counter\ninference_image_steps_total {}", m.image_steps);
        let _ = writeln!(o, "# TYPE inference_image_step_seconds_total counter\ninference_image_step_seconds_total {}", m.image_step_secs);
        let _ = writeln!(o, "# TYPE inference_image_job_seconds_total counter\ninference_image_job_seconds_total {}", m.image_seconds);
        let _ = writeln!(o, "# HELP inference_image_steps_per_second Denoiser steps per second of the last job.\n# TYPE inference_image_steps_per_second gauge\ninference_image_steps_per_second {}", m.last_image_steps_per_sec);
        let _ = writeln!(o, "# HELP inference_image_queue_depth Image jobs waiting for a GPU.\n# TYPE inference_image_queue_depth gauge\ninference_image_queue_depth {}", g.image_queue);
        let _ = writeln!(o, "# TYPE inference_image_jobs_running gauge\ninference_image_jobs_running {}", g.image_running);
        let _ = writeln!(o, "# HELP inference_duplex_sessions_total Duplex sessions started, by model.\n# TYPE inference_duplex_sessions_total counter");
        for (model, n) in &m.duplex_sessions {
            let _ = writeln!(o, "inference_duplex_sessions_total{{model=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_duplex_refused_total counter");
        for (reason, n) in &m.duplex_refused {
            let _ = writeln!(o, "inference_duplex_refused_total{{reason=\"{reason}\"}} {n}");
        }
        let _ = writeln!(o, "# TYPE inference_duplex_frames_in_total counter\ninference_duplex_frames_in_total {}", m.duplex_frames_in);
        let _ = writeln!(o, "# TYPE inference_duplex_frames_out_total counter\ninference_duplex_frames_out_total {}", m.duplex_frames_out);
        let _ = writeln!(o, "# HELP inference_duplex_late_frames_total Model frames that missed their slot.\n# TYPE inference_duplex_late_frames_total counter\ninference_duplex_late_frames_total {}", m.duplex_late);
        let _ = writeln!(o, "# TYPE inference_duplex_sessions_open gauge\ninference_duplex_sessions_open {}", g.duplex_sessions);
        self.render_vllm(&mut o, &m, g);
        o
    }
}

impl Metrics {
    fn render_vllm(&self, o: &mut String, m: &Inner, g: &Gauges) {
        let models: Vec<&str> = match g.loaded_models.is_empty() {
            true => m.prompt_tokens.keys().map(String::as_str).collect(),
            false => g.loaded_models.iter().map(|(id, _)| id.as_str()).collect(),
        };
        let first = models.first().copied().unwrap_or("");
        let _ = writeln!(o, "# HELP vllm:num_requests_running Number of requests in model execution batches.\n# TYPE vllm:num_requests_running gauge");
        let _ = writeln!(o, "vllm:num_requests_running{{engine=\"0\",model_name=\"{first}\"}} {}", g.in_flight);
        let _ = writeln!(o, "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n# TYPE vllm:num_requests_waiting gauge");
        let _ = writeln!(o, "vllm:num_requests_waiting{{engine=\"0\",model_name=\"{first}\"}} {}", g.queue_depth);
        let _ = writeln!(o, "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n# TYPE vllm:prompt_tokens_total counter");
        for (model, n) in &m.prompt_tokens {
            let _ = writeln!(o, "vllm:prompt_tokens_total{{engine=\"0\",model_name=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# HELP vllm:generation_tokens_total Number of generation tokens processed.\n# TYPE vllm:generation_tokens_total counter");
        for (model, n) in &m.completion_tokens {
            let _ = writeln!(o, "vllm:generation_tokens_total{{engine=\"0\",model_name=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# HELP vllm:request_success_total Count of successfully processed requests.\n# TYPE vllm:request_success_total counter");
        for ((model, reason), n) in &m.finished {
            let _ = writeln!(o, "vllm:request_success_total{{engine=\"0\",finished_reason=\"{reason}\",model_name=\"{model}\"}} {n}");
        }
        let _ = writeln!(o, "# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.\n# TYPE vllm:time_to_first_token_seconds histogram");
        let labels = format!("engine=\"0\",model_name=\"{first}\"");
        for (i, bound) in TTFT_BUCKETS.iter().enumerate() {
            let _ = writeln!(o, "vllm:time_to_first_token_seconds_bucket{{{labels},le=\"{bound}\"}} {}", m.ttft.counts[i]);
        }
        let _ = writeln!(o, "vllm:time_to_first_token_seconds_bucket{{{labels},le=\"+Inf\"}} {}", m.ttft.total);
        let _ = writeln!(o, "vllm:time_to_first_token_seconds_sum{{{labels}}} {}\nvllm:time_to_first_token_seconds_count{{{labels}}} {}", m.ttft.sum, m.ttft.total);
    }
}

pub fn label_vllm(text: &str, model: &str) -> String {
    let labels = format!("engine=\"0\",model_name=\"{model}\"");
    let mut out = String::with_capacity(text.len() + 64);
    for line in text.lines() {
        if !line.starts_with("vllm:") {
            out.push_str(line);
        } else if let Some(open) = line.find('{') {
            out.push_str(&line[..=open]);
            out.push_str(&labels);
            out.push(',');
            out.push_str(&line[open + 1..]);
        } else if let Some(space) = line.find(' ') {
            out.push_str(&line[..space]);
            out.push_str(&format!("{{{labels}}}"));
            out.push_str(&line[space..]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
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
    fn vllm_named_series_carry_the_dashboard_labels() {
        let m = Metrics::default();
        m.generation("qwen", 3, 5, 0.03, 4, 0.2);
        m.finished("qwen", "stop");
        let g = Gauges { queue_depth: 2, in_flight: 1, loaded_models: vec![("qwen".into(), 1)], ..Gauges::default() };
        let text = m.render(&g);
        for want in [
            "vllm:num_requests_running{engine=\"0\",model_name=\"qwen\"} 1",
            "vllm:num_requests_waiting{engine=\"0\",model_name=\"qwen\"} 2",
            "vllm:prompt_tokens_total{engine=\"0\",model_name=\"qwen\"} 3",
            "vllm:generation_tokens_total{engine=\"0\",model_name=\"qwen\"} 5",
            "vllm:request_success_total{engine=\"0\",finished_reason=\"stop\",model_name=\"qwen\"} 1",
            "vllm:time_to_first_token_seconds_bucket{engine=\"0\",model_name=\"qwen\",le=\"0.05\"} 1",
            "vllm:time_to_first_token_seconds_count{engine=\"0\",model_name=\"qwen\"} 1",
        ] {
            assert!(text.contains(want), "{want} missing from\n{text}");
        }
    }

    #[test]
    fn engine_series_get_the_model_label_added() {
        let text = label_vllm("# HELP vllm:a x\nvllm:a 1\nvllm:h_bucket{le=\"0.1\"} 2\ninference_x 3\n", "m");
        assert!(text.contains("vllm:a{engine=\"0\",model_name=\"m\"} 1"), "{text}");
        assert!(text.contains("vllm:h_bucket{engine=\"0\",model_name=\"m\",le=\"0.1\"} 2"), "{text}");
        assert!(text.contains("\ninference_x 3"), "{text}");
        assert!(text.contains("# HELP vllm:a x"));
    }

    #[test]
    fn percentiles() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 0.95), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
