use crate::chat::ChatRequest;
use crate::native::engine_error;
use crate::result::{BackendError, Generation};
use crate::target::{Capabilities, Request, Target};
use specrt::runtime::Sampling;
use specrt::sched::Finish;
use specrt::serve::{Histogram, ServeOptions, Serving, Stats};
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub struct ServingTarget {
    opts: ServeOptions,
    serving: Option<Serving>,
}

impl ServingTarget {
    pub fn new(opts: ServeOptions) -> Self {
        Self { opts, serving: None }
    }

    fn engine(&self) -> Result<&Serving, BackendError> {
        self.serving.as_ref().ok_or_else(|| BackendError::failed("no model is loaded"))
    }

    fn run(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: Sampling,
        streaming: bool,
        cancel: Option<Arc<AtomicBool>>,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let engine = self.engine()?;
        let cancel = cancel.unwrap_or_default();
        let c = engine
            .complete(prompt, max_tokens, sampling, streaming, cancel, &mut |t| {
                on_text(t);
                true
            })
            .map_err(|e| engine_error(&e))?;
        if c.finish == Some(Finish::Cancelled) {
            return Err(BackendError::new(crate::result::ErrorKind::Cancelled, "cancelled"));
        }
        Ok(Generation {
            text: c.text,
            prompt_tokens: c.prompt_tokens as u64,
            completion_tokens: c.completion_tokens as u64,
            ttft_secs: c.ttft_secs,
            prefill_secs: c.ttft_secs,
            decode_secs: c.decode_secs,
            peak_memory_bytes: None,
        })
    }
}

fn sampling(temperature: f32, seed: Option<u64>) -> Sampling {
    Sampling { temperature: temperature.max(0.0), seed: seed.unwrap_or(0) as u32 }
}

impl Target for ServingTarget {
    fn name(&self) -> &str {
        "specrt"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { streaming: true, logits: false, memory: false }
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        self.serving = None;
        let s = Serving::load(model, self.opts).map_err(|e| engine_error(&e))?;
        self.serving = Some(s);
        Ok(())
    }

    fn generate(&mut self, request: &Request, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let ids = self.engine()?.tokens(&request.prompt).map_err(|e| engine_error(&e))?;
        self.run(ids, request.max_tokens, Sampling::GREEDY, false, None, on_text)
    }

    fn chat(&mut self, request: &ChatRequest, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let list = request.messages_json().as_array().cloned().unwrap_or_default();
        let ids = self.engine()?.encode_messages(&list, request.use_template).map_err(|e| engine_error(&e))?;
        let s = request.sampling;
        self.run(ids, s.max_tokens, sampling(s.temperature, s.seed), true, request.cancel.clone(), on_text)
    }

    fn share(&self) -> Option<Box<dyn Target>> {
        let serving = self.serving.clone()?;
        Some(Box::new(ServingTarget { opts: self.opts, serving: Some(serving) }))
    }

    fn metrics(&self) -> Option<String> {
        self.serving.as_ref().map(|s| render(&s.stats()))
    }
}

fn histogram(o: &mut String, name: &str, help: &str, h: &Histogram) {
    let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} histogram");
    for (b, c) in h.bounds.iter().zip(&h.counts) {
        let _ = writeln!(o, "{name}_bucket{{le=\"{b}\"}} {c}");
    }
    let _ = writeln!(o, "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {}\n{name}_count {}", h.total, h.sum, h.total);
}

pub fn render(s: &Stats) -> String {
    let mut o = String::new();
    let gauge = |o: &mut String, name: &str, help: &str, v: f64| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
    };
    gauge(&mut o, "inference_running_seqs", "Sequences in the running batch.", s.running as f64);
    gauge(&mut o, "inference_waiting_seqs", "Sequences waiting for admission.", s.waiting as f64);
    gauge(&mut o, "inference_kv_blocks_used", "KV cache blocks in use.", s.kv_blocks_used as f64);
    gauge(&mut o, "inference_kv_blocks_total", "KV cache blocks in the pool.", s.kv_blocks_total as f64);
    gauge(&mut o, "inference_kv_block_tokens", "Tokens per KV cache block.", s.kv_block_tokens as f64);
    let _ = writeln!(
        o,
        "# HELP inference_preemptions_total Sequences preempted for KV space.\n# TYPE inference_preemptions_total counter\ninference_preemptions_total {}",
        s.preemptions
    );
    let _ = writeln!(o, "# TYPE inference_steps_total counter\ninference_steps_total {}", s.steps);
    histogram(&mut o, "inference_queue_seconds", "Time from arrival to the first scheduled step.", &s.queue_secs);
    histogram(&mut o, "inference_inter_token_seconds", "Time between streamed tokens of one sequence.", &s.inter_token_secs);
    histogram(&mut o, "inference_batch_size", "Decode rows per step.", &s.decode_batch);
    histogram(&mut o, "inference_prefill_tokens_per_step", "Prompt tokens prefilled per step.", &s.prefill_tokens_per_step);
    histogram(&mut o, "inference_step_seconds", "Wall time of one scheduler step.", &s.step_secs);
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_stats_render_as_prometheus_text() {
        let mut s = Stats { running: 3, waiting: 2, kv_blocks_used: 40, kv_blocks_total: 100, preemptions: 1, ..Stats::default() };
        s.inter_token_secs.observe(0.03);
        s.inter_token_secs.observe(0.3);
        let t = render(&s);
        for want in [
            "inference_running_seqs 3",
            "inference_waiting_seqs 2",
            "inference_kv_blocks_used 40",
            "inference_preemptions_total 1",
            "inference_inter_token_seconds_bucket{le=\"0.035\"} 1",
            "inference_inter_token_seconds_bucket{le=\"+Inf\"} 2",
            "inference_prefill_tokens_per_step_count 0",
            "inference_batch_size_count 0",
            "inference_queue_seconds_count 0",
        ] {
            assert!(t.contains(want), "{want} missing from\n{t}");
        }
    }
}
