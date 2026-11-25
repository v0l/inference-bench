use crate::chat::ChatRequest;
use crate::native::engine_error;
use crate::result::{BackendError, Generation};
use crate::target::{Capabilities, Request, Target};
use specrt::runtime::Sampling;
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
        salt: String,
        cancel: Option<Arc<AtomicBool>>,
        media: Vec<specrt::sched::MediaInput>,
        vision_tokens: usize,
        on_logprob: Option<crate::chat::LogprobSink>,
        schema: Option<&specrt::grammar::Schema>,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, BackendError> {
        let engine = self.engine()?;
        let cancel = cancel.unwrap_or_default();
        let c = engine
            .complete_with_logprobs(prompt, media, max_tokens, sampling, streaming, salt, cancel, schema, &mut |t| {
                on_text(t);
                true
            }, &mut |lp| {
                if let Some(sink) = &on_logprob {
                    sink(lp);
                }
            })
            .map_err(|e| engine_error(&e))?;
        Ok(Generation {
            text: c.text,
            prompt_tokens: c.prompt_tokens as u64,
            vision_tokens: vision_tokens as u64,
            cached_tokens: c.cached_tokens as u64,
            completion_tokens: c.completion_tokens as u64,
            ttft_secs: c.ttft_secs,
            prefill_secs: c.prefill_secs,
            queue_secs: c.queue_secs,
            decode_secs: c.decode_secs,
            peak_memory_bytes: None,
        })
    }
}

impl Target for ServingTarget {
    fn name(&self) -> &str {
        "specrt"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { streaming: true, logits: false, memory: false, vision: self.serving.as_ref().is_some_and(|s| s.accepts_media()), thinking: self.serving.as_ref().map(|s| s.thinking()).unwrap_or_default(), tools: self.serving.as_ref().map(|s| s.tool_format()).unwrap_or_default(), logprobs: true, batched: true, ..Capabilities::default() }
    }

    fn media_refusal(&self) -> Option<String> {
        self.serving.as_ref().and_then(|s| s.media_refusal())
    }

    fn set_progress(&mut self, sink: Option<specrt::progress::Sink>) {
        self.opts.progress = sink;
    }

    fn load(&mut self, model: &str) -> Result<(), BackendError> {
        self.serving = None;
        let s = Serving::load(model, self.opts.clone()).map_err(|e| engine_error(&e))?;
        self.serving = Some(s);
        Ok(())
    }

    fn generate(&mut self, request: &Request, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let ids = self.engine()?.tokens(&request.prompt).map_err(|e| engine_error(&e))?;
        self.run(ids, request.max_tokens, Sampling::GREEDY, false, String::new(), None, Vec::new(), 0, None, None, on_text)
    }

    fn chat(&mut self, request: &ChatRequest, on_text: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        let list = request.messages_json().as_array().cloned().unwrap_or_default();
        let (ids, media, vision_tokens) = if request.media.is_empty() {
            (self.engine()?.encode_messages(&list, request.template()).map_err(|e| engine_error(&e))?, Vec::new(), 0)
        } else {
            let budget = request.media_budget();
            self.engine()?
                .encode_media_messages(&list, request.template(), &request.media, budget)
                .map_err(|e| BackendError::new(crate::result::ErrorKind::InvalidRequest, format!("{e:#}")))?
        };
        let mut ids = ids;
        if let Some(tail) = request.continuation.as_deref().filter(|t| !t.is_empty()) {
            ids.extend(self.engine()?.tokens(tail).map_err(|e| engine_error(&e))?);
        }
        let s = request.sampling;
        let sampling = s.resolve(self.engine()?.sampling_defaults()).with_logprobs(request.logprobs);
        self.run(ids, s.max_tokens, sampling, !request.ignore_eos, request.cache_salt.clone().unwrap_or_default(), request.cancel.clone(), media, vision_tokens, request.on_logprob.clone(), request.response_format.as_ref(), on_text)
    }

    fn share(&self) -> Option<Box<dyn Target>> {
        let serving = self.serving.clone()?;
        Some(Box::new(ServingTarget { opts: self.opts.clone(), serving: Some(serving) }))
    }

    fn metrics(&self) -> Option<String> {
        self.serving.as_ref().map(|s| render(&s.stats()))
    }

    fn requires_serving(&self) -> bool {
        self.opts.solved.is_some() || self.opts.placement.as_ref().is_some_and(|p| !p.devices.is_empty())
    }

    fn failure(&self) -> Option<String> {
        self.serving.as_ref().and_then(|s| s.failure())
    }

    fn stats(&self) -> Option<Stats> {
        self.serving.as_ref().map(|s| s.stats())
    }

    fn sessions(&self) -> Option<std::sync::Arc<specrt::sessions::Shared>> {
        self.serving.as_ref()?.sessions()
    }

    fn props(&self) -> Option<serde_json::Value> {
        self.serving.as_ref().map(|s| s.props())
    }

    fn speculation(&self) -> Option<String> {
        let spec = self.serving.as_ref()?.spec();
        Some(match (&spec.mtp, spec.speculative()) {
            (None, _) => "off: the model has no MTP layer".to_string(),
            (Some(_), None) => "off: turned off in the engine options".to_string(),
            (Some(_), Some(m)) => {
                let head = spec.draft_head_quant().map_or("shared lm_head".to_string(), |q| format!("{q:?} head").to_lowercase());
                format!("on: {} draft tokens, {head}", m.draft_tokens)
            }
        })
    }
}

fn histogram(o: &mut String, name: &str, help: &str, h: &Histogram) {
    let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} histogram");
    for (b, c) in h.bounds.iter().zip(&h.counts) {
        let _ = writeln!(o, "{name}_bucket{{le=\"{b}\"}} {c}");
    }
    let _ = writeln!(o, "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {}\n{name}_count {}", h.total, h.sum, h.total);
}

fn chain(o: &mut String, c: &specrt::sched::ChainMetrics) {
    let _ = writeln!(o, "# HELP inference_pipeline_stages Stages in the pipeline chain.\n# TYPE inference_pipeline_stages gauge\ninference_pipeline_stages {}", c.stages.len());
    let per_stage: [(&str, &str, &str, fn(&specrt::sched::StageMetric) -> f64); 3] = [
        ("inference_pipeline_stage_busy_seconds_total", "counter", "Time a stage spent computing.", |m| m.busy_secs),
        ("inference_pipeline_stage_calls_total", "counter", "Prefill and decode calls a stage ran.", |m| m.calls as f64),
        ("inference_pipeline_stage_tokens_total", "counter", "Tokens a stage processed.", |m| m.tokens as f64),
    ];
    for (name, kind, help, get) in per_stage {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}");
        for (i, m) in c.stages.iter().enumerate() {
            let _ = writeln!(o, "{name}{{stage=\"{i}\",device=\"{}\",layers=\"{}-{}\"}} {}", if m.name.is_empty() { m.device.to_string() } else { m.name.clone() }, m.start, m.end, get(m));
        }
    }
    let _ = writeln!(o, "# HELP inference_pipeline_stage_lost A stage left the chain and the chain was dropped.\n# TYPE inference_pipeline_stage_lost gauge\ninference_pipeline_stage_lost {}", c.lost.is_some() as u8);
    let _ = writeln!(o, "# TYPE inference_pipeline_hops_total counter\ninference_pipeline_hops_total {}", c.hops);
    let _ = writeln!(o, "# TYPE inference_pipeline_hop_bytes_total counter\ninference_pipeline_hop_bytes_total {}", c.hop_bytes);
    let _ = writeln!(o, "# TYPE inference_pipeline_hop_seconds_total counter\ninference_pipeline_hop_seconds_total {}", c.hop_secs);
    let _ = writeln!(o, "# TYPE inference_pipeline_prefill_micro_batches_total counter\ninference_pipeline_prefill_micro_batches_total {}", c.micro_batches);
    let _ = writeln!(o, "# TYPE inference_pipeline_overlapped_prefills_total counter\ninference_pipeline_overlapped_prefills_total {}", c.overlapped_prefills);
}

fn vllm_names(o: &mut String, s: &Stats) {
    let usage = if s.kv_blocks_total > 0 { s.kv_blocks_used as f64 / s.kv_blocks_total as f64 } else { 0.0 };
    let one = |o: &mut String, name: &str, kind: &str, help: &str, v: f64| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
    };
    one(o, "vllm:kv_cache_usage_perc", "gauge", "KV cache usage, 1 means 100 percent.", usage);
    one(o, "vllm:num_preemptions_total", "counter", "Cumulative preemptions from the engine.", s.preemptions as f64);
    one(o, "vllm:prefix_cache_queries_total", "counter", "Prefix cache queries, in tokens.", s.prefix_queries as f64);
    one(o, "vllm:prefix_cache_hits_total", "counter", "Prefix cache hits, in tokens.", s.prefix_hits as f64);
    one(o, "vllm:spec_decode_num_draft_tokens_total", "counter", "Number of draft tokens.", s.spec_drafted as f64);
    one(o, "vllm:spec_decode_num_accepted_tokens_total", "counter", "Number of accepted tokens.", s.spec_accepted as f64);
    histogram(o, "vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", &s.inter_token_secs);
    histogram(o, "vllm:request_queue_time_seconds", "Histogram of time spent in the WAITING phase for a request.", &s.queue_secs);
}

fn sessions(o: &mut String, s: &Stats) {
    let Some(x) = &s.sessions else { return };
    let mut put = |name: &str, kind: &str, help: &str, line: String| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{line}");
    };
    put("inference_session_snapshots_total", "counter", "Session snapshots written to disk.", format!("inference_session_snapshots_total {}", x.snapshots));
    put("inference_session_snapshot_failures_total", "counter", "Session snapshots that could not be written.", format!("inference_session_snapshot_failures_total {}", x.snapshot_failures));
    put("inference_session_snapshot_bytes", "gauge", "Bytes of session snapshots on disk.", format!("inference_session_snapshot_bytes {}", x.stored_bytes));
    put("inference_session_snapshots_stored", "gauge", "Session snapshots on disk.", format!("inference_session_snapshots_stored {}", x.stored));
    put(
        "inference_session_restores_total",
        "counter",
        "Session restores by result.",
        format!(
            "inference_session_restores_total{{result=\"ok\"}} {}\ninference_session_restores_total{{result=\"corrupt\"}} {}\ninference_session_restores_total{{result=\"miss\"}} {}",
            x.restores_ok, x.restores_corrupt, x.restores_miss
        ),
    );
    put("inference_session_restore_seconds", "counter", "Seconds spent reading session snapshots.", format!("inference_session_restore_seconds {}", x.restore_seconds));
    put("inference_session_prefill_tokens_saved_total", "counter", "Prompt tokens not prefilled thanks to a restored session.", format!("inference_session_prefill_tokens_saved_total {}", x.prefill_tokens_saved));
}

pub fn render(s: &Stats) -> String {
    let mut o = String::new();
    let gauge = |o: &mut String, name: &str, help: &str, v: f64| {
        let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
    };
    gauge(&mut o, "inference_running_seqs", "Sequences in the running batch.", s.running as f64);
    gauge(&mut o, "inference_waiting_seqs", "Sequences waiting for admission.", s.waiting as f64);
    gauge(&mut o, "inference_kv_blocks_used", "KV cache blocks held by running sequences.", s.kv_blocks_used as f64);
    gauge(&mut o, "inference_kv_blocks_cached", "KV cache blocks held only by the prefix cache, evicted on demand.", s.kv_blocks_cached as f64);
    gauge(&mut o, "inference_kv_blocks_total", "KV cache blocks in the pool.", s.kv_blocks_total as f64);
    gauge(&mut o, "inference_kv_block_tokens", "Tokens per KV cache block.", s.kv_block_tokens as f64);
    let _ = writeln!(
        o,
        "# HELP inference_preemptions_total Sequences preempted for KV space.\n# TYPE inference_preemptions_total counter\ninference_preemptions_total {}",
        s.preemptions
    );
    let _ = writeln!(
        o,
        "# HELP specrt_preemptions_total Preemptions by how the sequence was freed.\n# TYPE specrt_preemptions_total counter\nspecrt_preemptions_total{{kind=\"swap\"}} {}\nspecrt_preemptions_total{{kind=\"recompute\"}} {}",
        s.swap_preemptions,
        s.preemptions.saturating_sub(s.swap_preemptions)
    );
    let _ = writeln!(o, "# TYPE specrt_swapped_bytes_total counter\nspecrt_swapped_bytes_total {}", s.swapped_bytes_total);
    let _ = writeln!(o, "# TYPE specrt_swap_seconds counter\nspecrt_swap_seconds {}", s.swap_secs);
    let _ = writeln!(o, "# TYPE specrt_reprefilled_tokens_total counter\nspecrt_reprefilled_tokens_total {}", s.reprefilled_tokens);
    let _ = writeln!(o, "# TYPE inference_steps_total counter\ninference_steps_total {}", s.steps);
    let _ = writeln!(
        o,
        "# HELP inference_prefix_cache_queries_total Prompt tokens looked up in the prefix cache.\n# TYPE inference_prefix_cache_queries_total counter\ninference_prefix_cache_queries_total {}",
        s.prefix_queries
    );
    let _ = writeln!(
        o,
        "# HELP inference_prefix_cache_hit_tokens_total Prompt tokens served from the prefix cache.\n# TYPE inference_prefix_cache_hit_tokens_total counter\ninference_prefix_cache_hit_tokens_total {}",
        s.prefix_hits
    );
    let rate = if s.prefix_queries > 0 { s.prefix_hits as f64 / s.prefix_queries as f64 } else { 0.0 };
    gauge(&mut o, "inference_prefix_cache_hit_rate", "Share of looked up prompt tokens that hit the prefix cache.", rate);
    gauge(&mut o, "inference_prefix_cache_blocks", "KV blocks held by the prefix cache.", s.prefix_blocks as f64);
    gauge(&mut o, "inference_prefix_cache_snapshots", "Recurrent state snapshots held by the prefix cache.", s.prefix_snapshots as f64);
    sessions(&mut o, s);
    if let Some(off) = &s.offload {
        gauge(&mut o, "inference_offload_hit_ratio", "Share of weight page requests served from GPU memory.", off.hit_ratio());
        let _ = writeln!(o, "# HELP inference_offload_read_bytes_total Weight bytes streamed to the card or read from disk.\n# TYPE inference_offload_read_bytes_total counter\ninference_offload_read_bytes_total {}", off.host_to_vram_bytes + off.nvme_read_bytes);
    }
    gauge(&mut o, "inference_prefix_cache_snapshot_bytes", "Host bytes of the recurrent state snapshots.", s.prefix_snapshot_bytes as f64);
    let _ = writeln!(
        o,
        "# HELP inference_spec_drafted_tokens_total MTP draft tokens proposed for verification.\n# TYPE inference_spec_drafted_tokens_total counter\ninference_spec_drafted_tokens_total {}",
        s.spec_drafted
    );
    let _ = writeln!(
        o,
        "# HELP inference_spec_accepted_tokens_total MTP draft tokens the target accepted.\n# TYPE inference_spec_accepted_tokens_total counter\ninference_spec_accepted_tokens_total {}",
        s.spec_accepted
    );
    vllm_names(&mut o, s);
    if let Some(c) = &s.chain {
        chain(&mut o, c);
    }
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
    fn offload_stats_render_the_two_issue_series() {
        let s = Stats { offload: Some(specrt::offload::OffloadStats { vram_hits: 3, nvme_reads: 1, nvme_read_bytes: 7, host_to_vram_bytes: 5, ..Default::default() }), ..Stats::default() };
        let t = render(&s);
        assert!(t.contains("inference_offload_hit_ratio 0.75"), "{t}");
        assert!(t.contains("inference_offload_read_bytes_total 12"), "{t}");
        assert!(!render(&Stats::default()).contains("inference_offload"));
    }

    #[test]
    fn session_counters_render_with_the_documented_names() {
        let s = Stats { sessions: Some(specrt::sessions::Stats { snapshots: 3, restores_ok: 2, restores_corrupt: 1, stored_bytes: 99, prefill_tokens_saved: 7000, restore_seconds: 1.5, ..Default::default() }), ..Stats::default() };
        let text = render(&s);
        for want in [
            "inference_session_snapshots_total 3",
            "inference_session_restores_total{result=\"ok\"} 2",
            "inference_session_restores_total{result=\"corrupt\"} 1",
            "inference_session_snapshot_bytes 99",
            "inference_session_restore_seconds 1.5",
            "inference_session_prefill_tokens_saved_total 7000",
        ] {
            assert!(text.contains(want), "{want}");
        }
        assert!(!render(&Stats::default()).contains("inference_session"));
    }

    #[test]
    fn serving_stats_render_as_prometheus_text() {
        let mut s = Stats { running: 3, waiting: 2, kv_blocks_used: 40, kv_blocks_total: 100, preemptions: 3, swap_preemptions: 2, swapped_bytes_total: 4096, swap_secs: 0.5, reprefilled_tokens: 700, spec_drafted: 10, spec_accepted: 7, prefix_queries: 200, prefix_hits: 50, ..Stats::default() };
        s.inter_token_secs.observe(0.03);
        s.inter_token_secs.observe(0.3);
        let t = render(&s);
        for want in [
            "inference_running_seqs 3",
            "inference_waiting_seqs 2",
            "inference_kv_blocks_used 40",
            "inference_preemptions_total 3",
            "specrt_preemptions_total{kind=\"swap\"} 2",
            "specrt_preemptions_total{kind=\"recompute\"} 1",
            "specrt_swapped_bytes_total 4096",
            "specrt_swap_seconds 0.5",
            "specrt_reprefilled_tokens_total 700",
            "inference_spec_drafted_tokens_total 10",
            "inference_prefix_cache_queries_total 200",
            "inference_prefix_cache_hit_tokens_total 50",
            "inference_prefix_cache_hit_rate 0.25",
            "inference_spec_accepted_tokens_total 7",
            "inference_inter_token_seconds_bucket{le=\"0.035\"} 1",
            "inference_inter_token_seconds_bucket{le=\"+Inf\"} 2",
            "vllm:kv_cache_usage_perc 0.4",
            "vllm:num_preemptions_total 3",
            "vllm:prefix_cache_hits_total 50",
            "vllm:spec_decode_num_draft_tokens_total 10",
            "vllm:spec_decode_num_accepted_tokens_total 7",
            "vllm:inter_token_latency_seconds_bucket{le=\"0.035\"} 1",
            "vllm:request_queue_time_seconds_count 0",
            "inference_prefill_tokens_per_step_count 0",
            "inference_batch_size_count 0",
            "inference_queue_seconds_count 0",
        ] {
            assert!(t.contains(want), "{want} missing from\n{t}");
        }
    }

    #[test]
    fn pipeline_chain_renders_per_stage_series() {
        use specrt::sched::{ChainMetrics, StageMetric};
        let chain = ChainMetrics {
            stages: vec![
                StageMetric { device: 0, name: String::new(), start: 0, end: 20, busy_secs: 1.5, calls: 10, tokens: 400 },
                StageMetric { device: 1, name: String::new(), start: 20, end: 40, busy_secs: 1.25, calls: 10, tokens: 400 },
            ],
            hops: 10,
            hop_bytes: 8000,
            micro_batches: 6,
            overlapped_prefills: 2,
            ..ChainMetrics::default()
        };
        let t = render(&Stats { chain: Some(chain), ..Stats::default() });
        for want in [
            "inference_pipeline_stages 2",
            "inference_pipeline_stage_busy_seconds_total{stage=\"1\",device=\"1\",layers=\"20-40\"} 1.25",
            "inference_pipeline_stage_tokens_total{stage=\"0\",device=\"0\",layers=\"0-20\"} 400",
            "inference_pipeline_hop_bytes_total 8000",
            "inference_pipeline_prefill_micro_batches_total 6",
        ] {
            assert!(t.contains(want), "{want} missing from\n{t}");
        }
        assert!(!render(&Stats::default()).contains("inference_pipeline"));
    }
}
