use crate::engine::Assets;
use crate::runtime::{Layout, Runtime, Sampling, kv_block_bytes};
use crate::sched::{self, ChainMetrics, Config, Event, Executor, Finish, GpuExecutor, PipelineExecutor, Scheduler, SeqId};
use crate::spec::Spec;
pub use crate::spec::{RuntimeOptions, TileChoice};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeOptions {
    pub enabled: Option<bool>,
    pub draft_tokens: Option<usize>,
    pub max_batch: Option<usize>,
}

impl SpeculativeOptions {
    pub fn apply(&self, spec: &mut Spec) -> Result<()> {
        let Some(m) = spec.mtp.as_mut() else {
            anyhow::ensure!(self.enabled != Some(true), "[speculative] enabled = true but {} has no [mtp] layer", spec.name);
            return Ok(());
        };
        if let Some(on) = self.enabled {
            m.speculate = on;
        }
        if let Some(k) = self.draft_tokens {
            m.draft_tokens = k;
        }
        if self.max_batch.is_some() {
            m.max_batch = self.max_batch;
        }
        spec.validate().context("[speculative]")
    }
}
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const DEFAULT_MAX_MODEL_LEN: usize = 131072;
pub const MAX_MODEL_LEN: usize = 262144;

#[derive(Debug, Clone, PartialEq)]
pub struct ServeOptions {
    pub max_model_len: usize,
    pub max_num_seqs: usize,
    pub kv_cache_bytes: Option<u64>,
    pub max_batched_tokens: usize,
    pub itl_target: Duration,
    pub runtime: Option<RuntimeOptions>,
    pub speculative: SpeculativeOptions,
    pub placement: Option<crate::placement::PlacementConfig>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions {
            max_model_len: DEFAULT_MAX_MODEL_LEN,
            max_num_seqs: 32,
            kv_cache_bytes: None,
            max_batched_tokens: 8192,
            itl_target: Duration::from_millis(100),
            runtime: None,
            speculative: SpeculativeOptions::default(),
            placement: None,
        }
    }
}

impl ServeOptions {
    pub fn layout(&self, spec: &Spec) -> Layout {
        let kv = self.kv_cache_bytes.map(|b| b as usize);
        let mut layout = Layout::serving(spec, self.max_model_len, self.max_num_seqs, kv);
        layout.prefill_chunk = layout.prefill_chunk.min(self.max_batched_tokens.max(64));
        layout
    }
}

pub const LATENCY_BUCKETS: [f64; 14] =
    [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 10.0];
pub const COUNT_BUCKETS: [f64; 12] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 256.0, 1024.0, 4096.0, 8192.0, 16384.0];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Histogram {
    pub bounds: Vec<f64>,
    pub counts: Vec<u64>,
    pub sum: f64,
    pub total: u64,
}

impl Histogram {
    fn new(bounds: &[f64]) -> Self {
        Histogram { bounds: bounds.to_vec(), counts: vec![0; bounds.len()], sum: 0.0, total: 0 }
    }

    pub fn observe(&mut self, v: f64) {
        for (i, b) in self.bounds.iter().enumerate() {
            if v <= *b {
                self.counts[i] += 1;
            }
        }
        self.sum += v;
        self.total += 1;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub running: usize,
    pub waiting: usize,
    pub kv_blocks_used: usize,
    pub kv_blocks_total: usize,
    pub kv_block_tokens: usize,
    pub preemptions: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub steps: u64,
    pub prefix_queries: u64,
    pub prefix_hits: u64,
    pub prefix_blocks: usize,
    pub prefix_snapshots: usize,
    pub prefix_snapshot_bytes: usize,
    pub queue_secs: Histogram,
    pub inter_token_secs: Histogram,
    pub decode_batch: Histogram,
    pub prefill_tokens_per_step: Histogram,
    pub step_secs: Histogram,
    pub chain: Option<ChainMetrics>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            running: 0,
            waiting: 0,
            kv_blocks_used: 0,
            kv_blocks_total: 0,
            kv_block_tokens: crate::codegen::PAGE,
            preemptions: 0,
            spec_drafted: 0,
            spec_accepted: 0,
            steps: 0,
            prefix_queries: 0,
            prefix_hits: 0,
            prefix_blocks: 0,
            prefix_snapshots: 0,
            prefix_snapshot_bytes: 0,
            queue_secs: Histogram::new(&LATENCY_BUCKETS),
            inter_token_secs: Histogram::new(&LATENCY_BUCKETS),
            decode_batch: Histogram::new(&COUNT_BUCKETS),
            prefill_tokens_per_step: Histogram::new(&COUNT_BUCKETS),
            step_secs: Histogram::new(&LATENCY_BUCKETS),
            chain: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SeqEvent {
    Started { queued: Duration },
    Token(u32),
    Preempted,
    Finished { finish: Finish, prompt_tokens: usize, completion_tokens: usize, cached_tokens: usize },
    Failed(String),
}

pub struct Job {
    pub request: sched::Request,
    pub events: Sender<SeqEvent>,
    pub cancel: Arc<AtomicBool>,
}

struct Live {
    events: Sender<SeqEvent>,
    cancel: Arc<AtomicBool>,
    last_token: Option<Instant>,
}

pub struct Core {
    jobs: Mutex<Option<Sender<Job>>>,
    stats: Arc<Mutex<Stats>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    max_model_len: usize,
    vision: bool,
}

impl Core {
    pub fn start(spec: &Spec, opts: ServeOptions) -> Result<Arc<Core>> {
        anyhow::ensure!(opts.max_model_len <= MAX_MODEL_LEN, "max_model_len above {MAX_MODEL_LEN}");
        anyhow::ensure!(opts.max_num_seqs >= 1, "max_num_seqs must be at least 1");
        let (jobs, inbox) = channel::<Job>();
        let (ready_tx, ready) = channel::<Result<bool>>();
        let stats = Arc::new(Mutex::new(Stats::default()));
        let shared = stats.clone();
        let spec = spec.clone();
        let max_model_len = opts.max_model_len;
        let thread = std::thread::Builder::new()
            .name("specrt-serve".into())
            .spawn(move || {
                let _ = start_engine(&spec, opts, ready_tx, inbox, shared);
            })
            .context("serving thread")?;
        let vision = ready.recv().context("the serving thread stopped while loading")??;
        Ok(Arc::new(Core {
            vision,
            jobs: Mutex::new(Some(jobs)),
            stats,
            thread: Mutex::new(Some(thread)),
            max_model_len,
        }))
    }

    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    pub fn vision(&self) -> bool {
        self.vision
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }

    pub fn submit(&self, job: Job) -> Result<()> {
        let jobs = self.jobs.lock().unwrap();
        let tx = jobs.as_ref().context("the serving engine is stopped")?;
        tx.send(job).map_err(|_| anyhow::anyhow!("the serving engine has stopped"))
    }

    pub fn stop(&self) {
        self.jobs.lock().unwrap().take();
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        self.stop();
    }
}

fn start_engine(spec: &Spec, opts: ServeOptions, ready: Sender<Result<bool>>, inbox: Receiver<Job>, stats: Arc<Mutex<Stats>>) -> Result<()> {
    if opts.placement.as_ref().is_some_and(|p| p.remote.is_some()) {
        match build_remote(spec, &opts) {
            Ok(mut sched) => {
                let _ = ready.send(Ok(false));
                run(&mut sched, inbox, stats);
            }
            Err(e) => {
                let _ = ready.send(Err(e.context("[placement.remote]")));
            }
        }
        return Ok(());
    }
    let planned = (|| -> Result<_> {
        Ok(match &opts.placement {
            Some(p) if !p.devices.is_empty() => {
                let mut resolved = spec.clone();
                opts.speculative.apply(&mut resolved)?;
                let placed = p.place(&resolved)?;
                (placed.stages.len() > 1).then_some((placed, p.micro_batch))
            }
            _ => None,
        })
    })();
    let pipelined = match planned {
        Ok(p) => p,
        Err(e) => {
            let _ = ready.send(Err(e.context("[placement]")));
            return Ok(());
        }
    };
    match pipelined {
        Some((placement, micro_batch)) => match build_pipeline(spec, &opts, &placement, micro_batch) {
            Ok(mut sched) => {
                let _ = ready.send(Ok(false));
                run(&mut sched, inbox, stats);
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        },
        None => match build(spec, opts) {
            Ok(mut sched) => {
                let _ = ready.send(Ok(sched.exec.rt.has_vision()));
                run(&mut sched, inbox, stats);
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        },
    }
    Ok(())
}

fn serving_config(spec: &Spec, opts: &ServeOptions) -> Config {
    Config {
        max_num_seqs: opts.max_num_seqs,
        max_batched_tokens: opts.max_batched_tokens,
        itl_target: opts.itl_target,
        prefix_cache: spec.runtime.prefix_cache,
        ..Config::default()
    }
}

fn build_pipeline(spec: &Spec, opts: &ServeOptions, placement: &crate::placement::Placement, micro_batch: usize) -> Result<Scheduler<PipelineExecutor>> {
    let mut spec = spec.clone();
    if let Some(r) = &opts.runtime {
        spec.runtime = r.clone();
    }
    opts.speculative.apply(&mut spec)?;
    if spec.mtp.take().is_some() {
        tracing::warn!(spec = %spec.name, stages = placement.stages.len(), "pipelined serving runs without MTP speculative decoding");
    }
    if spec.runtime.prefix_cache.enabled {
        tracing::warn!(spec = %spec.name, stages = placement.stages.len(), "pipelined serving runs without the prefix cache");
    }
    let spec = &spec;
    let layout = opts.layout(spec);
    let mut pipe = crate::pipeline::Pipeline::load(spec, placement, layout, crate::pipeline::Hop::Direct)?;
    pipe.set_micro_batch(micro_batch);
    tracing::info!(spec = %spec.name, stages = pipe.stages(), kv_blocks = pipe.kv_blocks(), "pipelined serving layout");
    let mut cfg = serving_config(spec, opts);
    cfg.prefix_cache.enabled = false;
    Ok(Scheduler::new(PipelineExecutor::new(pipe), cfg))
}

fn build_remote(spec: &Spec, opts: &ServeOptions) -> Result<Scheduler<crate::remote::ChainExecutor>> {
    let mut spec = spec.clone();
    opts.speculative.apply(&mut spec)?;
    spec.mtp = None;
    let placement = opts.placement.as_ref().context("remote serving needs [placement]")?;
    let remote = placement.remote.as_ref().context("remote serving needs [placement.remote]")?;
    let coordinator = crate::coordinator::Coordinator::new(
        &spec,
        placement.clone(),
        remote.key()?,
        Duration::from_millis(remote.timeout_ms),
        Duration::from_millis(remote.member_ttl_ms),
    )?;
    let layout = opts.layout(&spec);
    let mut exec = crate::remote::ChainExecutor::new(coordinator, layout.max_ctx, layout.slots, layout.prefill_chunk);
    if let Err(e) = exec.connect_within(Duration::from_millis(remote.connect_wait_ms)) {
        tracing::warn!(error = %format!("{e:#}"), "remote stages are not ready yet, connecting on the first request");
    }
    let mut cfg = serving_config(&spec, opts);
    cfg.prefix_cache.enabled = false;
    Ok(Scheduler::new(exec, cfg))
}

fn build(spec: &Spec, opts: ServeOptions) -> Result<Scheduler<GpuExecutor>> {
    let mut spec = spec.clone();
    if let Some(r) = &opts.runtime {
        spec.runtime = r.clone();
    }
    opts.speculative.apply(&mut spec)?;
    let spec = &spec;
    let layout = opts.layout(spec);
    let rt = Runtime::load_layout(spec, layout)?;
    let block = kv_block_bytes(spec, 1) as u64;
    tracing::info!(
        spec = %spec.name,
        max_model_len = layout.max_ctx,
        max_num_seqs = layout.slots,
        kv_blocks = rt.kv_blocks(),
        kv_bytes = rt.kv_blocks() as u64 * block,
        kv_tokens = rt.kv_blocks() * crate::codegen::PAGE,
        linear_state_bytes_per_slot = slot_state_bytes(spec),
        linear_state_mapped_bytes = rt.state_mapped_bytes() as u64,
        state_on_demand = layout.state_on_demand,
        device_bytes = rt.device_bytes,
        "serving layout"
    );
    let cfg = serving_config(spec, &opts);
    Ok(Scheduler::new(GpuExecutor::new(rt)?, cfg))
}

pub fn slot_state_bytes(spec: &Spec) -> u64 {
    match &spec.linear_attention {
        Some(la) => {
            let ssm = la.value_heads * la.key_head_dim * la.key_head_dim;
            let conv = la.conv_dim() * (la.conv_kernel - 1);
            (spec.linear_layers() * (ssm + conv) * 4) as u64
        }
        None => 0,
    }
}

fn run<E: Executor>(sched: &mut Scheduler<E>, inbox: Receiver<Job>, stats: Arc<Mutex<Stats>>) {
    let mut live: HashMap<SeqId, Live> = HashMap::new();
    let mut open = true;
    {
        let mut s = stats.lock().unwrap();
        s.kv_blocks_total = sched.blocks_total();
    }
    while open || !live.is_empty() {
        let first = if live.is_empty() {
            match inbox.recv_timeout(Duration::from_millis(50)) {
                Ok(j) => Some(j),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => {
                    open = false;
                    None
                }
            }
        } else {
            None
        };
        let mut incoming: Vec<Job> = first.into_iter().collect();
        loop {
            match inbox.try_recv() {
                Ok(j) => incoming.push(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        for job in incoming {
            match sched.submit(job.request) {
                Ok(id) => {
                    live.insert(id, Live { events: job.events, cancel: job.cancel, last_token: None });
                }
                Err(e) => {
                    let _ = job.events.send(SeqEvent::Failed(format!("{e:#}")));
                }
            }
        }
        let mut events = Vec::new();
        let gone: Vec<SeqId> =
            live.iter().filter(|(_, l)| l.cancel.load(Ordering::Relaxed)).map(|(id, _)| *id).collect();
        for id in gone {
            sched.cancel(id, &mut events);
        }
        if !open && live.is_empty() {
            break;
        }
        if let Some(reason) = sched.exec.refresh() {
            if !live.is_empty() {
                tracing::error!(%reason, "stage chain changed, failing in-flight requests");
                let mut drop_events = Vec::new();
                for id in live.keys().copied().collect::<Vec<_>>() {
                    sched.cancel(id, &mut drop_events);
                }
                for (_, l) in live.drain() {
                    let _ = l.events.send(SeqEvent::Failed(reason.clone()));
                }
                continue;
            }
        }
        let started = Instant::now();
        let report = if sched.idle() { Ok(None) } else { sched.step(&mut events).map(Some) };
        let report = match report {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "serving step failed, failing every request");
                let mut drop_events = Vec::new();
                for id in live.keys().copied().collect::<Vec<_>>() {
                    sched.cancel(id, &mut drop_events);
                }
                for (_, l) in live.drain() {
                    let _ = l.events.send(SeqEvent::Failed(format!("{e:#}")));
                }
                continue;
            }
        };
        let now = Instant::now();
        let mut s = stats.lock().unwrap();
        if let Some(r) = report {
            s.steps += 1;
            s.spec_drafted += r.drafted as u64;
            s.spec_accepted += r.accepted as u64;
            s.step_secs.observe(started.elapsed().as_secs_f64());
            if r.decode_rows > 0 {
                s.decode_batch.observe(r.decode_rows as f64);
            }
            if r.prefill_tokens > 0 {
                s.prefill_tokens_per_step.observe(r.prefill_tokens as f64);
            }
        }
        for e in events {
            let (id, ev) = match e {
                Event::Started { id, queued } => {
                    s.queue_secs.observe(queued.as_secs_f64());
                    (id, SeqEvent::Started { queued })
                }
                Event::Token { id, token } => {
                    if let Some(l) = live.get_mut(&id) {
                        if let Some(prev) = l.last_token {
                            s.inter_token_secs.observe(now.duration_since(prev).as_secs_f64());
                        }
                        l.last_token = Some(now);
                    }
                    (id, SeqEvent::Token(token))
                }
                Event::Preempted { id } => (id, SeqEvent::Preempted),
                Event::Finished { id, finish, prompt_tokens, completion_tokens, cached_tokens } => {
                    (id, SeqEvent::Finished { finish, prompt_tokens, completion_tokens, cached_tokens })
                }
            };
            let terminal = matches!(ev, SeqEvent::Finished { .. });
            let gone = live.get(&id).map(|l| l.events.send(ev).is_err()).unwrap_or(false);
            if terminal {
                live.remove(&id);
            } else if gone && let Some(l) = live.get(&id) {
                l.cancel.store(true, Ordering::Relaxed);
            }
        }
        s.running = sched.running();
        s.waiting = sched.waiting();
        s.kv_blocks_used = sched.blocks_used();
        s.preemptions = sched.preemptions();
        s.chain = sched.chain_metrics();
        let p = sched.prefix_stats();
        s.prefix_queries = p.queries;
        s.prefix_hits = p.hits;
        s.prefix_blocks = p.blocks;
        s.prefix_snapshots = p.snapshots;
        s.prefix_snapshot_bytes = p.snapshot_bytes;
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub finish: Option<Finish>,
    pub queue_secs: f64,
    pub ttft_secs: f64,
    pub decode_secs: f64,
    pub preemptions: usize,
}

pub struct Serving {
    core: Arc<Core>,
    assets: Arc<Assets>,
}

impl Clone for Serving {
    fn clone(&self) -> Self {
        Serving { core: self.core.clone(), assets: self.assets.clone() }
    }
}

impl Serving {
    pub fn sampling_defaults(&self) -> crate::engine::SamplingDefaults {
        self.assets.sampling
    }

    pub fn load(model: &str, opts: ServeOptions) -> Result<Self> {
        let assets = Assets::load(model)?;
        let core = Core::start(&assets.spec, opts)?;
        Ok(Serving { core, assets: Arc::new(assets) })
    }

    pub fn spec(&self) -> &Spec {
        &self.assets.spec
    }

    pub fn stats(&self) -> Stats {
        self.core.stats()
    }

    pub fn max_model_len(&self) -> usize {
        self.core.max_model_len()
    }

    pub fn encode_messages(&self, messages: &[Value], template: Option<&Value>) -> Result<Vec<u32>> {
        self.assets.encode_messages(messages, template)
    }

    pub fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        self.assets.tokens(text)
    }

    pub fn accepts_media(&self) -> bool {
        self.assets.media.is_some() && self.core.vision()
    }

    pub fn encode_media_messages(
        &self,
        messages: &[Value],
        template: Option<&Value>,
        items: &[crate::vision::MediaItem],
        budget: crate::vision::MediaBudget,
    ) -> Result<(Vec<u32>, Vec<sched::MediaInput>, usize)> {
        self.assets.encode_media_messages(messages, template, items, budget)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn complete(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: Sampling,
        use_stop: bool,
        salt: String,
        cancel: Arc<AtomicBool>,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion> {
        self.complete_media(prompt, Vec::new(), max_tokens, sampling, use_stop, salt, cancel, on_text)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn complete_media(
        &self,
        prompt: Vec<u32>,
        media: Vec<sched::MediaInput>,
        max_tokens: usize,
        sampling: Sampling,
        use_stop: bool,
        salt: String,
        cancel: Arc<AtomicBool>,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion> {
        let started = Instant::now();
        let (tx, rx) = channel();
        let stop = if use_stop { self.assets.stop.clone() } else { Vec::new() };
        let request = sched::Request { prompt, max_tokens, sampling, stop, salt, media };
        self.core.submit(Job { request, events: tx, cancel: cancel.clone() })?;
        let tokenizer = &self.assets.tokenizer;
        let mut out = Completion::default();
        let mut ids: Vec<u32> = Vec::new();
        let mut sent = String::new();
        let mut first: Option<Instant> = None;
        loop {
            let ev = rx.recv().map_err(|_| anyhow::anyhow!("the serving engine has stopped"))?;
            match ev {
                SeqEvent::Started { queued } => out.queue_secs = queued.as_secs_f64(),
                SeqEvent::Preempted => out.preemptions += 1,
                SeqEvent::Token(t) => {
                    if out.preemptions > 0 && ids.len() > out.completion_tokens {
                        continue;
                    }
                    first.get_or_insert_with(Instant::now);
                    ids.push(t);
                    if let Ok(text) = tokenizer.decode(&ids, true)
                        && text.len() > sent.len()
                        && text.starts_with(&sent)
                        && !text.ends_with('\u{fffd}')
                    {
                        if !on_text(&text[sent.len()..]) {
                            cancel.store(true, Ordering::Relaxed);
                        }
                        sent = text;
                    }
                }
                SeqEvent::Finished { finish, prompt_tokens, completion_tokens, cached_tokens } => {
                    out.cached_tokens = cached_tokens;
                    out.finish = Some(finish);
                    out.prompt_tokens = prompt_tokens;
                    out.completion_tokens = completion_tokens.min(ids.len());
                    break;
                }
                SeqEvent::Failed(e) => anyhow::bail!(e),
            }
        }
        ids.truncate(out.completion_tokens);
        let text = tokenizer.decode(&ids, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        if text.len() > sent.len() && text.starts_with(&sent) {
            on_text(&text[sent.len()..]);
        }
        out.text = text;
        let first = first.unwrap_or_else(Instant::now);
        out.ttft_secs = first.duration_since(started).as_secs_f64();
        out.decode_secs = first.elapsed().as_secs_f64();
        Ok(out)
    }
}
