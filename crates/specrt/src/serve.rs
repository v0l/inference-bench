use crate::engine::Assets;
use crate::runtime::{Layout, Runtime, Sampling, kv_block_bytes};
use crate::sched::{self, ChainMetrics, Config, Event, Executor, Finish, GpuExecutor, PipelineExecutor, Scheduler, SeqId};
use crate::spec::{PrefixCacheOptions, Spec};
pub use crate::spec::{RuntimeOptions, TileChoice};
pub use crate::sessions::SessionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftTokens {
    Auto,
    Fixed(usize),
}

impl serde::Serialize for DraftTokens {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            DraftTokens::Auto => s.serialize_str("auto"),
            DraftTokens::Fixed(n) => s.serialize_u64(*n as u64),
        }
    }
}

impl<'de> serde::Deserialize<'de> for DraftTokens {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Count(usize),
            Word(String),
        }
        match Raw::deserialize(d)? {
            Raw::Count(n) => Ok(DraftTokens::Fixed(n)),
            Raw::Word(w) if w == "auto" => Ok(DraftTokens::Auto),
            Raw::Word(w) => Err(serde::de::Error::custom(format!("draft_tokens is \"auto\" or a number, not {w:?}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeOptions {
    pub enabled: Option<bool>,
    pub draft_tokens: Option<DraftTokens>,
    pub max_batch: Option<usize>,
    pub head_quant: Option<crate::spec::Quant>,
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
        match self.draft_tokens {
            Some(DraftTokens::Auto) | None => m.draft_tokens = crate::spec::MAX_DRAFT_TOKENS,
            Some(DraftTokens::Fixed(k)) => m.draft_tokens = k,
        }
        if self.max_batch.is_some() {
            m.max_batch = self.max_batch;
        }
        if self.head_quant.is_some() {
            m.head_quant = self.head_quant;
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


#[derive(Debug, Clone, PartialEq)]
pub struct ServeOptions {
    pub max_model_len: Option<usize>,
    pub max_num_seqs: usize,
    pub kv_cache_bytes: Option<u64>,
    pub kv_cache_tokens: Option<usize>,
    pub max_batched_tokens: usize,
    pub itl_target: Duration,
    pub step_cost: f64,
    pub whole_prompt_tokens: usize,
    pub min_prefill_tokens: usize,
    pub swap_space_gb: f64,
    pub admit_lookahead: usize,
    pub prefill_burst_tokens: usize,
    pub max_long_in_flight: usize,
    pub stall_budget: Duration,
    pub min_stall_chunk_tokens: usize,
    pub runtime: Option<RuntimeOptions>,
    pub speculative: SpeculativeOptions,
    pub placement: Option<crate::placement::PlacementConfig>,
    pub progress: Option<crate::progress::Sink>,
    pub watchdog: Option<Duration>,
    pub sessions: Option<crate::sessions::SessionOptions>,
    pub tp: usize,
    pub grammar_cache: usize,
    pub solved: Option<Box<crate::solve::Solved>>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions {
            max_model_len: None,
            max_num_seqs: crate::options::DEFAULT_MAX_NUM_SEQS,
            kv_cache_bytes: None,
            kv_cache_tokens: None,
            max_batched_tokens: 8192,
            itl_target: Duration::ZERO,
            step_cost: 3.0,
            whole_prompt_tokens: 1024,
            min_prefill_tokens: 512,
            swap_space_gb: 0.0,
            admit_lookahead: 0,
            prefill_burst_tokens: 16384,
            max_long_in_flight: 4,
            stall_budget: Duration::ZERO,
            min_stall_chunk_tokens: 0,
            runtime: None,
            speculative: SpeculativeOptions::default(),
            placement: None,
            progress: None,
            watchdog: Some(Duration::from_secs(60)),
            sessions: None,
            tp: 1,
            grammar_cache: 32,
            solved: None,
        }
    }
}

impl ServeOptions {
    pub fn from_solved(solved: &crate::solve::Solved, base: ServeOptions) -> Result<ServeOptions> {
        let mut opts = crate::options::apply_sparse(&solved.options, base)?;
        opts.max_model_len = Some(solved.context);
        opts.max_num_seqs = solved.slots;
        opts.kv_cache_bytes = Some(solved.kv_pool_bytes);
        opts.kv_cache_tokens = None;
        opts.tp = solved.tp;
        opts.solved = Some(Box::new(solved.clone()));
        Ok(opts)
    }

    pub fn for_kv_context(&self, spec: &Spec, context: usize) -> (Spec, ServeOptions) {
        let mut opts = self.clone();
        if let Some(r) = opts.runtime.as_mut() {
            *r = Spec { runtime: r.clone(), ..spec.clone() }.for_kv_context(context).runtime;
        }
        (spec.for_kv_context(context), opts)
    }

    pub fn model_len(&self, spec: &Spec) -> Result<usize> {
        let model_max = match &self.runtime {
            Some(r) if r.rope_extension != spec.runtime.rope_extension => {
                let mut s = spec.clone();
                s.runtime.rope_extension = r.rope_extension;
                crate::engine::model_max_context(&s)
            }
            _ => crate::engine::model_max_context(spec),
        };
        match self.max_model_len {
            Some(n) => crate::engine::limit_context(n, model_max),
            None => Ok(model_max.unwrap_or(crate::engine::DEFAULT_CTX)),
        }
    }

    pub fn check_pool(&self, spec: &Spec) -> Result<()> {
        let mut resolved = spec.clone();
        if let Some(r) = &self.runtime {
            resolved.runtime = r.clone();
        }
        let layout = self.layout(&resolved);
        if layout.kv_blocks == 0 {
            return Ok(());
        }
        check_pool_tokens(layout.kv_blocks * crate::codegen::PAGE, layout.max_ctx)
    }

    pub fn stage_kv_bytes(&self, spec: &Spec, placement: &crate::placement::Placement) -> Vec<u64> {
        let blocks = self.layout(spec).kv_blocks;
        placement.stages.iter().map(|s| (blocks * kv_block_bytes(&spec.stage(s.start, s.end), 1)) as u64).collect()
    }

    pub fn layout(&self, spec: &Spec) -> Layout {
        let kv = self.kv_cache_bytes.map(|b| b as usize).or_else(|| {
            self.kv_cache_tokens.map(|t| (t.div_ceil(crate::codegen::PAGE) + self.max_num_seqs) * kv_block_bytes(spec, 1))
        });
        let len = self.model_len(spec).unwrap_or(crate::engine::DEFAULT_CTX);
        let mut layout = Layout::serving(spec, len, self.max_num_seqs, kv);
        layout.prefill_chunk = layout.prefill_chunk.min(self.max_batched_tokens.max(64));
        layout
    }
}

pub fn check_pool_tokens(pool_tokens: usize, max_model_len: usize) -> Result<()> {
    anyhow::ensure!(
        pool_tokens >= max_model_len,
        "the KV cache holds {pool_tokens} tokens but one request may be {max_model_len} tokens long: raise the KV cache size or lower the context length"
    );
    Ok(())
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
    pub fn new(bounds: &[f64]) -> Self {
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

    pub fn since(&self, older: &Histogram) -> Histogram {
        if older.bounds != self.bounds || older.total > self.total {
            return self.clone();
        }
        Histogram {
            bounds: self.bounds.clone(),
            counts: self.counts.iter().zip(&older.counts).map(|(a, b)| a.saturating_sub(*b)).collect(),
            sum: (self.sum - older.sum).max(0.0),
            total: self.total - older.total,
        }
    }

    pub fn quantile(&self, p: f64) -> Option<f64> {
        if self.total == 0 || self.bounds.is_empty() {
            return None;
        }
        let want = p.clamp(0.0, 1.0) * self.total as f64;
        let mut below = 0.0;
        let mut lower = 0.0;
        for (b, c) in self.bounds.iter().zip(&self.counts) {
            let c = *c as f64;
            if c >= want && c > below {
                return Some(lower + (b - lower) * ((want - below) / (c - below)).clamp(0.0, 1.0));
            }
            below = c;
            lower = *b;
        }
        self.bounds.last().copied()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub running: usize,
    pub waiting: usize,
    pub kv_blocks_used: usize,
    pub kv_blocks_cached: usize,
    pub kv_blocks_total: usize,
    pub kv_block_tokens: usize,
    pub kv_block_bytes: usize,
    pub preemptions: u64,
    pub swap_preemptions: u64,
    pub swapped_bytes_total: u64,
    pub swap_secs: f64,
    pub reprefilled_tokens: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub steps: u64,
    pub prefix_queries: u64,
    pub prefix_hits: u64,
    pub prefix_blocks: usize,
    pub prefix_snapshots: usize,
    pub prefix_snapshot_bytes: usize,
    pub generated_tokens: u64,
    pub queue_secs: Histogram,
    pub inter_token_secs: Histogram,
    pub decode_batch: Histogram,
    pub prefill_tokens_per_step: Histogram,
    pub step_secs: Histogram,
    pub chain: Option<ChainMetrics>,
    pub sessions: Option<crate::sessions::Stats>,
    pub offload: Option<crate::offload::OffloadStats>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            running: 0,
            waiting: 0,
            kv_blocks_used: 0,
            kv_blocks_cached: 0,
            kv_blocks_total: 0,
            kv_block_tokens: crate::codegen::PAGE,
            kv_block_bytes: 0,
            preemptions: 0,
            swap_preemptions: 0,
            swapped_bytes_total: 0,
            swap_secs: 0.0,
            reprefilled_tokens: 0,
            spec_drafted: 0,
            spec_accepted: 0,
            steps: 0,
            prefix_queries: 0,
            prefix_hits: 0,
            prefix_blocks: 0,
            prefix_snapshots: 0,
            prefix_snapshot_bytes: 0,
            generated_tokens: 0,
            queue_secs: Histogram::new(&LATENCY_BUCKETS),
            inter_token_secs: Histogram::new(&LATENCY_BUCKETS),
            decode_batch: Histogram::new(&COUNT_BUCKETS),
            prefill_tokens_per_step: Histogram::new(&COUNT_BUCKETS),
            step_secs: Histogram::new(&LATENCY_BUCKETS),
            chain: None,
            sessions: None,
            offload: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SeqEvent {
    Started { queued: Duration },
    Token(u32),
    Preempted,
    Finished { finish: Finish, prompt_tokens: usize, completion_tokens: usize, cached_tokens: usize },
    Logprob(crate::logprobs::TokenLogprob),
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

pub struct Pulse {
    origin: Instant,
    busy_since_ms: std::sync::atomic::AtomicU64,
}

impl Pulse {
    fn new() -> Self {
        Pulse { origin: Instant::now(), busy_since_ms: std::sync::atomic::AtomicU64::new(0) }
    }

    fn enter(&self) {
        self.busy_since_ms.store(self.origin.elapsed().as_millis() as u64 + 1, Ordering::SeqCst);
    }

    fn leave(&self) {
        self.busy_since_ms.store(0, Ordering::SeqCst);
    }

    fn stalled_for(&self) -> Option<Duration> {
        let since = self.busy_since_ms.load(Ordering::SeqCst);
        (since > 0).then(|| self.origin.elapsed().saturating_sub(Duration::from_millis(since - 1)))
    }
}

type Inflight = Arc<Mutex<Vec<(Sender<SeqEvent>, Arc<AtomicBool>)>>>;

pub const HUNG_MARKER: &str = "the engine stopped responding";

pub struct Core {
    jobs: Mutex<Option<Sender<Job>>>,
    stats: Arc<Mutex<Stats>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    max_model_len: usize,
    vision: bool,
    failure: Arc<Mutex<Option<String>>>,
    inflight: Inflight,
    stopping: Arc<AtomicBool>,
    sessions: Option<Arc<crate::sessions::Shared>>,
}

fn watch(pulse: Arc<Pulse>, limit: Duration, failure: Arc<Mutex<Option<String>>>, inflight: Inflight, stopping: Arc<AtomicBool>) {
    let poll = (limit / 4).clamp(Duration::from_millis(10), Duration::from_secs(1));
    while !stopping.load(Ordering::SeqCst) {
        std::thread::sleep(poll);
        let Some(stalled) = pulse.stalled_for() else { continue };
        if stalled < limit {
            continue;
        }
        let reason = format!("{HUNG_MARKER}: no progress for {} s, the model must be reloaded", stalled.as_secs());
        tracing::error!(%reason, "engine watchdog tripped");
        *failure.lock().unwrap() = Some(reason.clone());
        for (tx, _) in inflight.lock().unwrap().drain(..) {
            let _ = tx.send(SeqEvent::Failed(reason.clone()));
        }
        return;
    }
}

impl Core {
    pub fn start(spec: &Spec, opts: ServeOptions) -> Result<Arc<Core>> {
        let max_model_len = opts.model_len(spec)?;
        let (spec, opts) = opts.for_kv_context(spec, max_model_len);
        let spec = &spec;
        let effective = opts.runtime.as_ref().map_or_else(|| spec.clone(), |r| Spec { runtime: r.clone(), ..spec.clone() });
        match effective.kv_context_note(max_model_len) {
            Some((false, w)) => tracing::warn!(spec = %spec.name, "{w}"),
            Some((true, n)) => tracing::info!(spec = %spec.name, "{n}"),
            None => {}
        }
        anyhow::ensure!(opts.max_num_seqs >= 1, "max_num_seqs must be at least 1");
        opts.check_pool(spec)?;
        let kv_block = kv_block_bytes(spec, 1);
        let watchdog = opts.watchdog;
        let spec = spec.clone();
        let sessions = open_sessions(&spec, &opts)?;
        let shared = sessions.clone();
        Core::spawn(max_model_len, kv_block, watchdog, sessions, move |ready, inbox, stats, pulse| {
            let sink = opts.progress.clone();
            let _ = crate::progress::with_sink(sink, || start_engine(&spec, opts, shared, ready, inbox, stats, pulse));
        })
    }

    fn spawn(
        max_model_len: usize,
        kv_block_bytes: usize,
        watchdog: Option<Duration>,
        sessions: Option<Arc<crate::sessions::Shared>>,
        engine: impl FnOnce(Sender<Result<bool>>, Receiver<Job>, Arc<Mutex<Stats>>, Arc<Pulse>) + Send + 'static,
    ) -> Result<Arc<Core>> {
        let (jobs, inbox) = channel::<Job>();
        let (ready_tx, ready) = channel::<Result<bool>>();
        let stats = Arc::new(Mutex::new(Stats { kv_block_bytes, ..Stats::default() }));
        let shared = stats.clone();
        let pulse = Arc::new(Pulse::new());
        let thread_pulse = pulse.clone();
        let thread = std::thread::Builder::new()
            .name("specrt-serve".into())
            .spawn(move || engine(ready_tx, inbox, shared, thread_pulse))
            .context("serving thread")?;
        let vision = ready.recv().context("the serving thread stopped while loading")??;
        let failure = Arc::new(Mutex::new(None));
        let inflight: Inflight = Default::default();
        let stopping = Arc::new(AtomicBool::new(false));
        if let Some(limit) = watchdog.filter(|d| !d.is_zero()) {
            let (f, i, s) = (failure.clone(), inflight.clone(), stopping.clone());
            std::thread::Builder::new().name("specrt-watchdog".into()).spawn(move || watch(pulse, limit, f, i, s)).context("watchdog thread")?;
        }
        Ok(Arc::new(Core {
            vision,
            jobs: Mutex::new(Some(jobs)),
            stats,
            thread: Mutex::new(Some(thread)),
            max_model_len,
            failure,
            inflight,
            stopping,
            sessions,
        }))
    }

    pub fn failure(&self) -> Option<String> {
        self.failure.lock().unwrap().clone()
    }

    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    pub fn vision(&self) -> bool {
        self.vision
    }

    pub fn stats(&self) -> Stats {
        let mut s = self.stats.lock().unwrap().clone();
        s.sessions = self.sessions.as_ref().map(|x| x.stats());
        s
    }

    pub fn sessions(&self) -> Option<Arc<crate::sessions::Shared>> {
        self.sessions.clone()
    }

    pub fn submit(&self, job: Job) -> Result<()> {
        if let Some(reason) = self.failure() {
            anyhow::bail!(reason);
        }
        {
            let mut inflight = self.inflight.lock().unwrap();
            inflight.retain(|(_, cancel)| Arc::strong_count(cancel) > 1);
            inflight.push((job.events.clone(), job.cancel.clone()));
        }
        let jobs = self.jobs.lock().unwrap();
        let tx = jobs.as_ref().context("the serving engine is stopped")?;
        tx.send(job).map_err(|_| anyhow::anyhow!("the serving engine has stopped"))
    }

    pub fn stop(&self) {
        self.jobs.lock().unwrap().take();
        self.stopping.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.lock().unwrap().take() {
            if self.failure().is_some() {
                return;
            }
            let _ = t.join();
        }
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        self.stop();
    }
}

fn open_sessions(spec: &Spec, opts: &ServeOptions) -> Result<Option<Arc<crate::sessions::Shared>>> {
    let Some(o) = opts.sessions.clone().filter(|o| o.enabled) else { return Ok(None) };
    o.validate()?;
    let mut resolved = spec.clone();
    if let Some(r) = &opts.runtime {
        resolved.runtime = r.clone();
    }
    opts.speculative.apply(&mut resolved)?;
    let planes = if resolved.runtime.kv_dtype.is_fp8() { 1 } else { 2 };
    let root = if o.dir.is_empty() { crate::sessions::default_dir() } else { std::path::PathBuf::from(&o.dir) };
    let store = crate::sessions::SessionStore::open(&root, crate::sessions::compat_for(&resolved, planes))?;
    let mut o = o;
    if o.max_bytes == 0 {
        o.max_bytes = crate::sessions::free_space_budget(crate::sessions::available_bytes(store.dir()) + store.total_bytes(), 20);
    }
    store.expire(crate::sessions::now_seconds(), o.ttl_seconds());
    Ok(Some(Arc::new(crate::sessions::Shared::new(store, o))))
}

fn start_engine(spec: &Spec, opts: ServeOptions, sessions: Option<Arc<crate::sessions::Shared>>, ready: Sender<Result<bool>>, inbox: Receiver<Job>, stats: Arc<Mutex<Stats>>, pulse: Arc<Pulse>) -> Result<()> {
    if opts.placement.as_ref().is_some_and(|p| p.remote.is_some()) {
        match build_remote(spec, &opts) {
            Ok(mut sched) => {
                let _ = ready.send(Ok(false));
                run(&mut sched, inbox, stats, pulse);
            }
            Err(e) => {
                let _ = ready.send(Err(e.context("[placement.remote]")));
            }
        }
        return Ok(());
    }
    let solved = match crate::solve::for_options(spec, &opts) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(e.context("[placement]")));
            return Ok(());
        }
    };
    let Some(solved) = solved.map(Box::new) else {
        match build(spec, opts, 0) {
            Ok(mut sched) => {
                if let Some(shared) = sessions {
                    sched.enable_sessions(shared);
                }
                let _ = ready.send(Ok(sched.exec.rt.has_vision()));
                run(&mut sched, inbox, stats, pulse);
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        }
        return Ok(());
    };
    tracing::info!(plan = %solved.log_line(), "solved placement");
    tracing::debug!(solved = %serde_json::to_string(&*solved).unwrap_or_default(), "solved placement");
    if solved.tp > 1 {
        match build_tp(spec, &opts, &solved) {
            Ok(mut sched) => {
                let _ = ready.send(Ok(sched.exec.media_encoder().is_some()));
                run(&mut sched, inbox, stats, pulse);
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        }
        return Ok(());
    }
    if solved.spans() {
        let micro_batch = opts.placement.as_ref().map_or(0, |p| p.micro_batch_in_process());
        match build_pipeline(spec, &opts, &solved.placement, micro_batch) {
            Ok(mut sched) => {
                let _ = ready.send(Ok(sched.exec.media_encoder().is_some()));
                run(&mut sched, inbox, stats, pulse);
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        }
        return Ok(());
    }
    let device = solved.single_device();
    match build(spec, opts, device) {
        Ok(mut sched) => {
            if let Some(shared) = sessions {
                sched.enable_sessions(shared);
            }
            let _ = ready.send(Ok(sched.exec.rt.has_vision()));
            run(&mut sched, inbox, stats, pulse);
        }
        Err(e) => {
            let _ = ready.send(Err(e));
        }
    }
    Ok(())
}

fn serving_config(spec: &Spec, opts: &ServeOptions) -> Config {
    let mut prefix_cache = spec.runtime.prefix_cache;
    if prefix_cache.enabled {
        let mem_available = std::fs::read_to_string("/proc/meminfo").map_or(0, |t| crate::hardware::parse_meminfo(&t).1);
        prefix_cache.max_snapshots = auto_snapshots(prefix_cache.max_snapshots, opts.max_num_seqs, slot_state_bytes(spec), mem_available);
        tracing::info!(spec = %spec.name, max_snapshots = prefix_cache.max_snapshots, max_num_seqs = opts.max_num_seqs, "prefix cache snapshot budget");
    }
    Config {
        max_num_seqs: opts.max_num_seqs,
        max_batched_tokens: opts.max_batched_tokens,
        itl_target: opts.itl_target,
        step_cost: opts.step_cost,
        whole_prompt_tokens: opts.whole_prompt_tokens,
        min_prefill_tokens: opts.min_prefill_tokens,
        swap_space_bytes: (opts.swap_space_gb * (1u64 << 30) as f64) as u64,
        admit_lookahead: opts.admit_lookahead,
        prefill_burst_tokens: opts.prefill_burst_tokens,
        max_long_in_flight: opts.max_long_in_flight,
        stall_budget: opts.stall_budget,
        min_stall_chunk: opts.min_stall_chunk_tokens,
        adaptive_drafts: !matches!(opts.speculative.draft_tokens, Some(DraftTokens::Fixed(_))),
        prefix_cache,
        ..Config::default()
    }
}

pub fn auto_snapshots(configured: usize, seqs: usize, snapshot_bytes: u64, mem_available: u64) -> usize {
    if configured != PrefixCacheOptions::default().max_snapshots || snapshot_bytes == 0 {
        return configured;
    }
    let wanted = configured.max(seqs * 4);
    if mem_available == 0 {
        return wanted;
    }
    let affordable = (mem_available / 4 / snapshot_bytes) as usize;
    wanted.min(affordable)
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
    tracing::info!(spec = %spec.name, stages = pipe.stages(), kv_blocks = pipe.kv_blocks(), stage_kv_bytes = ?opts.stage_kv_bytes(spec, placement), "pipelined serving layout");
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
    let mut coordinator = crate::coordinator::Coordinator::new(
        &spec,
        placement.clone(),
        remote.key()?,
        Duration::from_millis(remote.timeout_ms),
        Duration::from_millis(remote.member_ttl_ms),
    )?;
    coordinator.set_link(remote.link());
    let layout = opts.layout(&spec);
    let mut exec = crate::remote::ChainExecutor::new(coordinator, layout.max_ctx, layout.slots, layout.prefill_chunk);
    exec.set_micro_batch(placement.micro_batch_for_remote());
    if let Err(e) = exec.connect_within(Duration::from_millis(remote.connect_wait_ms)) {
        tracing::warn!(error = %format!("{e:#}"), "remote stages are not ready yet, connecting on the first request");
    }
    let mut cfg = serving_config(&spec, opts);
    cfg.prefix_cache.enabled = false;
    Ok(Scheduler::new(exec, cfg))
}

fn build_tp(spec: &Spec, opts: &ServeOptions, solved: &crate::solve::Solved) -> Result<Scheduler<crate::tp_exec::TpExecutor>> {
    let mut spec = spec.clone();
    if let Some(r) = &opts.runtime {
        spec.runtime = r.clone();
    }
    opts.speculative.apply(&mut spec)?;
    let spec = &spec;
    let rank_spec = crate::tp::rank_spec(spec, opts.tp)?;
    let mut layout = opts.layout(&rank_spec);
    if let Some(bytes) = opts.kv_cache_bytes {
        layout.kv_blocks = ((bytes as usize / opts.tp) / kv_block_bytes(&rank_spec, 1)).max(1);
    }
    let devices = solved.ordinals();
    anyhow::ensure!(devices.len() == opts.tp, "the solved plan has tp {} on {} devices", opts.tp, devices.len());
    let exec = crate::tp_exec::TpExecutor::load(spec, layout, &devices)?;
    check_pool_tokens(exec.kv_blocks() * crate::codegen::PAGE, layout.max_ctx)?;
    tracing::info!(spec = %spec.name, tp = opts.tp, devices = ?devices, max_model_len = layout.max_ctx, max_num_seqs = layout.slots, kv_blocks = exec.kv_blocks(), "tensor parallel serving layout");
    let cfg = serving_config(spec, opts);
    Ok(Scheduler::new(exec, cfg))
}

pub fn scheduler(spec: &Spec, opts: ServeOptions, device: usize) -> Result<Scheduler<GpuExecutor>> {
    build(spec, opts, device)
}

fn build(spec: &Spec, opts: ServeOptions, device: usize) -> Result<Scheduler<GpuExecutor>> {
    let mut spec = spec.clone();
    if let Some(r) = &opts.runtime {
        spec.runtime = r.clone();
    }
    opts.speculative.apply(&mut spec)?;
    let spec = &spec;
    let layout = opts.layout(spec);
    let rt = Runtime::load_stage(spec, layout, crate::runtime::StageRange { device, ..crate::runtime::StageRange::whole(spec) })?;
    check_pool_tokens(rt.kv_blocks() * crate::codegen::PAGE, layout.max_ctx)?;
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
    crate::state::sequence_bytes(spec)
}

struct Encoded {
    job: Job,
    plan: Result<Arc<crate::vision::MediaPlan>>,
}

struct EncodeLane {
    jobs: Option<Sender<Job>>,
    done: Receiver<Encoded>,
    thread: Option<std::thread::JoinHandle<()>>,
    pending: usize,
}

impl EncodeLane {
    fn start(encoder: Arc<crate::media_encode::MediaEncoder>) -> Result<Self> {
        let (jobs, inbox) = channel::<Job>();
        let (done_tx, done) = channel::<Encoded>();
        let thread = std::thread::Builder::new()
            .name("specrt-vision".into())
            .spawn(move || {
                while let Ok(first) = inbox.recv() {
                    let mut batch = vec![first];
                    while let Ok(j) = inbox.try_recv() {
                        batch.push(j);
                    }
                    let (gone, keep): (Vec<Job>, Vec<Job>) = batch.into_iter().partition(|j| j.cancel.load(Ordering::Relaxed));
                    for job in gone {
                        if done_tx.send(Encoded { job, plan: Err(anyhow::anyhow!(CANCELLED_BEFORE_ENCODE)) }).is_err() {
                            return;
                        }
                    }
                    let requests: Vec<crate::media_encode::MediaRequest> =
                        keep.iter().map(|j| crate::media_encode::MediaRequest { prompt: &j.request.prompt, media: &j.request.media }).collect();
                    let plans = encoder.plan_batch(&requests);
                    drop(requests);
                    for (job, plan) in keep.into_iter().zip(plans) {
                        if done_tx.send(Encoded { job, plan: plan.map(Arc::new) }).is_err() {
                            return;
                        }
                    }
                }
            })
            .context("vision encode thread")?;
        Ok(EncodeLane { jobs: Some(jobs), done, thread: Some(thread), pending: 0 })
    }

    fn send(&mut self, job: Job) {
        self.pending += 1;
        if let Some(tx) = &self.jobs {
            let _ = tx.send(job);
        }
    }

    fn ready(&mut self, wait: Option<Duration>) -> Vec<Encoded> {
        let mut out = Vec::new();
        if let Some(d) = wait.filter(|_| self.pending > 0)
            && let Ok(e) = self.done.recv_timeout(d)
        {
            out.push(e);
        }
        while let Ok(e) = self.done.try_recv() {
            out.push(e);
        }
        self.pending -= out.len();
        out
    }
}

impl Drop for EncodeLane {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

const CANCELLED_BEFORE_ENCODE: &str = "cancelled before encoding";

fn cancelled(job: &Job) {
    let _ = job.events.send(SeqEvent::Finished { finish: Finish::Cancelled, prompt_tokens: 0, completion_tokens: 0, cached_tokens: 0 });
}

fn run<E: Executor>(sched: &mut Scheduler<E>, inbox: Receiver<Job>, stats: Arc<Mutex<Stats>>, pulse: Arc<Pulse>) {
    let mut live: HashMap<SeqId, Live> = HashMap::new();
    let mut open = true;
    {
        let mut s = stats.lock().unwrap();
        s.kv_blocks_total = sched.blocks_total();
    }
    let mut lane = match sched.exec.media_encoder().map(EncodeLane::start) {
        Some(Ok(l)) => Some(l),
        Some(Err(e)) => {
            tracing::error!(error = %format!("{e:#}"), "no vision encode thread, encoding on the engine thread");
            None
        }
        None => None,
    };
    let encoding = |lane: &Option<EncodeLane>| lane.as_ref().map_or(0, |l| l.pending);
    while open || !live.is_empty() || encoding(&lane) > 0 {
        let mut ready: Vec<Encoded> = Vec::new();
        if let Some(l) = lane.as_mut() {
            ready = l.ready(live.is_empty().then_some(Duration::from_millis(20)));
        }
        let first = if live.is_empty() && ready.is_empty() && encoding(&lane) == 0 {
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
            if job.cancel.load(Ordering::Relaxed) {
                cancelled(&job);
                continue;
            }
            if let Some(l) = lane.as_mut().filter(|_| !job.request.media.is_empty()) {
                match sched.check(&job.request) {
                    Ok(()) => l.send(job),
                    Err(e) => {
                        let _ = job.events.send(SeqEvent::Failed(format!("{e:#}")));
                    }
                }
                continue;
            }
            pulse.enter();
            let submitted = sched.submit(job.request);
            pulse.leave();
            match submitted {
                Ok(id) => {
                    live.insert(id, Live { events: job.events, cancel: job.cancel, last_token: None });
                }
                Err(e) => {
                    let _ = job.events.send(SeqEvent::Failed(format!("{e:#}")));
                }
            }
        }
        for Encoded { job, plan } in ready {
            if job.cancel.load(Ordering::Relaxed) {
                cancelled(&job);
                continue;
            }
            let submitted = plan.and_then(|p| sched.submit_planned(job.request, Some(p)));
            match submitted {
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
        if !open && live.is_empty() && encoding(&lane) == 0 {
            break;
        }
        if let Some(reason) = sched.exec.refresh()
            && !live.is_empty() {
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
        pulse.enter();
        sched.tick_sessions();
        pulse.leave();
        let started = Instant::now();
        pulse.enter();
        let report = if sched.idle() { Ok(None) } else { sched.step(&mut events).map(Some) };
        pulse.leave();
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
            tracing::debug!(
                rows = r.decode_rows,
                mtp = r.speculated,
                prefill_tokens = r.prefill_tokens,
                step_ms = started.elapsed().as_secs_f64() * 1e3,
                "step"
            );
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
                    s.generated_tokens += 1;
                    if let Some(l) = live.get_mut(&id) {
                        if let Some(prev) = l.last_token {
                            s.inter_token_secs.observe(now.duration_since(prev).as_secs_f64());
                        }
                        l.last_token = Some(now);
                    }
                    (id, SeqEvent::Token(token))
                }
                Event::Preempted { id } => (id, SeqEvent::Preempted),
                Event::Logprob { id, logprob } => (id, SeqEvent::Logprob(logprob)),
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
        s.waiting = sched.waiting() + encoding(&lane);
        s.kv_blocks_used = sched.blocks_live();
        s.kv_blocks_cached = sched.blocks_cached();
        s.preemptions = sched.preemptions();
        s.swap_preemptions = sched.swapped_out();
        s.swapped_bytes_total = sched.swapped_bytes_total();
        s.swap_secs = sched.swap_secs();
        s.reprefilled_tokens = sched.recomputed_tokens();
        s.chain = sched.chain_metrics();
        s.offload = sched.offload_stats();
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
    pub logprobs: Vec<TokenLogprobText>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobText {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
    pub top: Vec<(String, Vec<u8>, f32)>,
}

pub struct Serving {
    core: Arc<Core>,
    assets: Arc<Assets>,
    tokens: Arc<std::sync::OnceLock<Arc<crate::grammar::TokenTable>>>,
    grammars: Arc<Mutex<GrammarCache>>,
}

pub struct GrammarCache {
    cap: usize,
    order: Vec<(String, Arc<crate::grammar::Grammar>)>,
}

impl GrammarCache {
    pub fn new(cap: usize) -> Self {
        GrammarCache { cap, order: Vec::new() }
    }

    pub fn get_or_build(&mut self, key: String, build: impl FnOnce() -> Arc<crate::grammar::Grammar>) -> Arc<crate::grammar::Grammar> {
        if let Some(i) = self.order.iter().position(|(k, _)| *k == key) {
            let hit = self.order.remove(i);
            let g = hit.1.clone();
            self.order.push(hit);
            return g;
        }
        let g = build();
        if self.cap > 0 {
            if self.order.len() >= self.cap {
                self.order.remove(0);
            }
            self.order.push((key, g.clone()));
        }
        g
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

impl Clone for Serving {
    fn clone(&self) -> Self {
        Serving { core: self.core.clone(), assets: self.assets.clone(), tokens: self.tokens.clone(), grammars: self.grammars.clone() }
    }
}

impl Serving {
    pub fn sampling_defaults(&self) -> crate::engine::SamplingDefaults {
        self.assets.sampling
    }

    pub fn thinking(&self) -> crate::chat::Thinking {
        self.assets.chat.as_ref().map(|c| c.thinking()).unwrap_or_default()
    }

    pub fn tool_format(&self) -> crate::chat::ToolFormat {
        self.assets.chat.as_ref().map(|c| c.tool_format()).unwrap_or_default()
    }

    pub fn props(&self) -> serde_json::Value {
        use serde_json::json;
        let d = self.assets.sampling;
        let chat = self.assets.chat.as_ref();
        json!({
            "chat_template": chat.map(|c| c.source()).unwrap_or(""),
            "bos_token": chat.map(|c| c.bos_token()).unwrap_or(""),
            "eos_token": chat.map(|c| c.eos_token()).unwrap_or(""),
            "chat_template_caps": chat.map(|c| c.capabilities()).unwrap_or_else(|| json!({})),
            "thinking": format!("{:?}", self.thinking()),
            "tool_format": format!("{:?}", self.tool_format()),
            "n_ctx": self.max_model_len(),
            "generation": {"top_k": d.top_k, "top_p": d.top_p, "min_p": d.min_p, "max_new_tokens": d.max_new_tokens},
            "vision": self.accepts_media(),
        })
    }

    pub fn load(model: &str, opts: ServeOptions) -> Result<Self> {
        let assets = Assets::load(model)?;
        let grammars = Arc::new(Mutex::new(GrammarCache::new(opts.grammar_cache)));
        let core = Core::start(&assets.spec, opts)?;
        Ok(Serving { core, assets: Arc::new(assets), tokens: Default::default(), grammars })
    }

    pub fn spec(&self) -> &Spec {
        &self.assets.spec
    }

    pub fn stats(&self) -> Stats {
        self.core.stats()
    }

    pub fn sessions(&self) -> Option<Arc<crate::sessions::Shared>> {
        self.core.sessions()
    }

    pub fn max_model_len(&self) -> usize {
        self.core.max_model_len()
    }

    pub fn failure(&self) -> Option<String> {
        self.core.failure()
    }

    pub fn encode_messages(&self, messages: &[Value], template: Option<&Value>) -> Result<Vec<u32>> {
        self.assets.encode_messages(messages, template)
    }

    pub fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        self.assets.tokens(text)
    }

    pub fn media_refusal(&self) -> Option<String> {
        if self.accepts_media() {
            return None;
        }
        Some(match (&self.assets.spec.vision, &self.assets.media) {
            (None, _) => "this model's config has no vision_config".to_string(),
            (Some(_), None) => "the engine did not load the image preprocessing assets (the preprocessor configs are missing)".to_string(),
            (Some(_), Some(_)) => "the engine did not load a vision encoder (the checkpoint has no vision tensors)".to_string(),
        })
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
        self.complete_with_logprobs(prompt, media, max_tokens, sampling, use_stop, salt, cancel, None, on_text, &mut |_| {})
    }

    pub fn grammar(&self, schema: &crate::grammar::Schema) -> Arc<crate::grammar::Grammar> {
        let table = self.tokens.get_or_init(|| Arc::new(crate::grammar::TokenTable::from_tokenizer(&self.assets.tokenizer))).clone();
        let build = || schema.build(table, self.assets.spec.dims.vocab, self.assets.stop.clone());
        self.grammars.lock().unwrap().get_or_build(schema.key(), build)
    }

    pub fn token_text(&self, id: u32) -> (String, Vec<u8>) {
        let text = self.assets.tokenizer.decode(&[id], false).unwrap_or_default();
        let bytes = text.as_bytes().to_vec();
        (text, bytes)
    }

    fn logprob_text(&self, lp: &crate::logprobs::TokenLogprob) -> TokenLogprobText {
        let (token, bytes) = self.token_text(lp.token);
        let top = lp.top.iter().map(|&(t, l)| {
            let (s, b) = self.token_text(t);
            (s, b, l)
        }).collect();
        TokenLogprobText { token, bytes, logprob: lp.logprob, top }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn complete_with_logprobs(
        &self,
        prompt: Vec<u32>,
        media: Vec<sched::MediaInput>,
        max_tokens: usize,
        sampling: Sampling,
        use_stop: bool,
        salt: String,
        cancel: Arc<AtomicBool>,
        schema: Option<&crate::grammar::Schema>,
        on_text: &mut dyn FnMut(&str) -> bool,
        on_logprob: &mut dyn FnMut(&TokenLogprobText),
    ) -> Result<Completion> {
        let started = Instant::now();
        let max_tokens = crate::engine::output_budget(max_tokens, prompt.len(), self.core.max_model_len(), crate::spec::MAX_DRAFT_TOKENS, &self.assets.sampling);
        let (tx, rx) = channel();
        let stop = if use_stop { self.assets.stop.clone() } else { Vec::new() };
        let grammar = schema.map(|s| self.grammar(s));
        let request = sched::Request { prompt, max_tokens, sampling, stop, salt, media, grammar };
        self.core.submit(Job { request, events: tx, cancel: cancel.clone() })?;
        let tokenizer = &self.assets.tokenizer;
        let skip = self.assets.chat.as_ref().is_none_or(|c| c.skips_special_tokens());
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
                    if let Ok(text) = tokenizer.decode(&ids, skip)
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
                SeqEvent::Logprob(lp) => {
                    let entry = self.logprob_text(&lp);
                    on_logprob(&entry);
                    out.logprobs.push(entry);
                }
                SeqEvent::Failed(e) => anyhow::bail!(e),
            }
        }
        out.logprobs.truncate(out.completion_tokens);
        ids.truncate(out.completion_tokens);
        let text = tokenizer.decode(&ids, skip).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
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

#[cfg(test)]
mod snapshot_budget_tests {
    use super::*;

    #[test]
    fn the_default_scales_with_seqs_and_is_capped_by_host_memory_and_an_explicit_value_wins() {
        let snap = 157u64 << 20;
        let gb = 1u64 << 30;
        assert_eq!(auto_snapshots(32, 8, snap, 60 * gb), 32);
        assert_eq!(auto_snapshots(32, 37, snap, 100 * gb), 148);
        assert_eq!(auto_snapshots(32, 37, snap, 60 * gb), 97);
        assert_eq!(auto_snapshots(32, 37, snap, 8 * gb), 13);
        assert_eq!(auto_snapshots(32, 37, snap, 0), 148);
        assert_eq!(auto_snapshots(7, 37, snap, 60 * gb), 7);
        assert_eq!(auto_snapshots(0, 37, snap, 60 * gb), 0);
    }
}

#[cfg(test)]
mod watchdog_tests {
    use super::*;

    fn job() -> (Job, std::sync::mpsc::Receiver<SeqEvent>) {
        let (tx, rx) = channel();
        let request = sched::Request { prompt: vec![1], max_tokens: 1, sampling: crate::runtime::Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None };
        (Job { request, events: tx, cancel: Arc::new(AtomicBool::new(false)) }, rx)
    }

    fn hanging_core(watchdog: Option<Duration>, release: Arc<AtomicBool>) -> Arc<Core> {
        Core::spawn(512, 1, watchdog, None, move |ready, inbox, _stats, pulse| {
            let _ = ready.send(Ok(false));
            let Ok(_job) = inbox.recv() else { return };
            pulse.enter();
            while !release.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(5));
            }
            pulse.leave();
        })
        .unwrap()
    }

    #[test]
    fn an_engine_that_never_returns_fails_requests_and_marks_the_core_failed() {
        let release = Arc::new(AtomicBool::new(false));
        let core = hanging_core(Some(Duration::from_millis(100)), release.clone());
        let (j, rx) = job();
        core.submit(j).unwrap();
        match rx.recv_timeout(Duration::from_secs(5)).expect("the watchdog must answer") {
            SeqEvent::Failed(e) => assert!(e.contains(HUNG_MARKER), "{e}"),
            other => panic!("{other:?}"),
        }
        assert!(core.failure().unwrap().contains(HUNG_MARKER));
        let (j, _rx) = job();
        let e = core.submit(j).unwrap_err().to_string();
        assert!(e.contains(HUNG_MARKER), "{e}");
        release.store(true, Ordering::SeqCst);
        core.stop();
    }

    struct CountingExec {
        plans: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Executor for CountingExec {
        fn max_ctx(&self) -> usize {
            4096
        }
        fn slots(&self) -> usize {
            2
        }
        fn kv_blocks(&self) -> usize {
            64
        }
        fn chunk_limit(&self) -> usize {
            64
        }
        fn set_table(&mut self, _slot: usize, _blocks: &[u32]) -> Result<()> {
            Ok(())
        }
        fn prefill(&mut self, _slot: usize, _tokens: &[u32], _start: usize, _sample: Option<(crate::runtime::Sampling, u32)>) -> Result<Option<u32>> {
            Ok(Some(1))
        }
        fn decode(&mut self, rows: &[crate::runtime::DecodeRow]) -> Result<Vec<u32>> {
            Ok(vec![1; rows.len()])
        }
        fn plan_media(&mut self, _prompt: &[u32], _media: &[sched::MediaInput]) -> Result<Arc<crate::vision::MediaPlan>> {
            self.plans.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("counted")
        }
    }

    fn media_job(cancelled: bool) -> (Job, std::sync::mpsc::Receiver<SeqEvent>) {
        let (mut j, rx) = job();
        j.cancel = Arc::new(AtomicBool::new(cancelled));
        j.request.media = vec![sched::MediaInput {
            processed: crate::vision::Processed::Image(crate::vision::Patches { grid: [1, 80, 60], data: Vec::new() }),
            hash: [7; 32],
        }];
        (j, rx)
    }

    #[test]
    fn a_classify_burst_whose_clients_timed_out_is_never_encoded() {
        let plans = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut sched = Scheduler::new(CountingExec { plans: plans.clone() }, Config::default());
        let (tx, inbox) = channel::<Job>();
        let mut gone_rx = Vec::new();
        for _ in 0..7 {
            let (gone, rx) = media_job(true);
            tx.send(gone).unwrap();
            gone_rx.push(rx);
        }
        let (stays, stays_rx) = media_job(false);
        tx.send(stays).unwrap();
        drop(tx);
        run(&mut sched, inbox, Arc::new(Mutex::new(Stats::default())), Arc::new(Pulse::new()));
        for rx in gone_rx {
            assert!(matches!(rx.try_recv().unwrap(), SeqEvent::Finished { finish: Finish::Cancelled, .. }));
        }
        assert!(matches!(stays_rx.try_recv().unwrap(), SeqEvent::Failed(_)));
        assert_eq!(plans.load(Ordering::SeqCst), 1, "only the live request may reach the vision encoder");
    }

    struct HeldEncoder {
        release: Arc<AtomicBool>,
        started: Arc<AtomicBool>,
    }

    impl crate::media_encode::EncodeImages for HeldEncoder {
        fn encode_images(&self, items: &[&crate::vision::Patches]) -> Result<Vec<Vec<f32>>> {
            self.started.store(true, Ordering::SeqCst);
            while !self.release.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(items.iter().map(|p| vec![0.5; p.grid.iter().product::<usize>() / 4 * 8]).collect())
        }
    }

    struct VisionExec {
        encoder: Arc<crate::media_encode::MediaEncoder>,
    }

    impl Executor for VisionExec {
        fn max_ctx(&self) -> usize {
            4096
        }
        fn slots(&self) -> usize {
            2
        }
        fn kv_blocks(&self) -> usize {
            64
        }
        fn chunk_limit(&self) -> usize {
            64
        }
        fn set_table(&mut self, _slot: usize, _blocks: &[u32]) -> Result<()> {
            Ok(())
        }
        fn prefill(&mut self, _slot: usize, _tokens: &[u32], _start: usize, _sample: Option<(crate::runtime::Sampling, u32)>) -> Result<Option<u32>> {
            Ok(Some(1))
        }
        fn decode(&mut self, rows: &[crate::runtime::DecodeRow]) -> Result<Vec<u32>> {
            Ok(vec![1; rows.len()])
        }
        fn plan_media(&mut self, _prompt: &[u32], _media: &[sched::MediaInput]) -> Result<Arc<crate::vision::MediaPlan>> {
            anyhow::bail!("the engine thread must not encode")
        }
        fn media_encoder(&self) -> Option<Arc<crate::media_encode::MediaEncoder>> {
            Some(self.encoder.clone())
        }
    }

    #[test]
    fn decode_keeps_stepping_while_an_image_encodes() {
        let release = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        let tokens = crate::vision::VisionTokens { image: 9, video: 10, start: 11, end: 12 };
        let encoder = crate::media_encode::MediaEncoder::new(
            Box::new(HeldEncoder { release: release.clone(), started: started.clone() }),
            tokens,
            2,
            crate::spec::VisionOptions::default(),
        );
        let mut sched = Scheduler::new(VisionExec { encoder: Arc::new(encoder) }, Config::default());
        let (tx, inbox) = channel::<Job>();
        let (mut image, image_rx) = job();
        image.request.prompt = vec![1, 11, 9, 9, 9, 9, 12, 2];
        image.request.max_tokens = 4;
        image.request.media = vec![sched::MediaInput {
            processed: crate::vision::Processed::Image(crate::vision::Patches { grid: [1, 4, 4], data: vec![0.0; 16] }),
            hash: [3; 32],
        }];
        tx.send(image).unwrap();
        let engine = std::thread::spawn(move || run(&mut sched, inbox, Arc::new(Mutex::new(Stats::default())), Arc::new(Pulse::new())));
        let deadline = Instant::now() + Duration::from_secs(5);
        while !started.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the encoder never started");
            std::thread::sleep(Duration::from_millis(1));
        }
        let (mut text, text_rx) = job();
        text.request.max_tokens = 40;
        tx.send(text).unwrap();
        let mut generated = 0;
        loop {
            match text_rx.recv_timeout(Duration::from_secs(5)).expect("the text request must finish while the image is still encoding") {
                SeqEvent::Token(_) => generated += 1,
                SeqEvent::Finished { completion_tokens, .. } => {
                    assert_eq!(completion_tokens, 40);
                    break;
                }
                SeqEvent::Failed(e) => panic!("{e}"),
                _ => {}
            }
        }
        assert_eq!(generated, 40);
        assert!(image_rx.try_recv().is_err(), "the image request is still encoding");
        release.store(true, Ordering::SeqCst);
        loop {
            match image_rx.recv_timeout(Duration::from_secs(5)).expect("the image request finishes once encoded") {
                SeqEvent::Finished { completion_tokens, .. } => {
                    assert_eq!(completion_tokens, 4);
                    break;
                }
                SeqEvent::Failed(e) => panic!("{e}"),
                _ => {}
            }
        }
        drop(tx);
        engine.join().unwrap();
    }

    #[test]
    fn a_slow_but_finishing_engine_is_not_failed() {
        let release = Arc::new(AtomicBool::new(false));
        let core = hanging_core(Some(Duration::from_secs(30)), release.clone());
        let (j, rx) = job();
        core.submit(j).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(core.failure().is_none());
        assert!(rx.try_recv().is_err());
        release.store(true, Ordering::SeqCst);
        core.stop();
    }
}

#[cfg(test)]
mod histogram_tests {
    use super::*;

    #[test]
    fn quantiles_interpolate_inside_the_bucket() {
        let mut h = Histogram::new(&LATENCY_BUCKETS);
        for _ in 0..50 {
            h.observe(0.03);
        }
        for _ in 0..50 {
            h.observe(0.4);
        }
        let p50 = h.quantile(0.5).unwrap();
        assert!((0.02..=0.035).contains(&p50), "{p50}");
        let p95 = h.quantile(0.95).unwrap();
        assert!((0.3..=0.5).contains(&p95), "{p95}");
        assert_eq!(Histogram::new(&LATENCY_BUCKETS).quantile(0.5), None);
    }

    #[test]
    fn a_window_is_the_difference_of_two_snapshots() {
        let mut h = Histogram::new(&LATENCY_BUCKETS);
        h.observe(0.03);
        let before = h.clone();
        h.observe(2.0);
        h.observe(2.0);
        let window = h.since(&before);
        assert_eq!(window.total, 2);
        assert!(window.quantile(0.5).unwrap() > 1.0);
        assert_eq!(h.since(&h).quantile(0.5), None);
    }
}

#[cfg(test)]
mod placement_tests {
    use super::*;
    use crate::options::Profile;
    use crate::placement::{DeviceConfig, LinkConfig};
    use crate::solve::{PlacementRequest, Target, solve};

    fn spec() -> Spec {
        Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap()
    }

    fn device(i: usize, gib: f64) -> DeviceConfig {
        DeviceConfig { name: format!("box/gpu{i}"), memory_gib: gib, bandwidth_gbs: 1792.0, tflops: 500.0, ordinal: Some(i), address: None, reserve_gib: Some(0.0), link: LinkConfig::default(), offload: None, host: Some("box".into()), peer: None, wgpu: false }
    }

    fn solved(mode: Profile, devices: &[DeviceConfig]) -> crate::solve::Solved {
        let spec = spec();
        solve(&PlacementRequest { mode, target: Target::default(), pins: Default::default(), devices, spec: &spec }).unwrap()
    }

    #[test]
    fn serve_options_come_from_a_solved_plan_alone_without_a_placement_config() {
        let s = solved(Profile::MaxPerformance, &[device(0, 86.0), device(1, 86.0)]);
        let opts = ServeOptions::from_solved(&s, ServeOptions::default()).unwrap();
        assert!(opts.placement.is_none());
        assert_eq!(opts.tp, s.tp);
        assert_eq!(opts.max_num_seqs, s.slots);
        assert_eq!(opts.max_model_len, Some(s.context));
        assert_eq!(opts.kv_cache_bytes, Some(s.kv_pool_bytes));
        assert_eq!(opts.runtime.as_ref().unwrap().kv_dtype, s.kv_dtype);
        assert_eq!(opts.solved.as_deref(), Some(&s));
    }

    #[test]
    fn a_pipeline_solved_plan_sizes_each_stage_kv_from_its_own_layers_within_the_pool() {
        let mut s = spec();
        s.attention.kv_heads = 3;
        let devices = [device(0, 12.0), device(1, 12.0)];
        let plan = solve(&PlacementRequest { mode: Profile::MaxPerformance, target: Target::default(), pins: Default::default(), devices: &devices, spec: &s }).unwrap();
        assert_eq!(plan.placement.stages.len(), 2);
        let opts = ServeOptions::from_solved(&plan, ServeOptions::default()).unwrap();
        let mut resolved = s.clone();
        resolved.runtime = opts.runtime.clone().unwrap();
        let per_stage = opts.stage_kv_bytes(&resolved, &plan.placement);
        assert!(per_stage.iter().sum::<u64>() <= plan.kv_pool_bytes, "{per_stage:?} against {}", plan.kv_pool_bytes);
        for (bytes, stage) in per_stage.iter().zip(&plan.placement.stages) {
            let block = kv_block_bytes(&resolved.stage(stage.start, stage.end), 1) as u64;
            assert!(*bytes <= stage.kv_bytes + block, "stage {}..{}: {bytes} against planned {}", stage.start, stage.end, stage.kv_bytes);
        }
    }

    #[test]
    fn a_stage_holds_the_f16_kv_layers_that_fall_inside_its_range() {
        let mut s = spec();
        s.mtp = None;
        s.runtime.kv_dtype = crate::spec::KvDtype::Fp8;
        s.runtime.kv_f16_layers = Some(10);
        let whole = kv_block_bytes(&s, 1);
        let parts: usize = [(0, 36), (36, 64)].iter().map(|&(a, b)| kv_block_bytes(&s.stage(a, b), 1)).sum();
        assert_eq!(whole, parts);
    }

    #[test]
    fn the_grammar_cache_keeps_the_most_recent_schemas() {
        let table = Arc::new(crate::grammar::TokenTable::new(vec![Vec::new(), b"{".to_vec()]));
        let build = || crate::grammar::Grammar::json_object(table.clone(), 2, vec![0]);
        let mut cache = GrammarCache::new(2);
        let a = cache.get_or_build("a".into(), build);
        cache.get_or_build("b".into(), build);
        assert!(Arc::ptr_eq(&a, &cache.get_or_build("a".into(), build)));
        cache.get_or_build("c".into(), build);
        assert_eq!(cache.len(), 2);
        let again = cache.get_or_build("b".into(), build);
        assert!(!Arc::ptr_eq(&a, &again));
        assert!(Arc::ptr_eq(&a, &cache.get_or_build("a".into(), build)) || cache.len() == 2);
        let mut off = GrammarCache::new(0);
        off.get_or_build("a".into(), build);
        assert!(off.is_empty());
    }
}
