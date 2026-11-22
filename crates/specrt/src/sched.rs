use crate::codegen::PAGE;
use crate::prefix::{self, Hash, PrefixCache, StateSnapshot};
use crate::runtime::{Mask, DecodeRow, PrefillPart, Runtime, Sampling, SpecRow, Stepper};
use crate::spec::PrefixCacheOptions;
use anyhow::{Result, bail};
use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub type SeqId = u64;

pub fn blocks_for(tokens: usize) -> usize {
    tokens.div_ceil(PAGE)
}

#[derive(Debug, Clone)]
pub struct BlockPool {
    free: Vec<u32>,
    refs: Vec<u32>,
    cached: Vec<bool>,
    cache_only: usize,
    total: usize,
}

impl BlockPool {
    pub fn new(total: usize) -> Self {
        BlockPool { free: (0..total as u32).rev().collect(), refs: vec![0; total], cached: vec![false; total], cache_only: 0, total }
    }

    pub fn refs(&self, block: u32) -> u32 {
        self.refs[block as usize]
    }

    pub fn retain(&mut self, block: u32) {
        let i = block as usize;
        debug_assert!(self.refs[i] > 0);
        if self.cached[i] && self.refs[i] == 1 {
            self.cache_only -= 1;
        }
        self.refs[i] += 1;
    }

    pub fn retain_for_cache(&mut self, block: u32) {
        self.retain(block);
        self.cached[block as usize] = true;
    }

    pub fn release_from_cache(&mut self, block: u32) {
        let i = block as usize;
        if self.cached[i] && self.refs[i] == 1 {
            self.cache_only -= 1;
        }
        self.cached[i] = false;
        self.release(&mut vec![block]);
    }

    pub fn cache_only(&self) -> usize {
        self.cache_only
    }

    pub fn live(&self) -> usize {
        self.used() - self.cache_only
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    pub fn used(&self) -> usize {
        self.total - self.free.len()
    }

    pub fn alloc(&mut self) -> Option<u32> {
        let block = self.free.pop()?;
        self.refs[block as usize] = 1;
        Some(block)
    }

    pub fn release(&mut self, blocks: &mut Vec<u32>) {
        for b in blocks.drain(..) {
            debug_assert!((b as usize) < self.total && self.refs[b as usize] > 0);
            self.refs[b as usize] -= 1;
            match self.refs[b as usize] {
                0 => self.free.push(b),
                1 if self.cached[b as usize] => self.cache_only += 1,
                _ => {}
            }
        }
        debug_assert!(self.free.len() <= self.total);
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageMetric {
    pub name: String,
    pub device: usize,
    pub start: usize,
    pub end: usize,
    pub busy_secs: f64,
    pub calls: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChainMetrics {
    pub stages: Vec<StageMetric>,
    pub hops: u64,
    pub hop_bytes: u64,
    pub hop_secs: f64,
    pub micro_batches: u64,
    pub overlapped_prefills: u64,
    pub lost: Option<String>,
}

pub struct HostSeq {
    pub bytes: usize,
    pub blocks: usize,
    pub payload: Box<dyn Any + Send>,
}

pub trait Executor {
    fn swap_bytes(&self, _blocks: usize) -> usize {
        0
    }
    fn set_host_limit(&mut self, _bytes: usize) {}
    fn host_used(&self) -> usize {
        0
    }
    fn park(&mut self, _slot: usize, _blocks: &[u32]) -> Result<Option<HostSeq>> {
        Ok(None)
    }
    fn resume(&mut self, _slot: usize, _blocks: &[u32], _host: &HostSeq) -> Result<()> {
        bail!("this executor cannot resume a parked sequence")
    }
    fn chain_metrics(&self) -> Option<ChainMetrics> {
        None
    }
    fn max_ctx(&self) -> usize;
    fn slots(&self) -> usize;
    fn kv_blocks(&self) -> usize;
    fn chunk_limit(&self) -> usize;
    fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()>;
    fn acquire_state(&mut self, _slot: usize) -> Result<()> {
        Ok(())
    }
    fn release_state(&mut self, _slot: usize) {}
    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>>;
    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>>;
    fn fused(
        &mut self,
        rows: &[DecodeRow],
        slot: usize,
        tokens: &[u32],
        start: usize,
        sample: Option<(Sampling, u32)>,
    ) -> Result<(Vec<u32>, Option<u32>)> {
        let next = self.decode(rows)?;
        let first = self.prefill(slot, tokens, start, sample)?;
        Ok((next, first))
    }
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
    fn refresh(&mut self) -> Option<String> {
        None
    }
    fn draft_tokens(&self) -> usize {
        0
    }
    fn speculation_cap(&self) -> usize {
        0
    }
    fn speculate(&mut self, _rows: &[SpecRow]) -> Result<Vec<Vec<u32>>> {
        bail!("this executor does not speculate")
    }
    fn supports_prefix_cache(&self) -> bool {
        false
    }
    fn plan_media(&mut self, _prompt: &[u32], _media: &[MediaInput]) -> Result<Arc<crate::vision::MediaPlan>> {
        bail!("this executor has no vision encoder")
    }
    fn set_media(&mut self, _slot: usize, _plan: Option<&crate::vision::MediaPlan>) -> Result<()> {
        Ok(())
    }
    fn media_encoder(&self) -> Option<Arc<crate::media_encode::MediaEncoder>> {
        None
    }
    fn hybrid(&self) -> bool {
        false
    }
    fn prefix_root(&self, salt: &str) -> Hash {
        prefix::bare_namespace(salt)
    }
    fn snapshot(&mut self, _slot: usize) -> Result<StateSnapshot> {
        bail!("this executor cannot snapshot recurrent state")
    }
    fn restore(&mut self, _slot: usize, _state: &StateSnapshot) -> Result<()> {
        bail!("this executor cannot restore recurrent state")
    }
    fn copy_block(&mut self, _src: u32, _dst: u32) -> Result<()> {
        bail!("this executor cannot copy KV blocks")
    }
    fn session_compat(&self) -> Option<crate::sessions::Compat> {
        None
    }
    fn save_session(&mut self, _blocks: &[u32], _state: Option<&StateSnapshot>) -> Result<Vec<Vec<u8>>> {
        bail!("this executor cannot save sessions")
    }
    fn restore_session(&mut self, _slot: usize, _blocks: &[u32], _sections: &[Vec<u8>]) -> Result<()> {
        bail!("this executor cannot restore sessions")
    }
    fn supports_logprobs(&self) -> bool {
        false
    }
    fn supports_grammar(&self) -> bool {
        false
    }
    fn set_masks(&mut self, _rows: Vec<Option<Mask>>, _first: Option<Mask>) {}
    fn logits_of(&mut self, _at: LogitsAt) -> Result<Vec<f32>> {
        bail!("this executor cannot return logits")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsAt {
    Decode(usize),
    Prefill,
}

pub struct GpuExecutor {
    pub rt: Runtime,
    st: Stepper,
    fuse: bool,
    fused_prefill: Option<usize>,
}

impl GpuExecutor {
    pub fn new(rt: Runtime) -> Result<Self> {
        let st = rt.stepper()?;
        let fuse = rt.spec().runtime.fuse && rt.draft_tokens().is_none() && !rt.spec().uses_branch() && rt.spec().mixer.is_none() && !rt.spec().layers.split;
        Ok(GpuExecutor { rt, st, fuse, fused_prefill: None })
    }

    pub fn capture_secs(&self) -> &[(usize, f64)] {
        self.st.capture_secs()
    }
}

impl Executor for GpuExecutor {
    fn max_ctx(&self) -> usize {
        self.rt.max_ctx()
    }

    fn slots(&self) -> usize {
        self.rt.slots()
    }

    fn kv_blocks(&self) -> usize {
        self.rt.kv_blocks()
    }

    fn chunk_limit(&self) -> usize {
        self.rt.prefill_chunk_limit()
    }

    fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
        self.rt.set_block_table(slot, blocks)
    }

    fn acquire_state(&mut self, slot: usize) -> Result<()> {
        self.rt.acquire_state(slot)
    }

    fn release_state(&mut self, slot: usize) {
        self.rt.release_state(slot)
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
        self.fused_prefill = None;
        self.rt.prefill_slot(&mut self.st, slot, tokens, start, sample)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        self.rt.decode(&mut self.st, rows)
    }

    fn supports_logprobs(&self) -> bool {
        true
    }

    fn supports_grammar(&self) -> bool {
        true
    }

    fn set_masks(&mut self, rows: Vec<Option<Mask>>, first: Option<Mask>) {
        self.st.set_masks(rows, first)
    }

    fn logits_of(&mut self, at: LogitsAt) -> Result<Vec<f32>> {
        match (at, self.fused_prefill) {
            (LogitsAt::Decode(i), _) | (LogitsAt::Prefill, Some(i)) => self.rt.step_logits(&self.st, i, 1),
            (LogitsAt::Prefill, None) => self.rt.prefill_logits(),
        }
    }

    fn fused(
        &mut self,
        rows: &[DecodeRow],
        slot: usize,
        tokens: &[u32],
        start: usize,
        sample: Option<(Sampling, u32)>,
    ) -> Result<(Vec<u32>, Option<u32>)> {
        self.fused_prefill = self.fuse.then_some(rows.len());
        if !self.fuse {
            let next = self.rt.decode(&mut self.st, rows)?;
            let first = self.rt.prefill_slot(&mut self.st, slot, tokens, start, sample)?;
            return Ok((next, first));
        }
        self.rt.fused_step(&mut self.st, rows, PrefillPart { slot, tokens, start, sample })
    }

    fn flush(&mut self) -> Result<()> {
        self.rt.synchronize()
    }

    fn draft_tokens(&self) -> usize {
        self.rt.draft_tokens().unwrap_or(0)
    }

    fn speculation_cap(&self) -> usize {
        self.rt.speculation_cap()
    }

    fn speculate(&mut self, rows: &[SpecRow]) -> Result<Vec<Vec<u32>>> {
        self.rt.speculate(rows)
    }

    fn supports_prefix_cache(&self) -> bool {
        true
    }

    fn plan_media(&mut self, prompt: &[u32], media: &[MediaInput]) -> Result<Arc<crate::vision::MediaPlan>> {
        self.rt.plan_media(prompt, media)
    }

    fn media_encoder(&self) -> Option<Arc<crate::media_encode::MediaEncoder>> {
        self.rt.media_encoder()
    }

    fn set_media(&mut self, slot: usize, plan: Option<&crate::vision::MediaPlan>) -> Result<()> {
        self.rt.set_media(slot, plan)
    }

    fn hybrid(&self) -> bool {
        self.rt.spec().linear_layers() > 0
    }

    fn prefix_root(&self, salt: &str) -> Hash {
        prefix::namespace(self.rt.spec(), self.rt.kv_planes(), salt)
    }

    fn snapshot(&mut self, slot: usize) -> Result<StateSnapshot> {
        self.rt.snapshot_state(slot)
    }

    fn restore(&mut self, slot: usize, state: &StateSnapshot) -> Result<()> {
        self.rt.restore_state(slot, state)
    }

    fn copy_block(&mut self, src: u32, dst: u32) -> Result<()> {
        self.rt.copy_kv_block(src, dst)
    }

    fn swap_bytes(&self, blocks: usize) -> usize {
        self.rt.swap_bytes(blocks)
    }

    fn set_host_limit(&mut self, bytes: usize) {
        self.rt.host_pool().set_limit(bytes);
    }

    fn host_used(&self) -> usize {
        self.rt.host_pool().used()
    }

    fn park(&mut self, slot: usize, blocks: &[u32]) -> Result<Option<HostSeq>> {
        self.rt.park(slot, blocks).map(Some)
    }

    fn resume(&mut self, slot: usize, blocks: &[u32], host: &HostSeq) -> Result<()> {
        self.rt.resume(slot, blocks, host)
    }

    fn session_compat(&self) -> Option<crate::sessions::Compat> {
        Some(self.rt.session_compat())
    }

    fn save_session(&mut self, blocks: &[u32], state: Option<&StateSnapshot>) -> Result<Vec<Vec<u8>>> {
        let (k, v) = self.rt.save_kv_blocks(blocks)?;
        Ok(crate::sessions::pack_sections(k, v, state))
    }

    fn restore_session(&mut self, slot: usize, blocks: &[u32], sections: &[Vec<u8>]) -> Result<()> {
        let state = crate::sessions::unpack_state(sections)?;
        self.rt.load_kv_blocks(blocks, &sections[crate::sessions::SECTION_K], &sections[crate::sessions::SECTION_V])?;
        if let Some(state) = state {
            self.rt.restore_state(slot, &state)?;
        }
        Ok(())
    }
}

pub struct PipelineExecutor {
    pub pipe: crate::pipeline::Pipeline,
}

impl PipelineExecutor {
    pub fn new(pipe: crate::pipeline::Pipeline) -> Self {
        PipelineExecutor { pipe }
    }
}

impl Executor for PipelineExecutor {
    fn max_ctx(&self) -> usize {
        self.pipe.max_ctx()
    }

    fn slots(&self) -> usize {
        self.pipe.slots()
    }

    fn kv_blocks(&self) -> usize {
        self.pipe.kv_blocks()
    }

    fn chunk_limit(&self) -> usize {
        self.pipe.chunk_limit()
    }

    fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
        self.pipe.set_block_table(slot, blocks)
    }

    fn acquire_state(&mut self, slot: usize) -> Result<()> {
        self.pipe.acquire_state(slot)
    }

    fn release_state(&mut self, slot: usize) {
        self.pipe.release_state(slot)
    }

    fn plan_media(&mut self, prompt: &[u32], media: &[MediaInput]) -> Result<Arc<crate::vision::MediaPlan>> {
        self.pipe.plan_media(prompt, media)
    }

    fn media_encoder(&self) -> Option<Arc<crate::media_encode::MediaEncoder>> {
        self.pipe.media_encoder()
    }

    fn set_media(&mut self, slot: usize, plan: Option<&crate::vision::MediaPlan>) -> Result<()> {
        self.pipe.set_media(slot, plan)
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
        self.pipe.prefill(slot, tokens, start, sample)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        self.pipe.decode(rows)
    }

    fn flush(&mut self) -> Result<()> {
        self.pipe.synchronize()
    }

    fn chain_metrics(&self) -> Option<ChainMetrics> {
        let hops = self.pipe.stats();
        let prefill = self.pipe.prefill_stats();
        let stages = self
            .pipe
            .ranges()
            .iter()
            .zip(self.pipe.stage_stats())
            .map(|(&(device, start, end), s)| StageMetric { name: String::new(), device, start, end, busy_secs: s.busy.as_secs_f64(), calls: s.calls, tokens: s.tokens })
            .collect();
        Some(ChainMetrics {
            stages,
            hops: hops.hops,
            hop_bytes: hops.bytes,
            hop_secs: hops.time.as_secs_f64(),
            micro_batches: prefill.micro_batches,
            overlapped_prefills: prefill.overlapped_calls,
            lost: None,
        })
    }
}

const LONG_PROMPT: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub max_num_seqs: usize,
    pub max_batched_tokens: usize,
    pub itl_target: Duration,
    pub step_cost: f64,
    pub reserve_blocks: Option<usize>,
    pub min_prefill_chunk: usize,
    pub min_prefill_tokens: usize,
    pub shortest_first: bool,
    pub whole_prompt_tokens: usize,
    pub prefix_cache: PrefixCacheOptions,
    pub swap_space_bytes: u64,
    pub admit_lookahead: usize,
    pub prefill_burst_tokens: usize,
    pub max_long_in_flight: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_num_seqs: 32,
            max_batched_tokens: 8192,
            itl_target: Duration::ZERO,
            step_cost: 3.0,
            reserve_blocks: None,
            min_prefill_chunk: 64,
            min_prefill_tokens: 0,
            shortest_first: true,
            whole_prompt_tokens: 0,
            prefix_cache: PrefixCacheOptions::default(),
            swap_space_bytes: 0,
            admit_lookahead: 0,
            prefill_burst_tokens: 0,
            max_long_in_flight: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop: Vec<u32>,
    pub salt: String,
    pub media: Vec<MediaInput>,
    pub grammar: Option<Arc<crate::grammar::Grammar>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaInput {
    pub processed: crate::vision::Processed,
    pub hash: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Started { id: SeqId, queued: Duration },
    Token { id: SeqId, token: u32 },
    Preempted { id: SeqId },
    Finished { id: SeqId, finish: Finish, prompt_tokens: usize, completion_tokens: usize, cached_tokens: usize },
    Logprob { id: SeqId, logprob: crate::logprobs::TokenLogprob },
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepReport {
    pub decode_rows: usize,
    pub prefill_tokens: usize,
    pub preemptions: usize,
    pub decode_secs: f64,
    pub prefill_secs: f64,
    pub drafted: usize,
    pub accepted: usize,
    pub speculated: bool,
}

struct Seq {
    id: SeqId,
    req: Request,
    tokens: Vec<u32>,
    computed: usize,
    slot: Option<usize>,
    blocks: Vec<u32>,
    arrived: Instant,
    admitted: u64,
    started: bool,
    root: Hash,
    cached: usize,
    keys: Vec<u32>,
    media: Option<Arc<crate::vision::MediaPlan>>,
    cursor: Option<crate::grammar::Cursor>,
    parked: Option<HostSeq>,
}

impl Seq {
    fn generated(&self) -> usize {
        self.tokens.len() - self.req.prompt.len()
    }

    fn prefill_target(&self) -> usize {
        if self.generated() == 0 { self.tokens.len() } else { self.tokens.len() - 1 }
    }

    fn decoding(&self) -> bool {
        self.generated() > 0 && self.computed == self.tokens.len() - 1
    }
}

pub struct Scheduler<E: Executor> {
    pub exec: E,
    cfg: Config,
    pool: BlockPool,
    free_slots: Vec<usize>,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    next_id: SeqId,
    admissions: u64,
    preemptions: u64,
    swapped_out: u64,
    swapped_in: u64,
    swapped_bytes: u64,
    swapped_bytes_total: u64,
    swap_secs: f64,
    recomputed_tokens: u64,
    state_refusals: u64,
    prefill_rate: f64,
    decode_secs: f64,
    cache: Option<PrefixCache>,
    prefix_queries: u64,
    prefix_hits: u64,
    sessions: Option<Sessions>,
}

struct Held {
    scope: Hash,
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    state: Option<Arc<StateSnapshot>>,
}

struct Sessions {
    shared: Arc<crate::sessions::Shared>,
    janitor: crate::sessions::Janitor,
    held: std::collections::HashMap<String, Held>,
    clock: Box<dyn Fn() -> u64 + Send>,
    last_tick: u64,
}

struct Capturing<'a, E: Executor> {
    exec: &'a mut E,
    pool: &'a mut BlockPool,
    held: &'a mut std::collections::HashMap<String, Held>,
}

impl<E: Executor> crate::sessions::Capture for Capturing<'_, E> {
    fn capture(&mut self, key: &str) -> Result<crate::sessions::Session> {
        let h = self.held.get(key).ok_or_else(|| anyhow::anyhow!("session {key} is not resident"))?;
        let sections = self.exec.save_session(&h.blocks, h.state.as_deref())?;
        Ok(crate::sessions::Session { scope: h.scope, tokens: h.tokens.clone(), sections })
    }

    fn release(&mut self, key: &str) {
        if let Some(mut h) = self.held.remove(key) {
            self.pool.release(&mut h.blocks);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub queries: u64,
    pub hits: u64,
    pub blocks: usize,
    pub snapshots: usize,
    pub snapshot_bytes: usize,
    pub evictions: u64,
}

fn ewma(old: f64, new: f64) -> f64 {
    if old <= 0.0 { new } else { 0.8 * old + 0.2 * new }
}

impl<E: Executor> Scheduler<E> {
    pub fn new(exec: E, cfg: Config) -> Self {
        let pool = BlockPool::new(exec.kv_blocks());
        let free_slots = (0..exec.slots()).rev().collect();
        let cache = (cfg.prefix_cache.enabled && exec.supports_prefix_cache()).then(|| PrefixCache::new(&cfg.prefix_cache, exec.hybrid()));
        let mut exec = exec;
        if cfg.swap_space_bytes > 0 {
            exec.set_host_limit(cfg.swap_space_bytes as usize);
        }
        Scheduler {
            exec,
            cfg,
            pool,
            free_slots,
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 1,
            admissions: 0,
            preemptions: 0,
            swapped_out: 0,
            swapped_in: 0,
            swapped_bytes: 0,
            swapped_bytes_total: 0,
            swap_secs: 0.0,
            recomputed_tokens: 0,
            state_refusals: 0,
            prefill_rate: 0.0,
            decode_secs: 0.0,
            cache,
            prefix_queries: 0,
            prefix_hits: 0,
            sessions: None,
        }
    }

    pub fn enable_sessions(&mut self, shared: Arc<crate::sessions::Shared>) {
        let janitor = crate::sessions::Janitor::new(shared.options.clone());
        self.sessions = Some(Sessions { shared, janitor, held: Default::default(), clock: Box::new(crate::sessions::now_seconds), last_tick: 0 });
    }

    pub fn set_session_clock(&mut self, clock: Box<dyn Fn() -> u64 + Send>) {
        if let Some(s) = self.sessions.as_mut() {
            s.clock = clock;
        }
    }

    pub fn session_stats(&self) -> Option<crate::sessions::Stats> {
        self.sessions.as_ref().map(|s| s.shared.stats())
    }

    pub fn resident_sessions(&self) -> usize {
        self.sessions.as_ref().map_or(0, |s| s.held.len())
    }

    fn track_session(&mut self, i: usize) {
        let Some(sessions) = self.sessions.as_mut() else { return };
        let s = &self.running[i];
        let n = s.computed;
        if n < sessions.shared.options.min_tokens.max(1) || !s.req.media.is_empty() || s.slot.is_none() {
            return;
        }
        let Some(compat) = self.exec.session_compat() else { return };
        let state = if self.exec.hybrid() {
            match self.exec.snapshot(s.slot.expect("slot")) {
                Ok(st) => Some(Arc::new(st)),
                Err(e) => {
                    tracing::warn!(error = %format!("{e:#}"), "session state not captured");
                    return;
                }
            }
        } else {
            None
        };
        let tokens = s.keys[..n].to_vec();
        let scope = s.root;
        let id = crate::sessions::session_id(&compat, &scope, &tokens);
        let mut blocks = s.blocks[..blocks_for(n)].to_vec();
        for &b in &blocks {
            self.pool.retain(b);
        }
        let superseded: Vec<String> = sessions.held.iter().filter(|(_, h)| h.scope == scope && tokens.starts_with(&h.tokens)).map(|(k, _)| k.clone()).collect();
        for k in superseded {
            if let Some(mut h) = sessions.held.remove(&k) {
                self.pool.release(&mut h.blocks);
            }
        }
        if let Some(mut old) = sessions.held.remove(&id) {
            self.pool.release(&mut old.blocks);
        }
        let now = (sessions.clock)();
        sessions.janitor.tracker.touch(&id, now, n);
        sessions.held.insert(id, Held { scope, tokens, blocks: std::mem::take(&mut blocks), state });
        self.publish_residents();
    }

    fn publish_residents(&self) {
        let Some(sessions) = self.sessions.as_ref() else { return };
        let mut list: Vec<crate::sessions::Resident> = sessions
            .held
            .iter()
            .map(|(id, h)| crate::sessions::Resident { id: id.clone(), tokens: h.tokens.len(), last_used: sessions.janitor.tracker.last_request(id).map_or(0, |l| l.0) })
            .collect();
        list.sort_by(|a, b| a.id.cmp(&b.id));
        *sessions.shared.resident.lock().expect("residents") = list;
    }

    pub fn tick_sessions(&mut self) {
        let Some(sessions) = self.sessions.as_mut() else { return };
        let now = (sessions.clock)();
        if now == sessions.last_tick && sessions.shared.force.lock().expect("force").is_empty() {
            return;
        }
        let forced: Vec<String> = std::mem::take(&mut *sessions.shared.force.lock().expect("force"));
        let mut keys: Vec<String> = forced.into_iter().filter(|k| sessions.held.contains_key(k)).collect();
        keys.extend(sessions.janitor.tracker.due(now, sessions.janitor.options.idle_seconds, sessions.janitor.options.min_tokens).into_iter().filter(|k| sessions.held.contains_key(k)));
        keys.dedup();
        let store = &sessions.shared.store;
        store.expire(now, sessions.janitor.options.ttl_seconds());
        let mut cap = Capturing { exec: &mut self.exec, pool: &mut self.pool, held: &mut sessions.held };
        let written = sessions.janitor.save_keys(keys, now, store, &mut cap);
        for e in &written {
            tracing::info!(session = %e.id, tokens = e.tokens, bytes = e.bytes, "session snapshot written");
        }
        store.enforce_budget(sessions.janitor.options.max_bytes);
        sessions.last_tick = now;
        self.publish_residents();
    }

    fn spill_one_session(&mut self) -> bool {
        let Some(sessions) = self.sessions.as_mut() else { return false };
        let oldest = sessions.held.keys().min_by_key(|k| sessions.janitor.tracker.last_request(k).map_or(0, |l| l.0)).cloned();
        let Some(key) = oldest else { return false };
        let now = (sessions.clock)();
        let mut cap = Capturing { exec: &mut self.exec, pool: &mut self.pool, held: &mut sessions.held };
        let written = sessions.janitor.save_keys(vec![key.clone()], now, &sessions.shared.store, &mut cap);
        if written.is_empty()
            && let Some(mut h) = sessions.held.remove(&key) {
                self.pool.release(&mut h.blocks);
            }
        self.publish_residents();
        true
    }

    fn session_hit(&self, s: &Seq, target: usize, better_than: usize) -> Option<crate::sessions::Entry> {
        let sessions = self.sessions.as_ref()?;
        if s.generated() != 0 || !s.req.media.is_empty() {
            return None;
        }
        let e = sessions.shared.store.find_prefix(&s.root, &s.keys[..target])?;
        (e.tokens > better_than && e.tokens < target).then_some(e)
    }

    fn restore_session(&mut self, entry: &crate::sessions::Entry, s: &mut Seq, slot: usize) {
        let Some(sessions) = self.sessions.as_ref() else { return };
        let now = (sessions.clock)();
        let started = Instant::now();
        let loaded = sessions.shared.store.load(&entry.id, now);
        let outcome = loaded.and_then(|session| {
            anyhow::ensure!(session.tokens.len() == entry.tokens && s.keys.starts_with(&session.tokens), "snapshot tokens differ from the prompt");
            self.exec.restore_session(slot, &s.blocks[..blocks_for(entry.tokens)], &session.sections)
        });
        match outcome {
            Ok(()) => {
                s.computed = entry.tokens;
                s.cached = entry.tokens;
                tracing::info!(session = %entry.id, tokens = entry.tokens, secs = started.elapsed().as_secs_f64(), "session restored without prefill");
                if !sessions.shared.options.keep_after_restore {
                    let _ = sessions.shared.store.delete(&entry.id);
                }
            }
            Err(e) => {
                tracing::warn!(session = %entry.id, error = %format!("{e:#}"), "session restore failed, prefilling the prompt");
                let _ = sessions.shared.store.delete(&entry.id);
            }
        }
    }

    pub fn prefix_stats(&self) -> PrefixStats {
        let c = self.cache.as_ref();
        PrefixStats {
            queries: self.prefix_queries,
            hits: self.prefix_hits,
            blocks: c.map_or(0, |c| c.blocks()),
            snapshots: c.map_or(0, |c| c.snapshots()),
            snapshot_bytes: c.map_or(0, |c| c.snapshot_bytes()),
            evictions: c.map_or(0, |c| c.evictions()),
        }
    }

    fn alloc_block(&mut self) -> Option<u32> {
        loop {
            if let Some(b) = self.pool.alloc() {
                return Some(b);
            }
            let evicted = self.cache.as_mut().is_some_and(|c| c.evict_one(&mut self.pool));
            if !evicted && !self.spill_one_session() {
                return None;
            }
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn running(&self) -> usize {
        self.running.len()
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn blocks_used(&self) -> usize {
        self.pool.used()
    }

    pub fn blocks_live(&self) -> usize {
        self.pool.live()
    }

    pub fn blocks_cached(&self) -> usize {
        self.pool.cache_only()
    }

    pub fn blocks_total(&self) -> usize {
        self.pool.total()
    }

    pub fn chain_metrics(&self) -> Option<ChainMetrics> {
        self.exec.chain_metrics()
    }

    pub fn state_refusals(&self) -> u64 {
        self.state_refusals
    }

    pub fn preemptions(&self) -> u64 {
        self.preemptions
    }

    pub fn swapped_out(&self) -> u64 {
        self.swapped_out
    }

    pub fn swapped_in(&self) -> u64 {
        self.swapped_in
    }

    pub fn swapped_bytes(&self) -> u64 {
        self.swapped_bytes
    }

    pub fn swapped_bytes_total(&self) -> u64 {
        self.swapped_bytes_total
    }

    pub fn swap_secs(&self) -> f64 {
        self.swap_secs
    }

    pub fn recomputed_tokens(&self) -> u64 {
        self.recomputed_tokens
    }

    pub fn idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    fn reserve(&self) -> usize {
        self.cfg.reserve_blocks.unwrap_or((self.pool.total() / 100).max(1))
    }

    pub fn submit(&mut self, req: Request) -> Result<SeqId> {
        self.submit_planned(req, None)
    }

    pub fn check(&self, req: &Request) -> Result<()> {
        let max_ctx = self.exec.max_ctx();
        if req.prompt.is_empty() || req.max_tokens == 0 {
            bail!("empty prompt or nothing to generate");
        }
        if req.prompt.len() + req.max_tokens + self.exec.draft_tokens() > max_ctx {
            bail!(
                "prompt of {} tokens plus {} new tokens (and {} draft positions) exceeds the context of {max_ctx}",
                req.prompt.len(),
                req.max_tokens,
                self.exec.draft_tokens()
            );
        }
        let need = blocks_for(req.prompt.len() + req.max_tokens + self.exec.draft_tokens());
        if need > self.pool.total() {
            bail!("a request of {need} KV blocks can never fit the pool of {}", self.pool.total());
        }
        if req.sampling.logprobs.is_some() && !self.exec.supports_logprobs() {
            bail!("logprobs are not available on this executor");
        }
        if req.grammar.is_some() && !self.exec.supports_grammar() {
            bail!("structured output is not available on this executor");
        }
        Ok(())
    }

    pub fn submit_planned(&mut self, req: Request, plan: Option<Arc<crate::vision::MediaPlan>>) -> Result<SeqId> {
        self.check(&req)?;
        let id = self.next_id;
        self.next_id += 1;
        let cursor = req.grammar.as_ref().map(|g| g.cursor());
        let tokens = req.prompt.clone();
        let root = self.exec.prefix_root(&req.salt);
        let (media, keys) = if req.media.is_empty() {
            (None, tokens.clone())
        } else {
            let plan = match plan {
                Some(p) => p,
                None => self.exec.plan_media(&req.prompt, &req.media)?,
            };
            let keys = tokens.iter().zip(&plan.tags).map(|(&t, &tag)| if tag != 0 { tag } else { t }).collect();
            (Some(plan), keys)
        };
        self.waiting.push_back(Seq {
            id,
            req,
            tokens,
            computed: 0,
            slot: None,
            blocks: Vec::new(),
            arrived: Instant::now(),
            admitted: 0,
            started: false,
            root,
            cached: 0,
            keys,
            media,
            cursor,
            parked: None,
        });
        Ok(id)
    }

    pub fn cancel(&mut self, id: SeqId, events: &mut Vec<Event>) {
        if let Some(i) = self.waiting.iter().position(|s| s.id == id) {
            let s = self.waiting.remove(i).expect("index from position");
            if let Some(h) = &s.parked {
                self.swapped_bytes -= h.bytes as u64;
            }
            events.push(finished(&s, Finish::Cancelled));
            return;
        }
        if let Some(i) = self.running.iter().position(|s| s.id == id) {
            let mut s = self.running.remove(i);
            self.release(&mut s);
            events.push(finished(&s, Finish::Cancelled));
        }
    }

    fn release(&mut self, s: &mut Seq) {
        self.pool.release(&mut s.blocks);
        if let Some(slot) = s.slot.take() {
            self.exec.release_state(slot);
            self.free_slots.push(slot);
        }
    }

    fn try_park(&mut self, s: &Seq) -> Option<HostSeq> {
        let slot = s.slot?;
        if s.computed == 0 {
            return None;
        }
        let bytes = self.exec.swap_bytes(s.blocks.len()) as u64;
        if bytes == 0 || self.swapped_bytes + bytes > self.cfg.swap_space_bytes || self.exec.host_used() as u64 + bytes > self.cfg.swap_space_bytes {
            return None;
        }
        let started = Instant::now();
        let parked = self.exec.park(slot, &s.blocks);
        self.swap_secs += started.elapsed().as_secs_f64();
        match parked {
            Ok(Some(h)) => {
                self.swapped_bytes += h.bytes as u64;
                self.swapped_bytes_total += h.bytes as u64;
                self.swapped_out += 1;
                Some(h)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "parking a sequence failed, it will be recomputed");
                None
            }
        }
    }

    fn promise(&self, s: &Seq, held: usize) -> usize {
        let ahead = (s.req.max_tokens - s.generated()).min(self.cfg.admit_lookahead);
        blocks_for(s.tokens.len() + ahead + self.exec.draft_tokens()).saturating_sub(held)
    }

    fn running_promise(&self) -> usize {
        self.running.iter().map(|s| self.promise(s, s.blocks.len())).sum()
    }

    fn preempt_latest(&mut self, events: &mut Vec<Event>) -> SeqId {
        let i = (0..self.running.len()).max_by_key(|&i| self.running[i].admitted).expect("something is running");
        let mut s = self.running.remove(i);
        s.parked = self.try_park(&s);
        self.release(&mut s);
        if s.parked.is_none() {
            self.recomputed_tokens += s.computed as u64;
            s.computed = 0;
        }
        self.preemptions += 1;
        events.push(Event::Preempted { id: s.id });
        let id = s.id;
        self.waiting.push_front(s);
        id
    }

    fn grow_decoders(&mut self, events: &mut Vec<Event>, report: &mut StepReport) -> Result<()> {
        let mut i = 0;
        while i < self.running.len() {
            let s = &self.running[i];
            if !s.decoding() || blocks_for(s.tokens.len() + self.exec.draft_tokens()) <= s.blocks.len() {
                i += 1;
                continue;
            }
            match self.alloc_block() {
                Some(b) => {
                    let s = &mut self.running[i];
                    s.blocks.push(b);
                    let slot = s.slot.expect("running has a slot");
                    self.exec.set_table(slot, &s.blocks)?;
                    i += 1;
                }
                None => {
                    self.preempt_latest(events);
                    report.preemptions += 1;
                    i = 0;
                }
            }
        }
        Ok(())
    }

    fn admit(&mut self, budget: usize) -> Result<()> {
        let cap = self.cfg.max_num_seqs.min(self.exec.slots());
        let mut planned = 0;
        let mut long_in_flight = if self.cfg.shortest_first { self.running.iter().filter(|s| s.prefill_target() - s.computed > LONG_PROMPT).count() } else { 0 };
        let long_cap = if self.cfg.shortest_first { self.cfg.max_long_in_flight.max(1) } else { usize::MAX };
        while let Some(s) = self.waiting.front() {
            if self.running.len() >= cap || self.free_slots.is_empty() || planned >= budget {
                break;
            }
            let target = s.prefill_target();
            if target > LONG_PROMPT && long_in_flight >= long_cap {
                break;
            }
            let need = blocks_for(target + 1);
            let reserve = if self.running.is_empty() { 0 } else { self.reserve() };
            if s.parked.is_some() {
                if !self.resume_front(reserve)? {
                    break;
                }
                continue;
            }
            let lookahead = if self.running.is_empty() { 0 } else { self.promise(s, 0).saturating_sub(need).min(self.cfg.admit_lookahead) + self.running_promise() };
            let hit = match self.cache.as_mut() {
                Some(cache) => {
                    let hashes = prefix::block_hashes(&s.root, &s.keys[..target]);
                    let limit = if s.generated() == 0 { target - 1 } else { target };
                    cache.lookup_with_tail(&s.root, &s.keys[..target], &hashes, limit, &mut self.pool)
                }
                None => prefix::Hit::default(),
            };
            let from_disk = self.session_hit(s, target, hit.tokens);
            let hit = if from_disk.is_some() {
                self.give_back(hit);
                prefix::Hit::default()
            } else {
                hit
            };
            let fresh = need - hit.blocks.len();
            while self.pool.free_count() < fresh + reserve + lookahead {
                let evicted = self.cache.as_mut().is_some_and(|c| c.evict_one(&mut self.pool)) || self.spill_one_session();
                if !evicted {
                    break;
                }
            }
            if self.pool.free_count() < fresh + reserve + lookahead {
                self.give_back(hit);
                break;
            }
            let slot = *self.free_slots.last().expect("a free slot");
            if self.exec.acquire_state(slot).is_err() {
                self.state_refusals += 1;
                self.give_back(hit);
                break;
            }
            let mut s = self.waiting.pop_front().expect("front exists");
            if let Some(state) = &hit.snapshot
                && let Err(e) = self.exec.restore(slot, state) {
                    self.exec.release_state(slot);
                    self.give_back(hit);
                    self.waiting.push_front(s);
                    return Err(e);
                }
            self.free_slots.pop();
            if s.generated() == 0 && self.cache.is_some() {
                self.prefix_queries += s.req.prompt.len() as u64;
                self.prefix_hits += hit.tokens as u64;
            }
            if s.generated() == 0 {
                s.cached = hit.tokens;
            }
            s.computed = hit.tokens;
            s.blocks = hit.blocks;
            if let Some(src) = hit.tail {
                let dst = self.pool.alloc().expect("checked free blocks");
                self.exec.copy_block(src, dst)?;
                s.blocks.push(dst);
                self.pool.release(&mut vec![src]);
            }
            for _ in s.blocks.len()..blocks_for(target) {
                s.blocks.push(self.pool.alloc().expect("checked free blocks"));
            }
            self.exec.set_table(slot, &s.blocks)?;
            self.exec.set_media(slot, s.media.as_deref())?;
            if let Some(entry) = from_disk {
                self.restore_session(&entry, &mut s, slot);
            }
            s.slot = Some(slot);
            self.admissions += 1;
            s.admitted = self.admissions;
            let remaining = target - s.computed;
            if remaining > LONG_PROMPT {
                long_in_flight += 1;
            } else {
                planned += remaining;
            }
            self.running.push(s);
        }
        Ok(())
    }

    fn resume_front(&mut self, reserve: usize) -> Result<bool> {
        let front = self.waiting.front().expect("front exists");
        let held = front.parked.as_ref().map_or(0, |h| h.blocks);
        let total = held.max(blocks_for(front.tokens.len() + self.exec.draft_tokens()));
        let promised = self.running_promise();
        while self.pool.free_count() < total + reserve + promised {
            let evicted = self.cache.as_mut().is_some_and(|c| c.evict_one(&mut self.pool));
            if !evicted {
                break;
            }
        }
        let tight = self.pool.free_count() < total + reserve + promised;
        if self.pool.free_count() < total + reserve || (tight && !self.running.is_empty()) {
            return Ok(false);
        }
        let slot = *self.free_slots.last().expect("a free slot");
        if self.exec.acquire_state(slot).is_err() {
            self.state_refusals += 1;
            return Ok(false);
        }
        let mut s = self.waiting.pop_front().expect("front exists");
        let host = s.parked.take().expect("parked");
        let mut blocks = Vec::with_capacity(total);
        for _ in 0..total {
            blocks.push(self.pool.alloc().expect("checked free blocks"));
        }
        let started = Instant::now();
        let restored = self.exec.set_table(slot, &blocks).and_then(|_| self.exec.resume(slot, &blocks[..held], &host));
        self.swap_secs += started.elapsed().as_secs_f64();
        if let Err(e) = restored {
            self.exec.release_state(slot);
            self.pool.release(&mut blocks);
            s.parked = Some(host);
            self.waiting.push_front(s);
            return Err(e);
        }
        self.swapped_bytes -= host.bytes as u64;
        self.swapped_in += 1;
        self.free_slots.pop();
        self.exec.set_media(slot, s.media.as_deref())?;
        s.blocks = blocks;
        s.slot = Some(slot);
        self.admissions += 1;
        s.admitted = self.admissions;
        self.running.push(s);
        Ok(true)
    }

    fn give_back(&mut self, hit: prefix::Hit) {
        let mut held = hit.blocks;
        held.extend(hit.tail);
        self.pool.release(&mut held);
    }

    fn cache_progress(&mut self, i: usize, final_state: bool) -> Result<()> {
        let hybrid = self.exec.hybrid();
        let s = &self.running[i];
        let upto = s.computed / PAGE;
        let rem = s.computed % PAGE;
        if self.cache.is_none() || s.computed == 0 {
            return Ok(());
        }
        let at_point = if final_state { true } else { rem == 0 && self.snapshot_points(i).contains(&s.computed) };
        if !final_state && upto == 0 {
            return Ok(());
        }
        let hashes = prefix::block_hashes(&s.root, &s.keys[..upto * PAGE]);
        let blocks = s.blocks[..upto].to_vec();
        let slot = s.slot.expect("running has a slot");
        let parent = if upto == 0 { s.root } else { hashes[upto - 1] };
        let tail_tokens = s.keys[upto * PAGE..s.computed].to_vec();
        let tail_block = (rem > 0).then(|| s.blocks[upto]);
        let have_state = upto > 0 && rem == 0 && self.cache.as_ref().is_some_and(|c| c.has_snapshot(&parent));
        let have_tail = rem > 0 && self.cache.as_ref().is_some_and(|c| c.has_tail(&parent, &tail_tokens));
        let take = hybrid && at_point && !have_state && !have_tail;
        let state = if take { Some(Arc::new(self.exec.snapshot(slot)?)) } else { None };
        let cache = self.cache.as_mut().expect("checked above");
        if !hybrid || final_state || take {
            cache.register(&hashes, &blocks, &mut self.pool);
        }
        if final_state && rem > 0 {
            if let Some(block) = tail_block {
                cache.register_tail(parent, upto == 0, &tail_tokens, block, state, &mut self.pool);
            }
        } else if let (Some(state), true) = (state, upto > 0) {
            cache.store_snapshot(parent, state, &mut self.pool);
        }
        Ok(())
    }

    fn snapshot_points(&self, i: usize) -> Vec<usize> {
        let s = &self.running[i];
        prefix::snapshot_points(s.cached, s.prefill_target(), self.cfg.prefix_cache.snapshot_interval)
    }

    fn apply_decode(&mut self, rows: &[(usize, DecodeRow)], next: Vec<u32>, done: &mut Vec<usize>, events: &mut Vec<Event>) -> Result<()> {
        for (row, ((i, _), t)) in rows.iter().zip(next).enumerate() {
            self.record_logprob(*i, LogitsAt::Decode(row), t, events)?;
            let s = &mut self.running[*i];
            s.computed += 1;
            if accept(s, t, self.exec.max_ctx(), events) {
                done.push(*i);
            }
        }
        Ok(())
    }

    fn row_masks(&self, rows: &[(usize, DecodeRow)]) -> Vec<Option<Mask>> {
        if rows.iter().all(|(i, _)| self.running[*i].cursor.is_none()) {
            return Vec::new();
        }
        rows.iter().map(|(i, _)| self.running[*i].cursor.as_ref().map(|c| c.mask())).collect()
    }

    fn record_logprob(&mut self, i: usize, at: LogitsAt, t: u32, events: &mut Vec<Event>) -> Result<()> {
        let s = &self.running[i];
        let Some(top) = s.req.sampling.logprobs else { return Ok(()) };
        if s.req.stop.contains(&t) {
            return Ok(());
        }
        let id = s.id;
        let row = self.exec.logits_of(at)?;
        events.push(Event::Logprob { id, logprob: crate::logprobs::token_logprob(&row, t, top) });
        Ok(())
    }

    fn prefill_budget(&self, decodes: usize) -> usize {
        let full = self.cfg.max_batched_tokens.saturating_sub(decodes);
        let align = self.cfg.min_prefill_chunk.max(1);
        if decodes == 0 {
            return full.max(align);
        }
        let floor = self.cfg.min_prefill_tokens.div_ceil(align) * align;
        if self.prefill_rate <= 0.0 {
            return align.max(floor).min(full.max(1));
        }
        let by_cost = if self.cfg.step_cost > 1.0 { (self.cfg.step_cost - 1.0) * self.decode_secs } else { f64::INFINITY };
        let by_cap = if self.cfg.itl_target.is_zero() { f64::INFINITY } else { (self.cfg.itl_target.as_secs_f64() - self.decode_secs).max(0.0) };
        let room = by_cost.min(by_cap);
        let fit = if room.is_finite() { (room * self.prefill_rate) as usize } else { full };
        let fit = if fit > align { fit / align * align } else { align };
        let fit = fit.max(floor);
        let fit = fit.max(self.whole_prompt_remaining().unwrap_or(0));
        let fit = if self.long_prompt_pending() { fit.max(self.cfg.prefill_burst_tokens) } else { fit };
        fit.min(full.max(align))
    }

    fn long_prompt_pending(&self) -> bool {
        self.cfg.prefill_burst_tokens > 0
            && (self.running.iter().any(|s| s.prefill_target().saturating_sub(s.computed) > LONG_PROMPT)
                || self.waiting.iter().any(|s| s.prefill_target() > LONG_PROMPT))
    }

    fn whole_prompt_remaining(&self) -> Option<usize> {
        let limit = self.cfg.whole_prompt_tokens;
        if limit == 0 {
            return None;
        }
        let pending = self.running.iter().filter(|s| s.computed < s.prefill_target()).map(|s| s.prefill_target() - s.computed);
        let next = if self.cfg.shortest_first {
            pending.chain(self.waiting.iter().map(|s| s.prefill_target())).min()?
        } else {
            pending.chain(self.waiting.front().map(|s| s.prefill_target())).next()?
        };
        (next <= limit).then_some(next)
    }

    pub fn step(&mut self, events: &mut Vec<Event>) -> Result<StepReport> {
        let mut report = StepReport::default();
        self.grow_decoders(events, &mut report)?;
        let decodes = self.running.iter().filter(|s| s.decoding()).count();
        let budget = self.prefill_budget(decodes);
        self.admit(budget)?;
        let rows: Vec<(usize, DecodeRow)> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.decoding())
            .map(|(i, s)| {
                let row = DecodeRow {
                    slot: s.slot.expect("running has a slot"),
                    token: *s.tokens.last().expect("decoders have tokens"),
                    pos: s.tokens.len() - 1,
                    sampling: s.req.sampling,
                    step: s.generated() as u32,
                };
                (i, row)
            })
            .collect();
        let mut done: Vec<usize> = Vec::new();
        let plain: Vec<DecodeRow> = rows.iter().map(|r| r.1).collect();
        let mut pending = !rows.is_empty();
        let cap = self.exec.speculation_cap();
        let wants_logprobs = rows.iter().any(|(i, _)| self.running[*i].req.sampling.logprobs.is_some());
        let constrained = rows.iter().any(|(i, _)| self.running[*i].cursor.is_some());
        if pending && rows.len() <= cap && !wants_logprobs && !constrained {
            pending = false;
            let started = Instant::now();
            let spec: Vec<SpecRow> = rows
                .iter()
                .map(|(i, r)| {
                    let s = &self.running[*i];
                    let room = self.exec.max_ctx() - s.tokens.len();
                    let left = (s.req.max_tokens - s.generated()).min(room);
                    SpecRow { slot: r.slot, token: r.token, pos: r.pos, sampling: r.sampling, step: r.step, max_accept: left.saturating_sub(1) }
                })
                .collect();
            let out = self.exec.speculate(&spec)?;
            report.decode_secs = started.elapsed().as_secs_f64();
            report.speculated = true;
            report.decode_rows = rows.len();
            report.drafted = rows.len() * self.exec.draft_tokens();
            for ((i, _), emitted) in rows.iter().zip(out) {
                report.accepted += emitted.len() - 1;
                for t in emitted {
                    let s = &mut self.running[*i];
                    s.computed += 1;
                    if accept(s, t, self.exec.max_ctx(), events) {
                        done.push(*i);
                        break;
                    }
                }
            }
        }
        let mut left = budget;
        let started = Instant::now();
        let limit = self.exec.chunk_limit();
        let mut order: Vec<usize> = (0..self.running.len()).collect();
        if self.cfg.shortest_first {
            order.sort_by_key(|&i| {
                let s = &self.running[i];
                s.prefill_target().saturating_sub(s.computed)
            });
        }
        for i in order {
            if left == 0 {
                break;
            }
            let s = &self.running[i];
            if s.computed >= s.prefill_target() || done.contains(&i) {
                continue;
            }
            if !s.started {
                let queued = s.arrived.elapsed();
                let id = s.id;
                self.running[i].started = true;
                events.push(Event::Started { id, queued });
            }
            while left > 0 {
                let s = &self.running[i];
                let target = s.prefill_target();
                if s.computed >= target {
                    break;
                }
                let mut n = (target - s.computed).min(limit).min(left);
                if self.cache.is_some() && self.exec.hybrid()
                    && let Some(p) = self.snapshot_points(i).into_iter().find(|&p| p > self.running[i].computed) {
                        n = n.min(p - self.running[i].computed);
                    }
                if let Some(m) = self.running[i].media.clone() {
                    let from = self.running[i].computed;
                    n = m.extend_chunk(from, from + n, limit) - from;
                }
                let s = &self.running[i];
                let last = s.computed + n == target;
                let sample = (last && s.generated() == 0).then_some((s.req.sampling, 0));
                let slot = s.slot.expect("running has a slot");
                let chunk = s.tokens[s.computed..s.computed + n].to_vec();
                let start = s.computed;
                let first = if sample.is_some() { s.cursor.as_ref().map(|c| c.mask()) } else { None };
                let t = if pending {
                    pending = false;
                    let masks = self.row_masks(&rows);
                    self.exec.set_masks(masks, first);
                    let (next, t) = self.exec.fused(&plain, slot, &chunk, start, sample)?;
                    report.decode_rows = rows.len();
                    self.apply_decode(&rows, next, &mut done, events)?;
                    t
                } else {
                    self.exec.set_masks(Vec::new(), first);
                    self.exec.prefill(slot, &chunk, start, sample)?
                };
                if let Some(t) = t {
                    self.record_logprob(i, LogitsAt::Prefill, t, events)?;
                }
                let s = &mut self.running[i];
                s.computed += n;
                left -= n;
                report.prefill_tokens += n;
                if let Some(t) = t
                    && accept(s, t, self.exec.max_ctx(), events)
                {
                    done.push(i);
                }
                self.cache_progress(i, false)?;
            }
        }
        if pending {
            let started = Instant::now();
            let masks = self.row_masks(&rows);
            self.exec.set_masks(masks, None);
            let next = self.exec.decode(&plain)?;
            report.decode_secs = started.elapsed().as_secs_f64();
            report.decode_rows = rows.len();
            self.apply_decode(&rows, next, &mut done, events)?;
        }
        if report.prefill_tokens > 0 {
            self.exec.flush()?;
            report.prefill_secs = started.elapsed().as_secs_f64();
            if report.prefill_tokens >= self.cfg.min_prefill_chunk.max(self.cfg.min_prefill_tokens / 2) {
                let marginal = (report.prefill_secs - if report.decode_rows > 0 { self.decode_secs } else { 0.0 }).max(report.prefill_secs * 0.1);
                self.prefill_rate = ewma(self.prefill_rate, report.prefill_tokens as f64 / marginal.max(1e-6));
            }
        }
        if report.decode_secs > 0.0 && report.prefill_tokens == 0 {
            self.decode_secs = ewma(self.decode_secs, report.decode_secs);
        }
        done.sort_unstable();
        done.dedup();
        for &i in done.iter().rev() {
            self.cache_progress(i, true)?;
            self.track_session(i);
            let mut s = self.running.remove(i);
            self.release(&mut s);
            let finish = finish_of(&s);
            events.push(finished(&s, finish));
        }
        Ok(report)
    }
}

fn stopped_by_token(s: &Seq) -> bool {
    s.tokens.last().is_some_and(|t| s.req.stop.contains(t))
}

fn finish_of(s: &Seq) -> Finish {
    if stopped_by_token(s) || s.cursor.as_ref().is_some_and(|c| c.finished()) {
        Finish::Stop
    } else {
        Finish::Length
    }
}

fn finished(s: &Seq, finish: Finish) -> Event {
    let completion = s.generated() - usize::from(stopped_by_token(s));
    Event::Finished { id: s.id, finish, prompt_tokens: s.req.prompt.len(), completion_tokens: completion, cached_tokens: s.cached }
}

fn accept(s: &mut Seq, t: u32, max_ctx: usize, events: &mut Vec<Event>) -> bool {
    s.tokens.push(t);
    s.keys.push(t);
    let stop = s.req.stop.contains(&t);
    if !stop {
        events.push(Event::Token { id: s.id, token: t });
    }
    let grammar_done = s.cursor.as_mut().is_some_and(|c| {
        c.advance(t);
        c.finished()
    });
    stop || grammar_done || s.generated() >= s.req.max_tokens || s.tokens.len() >= max_ctx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const VOCAB: u32 = 1000;

    fn mix(h: u64, t: u32) -> u64 {
        (h ^ t as u64).wrapping_mul(0x100000001b3).rotate_left(17)
    }

    fn next_token(h: u64) -> u32 {
        (h % VOCAB as u64) as u32
    }

    pub struct HostGuard(Arc<std::sync::atomic::AtomicUsize>, usize);

    impl Drop for HostGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(self.1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub struct Fake {
        pub host_limit: usize,
        pub host_used: Arc<std::sync::atomic::AtomicUsize>,
        max_ctx: usize,
        slots: usize,
        kv: Vec<Option<u32>>,
        tables: Vec<Vec<u32>>,
        state: Vec<(u64, usize)>,
        chunk: usize,
        pub token_cost: Duration,
        pub row_cost: Duration,
        pub decode_calls: Vec<usize>,
        pub prefill_calls: Vec<usize>,
        pub state_budget: usize,
        pub states_live: std::collections::HashSet<usize>,
        pub hybrid: bool,
        pub light: bool,
        pub prefix_enabled: bool,
        pub restores: usize,
        pub session_restores: usize,
        pub block_copies: usize,
        pub spec_k: usize,
        pub spec_cap: usize,
        pub spec_calls: Vec<usize>,
        pub accept_counts: Vec<usize>,
        last_decode: Vec<u32>,
        last_prefill: Option<u32>,
        masks: Vec<Option<Mask>>,
        first_mask: Option<Mask>,
        pub masked_calls: usize,
        pub grammar: bool,
        pub swap: bool,
        pub parks: usize,
        pub resumes: usize,
    }

    fn constrain(t: u32, mask: &Option<Mask>) -> u32 {
        let Some(m) = mask else { return t };
        (0..VOCAB).map(|d| (t + d) % VOCAB).find(|&c| m[c as usize / 32] >> (c % 32) & 1 == 1).unwrap_or(t)
    }

    impl Fake {
        pub fn new(max_ctx: usize, slots: usize, blocks: usize, chunk: usize) -> Self {
            Fake {
                host_limit: usize::MAX,
                host_used: Default::default(),
                max_ctx,
                slots,
                kv: vec![None; blocks * PAGE],
                tables: vec![Vec::new(); slots],
                state: vec![(0, 0); slots],
                chunk,
                token_cost: Duration::ZERO,
                row_cost: Duration::ZERO,
                decode_calls: Vec::new(),
                prefill_calls: Vec::new(),
                state_budget: usize::MAX,
                states_live: Default::default(),
                hybrid: true,
                light: false,
                prefix_enabled: true,
                restores: 0,
                session_restores: 0,
                block_copies: 0,
                spec_k: 0,
                spec_cap: 0,
                spec_calls: Vec::new(),
                accept_counts: Vec::new(),
                last_decode: Vec::new(),
                last_prefill: None,
                masks: Vec::new(),
                first_mask: None,
                masked_calls: 0,
                grammar: true,
                swap: false,
                parks: 0,
                resumes: 0,
            }
        }

        fn host_alloc(&self, bytes: usize) -> Result<HostGuard> {
            let before = self.host_used.fetch_add(bytes, std::sync::atomic::Ordering::SeqCst);
            if before.saturating_add(bytes) > self.host_limit {
                self.host_used.fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
                bail!("pinned host pool is full");
            }
            Ok(HostGuard(self.host_used.clone(), bytes))
        }

        fn write(&mut self, slot: usize, pos: usize, t: u32) -> u64 {
            let block = self.tables[slot][pos / PAGE] as usize;
            self.kv[block * PAGE + pos % PAGE] = Some(t);
            let (h, n) = self.state[slot];
            if self.hybrid {
                assert_eq!(n, pos, "recurrent state of slot {slot} is at {n}, write at {pos}");
            }
            let h = if self.hybrid { mix(h, t) } else { 0 };
            self.state[slot] = (h, pos + 1);
            if self.light {
                return 0;
            }
            let mut check = 0u64;
            for p in 0..=pos {
                let b = self.tables[slot][p / PAGE] as usize;
                check = mix(check, self.kv[b * PAGE + p % PAGE].expect("written KV"));
            }
            if self.hybrid {
                assert_eq!(check, h, "KV through the block table disagrees with the recurrent state");
            }
            check
        }
    }

    impl Executor for Fake {
        fn max_ctx(&self) -> usize {
            self.max_ctx
        }
        fn slots(&self) -> usize {
            self.slots
        }
        fn kv_blocks(&self) -> usize {
            self.kv.len() / PAGE
        }
        fn chunk_limit(&self) -> usize {
            self.chunk
        }
        fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
            self.tables[slot] = blocks.to_vec();
            Ok(())
        }
        fn acquire_state(&mut self, slot: usize) -> Result<()> {
            if self.states_live.len() >= self.state_budget {
                bail!("no device memory for a recurrent state");
            }
            assert!(self.states_live.insert(slot), "slot {slot} acquired twice");
            Ok(())
        }
        fn release_state(&mut self, slot: usize) {
            assert!(self.states_live.remove(&slot), "slot {slot} released without state");
        }
        fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
            assert!(tokens.len() <= self.chunk);
            if start == 0 {
                self.state[slot] = (0, 0);
            }
            self.prefill_calls.push(tokens.len());
            std::thread::sleep(self.token_cost * tokens.len() as u32);
            let mut h = 0;
            for (i, &t) in tokens.iter().enumerate() {
                h = self.write(slot, start + i, t);
            }
            let first = self.first_mask.take();
            self.last_prefill = sample.map(|_| constrain(next_token(h), &first));
            Ok(self.last_prefill)
        }
        fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
            let mut seen = std::collections::HashSet::new();
            assert!(rows.iter().all(|r| seen.insert(r.slot)), "a slot twice in one batch");
            self.decode_calls.push(rows.len());
            std::thread::sleep(self.row_cost * rows.len() as u32);
            let masks = std::mem::take(&mut self.masks);
            self.masked_calls += usize::from(!masks.is_empty());
            self.last_decode = rows.iter().enumerate().map(|(i, r)| constrain(next_token(self.write(r.slot, r.pos, r.token)), masks.get(i).unwrap_or(&None))).collect();
            Ok(self.last_decode.clone())
        }
        fn supports_logprobs(&self) -> bool {
            true
        }
        fn supports_grammar(&self) -> bool {
            self.grammar
        }
        fn set_masks(&mut self, rows: Vec<Option<Mask>>, first: Option<Mask>) {
            self.masks = rows;
            self.first_mask = first;
        }
        fn logits_of(&mut self, at: LogitsAt) -> Result<Vec<f32>> {
            let t = match at {
                LogitsAt::Decode(i) => self.last_decode[i],
                LogitsAt::Prefill => self.last_prefill.expect("a sampled prefill"),
            };
            let mut row = vec![0.0; VOCAB as usize];
            row[t as usize] = 5.0;
            row[(t as usize + 1) % VOCAB as usize] = 3.0;
            Ok(row)
        }
        fn supports_prefix_cache(&self) -> bool {
            self.prefix_enabled
        }
        fn hybrid(&self) -> bool {
            self.hybrid
        }
        fn snapshot(&mut self, slot: usize) -> Result<StateSnapshot> {
            let (h, n) = self.state[slot];
            Ok(StateSnapshot::new(Vec::new(), vec![f32::from_bits(h as u32), f32::from_bits((h >> 32) as u32), f32::from_bits(n as u32)], Vec::new()))
        }
        fn restore(&mut self, slot: usize, state: &StateSnapshot) -> Result<()> {
            let s = state.ssm();
            let h = s[0].to_bits() as u64 | (s[1].to_bits() as u64) << 32;
            self.state[slot] = (h, s[2].to_bits() as usize);
            self.restores += 1;
            Ok(())
        }
        fn session_compat(&self) -> Option<crate::sessions::Compat> {
            Some(crate::sessions::Compat { model_hash: [1; 32], spec_hash: [2; 32], kv_dtype: 0, layers: 1, block_size: PAGE as u32 })
        }
        fn save_session(&mut self, blocks: &[u32], state: Option<&StateSnapshot>) -> Result<Vec<Vec<u8>>> {
            let _pinned = self.host_alloc(blocks.len() * PAGE * 4)?;
            let mut k = Vec::new();
            for &b in blocks {
                for i in 0..PAGE {
                    k.extend(self.kv[b as usize * PAGE + i].unwrap_or(u32::MAX).to_le_bytes());
                }
            }
            Ok(crate::sessions::pack_sections(k, Vec::new(), state))
        }
        fn restore_session(&mut self, slot: usize, blocks: &[u32], sections: &[Vec<u8>]) -> Result<()> {
            let k = &sections[crate::sessions::SECTION_K];
            assert_eq!(k.len(), blocks.len() * PAGE * 4);
            let _pinned = self.host_alloc(k.len())?;
            for (j, &b) in blocks.iter().enumerate() {
                for i in 0..PAGE {
                    let at = (j * PAGE + i) * 4;
                    let t = u32::from_le_bytes(k[at..at + 4].try_into().unwrap());
                    self.kv[b as usize * PAGE + i] = (t != u32::MAX).then_some(t);
                }
            }
            if let Some(state) = crate::sessions::unpack_state(sections)? {
                self.restore(slot, &state)?;
            }
            self.session_restores += 1;
            Ok(())
        }
        fn copy_block(&mut self, src: u32, dst: u32) -> Result<()> {
            for i in 0..PAGE {
                self.kv[dst as usize * PAGE + i] = self.kv[src as usize * PAGE + i];
            }
            self.block_copies += 1;
            Ok(())
        }
        fn swap_bytes(&self, blocks: usize) -> usize {
            if self.swap { blocks * 1000 + 100 } else { 0 }
        }
        fn set_host_limit(&mut self, bytes: usize) {
            self.host_limit = bytes;
        }
        fn host_used(&self) -> usize {
            self.host_used.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn park(&mut self, slot: usize, blocks: &[u32]) -> Result<Option<HostSeq>> {
            if !self.swap {
                return Ok(None);
            }
            let bytes = blocks.len() * 1000 + 100;
            let guard = self.host_alloc(bytes)?;
            self.parks += 1;
            let kv: Vec<Vec<Option<u32>>> = blocks.iter().map(|&b| self.kv[b as usize * PAGE..(b as usize + 1) * PAGE].to_vec()).collect();
            Ok(Some(HostSeq { bytes: blocks.len() * 1000 + 100, blocks: blocks.len(), payload: Box::new((kv, self.state[slot], guard)) }))
        }
        fn resume(&mut self, slot: usize, blocks: &[u32], host: &HostSeq) -> Result<()> {
            self.resumes += 1;
            let (kv, state, _) = host.payload.downcast_ref::<(Vec<Vec<Option<u32>>>, (u64, usize), HostGuard)>().unwrap();
            for (&b, page) in blocks.iter().zip(kv) {
                self.kv[b as usize * PAGE..(b as usize + 1) * PAGE].copy_from_slice(page);
            }
            self.state[slot] = *state;
            Ok(())
        }
        fn draft_tokens(&self) -> usize {
            self.spec_k
        }
        fn speculation_cap(&self) -> usize {
            self.spec_cap
        }
        fn speculate(&mut self, rows: &[SpecRow]) -> Result<Vec<Vec<u32>>> {
            let mut seen = std::collections::HashSet::new();
            assert!(rows.iter().all(|r| seen.insert(r.slot)), "a slot twice in one batch");
            assert!(rows.len() <= self.spec_cap);
            self.spec_calls.push(rows.len());
            std::thread::sleep(self.row_cost * rows.len() as u32);
            let mut out = Vec::new();
            for r in rows {
                let pages = self.tables[r.slot].len() * PAGE;
                assert!(r.pos + self.spec_k < pages, "verify rows past the block table: pos {} k {} pages {pages}", r.pos, self.spec_k);
                let wanted = ((r.pos as u64 * 2654435761 + r.slot as u64) % (self.spec_k as u64 + 1)) as usize;
                let a = wanted.min(r.max_accept);
                self.accept_counts.push(a);
                let mut emitted = Vec::new();
                let mut t = r.token;
                for j in 0..=a {
                    let h = self.write(r.slot, r.pos + j, t);
                    t = next_token(h);
                    emitted.push(t);
                }
                out.push(emitted);
            }
            Ok(out)
        }
    }

    fn expected(prompt: &[u32], max_tokens: usize, stop: &[u32]) -> Vec<u32> {
        let mut h = 0;
        for &t in prompt {
            h = mix(h, t);
        }
        let mut out = Vec::new();
        loop {
            let t = next_token(h);
            if stop.contains(&t) {
                break;
            }
            out.push(t);
            if out.len() >= max_tokens {
                break;
            }
            h = mix(h, t);
        }
        out
    }

    fn request(len: usize, max_tokens: usize, salt: u32) -> Request {
        Request {
            prompt: (0..len as u32).map(|i| (i * 31 + salt * 977) % VOCAB).collect(),
            max_tokens,
            sampling: Sampling::GREEDY,
            stop: Vec::new(),
            salt: String::new(),
            media: Vec::new(),
            grammar: None,
        }
    }

    fn drain<E: Executor>(s: &mut Scheduler<E>) -> (HashMap<SeqId, Vec<u32>>, Vec<Event>, Vec<StepReport>) {
        let mut out: HashMap<SeqId, Vec<u32>> = HashMap::new();
        let mut all = Vec::new();
        let mut reports = Vec::new();
        let mut steps = 0;
        while !s.idle() {
            let mut events = Vec::new();
            reports.push(s.step(&mut events).unwrap());
            for e in &events {
                if let Event::Token { id, token } = e { out.entry(*id).or_default().push(*token) }
            }
            all.extend(events);
            steps += 1;
            assert!(steps < 100_000, "the scheduler made no progress");
        }
        (out, all, reports)
    }

    fn json_grammar() -> Arc<crate::grammar::Grammar> {
        let mut tokens = vec![Vec::new()];
        tokens.extend((0u8..=255).map(|b| vec![b]));
        tokens.resize(VOCAB as usize, Vec::new());
        crate::grammar::Grammar::json_object(Arc::new(crate::grammar::TokenTable::new(tokens)), VOCAB as usize, vec![0])
    }

    fn is_json_object(tokens: &[u32]) -> bool {
        let bytes: Vec<u8> = tokens.iter().filter(|&&t| (1..=256).contains(&t)).map(|&t| (t - 1) as u8).collect();
        serde_json::from_slice::<serde_json::Value>(&bytes).is_ok_and(|v| v.is_object())
    }

    #[test]
    fn constrained_requests_emit_a_json_object_and_finish_by_stop_even_with_a_spec_cap() {
        let mut fake = Fake::new(4096, 4, 256, 64);
        fake.spec_cap = 4;
        fake.spec_k = 2;
        let mut s = Scheduler::new(fake, Config { max_num_seqs: 4, ..Config::default() });
        let g = json_grammar();
        let ids: Vec<SeqId> = (0..3).map(|i| s.submit(Request { grammar: Some(g.clone()), ..request(20 + i, 300, i as u32) }).unwrap()).collect();
        let plain = s.submit(request(20, 30, 9)).unwrap();
        let (out, events, _) = drain(&mut s);
        for id in ids {
            assert!(is_json_object(&out[&id]), "{:?}", out[&id]);
            assert!(events.iter().any(|e| matches!(e, Event::Finished { id: i, finish: Finish::Stop, .. } if *i == id)));
        }
        assert_eq!(out[&plain].len(), 30);
        assert!(s.exec.masked_calls > 0);
    }

    #[test]
    fn an_executor_without_grammar_support_refuses_a_constrained_request() {
        let mut fake = Fake::new(4096, 2, 64, 64);
        fake.grammar = false;
        let mut s = Scheduler::new(fake, Config::default());
        assert!(s.submit(Request { grammar: Some(json_grammar()), ..request(8, 8, 1) }).is_err());
        assert!(s.submit(request(8, 8, 1)).is_ok());
    }

    #[test]
    fn a_request_that_asks_gets_one_logprob_per_generated_token_and_others_get_none() {
        let mut s = Scheduler::new(Fake::new(1024, 4, 64, 64), Config { max_num_seqs: 4, ..Config::default() });
        let mut asks = request(100, 6, 1);
        asks.sampling = Sampling::GREEDY.with_logprobs(Some(2));
        let a = s.submit(asks.clone()).unwrap();
        let b = s.submit(request(90, 6, 2)).unwrap();
        let (out, events, _) = drain(&mut s);
        let lps: Vec<&crate::logprobs::TokenLogprob> =
            events.iter().filter_map(|e| match e { Event::Logprob { id, logprob } if *id == a => Some(logprob), _ => None }).collect();
        assert_eq!(lps.iter().map(|l| l.token).collect::<Vec<_>>(), out[&a]);
        assert!(lps.iter().all(|l| l.top.len() == 2 && l.top[0].0 == l.token && (l.top[0].1 - l.logprob).abs() < 1e-6));
        let z = 5f32.exp() + 3f32.exp() + (VOCAB - 2) as f32;
        assert!((lps[0].logprob - (5f32.exp() / z).ln()).abs() < 1e-4, "{}", lps[0].logprob);
        assert!(!events.iter().any(|e| matches!(e, Event::Logprob { id, .. } if *id == b)));
        assert_eq!(out[&a], expected(&asks.prompt, 6, &[]));
    }

    #[test]
    fn sixty_four_mixed_sequences_match_single_sequence_output_under_preemption() {
        let fake = Fake::new(2048, 16, 120, 256);
        let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), admit_lookahead: 0, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        let mut want = HashMap::new();
        for q in 0..64u32 {
            let r = request([3, 17, 90, 400, 1100][q as usize % 5], 20 + (q as usize * 13) % 200, q);
            want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
        }
        let (got, events, _) = drain(&mut s);
        assert!(s.preemptions() > 0, "the pool was meant to run out");
        assert_eq!(s.blocks_used(), 0);
        assert_eq!(got, want);
        let finished = events.iter().filter(|e| matches!(e, Event::Finished { finish: Finish::Length, .. })).count();
        assert_eq!(finished, 64);
    }

    fn pressured(swap_gb: u64, lookahead: usize) -> (Scheduler<Fake>, HashMap<SeqId, Vec<u32>>) {
        let mut fake = Fake::new(2048, 16, 120, 256);
        fake.swap = true;
        let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), swap_space_bytes: swap_gb, admit_lookahead: lookahead, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        let mut want = HashMap::new();
        for q in 0..64u32 {
            let r = request([3, 17, 90, 400, 1100][q as usize % 5], 20 + (q as usize * 13) % 200, q);
            want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
        }
        (s, want)
    }

    #[test]
    fn swap_preemption_keeps_outputs_and_reprefills_nothing_when_the_host_pool_fits() {
        let (mut s, want) = pressured(1 << 30, 0);
        let (got, events, reports) = drain(&mut s);
        assert!(s.preemptions() > 0, "the pool was meant to run out");
        assert_eq!(s.swapped_out(), s.preemptions());
        assert_eq!(s.swapped_in(), s.swapped_out());
        assert_eq!(s.recomputed_tokens(), 0);
        assert!(s.swapped_bytes_total() >= s.exec.parks as u64 * 1100);
        assert!(s.swap_secs() >= 0.0);
        assert_eq!(s.exec.parks, s.exec.resumes);
        assert_eq!(s.swapped_bytes(), 0);
        assert_eq!(s.blocks_used(), 0);
        assert_eq!(got, want);
        let prompt_tokens: usize = (0..64usize).map(|q| [3, 17, 90, 400, 1100][q % 5]).sum();
        let prefilled: usize = reports.iter().map(|r| r.prefill_tokens).sum();
        assert_eq!(prefilled, prompt_tokens);
        assert_eq!(events.iter().filter(|e| matches!(e, Event::Finished { finish: Finish::Length, .. })).count(), 64);
    }

    #[test]
    fn swap_falls_back_to_recompute_when_the_host_pool_is_full() {
        let (mut s, want) = pressured(2500, 0);
        let (got, _, _) = drain(&mut s);
        assert!(s.swapped_out() > 0);
        assert!(s.recomputed_tokens() > 0);
        assert!(s.preemptions() > s.swapped_out());
        assert_eq!(s.swapped_bytes(), 0);
        assert_eq!(got, want);
        let (mut none, want) = pressured(0, 0);
        let (got, _, _) = drain(&mut none);
        assert_eq!(none.swapped_out(), 0);
        assert_eq!(got, want);
    }

    #[test]
    fn cancelling_a_parked_sequence_returns_its_host_space() {
        let (mut s, _) = pressured(1 << 30, 0);
        let mut events = Vec::new();
        while s.swapped_bytes() == 0 {
            s.step(&mut events).unwrap();
        }
        let ids: Vec<SeqId> = s.waiting.iter().filter(|q| q.parked.is_some()).map(|q| q.id).collect();
        for id in ids {
            s.cancel(id, &mut events);
        }
        assert_eq!(s.swapped_bytes(), 0);
    }

    #[test]
    fn lookahead_admission_preempts_less_than_prompt_only_admission() {
        let run = |lookahead: usize| {
            let fake = Fake::new(2048, 16, 120, 256);
            let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), admit_lookahead: lookahead, ..Config::default() };
            let mut s = Scheduler::new(fake, cfg);
            let mut want = HashMap::new();
            for q in 0..64u32 {
                let r = request([3, 17, 90, 400, 1100][q as usize % 5], 20 + (q as usize * 13) % 200, q);
                want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
            }
            let (got, _, _) = drain(&mut s);
            assert_eq!(got, want);
            s.preemptions()
        };
        assert!(run(512) < run(0));
    }

    #[test]
    fn speculative_steps_with_different_accept_counts_match_single_sequence_output_under_preemption() {
        for (k, cap) in [(1, 16), (2, 16), (3, 16), (2, 4)] {
            let mut fake = Fake::new(2048, 16, 120, 256);
            fake.spec_k = k;
            fake.spec_cap = cap;
            let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), admit_lookahead: 0, ..Config::default() };
            let mut s = Scheduler::new(fake, cfg);
            let mut want = HashMap::new();
            for q in 0..64u32 {
                let mut r = request([3, 17, 90, 400, 1100][q as usize % 5], 20 + (q as usize * 13) % 200, q);
                if q % 7 == 3 {
                    let full = expected(&r.prompt, r.max_tokens, &[]);
                    r.stop = vec![full[full.len() / 2]];
                }
                want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &r.stop));
            }
            let (got, events, reports) = drain(&mut s);
            assert!(s.preemptions() > 0, "the pool was meant to run out");
            assert_eq!(s.blocks_used(), 0);
            assert_eq!(got, want, "k = {k}, cap = {cap}");
            let finished = events.iter().filter(|e| matches!(e, Event::Finished { .. })).count();
            assert_eq!(finished, 64);
            let counts = &s.exec.accept_counts;
            for a in 0..=k {
                assert!(counts.contains(&a), "k = {k}: no step accepted {a}");
            }
            assert!(s.exec.spec_calls.iter().any(|&n| n > 1), "speculation ran batched");
            if cap < 16 {
                assert!(!s.exec.decode_calls.is_empty() && s.exec.decode_calls.iter().all(|&n| n > cap));
            }
            let drafted: usize = reports.iter().map(|r| r.drafted).sum();
            let accepted: usize = reports.iter().map(|r| r.accepted).sum();
            assert_eq!(drafted, s.exec.spec_calls.iter().sum::<usize>() * k);
            assert!(accepted <= drafted && accepted >= counts.iter().sum::<usize>() / 2);
        }
    }

    #[test]
    fn speculation_reserves_k_positions_of_context_and_blocks() {
        let mut fake = Fake::new(64, 2, 8, 64);
        fake.spec_k = 3;
        fake.spec_cap = 2;
        let mut s = Scheduler::new(fake, Config::default());
        assert!(s.submit(request(40, 22, 1)).is_err(), "40 + 22 + 3 drafts > 64");
        let r = request(40, 21, 1);
        let want = expected(&r.prompt, r.max_tokens, &[]);
        let id = s.submit(r).unwrap();
        let (got, _, _) = drain(&mut s);
        assert_eq!(got[&id], want);
    }

    #[test]
    fn decoders_run_every_step_while_a_long_prompt_prefills_in_chunks() {
        let fake = Fake::new(65536, 8, 8000, 2048);
        let cfg = Config { max_batched_tokens: 1024, step_cost: 0.0, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        for q in 0..4 {
            s.submit(request(50, 400, q)).unwrap();
        }
        let mut events = Vec::new();
        for _ in 0..3 {
            s.step(&mut events).unwrap();
        }
        let long = s.submit(request(40_000, 4, 99)).unwrap();
        let mut prefill_steps = 0;
        while s.running() < 5 || events.iter().all(|e| !matches!(e, Event::Token { id, .. } if *id == long)) {
            let r = s.step(&mut events).unwrap();
            assert_eq!(r.decode_rows, 4, "a decoder was skipped while the long prompt prefilled {r:?} {prefill_steps}");
            assert!(r.prefill_tokens + r.decode_rows <= 1024, "{r:?}");
            prefill_steps += 1;
        }
        assert!(prefill_steps >= 40, "40k tokens in {prefill_steps} steps of at most 1020");
    }

    fn cached_config(interval: usize, snapshots: usize) -> Config {
        Config {
            prefix_cache: PrefixCacheOptions { enabled: true, snapshot_interval: interval, max_snapshots: snapshots, max_blocks: 0, async_snapshots: true },
            ..Config::default()
        }
    }

    fn run_one(s: &mut Scheduler<Fake>, r: Request) -> (Vec<u32>, usize, usize) {
        let before: usize = s.exec.prefill_calls.iter().sum();
        let id = s.submit(r).unwrap();
        let (got, events, _) = drain(s);
        let cached = events
            .iter()
            .find_map(|e| match e {
                Event::Finished { id: i, cached_tokens, .. } if *i == id => Some(*cached_tokens),
                _ => None,
            })
            .unwrap();
        (got.get(&id).cloned().unwrap_or_default(), cached, s.exec.prefill_calls.iter().sum::<usize>() - before)
    }

    #[test]
    fn a_second_request_with_a_shared_prefix_prefills_only_the_remainder_and_matches() {
        for hybrid in [false, true] {
            let mut fake = Fake::new(2048, 4, 200, 256);
            fake.hybrid = hybrid;
            let mut s = Scheduler::new(fake, cached_config(64, 4));
            let base = request(300, 12, 1);
            let (first, cached, prefilled) = run_one(&mut s, base.clone());
            assert_eq!((cached, prefilled), (0, 300), "hybrid = {hybrid}");
            assert_eq!(first, expected(&base.prompt, 12, &[]));
            let mut longer = base.clone();
            longer.prompt.extend((0..70u32).map(|i| (i * 13 + 5) % VOCAB));
            let (second, cached, prefilled) = run_one(&mut s, longer.clone());
            assert_eq!(second, expected(&longer.prompt, 12, &[]), "hybrid = {hybrid}");
            assert_eq!(cached, 288, "hybrid = {hybrid}: the last full block of the first prompt");
            assert_eq!(prefilled, 370 - 288, "hybrid = {hybrid}");
            let stats = s.prefix_stats();
            assert_eq!((stats.queries, stats.hits), (300 + 370, 288));
            let (again, cached, _) = run_one(&mut s, longer.clone());
            assert_eq!(again, second);
            assert_eq!(cached, 368, "an identical prompt hits all but its last token's block");
            assert_eq!(s.blocks_used(), s.prefix_stats().blocks, "only the cache holds blocks when idle");
        }
    }

    fn session_scheduler(hybrid: bool, tag: &str) -> (Scheduler<Fake>, Arc<std::sync::atomic::AtomicU64>, Arc<crate::sessions::Shared>) {
        let dir = std::env::temp_dir().join(format!("sched-sessions-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut fake = Fake::new(4096, 4, 200, 256);
        fake.hybrid = hybrid;
        let compat = fake.session_compat().unwrap();
        let options = crate::sessions::SessionOptions { enabled: true, min_tokens: 64, ..Default::default() };
        let shared = Arc::new(crate::sessions::Shared::new(crate::sessions::SessionStore::open(&dir, compat).unwrap(), options));
        let mut s = Scheduler::new(fake, Config::default());
        s.enable_sessions(shared.clone());
        let clock = Arc::new(std::sync::atomic::AtomicU64::new(1_000));
        let c = clock.clone();
        s.set_session_clock(Box::new(move || c.load(std::sync::atomic::Ordering::SeqCst)));
        (s, clock, shared)
    }

    #[test]
    fn a_turn_after_six_idle_minutes_restores_from_disk_and_prefills_only_the_suffix() {
        for hybrid in [false, true] {
            let (mut s, clock, shared) = session_scheduler(hybrid, if hybrid { "h" } else { "p" });
            let first = request(150, 40, 3);
            let (out, cached, prefilled) = run_one(&mut s, first.clone());
            assert_eq!((cached, prefilled), (0, 150));
            let held = 150 + 40 - 1;
            assert_eq!(s.resident_sessions(), 1);
            let used_with_session = s.blocks_used();
            assert!(used_with_session >= blocks_for(held));
            clock.fetch_add(299, std::sync::atomic::Ordering::SeqCst);
            s.tick_sessions();
            assert!(shared.store.list().is_empty(), "4:59 idle is not written");
            clock.fetch_add(2, std::sync::atomic::Ordering::SeqCst);
            s.tick_sessions();
            let listed = shared.store.list();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].tokens, held);
            assert_eq!(s.resident_sessions(), 0);
            assert_eq!(s.blocks_used(), 0, "the GPU blocks are free after the write");
            clock.fetch_add(60, std::sync::atomic::Ordering::SeqCst);
            let mut turn2 = first.clone();
            turn2.prompt.extend(&out);
            turn2.prompt.extend((0..30u32).map(|i| (i * 7 + 1) % VOCAB));
            let (got, cached, prefilled) = run_one(&mut s, turn2.clone());
            assert_eq!(got, expected(&turn2.prompt, 40, &[]), "hybrid = {hybrid}");
            assert_eq!(cached, held, "hybrid = {hybrid}");
            assert_eq!(prefilled, turn2.prompt.len() - held);
            assert_eq!(s.exec.session_restores, 1);
            assert!(shared.store.list().is_empty(), "keep_after_restore is off");
        }
    }

    #[test]
    fn a_held_session_and_parked_sequences_compete_for_one_host_pool() {
        let dir = std::env::temp_dir().join(format!("sched-sessions-pool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut fake = Fake::new(4096, 16, 120, 256);
        fake.swap = true;
        fake.hybrid = true;
        let compat = fake.session_compat().unwrap();
        let options = crate::sessions::SessionOptions { enabled: true, min_tokens: 64, ..Default::default() };
        let shared = Arc::new(crate::sessions::Shared::new(crate::sessions::SessionStore::open(&dir, compat).unwrap(), options));
        let used = fake.host_used.clone();
        let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), swap_space_bytes: 6000, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        assert_eq!(s.exec.host_limit, 6000, "one limit for parking and sessions");
        s.enable_sessions(shared.clone());
        let clock = Arc::new(std::sync::atomic::AtomicU64::new(1_000));
        let c = clock.clone();
        s.set_session_clock(Box::new(move || c.load(std::sync::atomic::Ordering::SeqCst)));
        let first = request(150, 10, 9);
        let (out, _, _) = run_one(&mut s, first.clone());
        assert_eq!(s.resident_sessions(), 1);

        let mut want = HashMap::new();
        for q in 0..64u32 {
            let r = request([3, 17, 90, 400, 1100][q as usize % 5], 20 + (q as usize * 13) % 200, q + 100);
            want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
        }
        let mut events = Vec::new();
        let mut early: HashMap<SeqId, Vec<u32>> = HashMap::new();
        while s.swapped_bytes() == 0 {
            s.step(&mut events).unwrap();
            for e in events.drain(..) {
                if let Event::Token { id, token } = e {
                    early.entry(id).or_default().push(token);
                }
            }
        }
        let parked = used.load(std::sync::atomic::Ordering::SeqCst);
        assert!(parked > 0 && parked <= 6000);
        clock.fetch_add(301, std::sync::atomic::Ordering::SeqCst);
        s.tick_sessions();
        let session_cost = blocks_for(150 + 10 - 1) * PAGE * 4;
        if parked + session_cost > 6000 {
            assert!(shared.store.list().is_empty(), "the session write waits while parked sequences hold the pool");
            assert_eq!(s.resident_sessions(), 1);
        }
        assert!(used.load(std::sync::atomic::Ordering::SeqCst) <= 6000);
        let (got, _, _) = drain(&mut s);
        for (id, w) in &want {
            let mut all = early.get(id).cloned().unwrap_or_default();
            all.extend(got.get(id).cloned().unwrap_or_default());
            assert_eq!(&all, w);
        }
        assert_eq!(used.load(std::sync::atomic::Ordering::SeqCst), 0, "every pinned byte is returned");
        clock.fetch_add(10, std::sync::atomic::Ordering::SeqCst);
        s.tick_sessions();
        assert!(shared.store.list().iter().any(|e| e.tokens == 150 + 10 - 1), "once the pool is free the session is written");
        let mut turn2 = first.clone();
        turn2.prompt.extend(&out);
        turn2.prompt.extend([1, 2, 3]);
        let (_, cached, _) = run_one(&mut s, turn2);
        assert_eq!(cached, 150 + 10 - 1);
    }

    #[test]
    fn a_session_is_deleted_after_eight_hours_and_then_the_prompt_is_prefilled_in_full() {
        let (mut s, clock, shared) = session_scheduler(true, "ttl");
        let first = request(150, 10, 5);
        let (out, _, _) = run_one(&mut s, first.clone());
        clock.fetch_add(301, std::sync::atomic::Ordering::SeqCst);
        s.tick_sessions();
        assert_eq!(shared.store.list().len(), 1);
        clock.fetch_add(8 * 3600 + 301, std::sync::atomic::Ordering::SeqCst);
        s.tick_sessions();
        assert!(shared.store.list().is_empty());
        let mut turn2 = first.clone();
        turn2.prompt.extend(&out);
        turn2.prompt.extend([1, 2, 3]);
        let (got, cached, prefilled) = run_one(&mut s, turn2.clone());
        assert_eq!((cached, prefilled), (0, turn2.prompt.len()));
        assert_eq!(got, expected(&turn2.prompt, 10, &[]));
    }

    #[test]
    fn a_corrupt_snapshot_falls_back_to_a_full_prefill_with_the_same_output() {
        let (mut s, clock, shared) = session_scheduler(true, "corrupt");
        let first = request(150, 10, 6);
        let (out, _, _) = run_one(&mut s, first.clone());
        clock.fetch_add(301, std::sync::atomic::Ordering::SeqCst);
        s.tick_sessions();
        let id = shared.store.list()[0].id.clone();
        let path = shared.store.dir().join(format!("{id}.kvs"));
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.len() / 2;
        bytes[at] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        let mut turn2 = first.clone();
        turn2.prompt.extend(&out);
        turn2.prompt.extend([1, 2, 3]);
        let (got, cached, prefilled) = run_one(&mut s, turn2.clone());
        assert_eq!((cached, prefilled), (0, turn2.prompt.len()));
        assert_eq!(got, expected(&turn2.prompt, 10, &[]));
        assert!(shared.store.list().is_empty());
    }

    #[test]
    fn a_different_salt_never_restores_another_sessions_snapshot() {
        let (mut s, clock, shared) = session_scheduler(false, "salt");
        let first = request(150, 10, 7);
        let (out, _, _) = run_one(&mut s, first.clone());
        clock.fetch_add(301, std::sync::atomic::Ordering::SeqCst);
        s.tick_sessions();
        assert_eq!(shared.store.list().len(), 1);
        let mut other = first.clone();
        other.salt = "tenant-b".into();
        other.prompt.extend(&out);
        other.prompt.extend([1, 2, 3]);
        let (_, cached, _) = run_one(&mut s, other);
        assert_eq!(cached, 0);
    }

    #[test]
    fn the_next_chat_turn_hits_the_end_of_the_previous_request_even_when_it_is_not_block_aligned() {
        for (hybrid, want) in [(false, 189), (true, 189)] {
            let mut fake = Fake::new(2048, 4, 200, 256);
            fake.hybrid = hybrid;
            let mut s = Scheduler::new(fake, cached_config(512, 4));
            let first = request(150, 40, 3);
            let (out, _, _) = run_one(&mut s, first.clone());
            let mut turn2 = first.clone();
            turn2.prompt.extend(&out);
            turn2.prompt.extend((0..30u32).map(|i| (i * 7 + 1) % VOCAB));
            let (got, cached, prefilled) = run_one(&mut s, turn2.clone());
            assert_eq!(got, expected(&turn2.prompt, 40, &[]), "hybrid = {hybrid}");
            assert_eq!(cached, want, "hybrid = {hybrid}: the whole previous request, tail block included");
            assert_eq!(prefilled, turn2.prompt.len() - want);
        }
    }

    #[test]
    fn many_concurrent_chats_each_hit_their_previous_turn_with_the_default_snapshot_budget() {
        let chats = 37;
        let mut fake = Fake::new(2048, 4, 6000, 256);
        fake.hybrid = true;
        let mut prefix_cache = PrefixCacheOptions { enabled: true, ..PrefixCacheOptions::default() };
        prefix_cache.max_snapshots = crate::serve::auto_snapshots(prefix_cache.max_snapshots, chats, 157 << 20, 100 << 30);
        assert_eq!(prefix_cache.max_snapshots, 148);
        let cfg = Config { prefix_cache, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        let mut turns = Vec::new();
        for c in 0..chats {
            let first = request(150, 40, 10 + c as u32);
            let (out, _, _) = run_one(&mut s, first.clone());
            let mut next = first;
            next.prompt.extend(&out);
            next.prompt.extend((0..30u32).map(|i| (i * 7 + c as u32) % VOCAB));
            turns.push(next);
        }
        for (c, turn) in turns.into_iter().enumerate() {
            let (_, cached, _) = run_one(&mut s, turn);
            assert_eq!(cached, 189, "chat {c} lost its previous turn");
        }
    }

    #[test]
    fn a_hybrid_hit_restores_the_snapshot_and_a_plain_model_needs_none() {
        let mut s = Scheduler::new(Fake::new(2048, 4, 200, 256), cached_config(128, 4));
        run_one(&mut s, request(300, 4, 2));
        assert!(s.prefix_stats().snapshots >= 2, "interval points and the end of the request");
        let before = s.exec.restores;
        let (_, cached, _) = run_one(&mut s, request(300, 4, 2));
        assert!(cached > 0);
        assert_eq!(s.exec.restores, before + 1);
        let mut plain = Fake::new(2048, 4, 200, 256);
        plain.hybrid = false;
        let mut s = Scheduler::new(plain, cached_config(128, 4));
        run_one(&mut s, request(300, 4, 2));
        assert_eq!(s.prefix_stats().snapshots, 0);
        let (_, cached, _) = run_one(&mut s, request(300, 4, 2));
        assert!(cached > 0);
        assert_eq!(s.exec.restores, 0);
    }

    #[test]
    fn prefill_stops_at_the_snapshot_interval_and_the_cache_is_off_by_default() {
        let mut s = Scheduler::new(Fake::new(2048, 4, 200, 1024), cached_config(128, 8));
        run_one(&mut s, request(400, 2, 5));
        assert_eq!(&s.exec.prefill_calls[..4], &[128, 128, 128, 16]);
        let mut off = Scheduler::new(Fake::new(2048, 4, 200, 1024), Config::default());
        run_one(&mut off, request(400, 2, 5));
        let (_, cached, prefilled) = run_one(&mut off, request(400, 2, 5));
        assert_eq!((cached, prefilled), (0, 400));
        assert_eq!(off.prefix_stats(), PrefixStats::default());
        assert_eq!(off.blocks_used(), 0);
    }

    #[test]
    fn different_salts_never_share_a_prefix() {
        let mut s = Scheduler::new(Fake::new(2048, 4, 200, 256), cached_config(64, 16));
        let a = Request { salt: "key-a".into(), ..request(200, 3, 4) };
        run_one(&mut s, a.clone());
        let (_, cached, _) = run_one(&mut s, Request { salt: "key-b".into(), ..a.clone() });
        assert_eq!(cached, 0);
        let (_, cached, _) = run_one(&mut s, a);
        assert!(cached > 0);
    }

    #[test]
    fn cached_blocks_are_evicted_when_new_requests_need_the_pool_and_output_stays_exact() {
        for hybrid in [false, true] {
            let mut fake = Fake::new(1024, 2, 40, 256);
            fake.hybrid = hybrid;
            let mut s = Scheduler::new(fake, cached_config(64, 4));
            for q in 0..12u32 {
                let r = request(200, 20, q);
                let (got, _, _) = run_one(&mut s, r.clone());
                assert_eq!(got, expected(&r.prompt, 20, &[]), "request {q}, hybrid = {hybrid}");
                assert!(s.prefix_stats().blocks <= 40);
            }
            assert!(s.prefix_stats().blocks > 0);
            let (_, cached, _) = run_one(&mut s, request(200, 20, 11));
            assert!(cached > 0, "the most recent prompt survived");
            let (_, cached, _) = run_one(&mut s, request(200, 20, 0));
            assert_eq!(cached, 0, "the oldest was evicted");
        }
    }

    #[test]
    fn concurrent_requests_with_one_shared_prefix_stay_exact_under_preemption() {
        let mut fake = Fake::new(2048, 8, 90, 256);
        fake.hybrid = true;
        let cfg = Config { max_num_seqs: 8, max_batched_tokens: 512, reserve_blocks: Some(1), ..cached_config(64, 6) };
        let mut s = Scheduler::new(fake, cfg);
        let shared: Vec<u32> = (0..200u32).map(|i| (i * 17 + 3) % VOCAB).collect();
        let mut want = HashMap::new();
        for q in 0..40u32 {
            let mut r = request(10 + (q as usize * 7) % 90, 30 + (q as usize * 11) % 100, q);
            r.prompt = [shared.clone(), r.prompt].concat();
            want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
        }
        let (got, _, _) = drain(&mut s);
        assert_eq!(got, want);
        assert!(s.prefix_stats().hits > 0);
        assert_eq!(s.blocks_used(), s.prefix_stats().blocks);
    }

    #[test]
    fn a_stop_token_ends_the_sequence_and_is_not_streamed() {
        let mut s = Scheduler::new(Fake::new(512, 2, 64, 128), Config::default());
        let r = request(10, 50, 1);
        let full = expected(&r.prompt, 50, &[]);
        let stop = full[5];
        let id = s.submit(Request { stop: vec![stop], ..r.clone() }).unwrap();
        let (got, events, _) = drain(&mut s);
        assert_eq!(got[&id], expected(&r.prompt, 50, &[stop]));
        assert!(events.contains(&Event::Finished {
            id,
            finish: Finish::Stop,
            prompt_tokens: 10,
            completion_tokens: got[&id].len(),
            cached_tokens: 0
        }));
    }

    #[test]
    fn cancelling_frees_blocks_and_the_slot_at_once() {
        let mut s = Scheduler::new(Fake::new(1024, 2, 64, 128), Config::default());
        let a = s.submit(request(200, 500, 1)).unwrap();
        let b = s.submit(request(30, 5, 2)).unwrap();
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        assert!(s.blocks_used() > 0);
        s.cancel(a, &mut events);
        s.cancel(b, &mut events);
        assert_eq!((s.blocks_used(), s.running(), s.waiting()), (0, 0, 0));
        let cancelled = events.iter().filter(|e| matches!(e, Event::Finished { finish: Finish::Cancelled, .. })).count();
        assert_eq!(cancelled, 2);
    }

    #[test]
    fn requests_that_never_fit_are_refused_up_front() {
        let mut s = Scheduler::new(Fake::new(1024, 2, 8, 128), Config::default());
        assert!(s.submit(request(1000, 100, 1)).is_err());
        assert!(s.submit(request(100, 100, 1)).is_err(), "13 blocks against a pool of 8");
        assert!(s.submit(request(60, 60, 1)).is_ok());
    }

    #[test]
    fn admission_waits_for_free_blocks_instead_of_overcommitting() {
        let mut s = Scheduler::new(Fake::new(1024, 4, 40, 512), Config { reserve_blocks: Some(2), ..Config::default() });
        s.submit(request(400, 100, 1)).unwrap();
        s.submit(request(400, 100, 2)).unwrap();
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        assert_eq!((s.running(), s.waiting()), (1, 1));
        let (_, _, _) = drain(&mut s);
        assert_eq!(s.blocks_used(), 0);
    }

    #[test]
    fn running_never_exceeds_max_num_seqs_even_when_the_executor_has_more_slots() {
        let mut s = Scheduler::new(Fake::new(1024, 42, 400, 512), Config { max_num_seqs: 21, ..Config::default() });
        for q in 0..42 {
            s.submit(request(20, 30, q)).unwrap();
        }
        let mut peak = 0;
        while !s.idle() {
            let mut events = Vec::new();
            s.step(&mut events).unwrap();
            peak = peak.max(s.running());
        }
        assert_eq!(peak, 21);
    }

    #[test]
    fn admission_stops_cleanly_when_no_recurrent_state_can_be_allocated() {
        let mut fake = Fake::new(1024, 8, 400, 512);
        fake.state_budget = 2;
        let mut s = Scheduler::new(fake, Config::default());
        for q in 0..5 {
            s.submit(request(20, 30, q)).unwrap();
        }
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        assert_eq!((s.running(), s.waiting()), (2, 3));
        assert!(s.state_refusals() > 0);
        assert_eq!(s.exec.states_live.len(), 2);
        let (_, _, _) = drain(&mut s);
        assert!(s.exec.states_live.is_empty());
        assert_eq!(s.blocks_used(), 0);
    }

    #[test]
    fn recurrent_states_are_freed_by_finish_cancel_and_preemption_with_no_growth() {
        let mut fake = Fake::new(512, 6, 60, 256);
        fake.state_budget = 4;
        let mut s = Scheduler::new(fake, Config { max_num_seqs: 6, reserve_blocks: Some(1), admit_lookahead: 0, ..Config::default() });
        let mut x = 12345u64;
        let mut events = Vec::new();
        for round in 0..1000u32 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if !(x >> 40).is_multiple_of(3) && s.waiting() < 8 {
                s.submit(request(1 + ((x >> 20) % 200) as usize, 1 + ((x >> 8) % 12) as usize, round)).unwrap();
            }
            s.step(&mut events).unwrap();
            assert!(s.exec.states_live.len() <= 4);
            assert_eq!(s.exec.states_live.len(), s.running());
        }
        let (_, _, _) = drain(&mut s);
        assert!(s.exec.states_live.is_empty());
        assert_eq!(s.blocks_used(), 0);
    }

    #[test]
    fn prefill_beside_decoders_costs_a_multiple_of_the_decode_step_from_the_first_step() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        fake.row_cost = Duration::from_millis(10);
        let cfg = Config { max_batched_tokens: 8192, step_cost: 3.0, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        s.submit(request(50, 400, 1)).unwrap();
        let mut events = Vec::new();
        for _ in 0..3 {
            s.step(&mut events).unwrap();
        }
        s.submit(request(20_000, 4, 2)).unwrap();
        let mut prefilled = 0;
        for _ in 0..12 {
            let started = Instant::now();
            let r = s.step(&mut events).unwrap();
            assert_eq!(r.decode_rows, 1);
            let took = started.elapsed();
            assert!(took < Duration::from_millis(48), "a step of {took:?} with {} prefill tokens", r.prefill_tokens);
            prefilled += r.prefill_tokens;
        }
        assert!(prefilled > 12 * 64, "{prefilled}");
    }

    #[test]
    fn a_slow_decode_step_still_gets_a_real_prefill_budget() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, step_cost: 3.0, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        s.submit(request(50, 400, 1)).unwrap();
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        s.submit(request(20_000, 4, 2)).unwrap();
        s.step(&mut events).unwrap();
        s.decode_secs = 0.095;
        s.prefill_rate = 10_000.0;
        let budget = s.prefill_budget(1);
        assert!(budget >= 1536, "{budget}");
        let capped = Scheduler::new(Fake::new(65536, 8, 8000, 2048), Config { step_cost: 3.0, itl_target: Duration::from_millis(100), ..Config::default() });
        let mut capped = capped;
        capped.decode_secs = 0.095;
        capped.prefill_rate = 10_000.0;
        assert!(capped.prefill_budget(1) <= 128, "{}", capped.prefill_budget(1));
    }

    #[test]
    fn prefill_beside_decoders_stays_inside_the_inter_token_target_from_the_first_step() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, step_cost: 0.0, itl_target: Duration::from_millis(50), ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        s.submit(request(50, 400, 1)).unwrap();
        let mut events = Vec::new();
        for _ in 0..3 {
            s.step(&mut events).unwrap();
        }
        s.submit(request(20_000, 4, 2)).unwrap();
        for _ in 0..30 {
            let started = Instant::now();
            let r = s.step(&mut events).unwrap();
            assert_eq!(r.decode_rows, 1);
            let took = started.elapsed();
            assert!(took < Duration::from_millis(75), "a step of {took:?} with {} prefill tokens", r.prefill_tokens);
        }
    }

    fn whole_prompt_steps(whole: usize) -> (usize, Vec<usize>, HashMap<SeqId, Vec<u32>>, HashMap<SeqId, Vec<u32>>) {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, step_cost: 0.0, itl_target: Duration::from_millis(50), whole_prompt_tokens: whole, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        let mut want = HashMap::new();
        for q in 0..4u32 {
            let r = request(50, 300, q);
            want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
        }
        let mut events = Vec::new();
        for _ in 0..4 {
            s.step(&mut events).unwrap();
        }
        let r = request(1500, 4, 9);
        want.insert(s.submit(r.clone()).unwrap(), expected(&r.prompt, r.max_tokens, &[]));
        let mut got: HashMap<SeqId, Vec<u32>> = HashMap::new();
        for e in &events {
            if let Event::Token { id, token } = e {
                got.entry(*id).or_default().push(*token);
            }
        }
        let (rest, _, reports) = drain(&mut s);
        for (id, v) in rest {
            got.entry(id).or_default().extend(v);
        }
        let prefills: Vec<usize> = reports.iter().map(|r| r.prefill_tokens).filter(|&n| n > 0).collect();
        (prefills.len(), prefills, got, want)
    }

    #[test]
    fn a_prompt_under_the_whole_prompt_limit_prefills_in_one_step_beside_decoders() {
        let (steps, prefills, got, want) = whole_prompt_steps(4096);
        assert_eq!(steps, 1, "{prefills:?}");
        assert_eq!(prefills[0], 1500);
        assert_eq!(got, want);
    }

    #[test]
    fn without_the_whole_prompt_limit_a_prompt_beside_decoders_takes_several_steps() {
        let (steps, prefills, got, want) = whole_prompt_steps(0);
        assert!(steps > 2, "{prefills:?}");
        assert_eq!(got, want);
    }

    #[test]
    fn a_prompt_over_the_whole_prompt_limit_is_still_chunked() {
        let (steps, _, got, want) = whole_prompt_steps(1000);
        assert!(steps > 2);
        assert_eq!(got, want);
    }

    #[test]
    fn a_slow_decode_step_shrinks_the_prefill_beside_it() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, step_cost: 0.0, itl_target: Duration::from_millis(60), ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        s.submit(request(50, 400, 1)).unwrap();
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        s.submit(request(20_000, 4, 2)).unwrap();
        s.step(&mut events).unwrap();
        let free = s.step(&mut events).unwrap().prefill_tokens;
        s.decode_secs = 0.045;
        let slow = s.step(&mut events).unwrap().prefill_tokens;
        assert!(slow * 2 < free, "free {free} slow {slow}");
    }

    #[test]
    fn a_first_token_mid_step_caps_the_rest_of_that_steps_prefill() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, step_cost: 0.0, itl_target: Duration::from_millis(50), ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        s.submit(request(500, 400, 1)).unwrap();
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        s.step(&mut events).unwrap();
        s.cancel(1, &mut events);
        let a = s.submit(request(300, 400, 2)).unwrap();
        s.submit(request(20_000, 4, 3)).unwrap();
        let mut last: Option<Instant> = None;
        let mut worst = Duration::ZERO;
        for _ in 0..200 {
            let mut ev = Vec::new();
            s.step(&mut ev).unwrap();
            if ev.iter().any(|e| matches!(e, Event::Token { id, .. } if *id == a)) {
                let now = Instant::now();
                if let Some(p) = last {
                    worst = worst.max(now - p);
                }
                last = Some(now);
            }
        }
        assert!(worst < Duration::from_millis(90), "a {worst:?} gap after the first token");
    }

    struct LoadResult {
        short_ttft: Vec<f64>,
        long_ttft: Vec<f64>,
        long_tok_per_step: f64,
        step_p95: f64,
    }

    fn pct(v: &mut [f64], p: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p) as usize]
    }

    const SCALE: f64 = 10.0;

    fn loaded(decoders: usize, cfg: Config) -> LoadResult {
        let mut fake = Fake::new(65536, 40, 12000, 2048);
        let (token_cost, row_cost) = (Duration::from_micros(20), Duration::from_micros(560));
        fake.token_cost = token_cost;
        fake.row_cost = row_cost;
        let mut s = Scheduler::new(fake, cfg);
        for q in 0..decoders as u32 {
            s.submit(request(50, 60_000, q)).unwrap();
        }
        let mut events = Vec::new();
        for _ in 0..6 {
            s.step(&mut events).unwrap();
        }
        let mut sent_at: HashMap<SeqId, (f64, bool)> = HashMap::new();
        let mut clock = 0.0f64;
        let mut short_ttft = Vec::new();
        let mut long_ttft = Vec::new();
        let mut steps = Vec::new();
        let mut long_tokens = 0usize;
        let mut long_steps = 0usize;
        let mut shorts = 0u32;
        let mut longs = 0u32;
        let id = s.submit(request(20_000, 4, 900)).unwrap();
        sent_at.insert(id, (clock, true));
        longs += 1;
        for n in 0..6000 {
            if n % 10 == 0 && shorts < 8 {
                let id = s.submit(request(1200, 2, 100 + shorts)).unwrap();
                sent_at.insert(id, (clock, false));
                shorts += 1;
            }
            if n % 25 == 5 && longs < 4 {
                let id = s.submit(request(6000 + 2000 * longs as usize, 2, 200 + longs)).unwrap();
                sent_at.insert(id, (clock, true));
                longs += 1;
            }
            let mut ev = Vec::new();
            let r = s.step(&mut ev).unwrap();
            let step_cost = (token_cost * r.prefill_tokens as u32 + row_cost * r.decode_rows as u32).as_secs_f64();
            clock += step_cost * SCALE;
            if r.decode_rows > 0 {
                steps.push(step_cost);
            }
            if r.prefill_tokens > 0 && sent_at.values().any(|v| v.1) {
                long_tokens += r.prefill_tokens;
                long_steps += 1;
            }
            for e in &ev {
                if let Event::Token { id, .. } = e
                    && let Some((t, long)) = sent_at.remove(id) {
                        if long { &mut long_ttft } else { &mut short_ttft }.push(clock - t);
                    }
            }
            if shorts == 8 && longs == 4 && sent_at.is_empty() {
                break;
            }
        }
        assert_eq!(short_ttft.len(), 8, "every short prompt got a first token");
        assert_eq!(long_ttft.len(), 4, "every long prompt completed its prefill");
        LoadResult { step_p95: pct(&mut steps, 0.95) * SCALE, short_ttft, long_ttft, long_tok_per_step: long_tokens as f64 / long_steps.max(1) as f64 }
    }

    fn load_cfg(min_prefill_tokens: usize, shortest_first: bool) -> Config {
        sweep_cfg(min_prefill_tokens, shortest_first, if shortest_first || min_prefill_tokens > 0 { 1024 } else { 4096 })
    }

    fn sweep_cfg(min_prefill_tokens: usize, shortest_first: bool, whole_prompt_tokens: usize) -> Config {
        Config { max_batched_tokens: 8192, step_cost: 0.0, itl_target: Duration::from_millis(10), whole_prompt_tokens, min_prefill_tokens, shortest_first, ..Config::default() }
    }

    #[test]
    fn prompts_stay_bounded_behind_each_other_with_full_decoders() {
        let mut r = loaded(16, load_cfg(512, true));
        let short = pct(&mut r.short_ttft, 0.95);
        let long = pct(&mut r.long_ttft, 1.0);
        assert!(short < 8.0, "short ttft p95 {short:.3}s");
        assert!(long < 20.0, "long ttft max {long:.3}s");
        assert!(r.long_tok_per_step > 400.0, "long prompts read {:.0} tokens per step", r.long_tok_per_step);
        assert!(r.step_p95 <= 0.35, "step p95 {:.3}s", r.step_p95);
    }

    #[test]
    #[ignore]
    fn print_sweep_table() {
        for decoders in [8usize, 16, 20] {
            for (whole, min) in [(4096usize, 512usize), (2048, 512), (1536, 512), (1024, 512), (0, 512), (0, 384), (0, 256), (1024, 256)] {
                let mut r = loaded(decoders, sweep_cfg(min, true, whole));
                println!(
                    "decoders {decoders} whole {whole} min {min}: short ttft p50 {:.2} p95 {:.2} long ttft p50 {:.2} max {:.2} long tok/step {:.0} step p95 {:.3}",
                    pct(&mut r.short_ttft, 0.5), pct(&mut r.short_ttft, 0.95), pct(&mut r.long_ttft, 0.5), pct(&mut r.long_ttft, 1.0), r.long_tok_per_step, r.step_p95
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn print_load_table() {
        for decoders in [8usize, 16, 20] {
            for (name, min, sf) in [("master", 0usize, false), ("shortest", 0, true), ("min512", 512, false), ("both", 512, true), ("both256", 256, true)] {
                let mut r = loaded(decoders, load_cfg(min, sf));
                println!(
                    "decoders {decoders} {name}: short ttft p50 {:.2} p95 {:.2} long ttft p50 {:.2} max {:.2} long tok/step {:.0} step p95 {:.3}",
                    pct(&mut r.short_ttft, 0.5), pct(&mut r.short_ttft, 0.95), pct(&mut r.long_ttft, 0.5), pct(&mut r.long_ttft, 1.0), r.long_tok_per_step, r.step_p95
                );
            }
        }
    }

    const LONG_TOKEN_COST: Duration = Duration::from_micros(20);
    const LONG_ROW_COST: Duration = Duration::from_micros(1250);

    fn long_fake() -> Fake {
        let mut fake = Fake::new(20480, 26, 14000, 4096);
        fake.light = true;
        fake.token_cost = LONG_TOKEN_COST;
        fake.row_cost = LONG_ROW_COST;
        fake.spec_k = 2;
        fake.spec_cap = 26;
        fake
    }

    fn long_cfg(min_prefill_tokens: usize) -> Config {
        Config { max_num_seqs: 26, max_batched_tokens: 16384, step_cost: 0.0, itl_target: Duration::from_millis(10), whole_prompt_tokens: 1024, min_prefill_tokens, ..Config::default() }
    }

    fn step_cost(r: &StepReport) -> f64 {
        (LONG_TOKEN_COST * r.prefill_tokens as u32 + LONG_ROW_COST * r.decode_rows as u32).as_secs_f64() * SCALE
    }

    struct Beside {
        steps_to_first_token: usize,
        long_running_max: usize,
    }

    fn long_prompts_beside_decoders(cfg: Config, longs: usize) -> Beside {
        let mut s = Scheduler::new(long_fake(), cfg);
        for q in 0..8u32 {
            s.submit(request(64, 5000, q)).unwrap();
        }
        let mut events = Vec::new();
        for _ in 0..6 {
            s.step(&mut events).unwrap();
        }
        let ids: Vec<SeqId> = (0..longs).map(|i| s.submit(request(16384, 4, 500 + i as u32)).unwrap()).collect();
        let mut steps = 0;
        let mut long_running_max = 0;
        loop {
            let mut ev = Vec::new();
            s.step(&mut ev).unwrap();
            steps += 1;
            let long_running = s.running.iter().filter(|q| ids.contains(&q.id) && q.prefill_target() - q.computed > LONG_PROMPT).count();
            long_running_max = long_running_max.max(long_running);
            if ev.iter().any(|e| matches!(e, Event::Token { id, .. } if *id == ids[0])) {
                return Beside { steps_to_first_token: steps, long_running_max };
            }
            assert!(steps < 400, "the long prompt never finished");
        }
    }

    #[test]
    fn a_long_prompt_beside_decoders_is_read_in_burst_sized_steps() {
        let old = long_prompts_beside_decoders(long_cfg(512), 1);
        assert!(old.steps_to_first_token >= 30, "the 512 floor reads 16k in about 32 steps, got {}", old.steps_to_first_token);
        let burst = long_prompts_beside_decoders(Config { prefill_burst_tokens: 16384, ..long_cfg(512) }, 1);
        assert!(burst.steps_to_first_token <= 6, "a 16k prompt took {} steps", burst.steps_to_first_token);
    }

    #[test]
    fn several_long_prompts_may_prefill_at_once() {
        let old = long_prompts_beside_decoders(long_cfg(512), 4);
        assert_eq!(old.long_running_max, 1, "one long prompt in flight by default");
        let wide = long_prompts_beside_decoders(Config { max_long_in_flight: 4, ..long_cfg(512) }, 4);
        assert_eq!(wide.long_running_max, 4);
    }

    struct LongLoad {
        ttft: Vec<f64>,
        gaps: Vec<f64>,
        stream_tok_s: Vec<f64>,
        prefill_tok_s: f64,
        running_mean: f64,
    }

    fn long_load(cfg: Config, requests: usize) -> LongLoad {
        let mut s = Scheduler::new(long_fake(), cfg);
        let mut clock = 0.0f64;
        let mut queued: VecDeque<(usize, f64)> = VecDeque::new();
        let mut sent: HashMap<SeqId, f64> = HashMap::new();
        let mut tokens_at: HashMap<SeqId, Vec<f64>> = HashMap::new();
        let (mut arrived, mut in_flight, mut done) = (0usize, 0usize, 0usize);
        let (mut prompt_tokens, mut prefill_time) = (0usize, 0.0f64);
        let (mut running_sum, mut steps) = (0usize, 0usize);
        let mut ttft = Vec::new();
        while done < requests {
            while arrived < requests && arrived as f64 <= clock {
                queued.push_back((arrived, clock));
                arrived += 1;
            }
            while in_flight < 8 && let Some((i, _)) = queued.pop_front() {
                let len = 15000 + (i * 7919) % 3000;
                let out = 256 + (i * 104729) % 256;
                let id = s.submit(request(len, out, 1000 + i as u32)).unwrap();
                sent.insert(id, clock);
                in_flight += 1;
            }
            if s.idle() {
                clock = clock.max(arrived as f64);
                continue;
            }
            let mut ev = Vec::new();
            let r = s.step(&mut ev).unwrap();
            let cost = step_cost(&r);
            clock += cost;
            steps += 1;
            running_sum += s.running();
            if r.prefill_tokens > 0 {
                prompt_tokens += r.prefill_tokens;
                prefill_time += cost;
            }
            for e in &ev {
                match e {
                    Event::Token { id, .. } => {
                        let at = tokens_at.entry(*id).or_default();
                        if at.is_empty() {
                            ttft.push(clock - sent[id]);
                        }
                        at.push(clock);
                    }
                    Event::Finished { .. } => {
                        in_flight -= 1;
                        done += 1;
                    }
                    _ => {}
                }
            }
            assert!(steps < 200_000, "the load never drained");
        }
        let mut gaps = Vec::new();
        let mut stream_tok_s = Vec::new();
        for at in tokens_at.values() {
            gaps.extend(at.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0));
            if at.len() > 1 {
                stream_tok_s.push((at.len() - 1) as f64 / (at[at.len() - 1] - at[0]));
            }
        }
        LongLoad { ttft, gaps, stream_tok_s, prefill_tok_s: prompt_tokens as f64 / prefill_time, running_mean: running_sum as f64 / steps as f64 }
    }

    #[test]
    #[ignore]
    fn print_long_prompt_table() {
        let base = long_cfg(512);
        for (name, cfg) in [
            ("default", base),
            ("min4096", Config { min_prefill_tokens: 4096, ..base }),
            ("min16384", Config { min_prefill_tokens: 16384, ..base }),
            ("burst16384", Config { prefill_burst_tokens: 16384, ..base }),
            ("long4", Config { max_long_in_flight: 4, ..base }),
            ("burst16384+long4", Config { prefill_burst_tokens: 16384, max_long_in_flight: 4, ..base }),
            ("burst8192+long4", Config { prefill_burst_tokens: 8192, max_long_in_flight: 4, ..base }),
        ] {
            let mut r = long_load(cfg, 24);
            let n = r.ttft.len();
            println!(
                "{name}: ttft p50 {:.1} p95 {:.1} last {:.1} s, prefill {:.0} tok/s, running mean {:.1}, stream decode p50 {:.1} tok/s, step gap p50 {:.0} p99 {:.0} max {:.0} ms",
                pct(&mut r.ttft.clone(), 0.5),
                pct(&mut r.ttft.clone(), 0.95),
                r.ttft[n - 1],
                r.prefill_tok_s,
                r.running_mean,
                pct(&mut r.stream_tok_s, 0.5),
                pct(&mut r.gaps.clone(), 0.5) * 1e3,
                pct(&mut r.gaps.clone(), 0.99) * 1e3,
                pct(&mut r.gaps, 1.0) * 1e3,
            );
        }
    }

    #[test]
    fn never_hit_cached_blocks_are_not_live_and_never_hold_up_admission() {
        let mut s = Scheduler::new(Fake::new(2048, 4, 64, 256), cached_config(64, 4));
        for q in 0..6 {
            run_one(&mut s, request(300, 4, 100 + q));
        }
        assert_eq!(s.blocks_live(), 0);
        assert!(s.blocks_cached() > 40, "the pool is mostly cache: {}", s.blocks_cached());
        assert_eq!(s.blocks_used(), s.blocks_cached());
        let big = request(900, 4, 777);
        let want = expected(&big.prompt, 4, &[]);
        let id = s.submit(big).unwrap();
        let mut events = Vec::new();
        s.step(&mut events).unwrap();
        assert_eq!(s.running(), 1, "admitted at once by evicting cache");
        assert_eq!(s.waiting(), 0);
        assert!(s.blocks_live() > 0);
        let (got, _, _) = drain(&mut s);
        let mut all: Vec<u32> = events.iter().filter_map(|e| match e { Event::Token { token, .. } => Some(*token), _ => None }).collect();
        all.extend(got.get(&id).cloned().unwrap_or_default());
        assert_eq!(all, want);
        assert_eq!(s.blocks_live(), 0);
    }

    struct Window {
        us_per_step: f64,
        evictions_per_step: f64,
        prefill_per_step: f64,
        preemptions: u64,
    }

    fn host_cost_per_step(pool_blocks: usize, steps_each: usize) -> (Window, Window, usize) {
        let mut fake = Fake::new(8192, 8, pool_blocks, 2048);
        fake.light = true;
        let cfg = Config { max_num_seqs: 8, prefix_cache: PrefixCacheOptions { enabled: true, snapshot_interval: 2048, max_snapshots: 4, max_blocks: 0, async_snapshots: true }, ..Config::default() };
        let mut s = Scheduler::new(fake, cfg);
        let mut next = 1u32;
        let window = |s: &mut Scheduler<Fake>, steps: usize, next: &mut u32| {
            let mut spent = Duration::ZERO;
            let mut events = Vec::new();
            let (evict0, pre0) = (s.prefix_stats().evictions, s.preemptions());
            let mut prefill = 0;
            for _ in 0..steps {
                while s.waiting() + s.running() < 12 {
                    *next += 1;
                    let r = if (*next).is_multiple_of(10) { request(1000, 2000 + (*next as usize * 7919) % 4000, *next) } else { request(1200, 8 + (*next as usize * 31) % 40, *next) };
                    s.submit(r).unwrap();
                }
                events.clear();
                let t = Instant::now();
                let report = s.step(&mut events).unwrap();
                spent += t.elapsed();
                prefill += report.prefill_tokens;
            }
            Window {
                us_per_step: spent.as_secs_f64() * 1e6 / steps as f64,
                evictions_per_step: (s.prefix_stats().evictions - evict0) as f64 / steps as f64,
                prefill_per_step: prefill as f64 / steps as f64,
                preemptions: s.preemptions() - pre0,
            }
        };
        let early = window(&mut s, steps_each, &mut next);
        let mut rounds = 0;
        while s.blocks_used() * 100 < pool_blocks * 98 {
            window(&mut s, 200, &mut next);
            rounds += 1;
            assert!(rounds < 3000, "the pool never filled");
        }
        let late = window(&mut s, steps_each, &mut next);
        (early, late, s.prefix_stats().blocks)
    }

    #[test]
    fn host_time_per_step_stays_flat_while_never_hit_cache_fills_the_pool() {
        for pool in [10_900, 58_066] {
            let (early, late, blocks) = host_cost_per_step(pool, 3000);
            println!(
                "pool {pool}: early {:.1} us/step, late {:.1} us/step; evictions/step {:.3} -> {:.3}; prefill tokens/step {:.0} -> {:.0}; preemptions {} -> {}; cached {blocks}",
                early.us_per_step, late.us_per_step, early.evictions_per_step, late.evictions_per_step, early.prefill_per_step, late.prefill_per_step, early.preemptions, late.preemptions
            );
            assert!(
                late.us_per_step < early.us_per_step * 1.5 + 5.0,
                "pool {pool}: {:.1} us per step early, {:.1} us late with {blocks} blocks cached",
                early.us_per_step,
                late.us_per_step
            );
        }
    }

    #[test]
    fn a_thousand_random_alloc_free_cycles_leak_nothing() {
        let mut pool = BlockPool::new(97);
        let mut held: Vec<Vec<u32>> = Vec::new();
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut rand = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..1000 {
            if rand() % 3 != 0 || held.is_empty() {
                let n = (rand() % 9) as usize;
                let mut got = Vec::new();
                for _ in 0..n {
                    match pool.alloc() {
                        Some(b) => got.push(b),
                        None => break,
                    }
                }
                held.push(got);
            } else {
                let i = (rand() as usize) % held.len();
                let mut v = held.swap_remove(i);
                pool.release(&mut v);
            }
            let mut all: Vec<u32> = held.iter().flatten().copied().chain(pool.free.iter().copied()).collect();
            all.sort_unstable();
            assert_eq!(all, (0..97).collect::<Vec<u32>>());
        }
        for mut v in held.drain(..) {
            pool.release(&mut v);
        }
        assert_eq!(pool.free_count(), 97);
    }
}
