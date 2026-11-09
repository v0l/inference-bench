use crate::codegen::PAGE;
use crate::prefix::{self, Hash, PrefixCache, StateSnapshot};
use crate::runtime::{DecodeRow, PrefillPart, Runtime, Sampling, SpecRow, Stepper};
use crate::spec::PrefixCacheOptions;
use anyhow::{Result, bail};
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
    total: usize,
}

impl BlockPool {
    pub fn new(total: usize) -> Self {
        BlockPool { free: (0..total as u32).rev().collect(), refs: vec![0; total], total }
    }

    pub fn refs(&self, block: u32) -> u32 {
        self.refs[block as usize]
    }

    pub fn retain(&mut self, block: u32) {
        debug_assert!(self.refs[block as usize] > 0);
        self.refs[block as usize] += 1;
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
            if self.refs[b as usize] == 0 {
                self.free.push(b);
            }
        }
        debug_assert!(self.free.len() <= self.total);
    }
}

pub trait Executor {
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
}

pub struct GpuExecutor {
    pub rt: Runtime,
    st: Stepper,
    fuse: bool,
}

impl GpuExecutor {
    pub fn new(rt: Runtime) -> Result<Self> {
        let st = rt.stepper()?;
        let fuse = rt.spec().runtime.fuse && rt.draft_tokens().is_none();
        Ok(GpuExecutor { rt, st, fuse })
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
        self.rt.prefill_slot(&mut self.st, slot, tokens, start, sample)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        self.rt.decode(&mut self.st, rows)
    }

    fn fused(
        &mut self,
        rows: &[DecodeRow],
        slot: usize,
        tokens: &[u32],
        start: usize,
        sample: Option<(Sampling, u32)>,
    ) -> Result<(Vec<u32>, Option<u32>)> {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub max_num_seqs: usize,
    pub max_batched_tokens: usize,
    pub itl_target: Duration,
    pub reserve_blocks: Option<usize>,
    pub min_prefill_chunk: usize,
    pub prefix_cache: PrefixCacheOptions,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_num_seqs: 32,
            max_batched_tokens: 8192,
            itl_target: Duration::from_millis(100),
            reserve_blocks: None,
            min_prefill_chunk: 64,
            prefix_cache: PrefixCacheOptions::default(),
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
    state_refusals: u64,
    prefill_rate: f64,
    cache: Option<PrefixCache>,
    prefix_queries: u64,
    prefix_hits: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub queries: u64,
    pub hits: u64,
    pub blocks: usize,
    pub snapshots: usize,
    pub snapshot_bytes: usize,
}

fn ewma(old: f64, new: f64) -> f64 {
    if old <= 0.0 { new } else { 0.8 * old + 0.2 * new }
}

impl<E: Executor> Scheduler<E> {
    pub fn new(exec: E, cfg: Config) -> Self {
        let pool = BlockPool::new(exec.kv_blocks());
        let free_slots = (0..exec.slots()).rev().collect();
        let cache = (cfg.prefix_cache.enabled && exec.supports_prefix_cache()).then(|| PrefixCache::new(&cfg.prefix_cache, exec.hybrid()));
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
            state_refusals: 0,
            prefill_rate: 0.0,
            cache,
            prefix_queries: 0,
            prefix_hits: 0,
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
        }
    }

    fn alloc_block(&mut self) -> Option<u32> {
        loop {
            if let Some(b) = self.pool.alloc() {
                return Some(b);
            }
            let cache = self.cache.as_mut()?;
            if !cache.evict_one(&mut self.pool) {
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

    pub fn blocks_total(&self) -> usize {
        self.pool.total()
    }

    pub fn state_refusals(&self) -> u64 {
        self.state_refusals
    }

    pub fn preemptions(&self) -> u64 {
        self.preemptions
    }

    pub fn idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    fn reserve(&self) -> usize {
        self.cfg.reserve_blocks.unwrap_or((self.pool.total() / 100).max(1))
    }

    pub fn submit(&mut self, req: Request) -> Result<SeqId> {
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
        let id = self.next_id;
        self.next_id += 1;
        let tokens = req.prompt.clone();
        let root = self.exec.prefix_root(&req.salt);
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
        });
        Ok(id)
    }

    pub fn cancel(&mut self, id: SeqId, events: &mut Vec<Event>) {
        if let Some(i) = self.waiting.iter().position(|s| s.id == id) {
            let s = self.waiting.remove(i).expect("index from position");
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

    fn preempt_latest(&mut self, events: &mut Vec<Event>) -> SeqId {
        let i = (0..self.running.len()).max_by_key(|&i| self.running[i].admitted).expect("something is running");
        let mut s = self.running.remove(i);
        self.release(&mut s);
        s.computed = 0;
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
        while let Some(s) = self.waiting.front() {
            if self.running.len() >= cap || self.free_slots.is_empty() || planned >= budget {
                break;
            }
            let target = s.prefill_target();
            let need = blocks_for(target + 1);
            let reserve = if self.running.is_empty() { 0 } else { self.reserve() };
            let hit = match self.cache.as_mut() {
                Some(cache) => {
                    let hashes = prefix::block_hashes(&s.root, &s.tokens[..target]);
                    let limit = if s.generated() == 0 { target - 1 } else { target };
                    cache.lookup_with_tail(&s.root, &s.tokens[..target], &hashes, limit, &mut self.pool)
                }
                None => prefix::Hit::default(),
            };
            let fresh = need - hit.blocks.len();
            while self.pool.free_count() < fresh + reserve {
                let evicted = self.cache.as_mut().is_some_and(|c| c.evict_one(&mut self.pool));
                if !evicted {
                    break;
                }
            }
            if self.pool.free_count() < fresh + reserve {
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
            if let Some(state) = &hit.snapshot {
                if let Err(e) = self.exec.restore(slot, state) {
                    self.exec.release_state(slot);
                    self.give_back(hit);
                    self.waiting.push_front(s);
                    return Err(e);
                }
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
            s.slot = Some(slot);
            self.admissions += 1;
            s.admitted = self.admissions;
            planned += target - s.computed;
            self.running.push(s);
        }
        Ok(())
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
        let hashes = prefix::block_hashes(&s.root, &s.tokens[..upto * PAGE]);
        let blocks = s.blocks[..upto].to_vec();
        let slot = s.slot.expect("running has a slot");
        let parent = if upto == 0 { s.root } else { hashes[upto - 1] };
        let tail_tokens = s.tokens[upto * PAGE..s.computed].to_vec();
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

    fn apply_decode(&mut self, rows: &[(usize, DecodeRow)], next: Vec<u32>, done: &mut Vec<usize>, events: &mut Vec<Event>) {
        for ((i, _), t) in rows.iter().zip(next) {
            let s = &mut self.running[*i];
            s.computed += 1;
            if accept(s, t, self.exec.max_ctx(), events) {
                done.push(*i);
            }
        }
    }

    fn prefill_budget(&self, decodes: usize) -> usize {
        let full = self.cfg.max_batched_tokens.saturating_sub(decodes);
        let align = self.cfg.min_prefill_chunk.max(1);
        if decodes == 0 {
            return full.max(align);
        }
        if self.prefill_rate <= 0.0 {
            return align.min(full.max(1));
        }
        let fit = (self.cfg.itl_target.as_secs_f64() * self.prefill_rate) as usize;
        let fit = if fit > align { fit / align * align } else { align };
        fit.min(full.max(align))
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
        if pending && rows.len() <= cap {
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
        for i in 0..self.running.len() {
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
                if self.cache.is_some() && self.exec.hybrid() {
                    if let Some(p) = self.snapshot_points(i).into_iter().find(|&p| p > self.running[i].computed) {
                        n = n.min(p - self.running[i].computed);
                    }
                }
                let s = &self.running[i];
                let last = s.computed + n == target;
                let sample = (last && s.generated() == 0).then_some((s.req.sampling, 0));
                let slot = s.slot.expect("running has a slot");
                let chunk = s.tokens[s.computed..s.computed + n].to_vec();
                let start = s.computed;
                let t = if pending {
                    pending = false;
                    let (next, t) = self.exec.fused(&plain, slot, &chunk, start, sample)?;
                    report.decode_rows = rows.len();
                    self.apply_decode(&rows, next, &mut done, events);
                    t
                } else {
                    self.exec.prefill(slot, &chunk, start, sample)?
                };
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
            let next = self.exec.decode(&plain)?;
            report.decode_secs = started.elapsed().as_secs_f64();
            report.decode_rows = rows.len();
            self.apply_decode(&rows, next, &mut done, events);
        }
        if report.prefill_tokens > 0 {
            self.exec.flush()?;
            report.prefill_secs = started.elapsed().as_secs_f64();
            if report.prefill_tokens >= self.cfg.min_prefill_chunk {
                self.prefill_rate = ewma(self.prefill_rate, report.prefill_tokens as f64 / report.prefill_secs.max(1e-6));
            }
        }
        done.sort_unstable();
        done.dedup();
        for &i in done.iter().rev() {
            self.cache_progress(i, true)?;
            let mut s = self.running.remove(i);
            self.release(&mut s);
            let finish = finish_of(&s);
            events.push(finished(&s, finish));
        }
        Ok(report)
    }
}

fn finish_of(s: &Seq) -> Finish {
    match s.tokens.last() {
        Some(t) if s.req.stop.contains(t) => Finish::Stop,
        _ => Finish::Length,
    }
}

fn finished(s: &Seq, finish: Finish) -> Event {
    let completion = s.generated() - usize::from(finish == Finish::Stop);
    Event::Finished { id: s.id, finish, prompt_tokens: s.req.prompt.len(), completion_tokens: completion, cached_tokens: s.cached }
}

fn accept(s: &mut Seq, t: u32, max_ctx: usize, events: &mut Vec<Event>) -> bool {
    s.tokens.push(t);
    let stop = s.req.stop.contains(&t);
    if !stop {
        events.push(Event::Token { id: s.id, token: t });
    }
    stop || s.generated() >= s.req.max_tokens || s.tokens.len() >= max_ctx
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

    pub struct Fake {
        max_ctx: usize,
        slots: usize,
        kv: Vec<Option<u32>>,
        tables: Vec<Vec<u32>>,
        state: Vec<(u64, usize)>,
        chunk: usize,
        pub token_cost: Duration,
        pub decode_calls: Vec<usize>,
        pub prefill_calls: Vec<usize>,
        pub state_budget: usize,
        pub states_live: std::collections::HashSet<usize>,
        pub hybrid: bool,
        pub prefix_enabled: bool,
        pub restores: usize,
        pub block_copies: usize,
        pub spec_k: usize,
        pub spec_cap: usize,
        pub spec_calls: Vec<usize>,
        pub accept_counts: Vec<usize>,
    }

    impl Fake {
        pub fn new(max_ctx: usize, slots: usize, blocks: usize, chunk: usize) -> Self {
            Fake {
                max_ctx,
                slots,
                kv: vec![None; blocks * PAGE],
                tables: vec![Vec::new(); slots],
                state: vec![(0, 0); slots],
                chunk,
                token_cost: Duration::ZERO,
                decode_calls: Vec::new(),
                prefill_calls: Vec::new(),
                state_budget: usize::MAX,
                states_live: Default::default(),
                hybrid: true,
                prefix_enabled: true,
                restores: 0,
                block_copies: 0,
                spec_k: 0,
                spec_cap: 0,
                spec_calls: Vec::new(),
                accept_counts: Vec::new(),
            }
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
            Ok(sample.map(|_| next_token(h)))
        }
        fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
            let mut seen = std::collections::HashSet::new();
            assert!(rows.iter().all(|r| seen.insert(r.slot)), "a slot twice in one batch");
            self.decode_calls.push(rows.len());
            Ok(rows.iter().map(|r| next_token(self.write(r.slot, r.pos, r.token))).collect())
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
        fn copy_block(&mut self, src: u32, dst: u32) -> Result<()> {
            for i in 0..PAGE {
                self.kv[dst as usize * PAGE + i] = self.kv[src as usize * PAGE + i];
            }
            self.block_copies += 1;
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
                match e {
                    Event::Token { id, token } => out.entry(*id).or_default().push(*token),
                    _ => {}
                }
            }
            all.extend(events);
            steps += 1;
            assert!(steps < 100_000, "the scheduler made no progress");
        }
        (out, all, reports)
    }

    #[test]
    fn sixty_four_mixed_sequences_match_single_sequence_output_under_preemption() {
        let fake = Fake::new(2048, 16, 120, 256);
        let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), ..Config::default() };
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

    #[test]
    fn speculative_steps_with_different_accept_counts_match_single_sequence_output_under_preemption() {
        for (k, cap) in [(1, 16), (2, 16), (3, 16), (2, 4)] {
            let mut fake = Fake::new(2048, 16, 120, 256);
            fake.spec_k = k;
            fake.spec_cap = cap;
            let cfg = Config { max_num_seqs: 16, max_batched_tokens: 512, reserve_blocks: Some(1), ..Config::default() };
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
        let cfg = Config { max_batched_tokens: 1024, ..Config::default() };
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
            assert_eq!(r.decode_rows, 4, "a decoder was skipped while the long prompt prefilled");
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
        let mut s = Scheduler::new(fake, Config { max_num_seqs: 6, reserve_blocks: Some(1), ..Config::default() });
        let mut x = 12345u64;
        let mut events = Vec::new();
        for round in 0..1000u32 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (x >> 40) % 3 != 0 && s.waiting() < 8 {
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
    fn prefill_beside_decoders_stays_inside_the_inter_token_target_from_the_first_step() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, itl_target: Duration::from_millis(50), ..Config::default() };
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

    #[test]
    fn a_first_token_mid_step_caps_the_rest_of_that_steps_prefill() {
        let mut fake = Fake::new(65536, 8, 8000, 2048);
        fake.token_cost = Duration::from_micros(100);
        let cfg = Config { max_batched_tokens: 8192, itl_target: Duration::from_millis(50), ..Config::default() };
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
        for _ in 0..40 {
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
