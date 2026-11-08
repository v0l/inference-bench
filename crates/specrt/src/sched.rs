use crate::codegen::PAGE;
use crate::runtime::{DecodeRow, Runtime, Sampling, Stepper};
use anyhow::{Result, bail};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub type SeqId = u64;

pub fn blocks_for(tokens: usize) -> usize {
    tokens.div_ceil(PAGE)
}

#[derive(Debug, Clone)]
pub struct BlockPool {
    free: Vec<u32>,
    total: usize,
}

impl BlockPool {
    pub fn new(total: usize) -> Self {
        BlockPool { free: (0..total as u32).rev().collect(), total }
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
        self.free.pop()
    }

    pub fn release(&mut self, blocks: &mut Vec<u32>) {
        debug_assert!(blocks.iter().all(|b| (*b as usize) < self.total));
        self.free.append(blocks);
        debug_assert!(self.free.len() <= self.total);
    }
}

pub trait Executor {
    fn max_ctx(&self) -> usize;
    fn slots(&self) -> usize;
    fn kv_blocks(&self) -> usize;
    fn chunk_limit(&self) -> usize;
    fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()>;
    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>>;
    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>>;
}

pub struct GpuExecutor {
    pub rt: Runtime,
    st: Stepper,
}

impl GpuExecutor {
    pub fn new(rt: Runtime) -> Result<Self> {
        let st = rt.stepper()?;
        Ok(GpuExecutor { rt, st })
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

    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
        self.rt.prefill_slot(&mut self.st, slot, tokens, start, sample)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        self.rt.decode(&mut self.st, rows)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub max_num_seqs: usize,
    pub max_batched_tokens: usize,
    pub itl_target: Duration,
    pub reserve_blocks: Option<usize>,
    pub min_prefill_chunk: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_num_seqs: 32,
            max_batched_tokens: 8192,
            itl_target: Duration::from_millis(100),
            reserve_blocks: None,
            min_prefill_chunk: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop: Vec<u32>,
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
    Finished { id: SeqId, finish: Finish, prompt_tokens: usize, completion_tokens: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepReport {
    pub decode_rows: usize,
    pub prefill_tokens: usize,
    pub preemptions: usize,
    pub decode_secs: f64,
    pub prefill_secs: f64,
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
    decode_ewma: f64,
    prefill_rate: f64,
}

fn ewma(old: f64, new: f64) -> f64 {
    if old <= 0.0 { new } else { 0.8 * old + 0.2 * new }
}

impl<E: Executor> Scheduler<E> {
    pub fn new(exec: E, cfg: Config) -> Self {
        let pool = BlockPool::new(exec.kv_blocks());
        let free_slots = (0..exec.slots()).rev().collect();
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
            decode_ewma: 0.0,
            prefill_rate: 0.0,
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
        if req.prompt.len() + req.max_tokens > max_ctx {
            bail!(
                "prompt of {} tokens plus {} new tokens exceeds the context of {max_ctx}",
                req.prompt.len(),
                req.max_tokens
            );
        }
        let need = blocks_for(req.prompt.len() + req.max_tokens);
        if need > self.pool.total() {
            bail!("a request of {need} KV blocks can never fit the pool of {}", self.pool.total());
        }
        let id = self.next_id;
        self.next_id += 1;
        let tokens = req.prompt.clone();
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
            if !s.decoding() || blocks_for(s.tokens.len()) <= s.blocks.len() {
                i += 1;
                continue;
            }
            match self.pool.alloc() {
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
            let need = blocks_for(s.prefill_target() + 1);
            let reserve = if self.running.is_empty() { 0 } else { self.reserve() };
            if self.pool.free_count() < need + reserve {
                break;
            }
            let mut s = self.waiting.pop_front().expect("front exists");
            let slot = self.free_slots.pop().expect("a free slot");
            for _ in 0..blocks_for(s.prefill_target()) {
                s.blocks.push(self.pool.alloc().expect("checked free blocks"));
            }
            self.exec.set_table(slot, &s.blocks)?;
            s.slot = Some(slot);
            self.admissions += 1;
            s.admitted = self.admissions;
            planned += s.prefill_target();
            self.running.push(s);
        }
        Ok(())
    }

    fn prefill_budget(&self, decodes: usize) -> usize {
        let full = self.cfg.max_batched_tokens.saturating_sub(decodes);
        if decodes == 0 || self.prefill_rate <= 0.0 {
            return full.max(self.cfg.min_prefill_chunk);
        }
        let room = (self.cfg.itl_target.as_secs_f64() - self.decode_ewma).max(0.0);
        let fit = (room * self.prefill_rate) as usize;
        let align = self.cfg.min_prefill_chunk.max(1);
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
        if !rows.is_empty() {
            let started = Instant::now();
            let plain: Vec<DecodeRow> = rows.iter().map(|r| r.1).collect();
            let next = self.exec.decode(&plain)?;
            report.decode_secs = started.elapsed().as_secs_f64();
            self.decode_ewma = ewma(self.decode_ewma, report.decode_secs);
            report.decode_rows = rows.len();
            for ((i, _), t) in rows.iter().zip(next) {
                let s = &mut self.running[*i];
                s.computed += 1;
                if accept(s, t, self.exec.max_ctx(), events) {
                    done.push(*i);
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
                let n = (target - s.computed).min(limit).min(left);
                let last = s.computed + n == target;
                let sample = (last && s.generated() == 0).then_some((s.req.sampling, 0));
                let slot = s.slot.expect("running has a slot");
                let chunk = s.tokens[s.computed..s.computed + n].to_vec();
                let start = s.computed;
                let t = self.exec.prefill(slot, &chunk, start, sample)?;
                let s = &mut self.running[i];
                s.computed += n;
                left -= n;
                report.prefill_tokens += n;
                if let Some(t) = t
                    && accept(s, t, self.exec.max_ctx(), events)
                {
                    done.push(i);
                }
            }
        }
        if report.prefill_tokens > 0 {
            report.prefill_secs = started.elapsed().as_secs_f64();
            if report.prefill_tokens >= self.cfg.min_prefill_chunk {
                self.prefill_rate = ewma(self.prefill_rate, report.prefill_tokens as f64 / report.prefill_secs.max(1e-6));
            }
        }
        done.sort_unstable();
        done.dedup();
        for &i in done.iter().rev() {
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
    Event::Finished { id: s.id, finish, prompt_tokens: s.req.prompt.len(), completion_tokens: completion }
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
        pub decode_calls: Vec<usize>,
        pub prefill_calls: Vec<usize>,
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
                decode_calls: Vec::new(),
                prefill_calls: Vec::new(),
            }
        }

        fn write(&mut self, slot: usize, pos: usize, t: u32) -> u64 {
            let block = self.tables[slot][pos / PAGE] as usize;
            self.kv[block * PAGE + pos % PAGE] = Some(t);
            let (h, n) = self.state[slot];
            assert_eq!(n, pos, "recurrent state of slot {slot} is at {n}, write at {pos}");
            let h = mix(h, t);
            self.state[slot] = (h, pos + 1);
            let mut check = 0u64;
            for p in 0..=pos {
                let b = self.tables[slot][p / PAGE] as usize;
                check = mix(check, self.kv[b * PAGE + p % PAGE].expect("written KV"));
            }
            assert_eq!(check, h, "KV through the block table disagrees with the recurrent state");
            h
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
        fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
            assert!(tokens.len() <= self.chunk);
            if start == 0 {
                self.state[slot] = (0, 0);
            }
            self.prefill_calls.push(tokens.len());
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
            completion_tokens: got[&id].len()
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
