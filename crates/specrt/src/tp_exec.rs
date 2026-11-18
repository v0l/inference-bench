use crate::allreduce::{Fabric, Transport, contexts};
use crate::prefix::{Hash, StateSnapshot};
use crate::runtime::{DecodeRow, Layout, Mask, PrefillPart, Runtime, Sampling, SpecRow, Stepper};
use crate::sched::{Executor, LogitsAt};
use crate::spec::Spec;
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

const RANK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

type Job = Box<dyn FnOnce(&mut Rank) -> Result<Reply> + Send>;

pub enum Reply {
    Unit,
    Token(Option<u32>),
    Tokens(Vec<u32>),
    Fused(Vec<u32>, Option<u32>),
    Spec(Vec<Vec<u32>>),
    Logits(Vec<f32>),
    Snapshot(StateSnapshot),
}

pub struct Rank {
    pub rt: Runtime,
    pub st: Stepper,
}

struct Worker {
    jobs: Sender<Job>,
    replies: Receiver<Result<Reply>>,
    handle: Option<JoinHandle<()>>,
}

pub struct TpExecutor {
    workers: Vec<Worker>,
    max_ctx: usize,
    slots: usize,
    kv_blocks: usize,
    chunk: usize,
    draft_tokens: usize,
    speculation_cap: usize,
    hybrid: bool,
    root: Hash,
    spec: Spec,
    fuse: bool,
    fused_prefill: Option<usize>,
    _fabric: std::sync::Arc<Fabric>,
}

pub fn fabric_bytes(spec: &Spec, layout: &Layout) -> usize {
    Runtime::chunk_rows(layout) * spec.dims.hidden * 4
}

impl TpExecutor {
    pub fn load(spec: &Spec, layout: Layout, devices: &[usize]) -> Result<TpExecutor> {
        Self::load_with(spec, layout, devices, None)
    }

    pub fn load_with(spec: &Spec, layout: Layout, devices: &[usize], force: Option<Transport>) -> Result<TpExecutor> {
        let tp = devices.len();
        ensure!(tp >= 2, "tensor parallel needs at least two ranks");
        crate::tp::supports(spec, tp)?;
        let mut distinct = devices.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        ensure!(layout.kv_blocks > 0 || distinct.len() == devices.len(), "virtual ranks share one GPU, so give the KV cache an explicit size");
        let ctxs = contexts(devices)?;
        let fabric = std::sync::Arc::new(Fabric::new(&ctxs, fabric_bytes(spec, &layout), force)?);
        let mut spec = spec.clone();
        if distinct.len() < devices.len() && spec.runtime.graphs {
            tracing::warn!(spec = %spec.name, "virtual ranks share one GPU's work queues, so a rank's captured graph can block its peer at the allreduce; they run without CUDA graphs");
            spec.runtime.graphs = false;
        }
        let spec = &spec;
        tracing::info!(spec = %spec.name, tp, devices = ?devices, transport = ?fabric.transport(), "tensor parallel ranks");
        let mut workers = Vec::with_capacity(tp);
        let (ready_tx, ready_rx) = channel::<(usize, Result<()>)>();
        for (rank, &device) in devices.iter().enumerate() {
            let (jobs, inbox) = channel::<Job>();
            let (out, replies) = channel::<Result<Reply>>();
            let (spec, ready, fabric) = (spec.clone(), ready_tx.clone(), fabric.clone());
            let sink = crate::progress::current();
            let handle = std::thread::Builder::new()
                .name(format!("specrt-tp{rank}"))
                .spawn(move || {
                    let loaded = crate::progress::with_sink(sink, || -> Result<Rank> {
                        let rt = Runtime::load_rank(&spec, layout, device, rank, &fabric)?;
                        let st = rt.stepper()?;
                        Ok(Rank { rt, st })
                    });
                    let mut me = match loaded {
                        Ok(r) => {
                            let _ = ready.send((rank, Ok(())));
                            r
                        }
                        Err(e) => {
                            let _ = ready.send((rank, Err(e)));
                            return;
                        }
                    };
                    let _ = me.rt.context().bind_to_thread();
                    while let Ok(job) = inbox.recv() {
                        if out.send(job(&mut me)).is_err() {
                            break;
                        }
                    }
                })
                .context("spawn a tensor parallel rank")?;
            workers.push(Worker { jobs, replies, handle: Some(handle) });
        }
        drop(ready_tx);
        let mut failed = None;
        for _ in 0..tp {
            let (rank, r) = ready_rx.recv().map_err(|_| anyhow!("a tensor parallel rank died while loading"))?;
            if let Err(e) = r {
                failed.get_or_insert(e.context(format!("rank {rank}")));
            }
        }
        let mut exec = TpExecutor {
            workers,
            max_ctx: 0,
            slots: 0,
            kv_blocks: 0,
            chunk: 0,
            draft_tokens: 0,
            speculation_cap: 0,
            hybrid: spec.linear_layers() > 0,
            root: crate::prefix::bare_namespace(""),
            spec: spec.clone(),
            fuse: false,
            fused_prefill: None,
            _fabric: fabric,
        };
        if let Some(e) = failed {
            return Err(e);
        }
        let facts = exec.all(|r| Ok(Reply::Tokens(vec![r.rt.max_ctx() as u32, r.rt.slots() as u32, r.rt.kv_blocks() as u32, r.rt.prefill_chunk_limit() as u32, r.rt.draft_tokens().unwrap_or(0) as u32, r.rt.speculation_cap() as u32])))?;
        let first = match &facts[0] {
            Reply::Tokens(v) => v.clone(),
            _ => unreachable!(),
        };
        let mut kv_blocks = first[2];
        for f in &facts {
            let Reply::Tokens(v) = f else { unreachable!() };
            ensure!(v[..2] == first[..2] && v[3..] == first[3..], "tensor parallel ranks disagree on their layout: {v:?} against {first:?}");
            kv_blocks = kv_blocks.min(v[2]);
        }
        exec.max_ctx = first[0] as usize;
        exec.slots = first[1] as usize;
        exec.kv_blocks = kv_blocks as usize;
        exec.chunk = first[3] as usize;
        exec.draft_tokens = first[4] as usize;
        exec.speculation_cap = first[5] as usize;
        exec.fuse = false;
        Ok(exec)
    }

    pub fn ranks(&self) -> usize {
        self.workers.len()
    }

    pub fn all(&self, job: impl Fn(&mut Rank) -> Result<Reply> + Send + Sync + Clone + 'static) -> Result<Vec<Reply>> {
        for w in &self.workers {
            let j = job.clone();
            w.jobs.send(Box::new(move |r: &mut Rank| j(r))).map_err(|_| anyhow!("a tensor parallel rank has stopped"))?;
        }
        let mut out = Vec::with_capacity(self.workers.len());
        let mut first_err = None;
        for (i, w) in self.workers.iter().enumerate() {
            match w.replies.recv_timeout(RANK_TIMEOUT).map_err(|e| match e {
                std::sync::mpsc::RecvTimeoutError::Timeout => anyhow!("tensor parallel rank {i} did not answer within {RANK_TIMEOUT:?}; the ranks are out of step"),
                std::sync::mpsc::RecvTimeoutError::Disconnected => anyhow!("tensor parallel rank {i} has stopped"),
            }) {
                Ok(Ok(r)) => out.push(r),
                Ok(Err(e)) => {
                    first_err.get_or_insert(e.context(format!("tensor parallel rank {i}")));
                    out.push(Reply::Unit);
                }
                Err(e) => {
                    first_err.get_or_insert(e);
                    out.push(Reply::Unit);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    fn lead(&self, job: impl Fn(&mut Rank) -> Result<Reply> + Send + Sync + Clone + 'static) -> Result<Reply> {
        let mut replies = self.all(job)?;
        Ok(replies.swap_remove(0))
    }

    fn agree_tokens(replies: Vec<Reply>) -> Result<Vec<u32>> {
        let mut it = replies.into_iter();
        let Some(Reply::Tokens(first)) = it.next() else { bail!("no tokens from rank 0") };
        for (i, r) in it.enumerate() {
            let Reply::Tokens(t) = r else { bail!("rank {} sent no tokens", i + 1) };
            ensure!(t == first, "tensor parallel ranks sampled different tokens: rank 0 {first:?}, rank {} {t:?}", i + 1);
        }
        Ok(first)
    }

    pub fn prompt_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let tokens = tokens.to_vec();
        match self.lead(move |r| {
            r.rt.acquire_state(0)?;
            let logits = r.rt.prompt_logits(&tokens);
            r.rt.release_state(0);
            Ok(Reply::Logits(logits?))
        })? {
            Reply::Logits(l) => Ok(l),
            _ => unreachable!(),
        }
    }


    pub fn set_graphs(&self, on: bool) -> Result<()> {
        self.all(move |r| {
            r.st.set_graphs(on);
            Ok(Reply::Unit)
        })?;
        Ok(())
    }
}

impl Drop for TpExecutor {
    fn drop(&mut self) {
        for w in &mut self.workers {
            let (tx, _) = channel();
            drop(std::mem::replace(&mut w.jobs, tx));
        }
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

impl Executor for TpExecutor {
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn slots(&self) -> usize {
        self.slots
    }

    fn kv_blocks(&self) -> usize {
        self.kv_blocks
    }

    fn chunk_limit(&self) -> usize {
        self.chunk
    }

    fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
        let blocks = blocks.to_vec();
        self.all(move |r| r.rt.set_block_table(slot, &blocks).map(|_| Reply::Unit))?;
        Ok(())
    }

    fn acquire_state(&mut self, slot: usize) -> Result<()> {
        self.all(move |r| r.rt.acquire_state(slot).map(|_| Reply::Unit))?;
        Ok(())
    }

    fn release_state(&mut self, slot: usize) {
        let _ = self.all(move |r| {
            r.rt.release_state(slot);
            Ok(Reply::Unit)
        });
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
        self.fused_prefill = None;
        let tokens = tokens.to_vec();
        let replies = self.all(move |r| {
            let Rank { rt, st } = r;
            Ok(Reply::Token(rt.prefill_slot(st, slot, &tokens, start, sample)?))
        })?;
        let tokens = Self::agree_tokens(replies.into_iter().map(|r| match r {
            Reply::Token(t) => Reply::Tokens(t.into_iter().collect()),
            other => other,
        }).collect())?;
        Ok(tokens.first().copied())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        let rows = rows.to_vec();
        Self::agree_tokens(self.all(move |r| {
            let Rank { rt, st } = r;
            Ok(Reply::Tokens(rt.decode(st, &rows)?))
        })?)
    }

    fn fused(&mut self, rows: &[DecodeRow], slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<(Vec<u32>, Option<u32>)> {
        if !self.fuse {
            let next = self.decode(rows)?;
            let first = self.prefill(slot, tokens, start, sample)?;
            return Ok((next, first));
        }
        self.fused_prefill = Some(rows.len());
        let (rows, tokens) = (rows.to_vec(), tokens.to_vec());
        let replies = self.all(move |r| {
            let Rank { rt, st } = r;
            let (d, f) = rt.fused_step(st, &rows, PrefillPart { slot, tokens: &tokens, start, sample })?;
            Ok(Reply::Fused(d, f))
        })?;
        let mut it = replies.into_iter();
        let Some(Reply::Fused(d0, f0)) = it.next() else { bail!("no fused step from rank 0") };
        for (i, r) in it.enumerate() {
            let Reply::Fused(d, f) = r else { bail!("rank {} sent no fused step", i + 1) };
            ensure!(d == d0 && f == f0, "tensor parallel ranks sampled different tokens in a fused step (rank {})", i + 1);
        }
        Ok((d0, f0))
    }

    fn flush(&mut self) -> Result<()> {
        self.all(|r| r.rt.synchronize().map(|_| Reply::Unit))?;
        Ok(())
    }

    fn draft_tokens(&self) -> usize {
        self.draft_tokens
    }

    fn speculation_cap(&self) -> usize {
        self.speculation_cap
    }

    fn speculate(&mut self, rows: &[SpecRow]) -> Result<Vec<Vec<u32>>> {
        let rows = rows.to_vec();
        let replies = self.all(move |r| Ok(Reply::Spec(r.rt.speculate(&rows)?)))?;
        let mut it = replies.into_iter();
        let Some(Reply::Spec(first)) = it.next() else { bail!("no speculation from rank 0") };
        for (i, r) in it.enumerate() {
            let Reply::Spec(s) = r else { bail!("rank {} sent no speculation", i + 1) };
            ensure!(s == first, "tensor parallel ranks accepted different drafts (rank {})", i + 1);
        }
        Ok(first)
    }

    fn supports_prefix_cache(&self) -> bool {
        true
    }

    fn hybrid(&self) -> bool {
        self.hybrid
    }

    fn prefix_root(&self, salt: &str) -> Hash {
        let _ = self.root;
        crate::prefix::namespace(&self.spec, 1, &format!("tp{}:{salt}", self.workers.len()))
    }

    fn snapshot(&mut self, slot: usize) -> Result<StateSnapshot> {
        let replies = self.all(move |r| Ok(Reply::Snapshot(r.rt.snapshot_state(slot)?)))?;
        let parts: Vec<StateSnapshot> = replies.into_iter().map(|r| match r {
            Reply::Snapshot(s) => s,
            _ => unreachable!(),
        }).collect();
        Ok(StateSnapshot::stacked(parts))
    }

    fn restore(&mut self, slot: usize, state: &StateSnapshot) -> Result<()> {
        let parts = std::sync::Arc::new(state.unstack(self.workers.len())?);
        self.all(move |r| {
            let rank = r.rt.tp_rank().map_or(0, |(_, k)| k);
            r.rt.restore_state(slot, &parts[rank]).map(|_| Reply::Unit)
        })?;
        Ok(())
    }

    fn copy_block(&mut self, src: u32, dst: u32) -> Result<()> {
        self.all(move |r| r.rt.copy_kv_block(src, dst).map(|_| Reply::Unit))?;
        Ok(())
    }

    fn supports_logprobs(&self) -> bool {
        true
    }

    fn supports_grammar(&self) -> bool {
        true
    }

    fn set_masks(&mut self, rows: Vec<Option<Mask>>, first: Option<Mask>) {
        let _ = self.all(move |r| {
            r.st.set_masks(rows.clone(), first.clone());
            Ok(Reply::Unit)
        });
    }

    fn logits_of(&mut self, at: LogitsAt) -> Result<Vec<f32>> {
        let fused = self.fused_prefill;
        match self.lead(move |r| {
            Ok(Reply::Logits(match (at, fused) {
                (LogitsAt::Decode(i), _) | (LogitsAt::Prefill, Some(i)) => r.rt.step_logits(&r.st, i, 1)?,
                (LogitsAt::Prefill, None) => r.rt.prefill_logits()?,
            }))
        })? {
            Reply::Logits(l) => Ok(l),
            _ => unreachable!(),
        }
    }
}
