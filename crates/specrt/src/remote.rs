use crate::codegen::PAGE;
use crate::coordinator::Coordinator;
use crate::runtime::{DecodeRow, Sampling};
use crate::sched::{ChainMetrics, Executor, StageMetric, blocks_for};
use crate::wire::{Chain, ChainTimes};
use anyhow::{Result, bail, ensure};

pub struct ChainExecutor {
    coordinator: Coordinator,
    chain: Option<Chain>,
    generation: u64,
    max_ctx: usize,
    slots: usize,
    chunk: usize,
    tables: Vec<Vec<u32>>,
    lost: Option<String>,
    metrics: ChainMetrics,
    hidden_row_bytes: u64,
    micro_batch: usize,
}

impl ChainExecutor {
    pub fn new(coordinator: Coordinator, max_ctx: usize, slots: usize, chunk: usize) -> Self {
        ChainExecutor { coordinator, chain: None, generation: 0, max_ctx, slots, chunk, tables: vec![Vec::new(); slots], lost: None, metrics: ChainMetrics::default(), hidden_row_bytes: 0, micro_batch: 0 }
    }

    pub fn set_micro_batch(&mut self, micro: usize) {
        self.micro_batch = micro;
    }

    pub fn coordinator(&self) -> &Coordinator {
        &self.coordinator
    }

    pub fn table(&self, slot: usize) -> &[u32] {
        &self.tables[slot]
    }

    pub fn connected(&self) -> bool {
        self.chain.is_some()
    }

    pub fn connect_within(&mut self, wait: std::time::Duration) -> Result<()> {
        if self.chain.is_none() {
            self.coordinator.wait_ready(wait)?;
        }
        self.connect()
    }

    pub fn connect(&mut self) -> Result<()> {
        if self.chain.is_some() {
            return Ok(());
        }
        let chain = self.coordinator.connect_chain()?;
        self.generation = self.coordinator.generation();
        self.hidden_row_bytes = 4 * chain.hidden() as u64;
        let ranges = chain.ranges();
        if self.metrics.stages.len() != ranges.len() {
            self.metrics.stages = ranges.iter().enumerate().map(|(i, &(a, b))| StageMetric { device: i, start: a as usize, end: b as usize, ..StageMetric::default() }).collect();
        }
        self.chain = Some(chain);
        Ok(())
    }

    fn record(&mut self, times: &ChainTimes, rows: usize) {
        for (i, m) in self.metrics.stages.iter_mut().enumerate() {
            let (Some(compute), Some(round)) = (times.compute.get(i), times.stage.get(i)) else { continue };
            m.busy_secs += compute.as_secs_f64();
            m.calls += 1;
            m.tokens += rows as u64;
            self.metrics.hop_secs += round.saturating_sub(*compute).as_secs_f64();
            self.metrics.hops += 1;
            if i > 0 {
                self.metrics.hop_bytes += rows as u64 * self.hidden_row_bytes;
            }
        }
    }

    fn drop_chain(&mut self, reason: String) {
        self.chain = None;
        self.lost = Some(reason);
    }

    fn forward_pipelined(&mut self, slot: usize, position: usize, tokens: &[u32]) -> Result<u32> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let covered = self.tables[slot].len() * PAGE;
        ensure!(position + tokens.len() <= covered, "slot {slot} has {} blocks, not enough for position {}", self.tables[slot].len(), position + tokens.len());
        self.connect()?;
        let micro = self.micro_batch;
        let chain = self.chain.as_mut().expect("connected");
        match chain.forward_pipelined(slot as u32, position as u32, tokens, micro) {
            Ok((token, times)) => {
                for t in &times {
                    self.record(t, micro);
                }
                self.metrics.micro_batches += times.len() as u64;
                self.metrics.overlapped_prefills += 1;
                Ok(token)
            }
            Err(e) => {
                self.drop_chain(format!("{e:#}"));
                Err(e.context("a stage was lost, the request is failed"))
            }
        }
    }

    fn forward(&mut self, slot: usize, position: usize, tokens: Vec<u32>, decode: bool) -> Result<u32> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let covered = self.tables[slot].len() * PAGE;
        ensure!(position + tokens.len() <= covered, "slot {slot} has {} blocks, not enough for position {}", self.tables[slot].len(), position + tokens.len());
        self.connect()?;
        let chain = self.chain.as_mut().expect("connected");
        let n_rows = tokens.len();
        match chain.forward(slot as u32, position as u32, tokens, decode) {
            Ok((token, times)) => {
                self.record(&times, n_rows);
                Ok(token)
            }
            Err(e) => {
                self.drop_chain(format!("{e:#}"));
                Err(e.context("a stage was lost, the request is failed"))
            }
        }
    }
}

impl Executor for ChainExecutor {
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn slots(&self) -> usize {
        self.slots
    }

    fn kv_blocks(&self) -> usize {
        self.slots * blocks_for(self.max_ctx)
    }

    fn chunk_limit(&self) -> usize {
        self.chunk
    }

    fn set_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let total = self.kv_blocks();
        if let Some(b) = blocks.iter().find(|b| **b as usize >= total) {
            bail!("block {b} is outside the {total} blocks of the remote chain");
        }
        self.tables[slot] = blocks.to_vec();
        Ok(())
    }

    fn release_state(&mut self, slot: usize) {
        if let Some(t) = self.tables.get_mut(slot) {
            t.clear();
        }
        if let Some(chain) = self.chain.as_mut() {
            if let Err(e) = chain.free(slot as u32) {
                self.drop_chain(format!("{e:#}"));
            }
        }
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], start: usize, sample: Option<(Sampling, u32)>) -> Result<Option<u32>> {
        ensure!(!tokens.is_empty() && tokens.len() <= self.chunk, "prefill chunk of {} tokens", tokens.len());
        let token = if self.micro_batch > 0 && self.micro_batch < tokens.len() {
            self.forward_pipelined(slot, start, tokens)?
        } else {
            self.forward(slot, start, tokens.to_vec(), false)?
        };
        Ok(sample.map(|_| token))
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        for r in rows {
            ensure!(r.slot < self.slots, "slot {} out of {}", r.slot, self.slots);
            let covered = self.tables[r.slot].len() * PAGE;
            ensure!(r.pos + 1 <= covered, "slot {} has {} blocks, not enough for position {}", r.slot, self.tables[r.slot].len(), r.pos + 1);
        }
        self.connect()?;
        let chain = self.chain.as_mut().expect("connected");
        let slots: Vec<u32> = rows.iter().map(|r| r.slot as u32).collect();
        let positions: Vec<u32> = rows.iter().map(|r| r.pos as u32).collect();
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        match chain.forward_batch(&slots, &positions, &tokens) {
            Ok((out, times)) => {
                self.record(&times, rows.len());
                Ok(out)
            }
            Err(e) => {
                self.drop_chain(format!("{e:#}"));
                Err(e.context("a stage was lost, the request is failed"))
            }
        }
    }

    fn chain_metrics(&self) -> Option<ChainMetrics> {
        Some(self.metrics.clone())
    }

    fn refresh(&mut self) -> Option<String> {
        self.coordinator.poll();
        if let Some(reason) = self.lost.take() {
            return Some(reason);
        }
        if self.chain.is_some() && self.coordinator.generation() != self.generation {
            self.chain = None;
            return Some("the stage placement changed, the chain is rebuilt".into());
        }
        None
    }
}
