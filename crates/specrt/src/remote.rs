use crate::codegen::PAGE;
use crate::coordinator::Coordinator;
use crate::runtime::{DecodeRow, Sampling};
use crate::sched::{Executor, blocks_for};
use crate::wire::Chain;
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
}

impl ChainExecutor {
    pub fn new(coordinator: Coordinator, max_ctx: usize, slots: usize, chunk: usize) -> Self {
        ChainExecutor { coordinator, chain: None, generation: 0, max_ctx, slots, chunk, tables: vec![Vec::new(); slots], lost: None }
    }

    pub fn coordinator(&self) -> &Coordinator {
        &self.coordinator
    }

    pub fn connected(&self) -> bool {
        self.chain.is_some()
    }

    pub fn connect(&mut self) -> Result<()> {
        if self.chain.is_some() {
            return Ok(());
        }
        let chain = self.coordinator.connect_chain()?;
        self.generation = self.coordinator.generation();
        self.chain = Some(chain);
        Ok(())
    }

    fn drop_chain(&mut self, reason: String) {
        self.chain = None;
        self.lost = Some(reason);
    }

    fn forward(&mut self, slot: usize, position: usize, tokens: Vec<u32>, decode: bool) -> Result<u32> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let covered = self.tables[slot].len() * PAGE;
        ensure!(position + tokens.len() <= covered, "slot {slot} has {} blocks, not enough for position {}", self.tables[slot].len(), position + tokens.len());
        self.connect()?;
        let chain = self.chain.as_mut().expect("connected");
        match chain.forward(slot as u32, position as u32, tokens, decode) {
            Ok((token, _)) => Ok(token),
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
        let token = self.forward(slot, start, tokens.to_vec(), false)?;
        Ok(sample.map(|_| token))
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        rows.iter().map(|r| self.forward(r.slot, r.pos, vec![r.token], true)).collect()
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
