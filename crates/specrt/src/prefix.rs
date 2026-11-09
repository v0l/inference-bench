use crate::codegen::PAGE;
use crate::sched::BlockPool;
use crate::spec::{PrefixCacheOptions, Spec};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

pub type Hash = [u8; 32];

#[derive(Clone, Debug, PartialEq)]
pub struct StateSnapshot {
    pub conv: Vec<f32>,
    pub ssm: Vec<f32>,
}

impl StateSnapshot {
    pub fn bytes(&self) -> usize {
        (self.conv.len() + self.ssm.len()) * 4
    }
}

pub fn namespace(spec: &Spec, kv_planes: usize, salt: &str) -> Hash {
    let mut keyed = spec.clone();
    keyed.runtime = Default::default();
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(&keyed).expect("a spec serialises"));
    h.update((kv_planes as u64).to_le_bytes());
    h.update((PAGE as u64).to_le_bytes());
    h.update((salt.len() as u64).to_le_bytes());
    h.update(salt.as_bytes());
    h.finalize().into()
}

pub fn block_hashes(root: &Hash, tokens: &[u32]) -> Vec<Hash> {
    let mut parent = *root;
    tokens
        .chunks_exact(PAGE)
        .map(|block| {
            let mut h = Sha256::new();
            h.update(parent);
            for t in block {
                h.update(t.to_le_bytes());
            }
            parent = h.finalize().into();
            parent
        })
        .collect()
}

pub fn bare_namespace(salt: &str) -> Hash {
    let mut h = Sha256::new();
    h.update((salt.len() as u64).to_le_bytes());
    h.update(salt.as_bytes());
    h.finalize().into()
}

pub fn snapshot_points(start: usize, prompt_len: usize, interval: usize) -> Vec<usize> {
    let last = prompt_len / PAGE * PAGE;
    let mut points: Vec<usize> = if interval == 0 { Vec::new() } else { (1..).map(|i| i * interval).take_while(|&p| p <= last).filter(|&p| p > start).collect() };
    if last > start && points.last() != Some(&last) {
        points.push(last);
    }
    points
}

#[derive(Debug)]
struct Entry {
    block: u32,
    tick: u64,
}

#[derive(Debug)]
struct Stored {
    state: Arc<StateSnapshot>,
    tick: u64,
}

#[derive(Debug, Default)]
pub struct Hit {
    pub tokens: usize,
    pub blocks: Vec<u32>,
    pub snapshot: Option<Arc<StateSnapshot>>,
}

#[derive(Debug)]
pub struct PrefixCache {
    hybrid: bool,
    max_blocks: usize,
    max_snapshots: usize,
    entries: HashMap<Hash, Entry>,
    states: HashMap<Hash, Stored>,
    tick: u64,
}

impl PrefixCache {
    pub fn new(opts: &PrefixCacheOptions, hybrid: bool) -> Self {
        PrefixCache { hybrid, max_blocks: opts.max_blocks, max_snapshots: opts.max_snapshots, entries: HashMap::new(), states: HashMap::new(), tick: 0 }
    }

    pub fn blocks(&self) -> usize {
        self.entries.len()
    }

    pub fn snapshots(&self) -> usize {
        self.states.len()
    }

    pub fn snapshot_bytes(&self) -> usize {
        self.states.values().map(|s| s.state.bytes()).sum()
    }

    fn next_ticks(&mut self, n: usize) -> u64 {
        let base = self.tick;
        self.tick += n as u64 + 1;
        base
    }

    pub fn register(&mut self, hashes: &[Hash], blocks: &[u32], pool: &mut BlockPool) {
        let n = hashes.len().min(blocks.len());
        let base = self.next_ticks(n);
        for i in 0..n {
            let tick = base + (n - i) as u64;
            if let Some(e) = self.entries.get_mut(&hashes[i]) {
                e.tick = tick;
                continue;
            }
            if self.max_blocks > 0 && self.entries.len() >= self.max_blocks && !self.evict_one(pool) {
                break;
            }
            pool.retain(blocks[i]);
            self.entries.insert(hashes[i], Entry { block: blocks[i], tick });
        }
    }

    pub fn store_snapshot(&mut self, hash: Hash, state: Arc<StateSnapshot>) -> bool {
        if !self.hybrid || self.max_snapshots == 0 || !self.entries.contains_key(&hash) {
            return false;
        }
        let tick = self.next_ticks(0);
        if !self.states.contains_key(&hash) && self.states.len() >= self.max_snapshots {
            let oldest = self.states.iter().min_by_key(|(_, s)| s.tick).map(|(h, _)| *h);
            if let Some(h) = oldest {
                self.states.remove(&h);
            }
        }
        self.states.insert(hash, Stored { state, tick });
        true
    }

    pub fn has_block(&self, hash: &Hash) -> bool {
        self.entries.contains_key(hash)
    }

    pub fn has_snapshot(&self, hash: &Hash) -> bool {
        self.states.contains_key(hash)
    }

    pub fn lookup(&mut self, hashes: &[Hash], max_tokens: usize, pool: &mut BlockPool) -> Hit {
        let limit = hashes.len().min(max_tokens / PAGE);
        let run = hashes[..limit].iter().take_while(|h| self.entries.contains_key(*h)).count();
        let depth = if self.hybrid { (0..run).rev().find(|&i| self.states.contains_key(&hashes[i])).map_or(0, |i| i + 1) } else { run };
        if depth == 0 {
            return Hit::default();
        }
        let base = self.next_ticks(depth);
        let mut blocks = Vec::with_capacity(depth);
        for (i, h) in hashes[..depth].iter().enumerate() {
            let e = self.entries.get_mut(h).expect("run is cached");
            e.tick = base + (depth - i) as u64;
            pool.retain(e.block);
            blocks.push(e.block);
        }
        let snapshot = if self.hybrid {
            let s = self.states.get_mut(&hashes[depth - 1]).expect("depth has a state");
            s.tick = base + depth as u64 + 1;
            Some(s.state.clone())
        } else {
            None
        };
        Hit { tokens: depth * PAGE, blocks, snapshot }
    }

    pub fn evict_one(&mut self, pool: &mut BlockPool) -> bool {
        let victim = self.entries.iter().filter(|(_, e)| pool.refs(e.block) == 1).min_by_key(|(_, e)| e.tick).map(|(h, _)| *h);
        let Some(h) = victim else { return false };
        let e = self.entries.remove(&h).expect("victim exists");
        self.states.remove(&h);
        pool.release(&mut vec![e.block]);
        true
    }
}
