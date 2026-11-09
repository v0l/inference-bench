use crate::codegen::PAGE;
use crate::sched::BlockPool;
use crate::spec::{PrefixCacheOptions, Spec};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

pub type Hash = [u8; 32];

pub trait SnapshotData: Send + Sync {
    fn floats(&self) -> &[f32];
    fn len(&self) -> usize;
}

#[derive(Clone)]
enum Store {
    Host(Vec<f32>),
    Shared(Arc<dyn SnapshotData>),
}

#[derive(Clone)]
pub struct StateSnapshot {
    store: Store,
    conv_len: usize,
    ssm_len: usize,
    total: usize,
}

impl StateSnapshot {
    pub fn new(conv: Vec<f32>, ssm: Vec<f32>, pending: Vec<f32>) -> Self {
        let (conv_len, ssm_len) = (conv.len(), ssm.len());
        let mut all = conv;
        all.extend(ssm);
        all.extend(pending);
        let total = all.len();
        StateSnapshot { store: Store::Host(all), conv_len, ssm_len, total }
    }

    pub fn shared(data: Arc<dyn SnapshotData>, conv_len: usize, ssm_len: usize) -> Self {
        let total = data.len();
        StateSnapshot { store: Store::Shared(data), conv_len, ssm_len, total }
    }

    fn floats(&self) -> &[f32] {
        match &self.store {
            Store::Host(v) => v,
            Store::Shared(d) => d.floats(),
        }
    }

    pub fn conv(&self) -> &[f32] {
        &self.floats()[..self.conv_len]
    }

    pub fn ssm(&self) -> &[f32] {
        &self.floats()[self.conv_len..self.conv_len + self.ssm_len]
    }

    pub fn pending(&self) -> &[f32] {
        &self.floats()[self.conv_len + self.ssm_len..]
    }

    pub fn bytes(&self) -> usize {
        self.total * 4
    }
}

impl std::fmt::Debug for StateSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StateSnapshot {{ conv: {}, ssm: {}, pending: {} }}", self.conv_len, self.ssm_len, self.total - self.conv_len - self.ssm_len)
    }
}

impl PartialEq for StateSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.conv_len == other.conv_len && self.ssm_len == other.ssm_len && self.floats() == other.floats()
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

pub fn bare_namespace(salt: &str) -> Hash {
    let mut h = Sha256::new();
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

pub fn tail_key(parent: &Hash, tokens: &[u32]) -> Hash {
    let mut h = Sha256::new();
    h.update(parent);
    h.update(b"tail");
    for t in tokens {
        h.update(t.to_le_bytes());
    }
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

#[derive(Debug)]
struct Tail {
    parent: Hash,
    tokens: Vec<u32>,
    block: u32,
    tick: u64,
    state: Option<Arc<StateSnapshot>>,
    state_tick: u64,
}

#[derive(Debug, Default)]
pub struct Hit {
    pub tokens: usize,
    pub blocks: Vec<u32>,
    pub tail: Option<u32>,
    pub snapshot: Option<Arc<StateSnapshot>>,
}

#[derive(Debug)]
pub struct PrefixCache {
    hybrid: bool,
    max_blocks: usize,
    max_snapshots: usize,
    entries: HashMap<Hash, Entry>,
    states: HashMap<Hash, Stored>,
    tails: HashMap<Hash, Tail>,
    tails_by_parent: HashMap<Hash, Vec<Hash>>,
    tick: u64,
}

impl PrefixCache {
    pub fn new(opts: &PrefixCacheOptions, hybrid: bool) -> Self {
        PrefixCache {
            hybrid,
            max_blocks: opts.max_blocks,
            max_snapshots: opts.max_snapshots,
            entries: HashMap::new(),
            states: HashMap::new(),
            tails: HashMap::new(),
            tails_by_parent: HashMap::new(),
            tick: 0,
        }
    }

    pub fn blocks(&self) -> usize {
        self.entries.len() + self.tails.len()
    }

    pub fn tails(&self) -> usize {
        self.tails.len()
    }

    pub fn snapshots(&self) -> usize {
        self.states.len() + self.tails.values().filter(|t| t.state.is_some()).count()
    }

    pub fn snapshot_bytes(&self) -> usize {
        self.states.values().map(|s| s.state.bytes()).sum::<usize>()
            + self.tails.values().filter_map(|t| t.state.as_ref()).map(|s| s.bytes()).sum::<usize>()
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
            if self.max_blocks > 0 && self.blocks() >= self.max_blocks && !self.evict_one(pool) {
                break;
            }
            pool.retain(blocks[i]);
            self.entries.insert(hashes[i], Entry { block: blocks[i], tick });
        }
    }

    fn make_room_for_state(&mut self, pool: &mut BlockPool) {
        while self.snapshots() >= self.max_snapshots {
            let oldest_state = self.states.iter().min_by_key(|(_, s)| s.tick).map(|(h, s)| (*h, s.tick));
            let oldest_tail = self.tails.iter().filter(|(_, t)| t.state.is_some()).min_by_key(|(_, t)| t.state_tick).map(|(k, t)| (*k, t.state_tick));
            match (oldest_state, oldest_tail) {
                (Some((h, st)), Some((_, tt))) if st <= tt => {
                    self.states.remove(&h);
                }
                (Some((h, _)), None) => {
                    self.states.remove(&h);
                }
                (_, Some((k, _))) => self.remove_tail(&k, pool),
                (None, None) => break,
            }
        }
    }

    pub fn store_snapshot(&mut self, hash: Hash, state: Arc<StateSnapshot>, pool: &mut BlockPool) -> bool {
        if !self.hybrid || self.max_snapshots == 0 || !self.entries.contains_key(&hash) {
            return false;
        }
        let tick = self.next_ticks(0);
        if !self.states.contains_key(&hash) {
            self.make_room_for_state(pool);
        }
        self.states.insert(hash, Stored { state, tick });
        true
    }

    pub fn register_tail(&mut self, parent: Hash, parent_is_root: bool, tokens: &[u32], block: u32, state: Option<Arc<StateSnapshot>>, pool: &mut BlockPool) -> bool {
        if tokens.is_empty() || tokens.len() >= PAGE || (self.hybrid && (state.is_none() || self.max_snapshots == 0)) {
            return false;
        }
        let parent_tick = match self.entries.get(&parent) {
            Some(e) => e.tick,
            None if parent_is_root => self.next_ticks(1),
            None => return false,
        };
        let key = tail_key(&parent, tokens);
        let tick = parent_tick.saturating_sub(1);
        let state_tick = self.next_ticks(0);
        if let Some(t) = self.tails.get_mut(&key) {
            t.tick = tick;
            if t.state.is_none() || state.is_some() {
                t.state = state;
                t.state_tick = state_tick;
            }
            return true;
        }
        if self.hybrid {
            self.make_room_for_state(pool);
        }
        if self.max_blocks > 0 && self.blocks() >= self.max_blocks && !self.evict_one(pool) {
            return false;
        }
        pool.retain(block);
        self.tails_by_parent.entry(parent).or_default().push(key);
        self.tails.insert(key, Tail { parent, tokens: tokens.to_vec(), block, tick, state, state_tick });
        true
    }

    fn remove_tail(&mut self, key: &Hash, pool: &mut BlockPool) {
        let Some(t) = self.tails.remove(key) else { return };
        if let Some(list) = self.tails_by_parent.get_mut(&t.parent) {
            list.retain(|k| k != key);
            if list.is_empty() {
                self.tails_by_parent.remove(&t.parent);
            }
        }
        pool.release(&mut vec![t.block]);
    }

    pub fn has_block(&self, hash: &Hash) -> bool {
        self.entries.contains_key(hash)
    }

    pub fn has_snapshot(&self, hash: &Hash) -> bool {
        self.states.contains_key(hash)
    }

    pub fn has_tail(&self, parent: &Hash, tokens: &[u32]) -> bool {
        self.tails.contains_key(&tail_key(parent, tokens))
    }

    pub fn lookup(&mut self, hashes: &[Hash], max_tokens: usize, pool: &mut BlockPool) -> Hit {
        self.find(hashes, None, max_tokens, pool)
    }

    pub fn lookup_with_tail(&mut self, root: &Hash, tokens: &[u32], hashes: &[Hash], max_tokens: usize, pool: &mut BlockPool) -> Hit {
        self.find(hashes, Some((root, tokens)), max_tokens, pool)
    }

    fn best_tail(&self, root: &Hash, tokens: &[u32], hashes: &[Hash], run: usize, max_tokens: usize) -> Option<(usize, Hash)> {
        for t in (0..=run).rev() {
            let parent = if t == 0 { root } else { &hashes[t - 1] };
            let Some(keys) = self.tails_by_parent.get(parent) else { continue };
            let rest = &tokens[(t * PAGE).min(tokens.len())..];
            let best = keys
                .iter()
                .filter_map(|k| self.tails.get(k).map(|tail| (k, tail)))
                .filter(|(_, tail)| rest.starts_with(&tail.tokens) && t * PAGE + tail.tokens.len() <= max_tokens)
                .filter(|(_, tail)| !self.hybrid || tail.state.is_some())
                .max_by_key(|(_, tail)| tail.tokens.len());
            if let Some((k, tail)) = best {
                return Some((t * PAGE + tail.tokens.len(), *k));
            }
        }
        None
    }

    fn find(&mut self, hashes: &[Hash], tail_ctx: Option<(&Hash, &[u32])>, max_tokens: usize, pool: &mut BlockPool) -> Hit {
        let limit = hashes.len().min(max_tokens / PAGE);
        let run = hashes[..limit].iter().take_while(|h| self.entries.contains_key(*h)).count();
        let depth = if self.hybrid { (0..run).rev().find(|&i| self.states.contains_key(&hashes[i])).map_or(0, |i| i + 1) } else { run };
        let tail = tail_ctx.and_then(|(root, tokens)| self.best_tail(root, tokens, hashes, run, max_tokens));
        if let Some((tail_tokens, key)) = tail.filter(|(n, _)| *n > depth * PAGE) {
            let t = tail_tokens / PAGE;
            let base = self.next_ticks(t);
            let mut blocks = Vec::with_capacity(t);
            for (i, h) in hashes[..t].iter().enumerate() {
                let e = self.entries.get_mut(h).expect("run is cached");
                e.tick = base + (t - i) as u64;
                pool.retain(e.block);
                blocks.push(e.block);
            }
            let state_tick = self.next_ticks(0);
            let tail = self.tails.get_mut(&key).expect("tail exists");
            tail.tick = base;
            tail.state_tick = state_tick;
            pool.retain(tail.block);
            return Hit { tokens: tail_tokens, blocks, tail: Some(tail.block), snapshot: tail.state.clone() };
        }
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
        Hit { tokens: depth * PAGE, blocks, tail: None, snapshot }
    }

    pub fn evict_one(&mut self, pool: &mut BlockPool) -> bool {
        let entry = self.entries.iter().filter(|(_, e)| pool.refs(e.block) == 1).min_by_key(|(_, e)| e.tick).map(|(h, e)| (*h, e.tick));
        let tail = self.tails.iter().filter(|(_, t)| pool.refs(t.block) == 1).min_by_key(|(_, t)| t.tick).map(|(k, t)| (*k, t.tick));
        match (entry, tail) {
            (_, Some((k, tt))) if entry.is_none_or(|(_, et)| tt <= et) => {
                self.remove_tail(&k, pool);
                true
            }
            (Some((h, _)), _) => {
                let e = self.entries.remove(&h).expect("victim exists");
                self.states.remove(&h);
                let orphans: Vec<Hash> = self.tails_by_parent.get(&h).cloned().unwrap_or_default();
                for k in orphans {
                    self.remove_tail(&k, pool);
                }
                pool.release(&mut vec![e.block]);
                true
            }
            _ => false,
        }
    }
}
