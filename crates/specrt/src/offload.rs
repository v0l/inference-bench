use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    Lru,
    Frequency,
    #[default]
    Score,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OffloadSettings {
    pub vram_budget: u64,
    pub host_budget: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvme_path: Option<String>,
    pub prefetch_depth: usize,
    pub policy: Policy,
    pub expert_slots: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub pointer_table: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hotlist: Option<String>,
    pub promote_margin: f64,
    pub score_half_life: u32,
    pub promote_per_step: usize,
}

impl Eq for OffloadSettings {}

impl OffloadSettings {
    pub fn is_default(&self) -> bool {
        *self == OffloadSettings::default()
    }

    pub fn enabled(&self) -> bool {
        self.vram_budget > 0
    }
}

fn default_prefetch_depth() -> usize {
    1
}

impl Default for OffloadSettings {
    fn default() -> Self {
        OffloadSettings {
            vram_budget: 0,
            host_budget: 0,
            nvme_path: None,
            prefetch_depth: default_prefetch_depth(),
            policy: Policy::Score,
            expert_slots: 0,
            pointer_table: false,
            hotlist: None,
            promote_margin: 1.15,
            score_half_life: 64,
            promote_per_step: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tier {
    Vram,
    Host,
    Nvme,
}

pub trait PageSource: Send + Sync {
    fn read(&self, page: usize, out: &mut Vec<u8>) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OffloadStats {
    pub vram_hits: u64,
    pub host_hits: u64,
    pub nvme_reads: u64,
    pub nvme_read_bytes: u64,
    #[serde(default)]
    pub nvme_read_failures: u64,
    #[serde(default)]
    pub nvme_wait_nanos: u64,
    pub host_to_vram_bytes: u64,
    pub evictions: u64,
    #[serde(default)]
    pub promotions: u64,
}

impl OffloadStats {
    pub fn plus(&self, o: &OffloadStats) -> OffloadStats {
        OffloadStats {
            vram_hits: self.vram_hits + o.vram_hits,
            host_hits: self.host_hits + o.host_hits,
            nvme_reads: self.nvme_reads + o.nvme_reads,
            nvme_read_bytes: self.nvme_read_bytes + o.nvme_read_bytes,
            nvme_read_failures: self.nvme_read_failures + o.nvme_read_failures,
            nvme_wait_nanos: self.nvme_wait_nanos + o.nvme_wait_nanos,
            host_to_vram_bytes: self.host_to_vram_bytes + o.host_to_vram_bytes,
            evictions: self.evictions + o.evictions,
            promotions: self.promotions + o.promotions,
        }
    }

    pub fn absorb_io(&mut self, other: &OffloadStats) {
        self.host_to_vram_bytes += other.host_to_vram_bytes;
        self.nvme_reads += other.nvme_reads;
        self.nvme_read_bytes += other.nvme_read_bytes;
        self.nvme_read_failures += other.nvme_read_failures;
        self.nvme_wait_nanos += other.nvme_wait_nanos;
    }

    pub fn hit_ratio(&self) -> f64 {
        let total = self.vram_hits + self.host_hits + self.nvme_reads;
        if total == 0 { 1.0 } else { self.vram_hits as f64 / total as f64 }
    }

    pub fn prometheus(&self) -> String {
        format!(
            "inference_offload_hit_ratio {}\ninference_offload_read_bytes_total {}\ninference_offload_upload_bytes_total {}\ninference_offload_evictions_total {}\n",
            self.hit_ratio(),
            self.nvme_read_bytes,
            self.host_to_vram_bytes,
            self.evictions
        )
    }
}

struct Entry {
    data: Arc<Vec<u8>>,
    last_use: u64,
    uses: u64,
}

struct Level {
    budget: u64,
    used: u64,
    entries: HashMap<usize, Entry>,
}

impl Level {
    fn new(budget: u64) -> Self {
        Level { budget, used: 0, entries: HashMap::new() }
    }

    fn victim(&self, policy: Policy, pinned: &[bool]) -> Option<usize> {
        self.entries
            .iter()
            .filter(|(id, _)| !pinned[**id])
            .min_by_key(|(id, e)| match policy {
                Policy::Lru => (e.last_use, 0, **id),
                Policy::Frequency | Policy::Score => (e.uses, e.last_use, **id),
            })
            .map(|(id, _)| *id)
    }
}

struct Inner {
    vram: Level,
    host: Level,
    clock: u64,
    stats: OffloadStats,
    prefetch_log: Vec<usize>,
}

pub struct TieredStore<S: PageSource> {
    source: S,
    sizes: Vec<u64>,
    pinned: Vec<bool>,
    policy: Policy,
    inner: Mutex<Inner>,
}

impl<S: PageSource> TieredStore<S> {
    pub fn new(source: S, sizes: Vec<u64>, settings: &OffloadSettings) -> Self {
        let pinned = vec![false; sizes.len()];
        TieredStore {
            source,
            sizes,
            pinned,
            policy: settings.policy,
            inner: Mutex::new(Inner { vram: Level::new(settings.vram_budget), host: Level::new(settings.host_budget), clock: 0, stats: OffloadStats::default(), prefetch_log: vec![] }),
        }
    }

    pub fn pin(&mut self, pages: &[usize]) {
        for p in pages {
            self.pinned[*p] = true;
        }
    }

    pub fn stats(&self) -> OffloadStats {
        self.inner.lock().unwrap().stats
    }

    pub fn prefetch_order(&self) -> Vec<usize> {
        self.inner.lock().unwrap().prefetch_log.clone()
    }

    pub fn tier_of(&self, page: usize) -> Tier {
        let g = self.inner.lock().unwrap();
        if g.vram.entries.contains_key(&page) {
            Tier::Vram
        } else if g.host.entries.contains_key(&page) {
            Tier::Host
        } else {
            Tier::Nvme
        }
    }

    pub fn resident_bytes(&self) -> (u64, u64) {
        let g = self.inner.lock().unwrap();
        (g.vram.used, g.host.used)
    }

    fn admit(level: &mut Level, id: usize, size: u64, data: Arc<Vec<u8>>, clock: u64, uses: u64, policy: Policy, pinned: &[bool], evicted: &mut Vec<(usize, Entry)>) -> bool {
        if size > level.budget {
            return false;
        }
        while level.used + size > level.budget {
            let Some(v) = level.victim(policy, pinned) else { return false };
            let e = level.entries.remove(&v).unwrap();
            level.used -= e.data.len() as u64;
            evicted.push((v, e));
        }
        level.used += size;
        level.entries.insert(id, Entry { data, last_use: clock, uses });
        true
    }

    pub fn get(&self, page: usize) -> Result<Arc<Vec<u8>>> {
        ensure!(page < self.sizes.len(), "page {page} out of range");
        self.fetch(page, false)
    }

    pub fn prefetch(&self, pages: &[usize]) -> Result<()> {
        for p in pages {
            ensure!(*p < self.sizes.len(), "page {p} out of range");
            self.inner.lock().unwrap().prefetch_log.push(*p);
            self.fetch(*p, true)?;
        }
        Ok(())
    }

    fn fetch(&self, page: usize, speculative: bool) -> Result<Arc<Vec<u8>>> {
        let size = self.sizes[page];
        let mut g = self.inner.lock().unwrap();
        g.clock += 1;
        let clock = g.clock;
        if let Some(e) = g.vram.entries.get_mut(&page) {
            if !speculative {
                e.last_use = clock;
                e.uses += 1;
                g.stats.vram_hits += 1;
            }
            return Ok(g.vram.entries[&page].data.clone());
        }
        let (data, uses) = if let Some(e) = g.host.entries.get(&page) {
            let r = (e.data.clone(), e.uses + 1);
            if !speculative {
                g.stats.host_hits += 1;
            }
            r
        } else {
            drop(g);
            let mut buf = Vec::new();
            self.source.read(page, &mut buf)?;
            if buf.len() as u64 != size {
                bail!("page {page} read {} bytes, expected {size}", buf.len());
            }
            g = self.inner.lock().unwrap();
            g.stats.nvme_reads += 1;
            g.stats.nvme_read_bytes += size;
            let data = Arc::new(buf);
            let mut evicted = vec![];
            let inner = &mut *g;
            if Self::admit(&mut inner.host, page, size, data.clone(), clock, 1, self.policy, &self.pinned, &mut evicted) {
                inner.stats.evictions += evicted.len() as u64;
            }
            (data, 1)
        };
        let mut evicted = vec![];
        let inner = &mut *g;
        if Self::admit(&mut inner.vram, page, size, data.clone(), clock, uses, self.policy, &self.pinned, &mut evicted) {
            inner.stats.host_to_vram_bytes += size;
            inner.stats.evictions += evicted.len() as u64;
        }
        Ok(data)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExpertHotlist {
    pub counts: Vec<Vec<u64>>,
}

impl ExpertHotlist {
    pub fn new(layers: usize, experts: usize) -> Self {
        ExpertHotlist { counts: vec![vec![0; experts]; layers] }
    }

    pub fn record(&mut self, layer: usize, expert: usize) {
        if let Some(c) = self.counts.get_mut(layer).and_then(|l| l.get_mut(expert)) {
            *c += 1;
        }
    }

    pub fn top(&self, layer: usize, k: usize) -> Vec<usize> {
        let Some(l) = self.counts.get(layer) else { return vec![] };
        let mut ids: Vec<usize> = (0..l.len()).filter(|&e| l[e] > 0).collect();
        ids.sort_by_key(|&e| (std::cmp::Reverse(l[e]), e));
        ids.truncate(k);
        ids
    }

    pub fn hottest(&self, k: usize) -> Vec<(usize, usize)> {
        let mut all: Vec<(u64, usize, usize)> = self.counts.iter().enumerate().flat_map(|(l, c)| c.iter().enumerate().filter(|(_, n)| **n > 0).map(move |(e, n)| (*n, l, e))).collect();
        all.sort_by_key(|&(n, l, e)| (std::cmp::Reverse(n), l, e));
        all.into_iter().take(k).map(|(_, l, e)| (l, e)).collect()
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string(self)?).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }
}

pub struct SlotBook {
    page_of: Vec<Option<usize>>,
    slot_of: HashMap<usize, usize>,
    last_use: Vec<u64>,
    uses: Vec<u64>,
    pinned: Vec<bool>,
    clock: u64,
    policy: Policy,
    pub stats: OffloadStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertRoute {
    Table,
    Staged,
}

pub fn expert_route(table_enabled: bool, every_routed_resident: bool, tokens: usize, table_max_tokens: usize) -> ExpertRoute {
    if table_enabled && every_routed_resident && tokens < table_max_tokens { ExpertRoute::Table } else { ExpertRoute::Staged }
}

pub fn pointer_table(book: &SlotBook, page_of: impl Fn(usize) -> usize, local: usize, shared: usize, shared_base: usize) -> (Vec<i32>, bool) {
    pointer_table_of(|p| book.resident(p), page_of, local, shared, shared_base)
}

pub fn pointer_table_of(resident: impl Fn(usize) -> Option<usize>, page_of: impl Fn(usize) -> usize, local: usize, shared: usize, shared_base: usize) -> (Vec<i32>, bool) {
    let mut table: Vec<i32> = (0..local).map(|e| resident(page_of(e)).map_or(0, |s| s as i32 + 1)).collect();
    let full = table.iter().all(|&s| s != 0);
    table.extend((0..shared).map(|s| (shared_base + s) as i32 + 1));
    (table, full)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acquired {
    pub slot: usize,
    pub copy: bool,
}

impl SlotBook {
    pub fn new(slots: usize, policy: Policy) -> Self {
        SlotBook { page_of: vec![None; slots], slot_of: HashMap::new(), last_use: vec![0; slots], uses: vec![0; slots], pinned: vec![false; slots], clock: 0, policy, stats: OffloadStats::default() }
    }

    pub fn slots(&self) -> usize {
        self.page_of.len()
    }

    pub fn resident(&self, page: usize) -> Option<usize> {
        self.slot_of.get(&page).copied()
    }

    pub fn evict(&mut self, page: usize) -> bool {
        let Some(slot) = self.slot_of.remove(&page) else { return false };
        self.page_of[slot] = None;
        self.stats.evictions += 1;
        true
    }

    pub fn touch(&mut self, page: usize) -> Option<usize> {
        let slot = *self.slot_of.get(&page)?;
        self.clock += 1;
        self.last_use[slot] = self.clock;
        Some(slot)
    }

    pub fn misses(&self, pages: impl Iterator<Item = usize>) -> usize {
        pages.filter(|p| !self.slot_of.contains_key(p)).count()
    }

    pub fn clear_unpinned(&mut self) {
        for s in 0..self.page_of.len() {
            if !self.pinned[s]
                && let Some(old) = self.page_of[s].take()
            {
                self.slot_of.remove(&old);
                self.stats.evictions += 1;
            }
        }
    }

    pub fn unpin_all(&mut self) {
        self.pinned.iter_mut().for_each(|p| *p = false);
    }

    pub fn acquire(&mut self, page: usize, pin: bool, speculative: bool) -> Option<Acquired> {
        self.clock += 1;
        if let Some(&slot) = self.slot_of.get(&page) {
            self.last_use[slot] = self.clock;
            if !speculative {
                self.uses[slot] += 1;
                self.stats.vram_hits += 1;
            }
            self.pinned[slot] |= pin;
            return Some(Acquired { slot, copy: false });
        }
        let slot = match self.page_of.iter().position(Option::is_none) {
            Some(free) => free,
            None => (0..self.page_of.len())
                .filter(|&s| !self.pinned[s])
                .min_by_key(|&s| match self.policy {
                    Policy::Lru | Policy::Score => (self.last_use[s], 0, s),
                    Policy::Frequency => (self.uses[s], self.last_use[s], s),
                })?,
        };
        if let Some(old) = self.page_of[slot].take() {
            self.slot_of.remove(&old);
            self.stats.evictions += 1;
        }
        self.page_of[slot] = Some(page);
        self.slot_of.insert(page, slot);
        self.last_use[slot] = self.clock;
        self.uses[slot] = u64::from(!speculative);
        self.pinned[slot] = pin;
        if !speculative {
            self.stats.host_hits += 1;
        }
        Some(Acquired { slot, copy: true })
    }
}

pub trait CopyEngine {
    type Ticket;
    fn copy(&mut self, page: usize, slot: usize) -> Result<Option<Self::Ticket>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Resident(usize),
    Landing(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub page: usize,
    pub slot: usize,
    pub copy: bool,
}

pub struct ExpertBook<T> {
    state: Vec<SlotState>,
    slot_of: Vec<Option<usize>>,
    landing: Vec<(usize, usize, T)>,
    score: Vec<f64>,
    scratch: usize,
    decay: f64,
    margin: f64,
    per_step: usize,
    pub stats: OffloadStats,
}

pub fn scratch_slots(slots: usize, top_k: usize) -> usize {
    top_k.max(slots / 16).clamp(1, slots.max(1))
}

pub fn score_pool(slots: usize, top_k: usize, cached_pages: usize, settings: &OffloadSettings) -> (usize, usize) {
    if settings.pointer_table && slots >= cached_pages { (0, 0) } else { (scratch_slots(slots, top_k), settings.promote_per_step) }
}

impl<T> ExpertBook<T> {
    pub fn new(pages: usize, slots: usize, scratch: usize, settings: &OffloadSettings) -> Self {
        let scratch = scratch.min(slots);
        ExpertBook {
            state: vec![SlotState::Free; slots - scratch],
            slot_of: vec![None; pages],
            landing: Vec::new(),
            score: vec![0.0; pages],
            scratch,
            decay: 0.5f64.powf(1.0 / f64::from(settings.score_half_life.max(1))),
            margin: settings.promote_margin.max(1.0),
            per_step: settings.promote_per_step,
            stats: OffloadStats::default(),
        }
    }

    pub fn cache_slots(&self) -> usize {
        self.state.len()
    }

    pub fn resident(&self, page: usize) -> Option<usize> {
        self.slot_of[page]
    }

    pub fn in_flight(&self) -> usize {
        self.landing.len()
    }

    pub fn score(&self, page: usize) -> f64 {
        self.score[page]
    }

    pub fn record(&mut self, page: usize, uses: f64) {
        self.score[page] += uses;
    }

    pub fn fill(&mut self, pages: &[usize]) -> Vec<(usize, usize)> {
        let mut placed = Vec::new();
        for &page in pages {
            if self.slot_of[page].is_some() {
                continue;
            }
            let Some(slot) = self.state.iter().position(|s| *s == SlotState::Free) else { break };
            self.state[slot] = SlotState::Resident(page);
            self.slot_of[page] = Some(slot);
            placed.push((page, slot));
        }
        placed
    }

    pub fn evict(&mut self, page: usize) -> bool {
        let Some(slot) = self.slot_of[page].take() else { return false };
        self.state[slot] = SlotState::Free;
        self.stats.evictions += 1;
        true
    }

    pub fn seed(&mut self, hot: &ExpertHotlist, local: usize, top_k: usize) -> Vec<(usize, usize)> {
        let steady = 1.0 / (1.0 - self.decay);
        for (layer, counts) in hot.counts.iter().enumerate() {
            let sum: u64 = counts.iter().sum();
            for (e, &c) in counts.iter().enumerate().take(local) {
                let page = layer * local + e;
                if c > 0 && page < self.score.len() {
                    let rate = (c as f64 * top_k as f64 / sum as f64).min(1.0);
                    self.score[page] = self.score[page].max(rate * steady);
                }
            }
        }
        let mut order: Vec<usize> = (0..self.score.len()).filter(|&p| self.score[p] > 0.0 && self.slot_of[p].is_none()).collect();
        order.sort_by(|&a, &b| self.score[b].total_cmp(&self.score[a]).then(a.cmp(&b)));
        let mut placed = Vec::new();
        for page in order {
            let Some(slot) = self.state.iter().position(|s| *s == SlotState::Free) else { break };
            self.state[slot] = SlotState::Resident(page);
            self.slot_of[page] = Some(slot);
            placed.push((page, slot));
        }
        placed.sort_unstable();
        placed
    }

    pub fn plan(&mut self, pages: &[usize], borrow: bool) -> Result<Vec<Vec<Placed>>> {
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for &page in pages {
            match self.slot_of[page] {
                Some(slot) => hits.push(Placed { page, slot, copy: false }),
                None => misses.push(page),
            }
        }
        self.stats.vram_hits += hits.len() as u64;
        self.stats.host_hits += misses.len() as u64;
        let base = self.state.len();
        let mut pool: Vec<usize> = (base..base + self.scratch).collect();
        if (borrow || pool.is_empty()) && misses.len() > pool.len() {
            let mut lenders: Vec<usize> = (0..base)
                .filter(|&s| match self.state[s] {
                    SlotState::Free => true,
                    SlotState::Resident(p) => !pages.contains(&p),
                    SlotState::Landing(_) => false,
                })
                .collect();
            lenders.sort_by(|&a, &b| self.slot_score(a).total_cmp(&self.slot_score(b)).then(a.cmp(&b)));
            for s in lenders.into_iter().take(misses.len() - pool.len()) {
                if let SlotState::Resident(p) = self.state[s] {
                    self.slot_of[p] = None;
                    self.stats.evictions += 1;
                }
                self.state[s] = SlotState::Free;
                pool.push(s);
            }
        }
        let mut groups = vec![hits];
        if misses.is_empty() {
            return Ok(groups);
        }
        ensure!(!pool.is_empty(), "expert cache has no slot free for {} missing experts", misses.len());
        for (i, chunk) in misses.chunks(pool.len()).enumerate() {
            let placed: Vec<Placed> = chunk.iter().zip(&pool).map(|(&page, &slot)| Placed { page, slot, copy: true }).collect();
            for p in placed.iter().filter(|p| p.slot < base) {
                if let SlotState::Resident(prev) = self.state[p.slot] {
                    self.slot_of[prev] = None;
                }
                self.state[p.slot] = SlotState::Resident(p.page);
                self.slot_of[p.page] = Some(p.slot);
            }
            if i == 0 {
                groups[0].extend(placed);
            } else {
                groups.push(placed);
            }
        }
        Ok(groups)
    }

    fn slot_score(&self, slot: usize) -> f64 {
        match self.state[slot] {
            SlotState::Resident(p) => self.score[p],
            _ => f64::NEG_INFINITY,
        }
    }

    pub fn poll(&mut self, done: impl Fn(&T) -> bool) -> usize {
        let mut landed = 0;
        let mut i = 0;
        while i < self.landing.len() {
            if done(&self.landing[i].2) {
                let (page, slot, _) = self.landing.swap_remove(i);
                if let Some(other) = self.slot_of[page] {
                    self.state[other] = SlotState::Free;
                }
                self.state[slot] = SlotState::Resident(page);
                self.slot_of[page] = Some(slot);
                landed += 1;
            } else {
                i += 1;
            }
        }
        landed
    }

    pub fn end_step<E: CopyEngine<Ticket = T>>(&mut self, engine: &mut E) -> Result<usize> {
        let landing_page = |book: &Self, p: usize| book.landing.iter().any(|(q, _, _)| *q == p);
        let mut wanted: Vec<usize> = (0..self.score.len()).filter(|&p| self.score[p] > 0.0 && self.slot_of[p].is_none() && !landing_page(self, p)).collect();
        let by_score = |book: &Self, a: &usize, b: &usize| book.score[*b].total_cmp(&book.score[*a]).then(a.cmp(b));
        if wanted.len() > self.per_step {
            wanted.select_nth_unstable_by(self.per_step, |a, b| by_score(self, a, b));
            wanted.truncate(self.per_step);
        }
        wanted.sort_by(|a, b| by_score(self, a, b));
        let mut victims: Vec<usize> = (0..self.state.len()).filter(|&s| !matches!(self.state[s], SlotState::Landing(_))).collect();
        victims.sort_by(|&a, &b| self.slot_score(a).total_cmp(&self.slot_score(b)).then(a.cmp(&b)));
        let mut promoted = 0;
        for (page, slot) in wanted.into_iter().zip(victims) {
            let old = match self.state[slot] {
                SlotState::Resident(p) => Some(p),
                _ => None,
            };
            if let Some(p) = old
                && self.score[page] <= self.margin * self.score[p]
            {
                break;
            }
            let Some(ticket) = engine.copy(page, slot)? else { continue };
            if let Some(p) = old {
                self.slot_of[p] = None;
                self.stats.evictions += 1;
            }
            self.state[slot] = SlotState::Landing(page);
            self.landing.push((page, slot, ticket));
            self.stats.promotions += 1;
            promoted += 1;
        }
        for s in &mut self.score {
            *s *= self.decay;
        }
        Ok(promoted)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerResidency {
    pub layer: usize,
    pub bytes: u64,
    pub tier: Tier,
    pub pinned: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OffloadPlan {
    pub layers: Vec<LayerResidency>,
    pub staging_bytes: u64,
    pub vram_bytes: u64,
    pub host_bytes: u64,
    pub nvme_bytes: u64,
    #[serde(default)]
    pub host_staging_bytes: u64,
    #[serde(default)]
    pub swap_staging_bytes: u64,
}

impl OffloadPlan {
    pub fn check_swap_fits(&self, host_budget: u64, in_use: u64, bytes: u64) -> Result<()> {
        if host_budget == 0 || host_budget == u64::MAX {
            return Ok(());
        }
        let pinned = self.host_bytes + self.host_staging_bytes.saturating_sub(self.swap_staging_bytes) + in_use + bytes;
        ensure!(pinned <= host_budget, "parking {bytes} bytes would pin {pinned} bytes, over [runtime.offload] host_budget {host_budget}");
        Ok(())
    }
}

pub fn swap_staging_for(sequence_kv_bytes: u64) -> u64 {
    sequence_kv_bytes.div_ceil(4096) * 4096 + 4096
}

pub fn plan_layers(layer_bytes: &[u64], always_resident: u64, settings: &OffloadSettings, swap_staging: u64) -> Result<OffloadPlan> {
    let mut plan = plan_layers_inner(layer_bytes, always_resident, settings, swap_staging)?;
    plan.swap_staging_bytes = swap_staging;
    Ok(plan)
}

fn plan_layers_inner(layer_bytes: &[u64], always_resident: u64, settings: &OffloadSettings, swap_staging: u64) -> Result<OffloadPlan> {
    let plan = assign_layers(layer_bytes, always_resident, settings, swap_staging)?;
    if plan.nvme_bytes == 0 {
        return Ok(plan);
    }
    let slot = layer_bytes.iter().copied().max().unwrap_or(0).div_ceil(4096) * 4096;
    let staging = slot * (settings.prefetch_depth.max(1) as u64 + 1);
    assign_layers(layer_bytes, always_resident, settings, staging + swap_staging)
}

fn assign_layers(layer_bytes: &[u64], always_resident: u64, settings: &OffloadSettings, host_staging: u64) -> Result<OffloadPlan> {
    let n = layer_bytes.len();
    let staging = layer_bytes.iter().copied().max().unwrap_or(0) * (settings.prefetch_depth.max(1) as u64 + 1);
    let fixed = always_resident + staging;
    ensure!(settings.vram_budget >= fixed, "vram_budget {} cannot hold resident tensors and {} staging bytes", settings.vram_budget, fixed);
    let mut vram_left = settings.vram_budget - fixed;
    let mut host_left = settings.host_budget.saturating_sub(host_staging);
    let mut order: Vec<usize> = (0..n).collect();
    if n > 1 {
        order = vec![0, n - 1].into_iter().chain(1..n - 1).collect();
    }
    let mut tiers = vec![Tier::Nvme; n];
    for &l in &order {
        let b = layer_bytes[l];
        if b <= vram_left {
            vram_left -= b;
            tiers[l] = Tier::Vram;
        } else if b <= host_left {
            host_left -= b;
            tiers[l] = Tier::Host;
        }
    }
    let layers: Vec<LayerResidency> = (0..n).map(|l| LayerResidency { layer: l, bytes: layer_bytes[l], tier: tiers[l], pinned: tiers[l] == Tier::Vram }).collect();
    let sum = |t: Tier| layers.iter().filter(|l| l.tier == t).map(|l| l.bytes).sum::<u64>();
    Ok(OffloadPlan { vram_bytes: sum(Tier::Vram) + fixed, host_bytes: sum(Tier::Host), nvme_bytes: sum(Tier::Nvme), staging_bytes: staging, host_staging_bytes: host_staging, swap_staging_bytes: 0, layers })
}

impl OffloadPlan {
    pub fn streamed_bytes_per_token(&self) -> u64 {
        self.layers.iter().filter(|l| l.tier != Tier::Vram).map(|l| l.bytes).sum()
    }

    pub fn tokens_per_second_ceiling(&self, host_gbs: f64, nvme_gbs: f64, resident_read_gbs: f64) -> f64 {
        let host: u64 = self.layers.iter().filter(|l| l.tier == Tier::Host).map(|l| l.bytes).sum();
        let nvme: u64 = self.layers.iter().filter(|l| l.tier == Tier::Nvme).map(|l| l.bytes).sum();
        let resident: u64 = self.layers.iter().filter(|l| l.tier == Tier::Vram).map(|l| l.bytes).sum();
        let stream = host as f64 / (host_gbs * 1e9) + nvme as f64 / (nvme_gbs * 1e9);
        let compute = (resident + host + nvme) as f64 / (resident_read_gbs * 1e9);
        1.0 / stream.max(compute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    struct Fake {
        sizes: Vec<u64>,
        reads: Arc<Mutex<Vec<usize>>>,
        bytes_per_sec: Option<f64>,
        slept: AtomicU64,
    }

    fn content(page: usize, len: u64) -> Vec<u8> {
        (0..len).map(|i| (page as u64 * 31 + i) as u8).collect()
    }

    impl PageSource for Fake {
        fn read(&self, page: usize, out: &mut Vec<u8>) -> Result<()> {
            self.reads.lock().unwrap().push(page);
            if let Some(bps) = self.bytes_per_sec {
                let d = Duration::from_secs_f64(self.sizes[page] as f64 / bps);
                self.slept.fetch_add(d.as_micros() as u64, Ordering::SeqCst);
                std::thread::sleep(d);
            }
            *out = content(page, self.sizes[page]);
            Ok(())
        }
    }

    fn store(sizes: Vec<u64>, vram: u64, host: u64, policy: Policy, bps: Option<f64>) -> (TieredStore<Fake>, Arc<Mutex<Vec<usize>>>) {
        let reads = Arc::new(Mutex::new(vec![]));
        let src = Fake { sizes: sizes.clone(), reads: reads.clone(), bytes_per_sec: bps, slept: AtomicU64::new(0) };
        let s = OffloadSettings { vram_budget: vram, host_budget: host, policy, ..Default::default() };
        (TieredStore::new(src, sizes, &s), reads)
    }

    #[test]
    fn residency_moves_nvme_to_vram_and_back() {
        let (s, _) = store(vec![100; 4], 200, 300, Policy::Lru, None);
        assert_eq!(s.tier_of(0), Tier::Nvme);
        s.get(0).unwrap();
        assert_eq!(s.tier_of(0), Tier::Vram);
        s.get(1).unwrap();
        s.get(2).unwrap();
        assert_eq!(s.tier_of(0), Tier::Host);
        assert_eq!(s.resident_bytes(), (200, 300));
    }

    #[test]
    fn resident_page_is_never_read_again() {
        let (s, reads) = store(vec![100; 4], 200, 400, Policy::Lru, None);
        for _ in 0..5 {
            s.get(0).unwrap();
        }
        s.get(1).unwrap();
        s.get(2).unwrap();
        s.get(0).unwrap();
        s.get(1).unwrap();
        assert_eq!(*reads.lock().unwrap(), vec![0, 1, 2]);
        let st = s.stats();
        assert_eq!(st.nvme_reads, 3);
        assert!(st.host_hits >= 2);
    }

    #[test]
    fn budgets_are_never_exceeded() {
        let (s, _) = store((1..=8).map(|i| i * 10).collect(), 90, 150, Policy::Lru, None);
        for r in 0..3 {
            for p in 0..8 {
                s.get((p + r) % 8).unwrap();
                let (v, h) = s.resident_bytes();
                assert!(v <= 90 && h <= 150);
            }
        }
        assert!(s.stats().evictions > 0);
    }

    #[test]
    fn pinned_pages_survive_eviction() {
        let (mut s, _) = store(vec![100; 4], 200, 0, Policy::Lru, None);
        s.pin(&[0]);
        s.get(0).unwrap();
        for p in 1..4 {
            s.get(p).unwrap();
        }
        assert_eq!(s.tier_of(0), Tier::Vram);
    }

    #[test]
    fn frequency_policy_keeps_the_hot_page() {
        let (s, _) = store(vec![100; 3], 200, 0, Policy::Frequency, None);
        for _ in 0..5 {
            s.get(0).unwrap();
        }
        s.get(1).unwrap();
        s.get(2).unwrap();
        assert_eq!(s.tier_of(0), Tier::Vram);
        assert_eq!(s.tier_of(1), Tier::Nvme);
    }

    #[test]
    fn lru_policy_drops_the_oldest() {
        let (s, _) = store(vec![100; 3], 200, 0, Policy::Lru, None);
        for _ in 0..5 {
            s.get(0).unwrap();
        }
        s.get(1).unwrap();
        s.get(2).unwrap();
        assert_eq!(s.tier_of(0), Tier::Nvme);
    }

    #[test]
    fn prefetch_keeps_order_and_makes_get_a_hit() {
        let (s, reads) = store(vec![100; 4], 400, 0, Policy::Lru, None);
        s.prefetch(&[2, 3, 1]).unwrap();
        assert_eq!(s.prefetch_order(), vec![2, 3, 1]);
        assert_eq!(*reads.lock().unwrap(), vec![2, 3, 1]);
        s.get(3).unwrap();
        let st = s.stats();
        assert_eq!((st.vram_hits, st.nvme_reads), (1, 3));
    }

    #[test]
    fn streamed_bytes_are_identical_to_the_source() {
        let (s, _) = store(vec![64, 128, 32], 64, 128, Policy::Lru, None);
        for _ in 0..3 {
            for p in 0..3 {
                assert_eq!(*s.get(p).unwrap(), content(p, [64, 128, 32][p]));
            }
        }
    }

    #[test]
    fn one_layer_resident_streams_through_with_stable_hit_ratio() {
        let (s, _) = store(vec![100; 6], 100, 0, Policy::Lru, None);
        for _ in 0..4 {
            for p in 0..6 {
                s.get(p).unwrap();
            }
        }
        let st = s.stats();
        assert_eq!(st.nvme_reads, 24);
        assert_eq!(st.hit_ratio(), 0.0);
    }

    #[test]
    fn throttled_nvme_read_time_follows_bandwidth() {
        let (s, _) = store(vec![1_000_000; 4], 4_000_000, 0, Policy::Lru, Some(100e6));
        let t = Instant::now();
        for p in 0..4 {
            s.get(p).unwrap();
        }
        let secs = t.elapsed().as_secs_f64();
        assert!(secs >= 0.039 && secs < 0.2, "{secs}");
        let t = Instant::now();
        for p in 0..4 {
            s.get(p).unwrap();
        }
        assert!(t.elapsed().as_secs_f64() < 0.01);
        assert_eq!(s.stats().nvme_read_bytes, 4_000_000);
    }

    #[test]
    fn wrong_sized_read_is_an_error() {
        let reads = Arc::new(Mutex::new(vec![]));
        let src = Fake { sizes: vec![10], reads, bytes_per_sec: None, slept: AtomicU64::new(0) };
        let s = TieredStore::new(src, vec![20], &OffloadSettings { vram_budget: 100, ..Default::default() });
        assert!(s.get(0).is_err());
    }

    #[test]
    fn swap_staging_is_charged_against_host_budget_and_a_park_over_it_is_refused() {
        let sizes = vec![10000u64; 10];
        let swap = swap_staging_for(5000);
        assert_eq!(swap, 12288);
        let s = OffloadSettings { vram_budget: 60000, host_budget: 12288 * 3 + 12288 + 20000, prefetch_depth: 2, ..Default::default() };
        let p = plan_layers(&sizes, 0, &s, swap).unwrap();
        assert_eq!(p.swap_staging_bytes, swap);
        assert_eq!(p.host_staging_bytes, 12288 * 3 + swap);
        assert_eq!(p.host_bytes, 20000);
        assert!(p.host_bytes + p.host_staging_bytes <= s.host_budget, "{p:?}");
        let without = plan_layers(&sizes, 0, &s, 0).unwrap();
        assert!(without.host_bytes > p.host_bytes, "swap staging takes room from host layers");
        assert!(p.check_swap_fits(s.host_budget, 0, 5000).is_ok());
        let err = p.check_swap_fits(s.host_budget, 12000, 5000).unwrap_err().to_string();
        assert!(err.contains("over [runtime.offload] host_budget"), "{err}");
        assert!(p.check_swap_fits(0, 1 << 40, 1 << 40).is_ok());
    }

    #[test]
    fn pinned_total_with_disk_staging_stays_inside_host_budget() {
        let sizes = vec![10000u64; 10];
        let staging = 12288u64 * 3;
        for host in [0u64, 30000, 90000, 200000] {
            let s = OffloadSettings { vram_budget: 60000, host_budget: host, prefetch_depth: 2, ..Default::default() };
            let p = plan_layers(&sizes, 0, &s, 0).unwrap();
            if p.nvme_bytes > 0 {
                assert_eq!(p.host_staging_bytes, staging);
                assert!(p.host_bytes + p.host_staging_bytes <= host.max(staging), "{p:?}");
            } else {
                assert_eq!(p.host_staging_bytes, 0);
            }
        }
        let tight = OffloadSettings { vram_budget: 60000, host_budget: staging + 20000, prefetch_depth: 2, ..Default::default() };
        let p = plan_layers(&sizes, 0, &tight, 0).unwrap();
        assert!(p.nvme_bytes > 0 && p.host_bytes == 20000 && p.host_bytes + p.host_staging_bytes <= tight.host_budget, "{p:?}");
    }

    #[test]
    fn planner_pins_first_and_last_then_fills_vram_then_host() {
        let layers = vec![10u64; 10];
        let s = OffloadSettings { vram_budget: 50, host_budget: 30, prefetch_depth: 1, ..Default::default() };
        let p = plan_layers(&layers, 10, &s, 0).unwrap();
        assert_eq!(p.staging_bytes, 20);
        assert_eq!(p.layers[0].tier, Tier::Vram);
        assert_eq!(p.layers[9].tier, Tier::Vram);
        let count = |t| p.layers.iter().filter(|l| l.tier == t).count();
        assert_eq!((count(Tier::Vram), count(Tier::Host), count(Tier::Nvme)), (2, 0, 8));
        assert_eq!(p.vram_bytes, 50);
        assert_eq!(p.streamed_bytes_per_token(), 80);
    }

    #[test]
    fn planner_rejects_a_budget_below_the_resident_set() {
        let s = OffloadSettings { vram_budget: 15, ..Default::default() };
        assert!(plan_layers(&[10, 10], 10, &s, 0).is_err());
    }

    #[test]
    fn ceiling_falls_as_vram_budget_shrinks() {
        let layers = vec![1_000_000_000u64; 27];
        let tps = |vram: u64| {
            let s = OffloadSettings { vram_budget: vram, host_budget: 100_000_000_000, ..Default::default() };
            plan_layers(&layers, 0, &s, 0).unwrap().tokens_per_second_ceiling(50.0, 10.0, 1500.0)
        };
        assert!(tps(30_000_000_000) > tps(14_000_000_000));
        assert!(tps(14_000_000_000) > tps(4_000_000_000));
        let full = tps(30_000_000_000);
        assert!((full - 1500.0 / 27.0).abs() < 1.0);
    }

    #[test]
    fn pointer_table_is_full_only_when_every_expert_has_an_address() {
        let mut b = SlotBook::new(4, Policy::Lru);
        for e in 0..3 {
            b.acquire(10 + e, false, false).unwrap();
        }
        let (t, full) = pointer_table(&b, |e| 10 + e, 3, 1, 4);
        assert!(full);
        assert_eq!(t, vec![1, 2, 3, 5]);
        b.acquire(99, false, false).unwrap();
        b.acquire(100, false, false).unwrap();
        let (t, full) = pointer_table(&b, |e| 10 + e, 3, 1, 4);
        assert!(!full);
        assert_eq!(t.iter().filter(|&&s| s == 0).count(), 1);
        assert_eq!(t.len(), 4);
        let (t, full) = pointer_table(&SlotBook::new(2, Policy::Lru), |e| e, 2, 0, 2);
        assert!(!full);
        assert_eq!(t, vec![0, 0]);
    }

    #[test]
    fn a_missing_routed_expert_always_takes_the_staged_path() {
        let mut b = SlotBook::new(4, Policy::Lru);
        for e in 0..3 {
            b.acquire(e, false, false).unwrap();
        }
        let (_, full) = pointer_table(&b, |e| e, 4, 0, 4);
        assert!(!full);
        for tokens in [1, 2, 7, 8, 9, 64] {
            assert_eq!(expert_route(true, full, tokens, 8), ExpertRoute::Staged, "{tokens} tokens");
        }
        b.acquire(3, false, false).unwrap();
        let (_, full) = pointer_table(&b, |e| e, 4, 0, 4);
        assert!(full);
        for tokens in [1, 2, 7] {
            assert_eq!(expert_route(true, full, tokens, 8), ExpertRoute::Table);
        }
        for tokens in [8, 9, 64] {
            assert_eq!(expert_route(true, full, tokens, 8), ExpertRoute::Staged);
        }
        assert_eq!(expert_route(false, full, 1, 8), ExpertRoute::Staged);
        assert!(b.evict(2));
        let (t, full) = pointer_table(&b, |e| e, 4, 0, 4);
        assert!(!full);
        assert_eq!(t[2], 0);
        assert_eq!(expert_route(true, full, 1, 8), ExpertRoute::Staged);
    }

    #[test]
    fn slot_book_hits_and_evicts_least_recent() {
        let mut b = SlotBook::new(2, Policy::Lru);
        assert_eq!(b.acquire(10, false, false), Some(Acquired { slot: 0, copy: true }));
        assert_eq!(b.acquire(11, false, false), Some(Acquired { slot: 1, copy: true }));
        assert_eq!(b.acquire(10, false, false), Some(Acquired { slot: 0, copy: false }));
        assert_eq!(b.acquire(12, false, false), Some(Acquired { slot: 1, copy: true }));
        assert_eq!(b.resident(11), None);
        assert_eq!((b.stats.vram_hits, b.stats.host_hits, b.stats.evictions), (1, 3, 1));
    }

    #[test]
    fn a_cleared_book_hands_out_slots_in_order_and_keeps_pinned_pages() {
        let mut b = SlotBook::new(4, Policy::Lru);
        b.acquire(10, true, false).unwrap();
        b.unpin_all();
        b.acquire(11, false, false).unwrap();
        b.acquire(12, true, false).unwrap();
        assert_eq!(b.misses([10, 11, 99].into_iter()), 1);
        b.clear_unpinned();
        assert_eq!(b.resident(12), Some(2));
        assert_eq!(b.resident(10), None);
        let slots: Vec<usize> = [1, 2, 3].iter().map(|&p| b.acquire(p, true, false).unwrap().slot).collect();
        assert_eq!(slots, vec![0, 1, 3]);
    }

    #[test]
    fn slot_book_never_evicts_pinned_pages() {
        let mut b = SlotBook::new(2, Policy::Lru);
        b.acquire(1, true, false).unwrap();
        b.acquire(2, true, false).unwrap();
        assert_eq!(b.acquire(3, true, false), None);
        b.unpin_all();
        assert!(b.acquire(3, true, false).is_some());
    }

    #[test]
    fn slot_book_frequency_keeps_the_busy_page() {
        let mut b = SlotBook::new(2, Policy::Frequency);
        b.acquire(1, false, false).unwrap();
        for _ in 0..3 {
            b.acquire(1, false, false).unwrap();
        }
        b.acquire(2, false, false).unwrap();
        b.acquire(3, false, false).unwrap();
        assert!(b.resident(1).is_some());
        assert!(b.resident(2).is_none());
    }

    #[test]
    fn speculative_acquire_is_not_counted_as_a_use() {
        let mut b = SlotBook::new(2, Policy::Lru);
        b.acquire(1, false, true).unwrap();
        assert_eq!(b.stats.host_hits, 0);
        assert_eq!(b.acquire(1, false, false), Some(Acquired { slot: 0, copy: false }));
        assert_eq!(b.stats.vram_hits, 1);
    }

    #[test]
    fn hotlist_ranks_by_count_and_round_trips() {
        let mut h = ExpertHotlist::new(2, 4);
        for _ in 0..3 {
            h.record(0, 2);
        }
        h.record(0, 1);
        h.record(1, 3);
        h.record(1, 3);
        assert_eq!(h.top(0, 2), vec![2, 1]);
        assert_eq!(h.hottest(2), vec![(0, 2), (1, 3)]);
        let path = std::env::temp_dir().join(format!("hotlist-{}.json", std::process::id()));
        h.save(&path).unwrap();
        assert_eq!(ExpertHotlist::load(&path).unwrap(), h);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn settings_parse_from_toml_with_defaults() {
        let s: OffloadSettings = toml::from_str("vram_budget = 12\npolicy = \"frequency\"\n").unwrap();
        assert_eq!(s.vram_budget, 12);
        assert_eq!(s.policy, Policy::Frequency);
        assert_eq!(s.prefetch_depth, 1);
    }

    use std::cell::Cell;
    use std::rc::Rc;

    struct FakeCopies {
        now: Rc<Cell<u64>>,
        latency: u64,
        copies: Vec<(usize, usize)>,
    }

    impl CopyEngine for FakeCopies {
        type Ticket = u64;
        fn copy(&mut self, page: usize, slot: usize) -> Result<Option<u64>> {
            self.copies.push((page, slot));
            Ok(Some(self.now.get() + self.latency))
        }
    }

    impl FakeCopies {
        fn done(&self, ticket: &u64) -> bool {
            self.now.get() >= *ticket
        }
    }

    fn engine(latency: u64) -> (FakeCopies, Rc<Cell<u64>>) {
        let now = Rc::new(Cell::new(0));
        (FakeCopies { now: now.clone(), latency, copies: vec![] }, now)
    }

    fn score_settings(margin: f64, half_life: u32) -> OffloadSettings {
        OffloadSettings { policy: Policy::Score, promote_margin: margin, score_half_life: half_life, promote_per_step: 16, ..Default::default() }
    }

    fn step(book: &mut ExpertBook<u64>, eng: &mut FakeCopies, now: &Rc<Cell<u64>>, used: &[usize]) -> Vec<Vec<Placed>> {
        book.poll(|t| eng.done(t));
        let groups = book.plan(used, false).unwrap();
        for &p in used {
            book.record(p, 1.0);
        }
        book.end_step(eng).unwrap();
        now.set(now.get() + 1);
        groups
    }

    #[test]
    fn promotion_evicts_a_resident_only_past_the_margin() {
        let (mut eng, _) = engine(0);
        let mut b: ExpertBook<u64> = ExpertBook::new(4, 2, 1, &score_settings(1.15, 1_000_000));
        b.record(0, 10.0);
        b.end_step(&mut eng).unwrap();
        b.poll(|t| eng.done(t));
        assert_eq!(b.resident(0), Some(0));
        b.record(1, 11.0);
        b.end_step(&mut eng).unwrap();
        b.poll(|t| eng.done(t));
        assert_eq!((b.resident(0), b.resident(1)), (Some(0), None), "11 is under 1.15 times 10");
        assert_eq!(b.stats.evictions, 0);
        b.record(1, 1.0);
        b.end_step(&mut eng).unwrap();
        b.poll(|t| eng.done(t));
        assert_eq!((b.resident(0), b.resident(1)), (None, Some(0)), "12 is over 1.15 times 10");
        assert_eq!((b.stats.evictions, b.stats.promotions), (1, 2));
    }

    #[test]
    fn an_expert_still_copying_is_served_from_scratch() {
        let (mut eng, now) = engine(3);
        let mut b: ExpertBook<u64> = ExpertBook::new(8, 4, 1, &score_settings(1.15, 64));
        let g = step(&mut b, &mut eng, &now, &[5]);
        assert_eq!(g, vec![vec![Placed { page: 5, slot: 3, copy: true }]]);
        assert_eq!((b.in_flight(), b.resident(5)), (1, None));
        for _ in 0..2 {
            let g = step(&mut b, &mut eng, &now, &[5]);
            assert_eq!(g, vec![vec![Placed { page: 5, slot: 3, copy: true }]], "in flight is never read as resident");
        }
        let g = step(&mut b, &mut eng, &now, &[5]);
        assert_eq!(g, vec![vec![Placed { page: 5, slot: 0, copy: false }]]);
        assert_eq!(eng.copies, vec![(5, 0)], "one promotion copy, never repeated while in flight");
    }

    #[test]
    fn scores_decay_by_half_each_half_life() {
        let (mut eng, _) = engine(0);
        let mut b: ExpertBook<u64> = ExpertBook::new(2, 2, 1, &score_settings(1.15, 4));
        b.record(1, 8.0);
        for _ in 0..4 {
            b.end_step(&mut eng).unwrap();
        }
        assert!((b.score(1) - 4.0).abs() < 1e-9, "{}", b.score(1));
        for _ in 0..4 {
            b.end_step(&mut eng).unwrap();
        }
        assert!((b.score(1) - 2.0).abs() < 1e-9, "{}", b.score(1));
    }

    fn alternating_evictions(margin: f64) -> u64 {
        let (mut eng, now) = engine(1);
        let mut b: ExpertBook<u64> = ExpertBook::new(4, 2, 1, &score_settings(margin, 64));
        for t in 0..1000 {
            step(&mut b, &mut eng, &now, &[t % 2]);
        }
        b.stats.evictions
    }

    #[test]
    fn alternating_experts_do_not_thrash_the_cache() {
        assert!(alternating_evictions(1.15) <= 2, "{}", alternating_evictions(1.15));
        assert!(alternating_evictions(1.0) > 400, "without a margin the same pattern thrashes: {}", alternating_evictions(1.0));
    }

    #[test]
    fn hotlist_seed_fills_the_cache_before_the_first_token() {
        let mut hot = ExpertHotlist::new(2, 4);
        for (l, e, n) in [(0, 1, 9), (0, 2, 3), (1, 0, 7), (1, 3, 1), (0, 0, 1)] {
            for _ in 0..n {
                hot.record(l, e);
            }
        }
        let mut b: ExpertBook<u64> = ExpertBook::new(8, 4, 1, &score_settings(1.15, 64));
        let placed = b.seed(&hot, 4, 2);
        assert_eq!(placed.len(), 3);
        assert!(b.resident(1).is_some() && b.resident(4).is_some() && b.resident(2).is_some(), "{placed:?}");
        let g = b.plan(&[1, 4], false).unwrap();
        assert!(g[0].iter().all(|p| !p.copy), "{g:?}");
        assert_eq!(b.stats.host_hits, 0);
    }

    #[test]
    fn misses_past_the_scratch_slots_split_into_groups_or_borrow_cold_slots() {
        let (mut eng, _) = engine(0);
        let mut b: ExpertBook<u64> = ExpertBook::new(16, 6, 2, &score_settings(1.15, 64));
        for p in 0..4 {
            b.record(p, 1.0 + p as f64);
        }
        b.end_step(&mut eng).unwrap();
        b.poll(|t| eng.done(t));
        let g = b.plan(&[0, 8, 9, 10, 11, 12], false).unwrap();
        assert_eq!(g.len(), 3);
        assert_eq!(g[0][0], Placed { page: 0, slot: b.resident(0).unwrap(), copy: false });
        assert!(g.iter().flatten().filter(|p| p.copy).all(|p| p.slot >= 4));
        let g = b.plan(&[3, 8, 9, 10, 11, 12], true).unwrap();
        assert_eq!(g.len(), 1);
        let slots: Vec<usize> = g[0].iter().map(|p| p.slot).collect();
        let mut uniq = slots.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 6);
        assert_eq!(b.resident(3), Some(g[0][0].slot), "the hottest routed resident stays");
        assert_eq!((b.resident(0), b.resident(1), b.resident(2)), (None, None, None));
        let borrowed: Vec<usize> = g[0].iter().filter(|p| p.copy && p.slot < 4).map(|p| p.page).collect();
        assert_eq!(borrowed.len(), 3);
        assert!(borrowed.iter().all(|&p| b.resident(p).is_some()), "a borrowed slot keeps the expert it was given");
    }

    #[test]
    fn a_borrowed_slot_reused_by_a_later_group_belongs_to_the_last_expert_copied() {
        let (mut eng, _) = engine(0);
        let mut b: ExpertBook<u64> = ExpertBook::new(32, 4, 1, &score_settings(1.15, 64));
        for p in 0..3 {
            b.record(p, 1.0);
        }
        b.end_step(&mut eng).unwrap();
        b.poll(|t| eng.done(t));
        let g = b.plan(&[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], true).unwrap();
        assert!(g.len() > 1, "{g:?}");
        let mut owner = std::collections::HashMap::new();
        for p in g.iter().flatten() {
            owner.insert(p.slot, p.page);
        }
        for (slot, page) in owner {
            if slot < b.cache_slots() {
                assert_eq!(b.resident(page), Some(slot));
            }
        }
        for page in 10..20 {
            if let Some(slot) = b.resident(page) {
                assert_eq!(g.iter().flatten().rfind(|p| p.slot == slot).unwrap().page, page, "slot {slot} holds the last copy");
            }
        }
    }

    #[test]
    fn the_table_changes_only_when_a_promotion_lands_or_an_expert_leaves() {
        let (mut eng, now) = engine(2);
        let mut b: ExpertBook<u64> = ExpertBook::new(4, 3, 1, &score_settings(1.15, 1_000_000));
        let table = |b: &ExpertBook<u64>| pointer_table_of(|p| b.resident(p), |e| e, 4, 0, 3);
        b.record(0, 5.0);
        b.record(1, 5.0);
        b.end_step(&mut eng).unwrap();
        assert_eq!(table(&b), (vec![0, 0, 0, 0], false), "copies in flight are not in the table");
        now.set(1);
        b.poll(|t| eng.done(t));
        assert_eq!(table(&b).0, vec![0, 0, 0, 0]);
        now.set(2);
        b.poll(|t| eng.done(t));
        let landed = table(&b).0;
        assert!(landed[0] > 0 && landed[1] > 0 && landed[2] == 0, "{landed:?}");
        b.record(2, 100.0);
        b.end_step(&mut eng).unwrap();
        let during = table(&b).0;
        assert!(during.iter().filter(|&&s| s > 0).count() == 1 && during[2] == 0, "an eviction leaves the table at once, the newcomer only on landing: {during:?}");
        now.set(10);
        b.poll(|t| eng.done(t));
        assert!(table(&b).0[2] > 0);
        assert!(b.evict(2));
        assert_eq!(table(&b).0[2], 0);
    }

    #[test]
    fn a_pool_that_holds_every_cached_expert_has_no_scratch_and_promotes_nothing() {
        let on = OffloadSettings { pointer_table: true, ..score_settings(1.15, 64) };
        assert_eq!(score_pool(24, 2, 24, &on), (0, 0));
        assert_eq!(score_pool(30, 2, 24, &on), (0, 0));
        assert_eq!(score_pool(12, 2, 24, &on), (2, 16));
        assert_eq!(score_pool(24, 2, 24, &score_settings(1.15, 64)), (2, 16), "without the table the margin still applies");
        let (scratch, per_step) = score_pool(8, 2, 8, &on);
        let (mut eng, _) = engine(0);
        let mut b: ExpertBook<u64> = ExpertBook::new(8, 8, scratch, &OffloadSettings { promote_per_step: per_step, ..on });
        assert_eq!(b.fill(&(0..8).collect::<Vec<_>>()).len(), 8);
        for p in 0..8 {
            b.record(p, p as f64 * 10.0);
        }
        assert_eq!(b.end_step(&mut eng).unwrap(), 0);
        assert!(eng.copies.is_empty());
        assert!(pointer_table_of(|p| b.resident(p), |e| e, 8, 0, 8).1);
        assert!(b.evict(3));
        let g = b.plan(&[3], false).unwrap();
        assert_eq!(g, vec![vec![Placed { page: 3, slot: 3, copy: true }]], "with no scratch a miss reuses a free cache slot");
    }

    #[test]
    fn scratch_slots_never_appear_in_the_pointer_table() {
        let (mut eng, _) = engine(0);
        let mut b: ExpertBook<u64> = ExpertBook::new(8, 4, 2, &score_settings(1.15, 64));
        b.record(0, 3.0);
        b.end_step(&mut eng).unwrap();
        b.poll(|t| eng.done(t));
        let g = b.plan(&[0, 5, 6], false).unwrap();
        let scratch: Vec<usize> = g.iter().flatten().filter(|p| p.copy).map(|p| p.slot).collect();
        assert_eq!(scratch, vec![2, 3]);
        let (t, full) = pointer_table_of(|p| b.resident(p), |e| e, 8, 0, 4);
        assert!(!full);
        assert!(t.iter().all(|&s| s == 0 || (s as usize - 1) < b.cache_slots()), "{t:?}");
        assert_eq!((b.resident(5), b.resident(6)), (None, None));
    }

    #[test]
    fn score_settings_parse_with_defaults() {
        let s: OffloadSettings = toml::from_str("policy = \"score\"\npromote_margin = 1.3\n").unwrap();
        assert_eq!((s.policy, s.promote_margin, s.score_half_life, s.promote_per_step), (Policy::Score, 1.3, 64, 16));
        assert_eq!(OffloadSettings::default().policy, Policy::Score);
        assert_eq!(OffloadSettings::default().promote_margin, 1.15);
    }

    #[test]
    fn metrics_text_names_the_issue_series() {
        let st = OffloadStats { vram_hits: 3, nvme_reads: 1, nvme_read_bytes: 7, ..Default::default() };
        let t = st.prometheus();
        assert!(t.contains("inference_offload_hit_ratio 0.75"));
        assert!(t.contains("inference_offload_read_bytes_total 7"));
    }
}
