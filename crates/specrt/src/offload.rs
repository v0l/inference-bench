use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    #[default]
    Lru,
    Frequency,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OffloadSettings {
    pub vram_budget: u64,
    pub host_budget: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvme_path: Option<String>,
    pub prefetch_depth: usize,
    pub policy: Policy,
    pub expert_slots: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hotlist: Option<String>,
}

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
        OffloadSettings { vram_budget: 0, host_budget: 0, nvme_path: None, prefetch_depth: default_prefetch_depth(), policy: Policy::Lru, expert_slots: 0, hotlist: None }
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
    pub host_to_vram_bytes: u64,
    pub evictions: u64,
}

impl OffloadStats {
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
                Policy::Frequency => (e.uses, e.last_use, **id),
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
                    Policy::Lru => (self.last_use[s], 0, s),
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
}

pub fn plan_layers(layer_bytes: &[u64], always_resident: u64, settings: &OffloadSettings) -> Result<OffloadPlan> {
    let plan = assign_layers(layer_bytes, always_resident, settings, 0)?;
    if plan.nvme_bytes == 0 {
        return Ok(plan);
    }
    let slot = layer_bytes.iter().copied().max().unwrap_or(0).div_ceil(4096) * 4096;
    let staging = slot * (settings.prefetch_depth.max(1) as u64 + 1);
    assign_layers(layer_bytes, always_resident, settings, staging)
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
    Ok(OffloadPlan { vram_bytes: sum(Tier::Vram) + fixed, host_bytes: sum(Tier::Host), nvme_bytes: sum(Tier::Nvme), staging_bytes: staging, host_staging_bytes: host_staging, layers })
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
    fn pinned_total_with_disk_staging_stays_inside_host_budget() {
        let sizes = vec![10000u64; 10];
        let staging = 12288u64 * 3;
        for host in [0u64, 30000, 90000, 200000] {
            let s = OffloadSettings { vram_budget: 60000, host_budget: host, prefetch_depth: 2, ..Default::default() };
            let p = plan_layers(&sizes, 0, &s).unwrap();
            if p.nvme_bytes > 0 {
                assert_eq!(p.host_staging_bytes, staging);
                assert!(p.host_bytes + p.host_staging_bytes <= host.max(staging), "{p:?}");
            } else {
                assert_eq!(p.host_staging_bytes, 0);
            }
        }
        let tight = OffloadSettings { vram_budget: 60000, host_budget: staging + 20000, prefetch_depth: 2, ..Default::default() };
        let p = plan_layers(&sizes, 0, &tight).unwrap();
        assert!(p.nvme_bytes > 0 && p.host_bytes == 20000 && p.host_bytes + p.host_staging_bytes <= tight.host_budget, "{p:?}");
    }

    #[test]
    fn planner_pins_first_and_last_then_fills_vram_then_host() {
        let layers = vec![10u64; 10];
        let s = OffloadSettings { vram_budget: 50, host_budget: 30, prefetch_depth: 1, ..Default::default() };
        let p = plan_layers(&layers, 10, &s).unwrap();
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
        assert!(plan_layers(&[10, 10], 10, &s).is_err());
    }

    #[test]
    fn ceiling_falls_as_vram_budget_shrinks() {
        let layers = vec![1_000_000_000u64; 27];
        let tps = |vram: u64| {
            let s = OffloadSettings { vram_budget: vram, host_budget: 100_000_000_000, ..Default::default() };
            plan_layers(&layers, 0, &s).unwrap().tokens_per_second_ceiling(50.0, 10.0, 1500.0)
        };
        assert!(tps(30_000_000_000) > tps(14_000_000_000));
        assert!(tps(14_000_000_000) > tps(4_000_000_000));
        let full = tps(30_000_000_000);
        assert!((full - 1500.0 / 27.0).abs() < 1.0);
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

    #[test]
    fn metrics_text_names_the_issue_series() {
        let st = OffloadStats { vram_hits: 3, nvme_reads: 1, nvme_read_bytes: 7, ..Default::default() };
        let t = st.prometheus();
        assert!(t.contains("inference_offload_hit_ratio 0.75"));
        assert!(t.contains("inference_offload_read_bytes_total 7"));
    }
}
