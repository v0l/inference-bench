use super::{StackWeights, Stacks};
use crate::hostpool::{HostPool, PinnedBuf};
use crate::offload::{Acquired, ExpertHotlist, OffloadStats, Policy, SlotBook};
use crate::weights::{MoeWeights, Stacked};
use anyhow::{Result, ensure};
use cudarc::driver::sys::{CUdeviceptr, CUevent_flags};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Bf16,
    Fp4,
    Fp8,
    Block,
    Ggml,
}

struct Part {
    offset: usize,
    per: usize,
}

enum Backing {
    Pinned(PinnedBuf),
    File { file: Arc<dyn crate::diskio::ReadAt>, path: std::path::PathBuf, len: u64 },
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::File { path, .. } = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[derive(Clone)]
pub(in crate::runtime) struct ExpertStore {
    pool: HostPool,
    dir: Option<std::path::PathBuf>,
}

impl ExpertStore {
    pub(in crate::runtime) fn new(ctx: Arc<cudarc::driver::CudaContext>, host_budget: u64, dir: Option<&str>) -> Self {
        let pool = HostPool::new(ctx);
        if host_budget > 0 {
            pool.set_limit(host_budget as usize);
        }
        ExpertStore { pool, dir: dir.map(std::path::PathBuf::from) }
    }
}

pub(in crate::runtime) struct HostExperts {
    buf: Backing,
    kinds: [Kind; 3],
    pub(in crate::runtime) input_globals: [f32; 3],
    parts: Vec<Part>,
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn comps(s: &Stacked) -> (Kind, Vec<&[u8]>) {
    match s {
        Stacked::Bf16(w) => (Kind::Bf16, vec![as_bytes(w)]),
        Stacked::Fp4 { packed, scales, inv_global, .. } => (Kind::Fp4, vec![packed, scales, as_bytes(inv_global)]),
        Stacked::Fp8 { values, grid } => (Kind::Fp8, vec![values, as_bytes(grid)]),
        Stacked::Block { packed, scales } => (Kind::Block, vec![packed, scales]),
        Stacked::Ggml { data, tables } => (Kind::Ggml, vec![data, as_bytes(tables)]),
    }
}

impl HostExperts {
    pub(in crate::runtime) fn pack(store: &ExpertStore, m: &MoeWeights, entries: usize) -> Result<Self> {
        let all: Vec<(Kind, Vec<&[u8]>)> = [&m.gate, &m.up, &m.down].into_iter().map(comps).collect();
        let total: usize = all.iter().flat_map(|(_, c)| c.iter()).map(|c| c.len()).sum();
        let mut parts = Vec::new();
        let mut at = 0;
        for (_, cs) in &all {
            for c in cs {
                ensure!(entries > 0 && c.len() % entries == 0, "expert tensor of {} bytes does not split into {entries} experts", c.len());
                parts.push(Part { offset: at, per: c.len() / entries });
                at += c.len();
            }
        }
        let flat = || all.iter().flat_map(|(_, cs)| cs.iter().copied());
        let buf = match store.pool.alloc(total) {
            Ok(mut b) => {
                let dst = b.as_mut_slice()?;
                let mut at = 0;
                for c in flat() {
                    dst[at..at + c.len()].copy_from_slice(c);
                    at += c.len();
                }
                Backing::Pinned(b)
            }
            Err(e) => {
                let Some(dir) = &store.dir else {
                    return Err(e.context("expert pages exceed [runtime.offload] host_budget and nvme_path is not set"));
                };
                std::fs::create_dir_all(dir)?;
                crate::diskio::remove_stale(dir, "experts");
                let path = crate::diskio::unique_path(dir, "experts");
                let mut f = std::fs::File::create(&path)?;
                for c in flat() {
                    std::io::Write::write_all(&mut f, c)?;
                }
                f.sync_all()?;
                drop(f);
                Backing::File { file: Arc::new(crate::diskio::DirectFile::open(&path)?), path, len: total as u64 }
            }
        };
        let ig = |s: &Stacked| if let Stacked::Fp4 { input_global, .. } = s { *input_global } else { 1.0 };
        Ok(HostExperts { buf, kinds: [all[0].0, all[1].0, all[2].0], input_globals: [ig(&m.gate), ig(&m.up), ig(&m.down)], parts })
    }

    pub(in crate::runtime) fn entry_bytes(&self) -> usize {
        self.parts.iter().map(|p| p.per).sum()
    }

    pub(in crate::runtime) fn host_bytes(&self) -> u64 {
        match &self.buf {
            Backing::Pinned(b) => b.len() as u64,
            Backing::File { len, .. } => *len,
        }
    }

    fn allocate(&self, stream: &Arc<CudaStream>, cap: usize) -> Result<Stacks> {
        let mut per = self.parts.iter().map(|p| p.per);
        let mut one = |kind: Kind| -> Result<StackWeights> {
            Ok(match kind {
                Kind::Bf16 => StackWeights::Bf16(stream.alloc_zeros(cap * per.next().unwrap() / 2)?),
                Kind::Fp8 => StackWeights::Fp8 { values: stream.alloc_zeros(cap * per.next().unwrap())?, grid: stream.alloc_zeros(cap * per.next().unwrap() / 4)? },
                Kind::Block => StackWeights::Block { packed: stream.alloc_zeros(cap * per.next().unwrap())?, scales: stream.alloc_zeros(cap * per.next().unwrap())? },
                Kind::Ggml => StackWeights::Ggml { data: stream.alloc_zeros(cap * per.next().unwrap())?, tables: stream.alloc_zeros(cap * per.next().unwrap() / 4)? },
                Kind::Fp4 => StackWeights::Fp4 {
                    input_global: 1.0,
                    packed: stream.alloc_zeros(cap * per.next().unwrap())?,
                    scales: stream.alloc_zeros(cap * per.next().unwrap())?,
                    inv_global: stream.alloc_zeros(cap * per.next().unwrap() / 4)?,
                },
            })
        };
        Ok(Stacks { gate: one(self.kinds[0])?, up: one(self.kinds[1])?, down: one(self.kinds[2])? })
    }
}

fn address<T: cudarc::driver::DeviceRepr>(stream: &Arc<CudaStream>, s: &CudaSlice<T>) -> CUdeviceptr {
    let (ptr, _sync) = s.device_ptr(stream);
    ptr
}

fn pointers(stream: &Arc<CudaStream>, s: &Stacks) -> Vec<CUdeviceptr> {
    let mut out = Vec::new();
    for w in [&s.gate, &s.up, &s.down] {
        match w {
            StackWeights::Bf16(a) => out.push(address(stream, a)),
            StackWeights::Fp8 { values, grid } => out.extend([address(stream, values), address(stream, grid)]),
            StackWeights::Block { packed, scales } => out.extend([address(stream, packed), address(stream, scales)]),
            StackWeights::Ggml { data, tables } => out.extend([address(stream, data), address(stream, tables)]),
            StackWeights::Fp4 { packed, scales, inv_global, .. } => out.extend([address(stream, packed), address(stream, scales), address(stream, inv_global)]),
        }
    }
    out
}

struct State {
    book: SlotBook,
    last: Vec<Vec<usize>>,
    seen: ExpertHotlist,
    prior: Option<ExpertHotlist>,
    mirror: Vec<Vec<i32>>,
}

pub(in crate::runtime) struct ExpertCache {
    copy: Arc<CudaStream>,
    mark: CudaEvent,
    copied: CudaEvent,
    pub(in crate::runtime) stacks: Stacks,
    addrs: Vec<CUdeviceptr>,
    pub(in crate::runtime) slots: usize,
    pub(in crate::runtime) bias: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    tables: Option<Vec<CudaSlice<i32>>>,
    table_launches: std::sync::atomic::AtomicU64,
    local: usize,
    shared: usize,
    state: Mutex<State>,
    staging: Mutex<Staging>,
    staging_pool: HostPool,
}

const BULK_MIN_EXPERTS: usize = 32;

pub(in crate::runtime) const TABLE_MAX_TOKENS: usize = super::MMA_MIN_TOKENS;

struct Staging {
    buf: Option<PinnedBuf>,
}

pub(in crate::runtime) struct Staged {
    pub mapping: Vec<(usize, usize)>,
}

impl ExpertCache {
    pub(in crate::runtime) fn new(compute: &Arc<CudaStream>, like: &HostExperts, layers: usize, slots: usize, local: usize, shared: usize, policy: Policy, prior: Option<ExpertHotlist>, pointer_table: bool, bias_dims: Option<(usize, usize)>) -> Result<Self> {
        let bias = match bias_dims {
            Some((inter, hidden)) => Some((compute.alloc_zeros::<f32>((slots + shared) * 2 * inter)?, compute.alloc_zeros::<f32>((slots + shared) * hidden)?)),
            None => None,
        };
        let ctx = compute.context().clone();
        let stacks = like.allocate(compute, slots + shared)?;
        let addrs = pointers(compute, &stacks);
        let tables = if pointer_table { Some((0..layers).map(|_| compute.alloc_zeros::<i32>(local + shared)).collect::<Result<Vec<_>, _>>()?) } else { None };
        Ok(ExpertCache {
            copy: ctx.new_stream()?,
            mark: ctx.new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
            copied: ctx.new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
            stacks,
            addrs,
            slots,
            bias,
            tables,
            table_launches: std::sync::atomic::AtomicU64::new(0),
            local,
            shared,
            staging: Mutex::new(Staging { buf: None }),
            staging_pool: HostPool::new(ctx.clone()),
            state: Mutex::new(State { book: SlotBook::new(slots, policy), last: vec![Vec::new(); layers], seen: ExpertHotlist::new(layers, local), prior, mirror: vec![Vec::new(); layers] }),
        })
    }

    pub(in crate::runtime) fn stats(&self) -> OffloadStats {
        self.state.lock().unwrap().book.stats
    }

    pub(in crate::runtime) fn hotlist(&self) -> ExpertHotlist {
        self.state.lock().unwrap().seen.clone()
    }

    fn page(&self, layer: usize, expert: usize) -> usize {
        layer * self.local + expert
    }

    fn enqueue(&self, host: &HostExperts, entry: usize, slot: usize, stats: &mut OffloadStats) -> Result<()> {
        self.enqueue_run(host, entry, slot, 1, stats)
    }

    fn enqueue_all(&self, host: &HostExperts, copies: &mut Vec<(usize, usize)>, stats: &mut OffloadStats) -> Result<()> {
        copies.sort_unstable();
        let mut runs = Vec::new();
        let mut i = 0;
        while i < copies.len() {
            let (entry, slot) = copies[i];
            let mut n = 1;
            while i + n < copies.len() && copies[i + n] == (entry + n, slot + n) {
                n += 1;
            }
            runs.push((entry, slot, n));
            i += n;
        }
        copies.clear();
        if let Backing::File { file, .. } = &host.buf {
            return self.read_runs(file, host, &runs, stats);
        }
        for (entry, slot, n) in runs {
            self.enqueue_run(host, entry, slot, n, stats)?;
        }
        Ok(())
    }

    fn read_runs(&self, file: &Arc<dyn crate::diskio::ReadAt>, host: &HostExperts, runs: &[(usize, usize, usize)], stats: &mut OffloadStats) -> Result<()> {
        let mut st = self.staging.lock().unwrap();
        let cap = (host.entry_bytes() * 64).max(64 << 20);
        if st.buf.is_none() {
            st.buf = Some(self.staging_pool.alloc(cap)?);
        }
        let mut jobs: Vec<(usize, usize, usize, usize)> = Vec::new();
        for &(entry, slot, count) in runs {
            for (pi, p) in host.parts.iter().enumerate() {
                let step = (cap / p.per).max(1);
                let mut done = 0;
                while done < count {
                    let take = step.min(count - done);
                    jobs.push((pi, entry + done, slot + done, take));
                    done += take;
                }
            }
        }
        let pool = crate::diskio::shared_pool();
        let mut j = 0;
        while j < jobs.len() {
            self.copy.synchronize()?;
            let base_ptr = st.buf.as_mut().unwrap().as_mut_slice()?.as_mut_ptr();
            let mut at = 0;
            let mut batch = Vec::new();
            let mut tickets = Vec::new();
            while j < jobs.len() {
                let (pi, entry, slot, take) = jobs[j];
                let p = &host.parts[pi];
                let bytes = take * p.per;
                if at + bytes > cap {
                    break;
                }
                let t = unsafe { submit_counted(&pool, file, (p.offset + entry * p.per) as u64, base_ptr.add(at), bytes, stats) };
                tickets.push(t);
                batch.push((pi, slot, at, bytes));
                at += bytes.next_multiple_of(4096);
                j += 1;
            }
            wait_counted(tickets, stats)?;
            let buf = st.buf.as_ref().unwrap().as_slice()?;
            for (pi, slot, off, bytes) in batch {
                let p = &host.parts[pi];
                unsafe { cudarc::driver::result::memcpy_htod_async(self.addrs[pi] + (slot * p.per) as u64, &buf[off..off + bytes], self.copy.cu_stream()) }?;
                stats.host_to_vram_bytes += bytes as u64;
            }
        }
        Ok(())
    }

    fn enqueue_run(&self, host: &HostExperts, entry: usize, slot: usize, count: usize, stats: &mut OffloadStats) -> Result<()> {
        match &host.buf {
            Backing::Pinned(b) => {
                let src = b.as_slice()?;
                for (p, base) in host.parts.iter().zip(&self.addrs) {
                    let from = &src[p.offset + entry * p.per..p.offset + (entry + count) * p.per];
                    unsafe { cudarc::driver::result::memcpy_htod_async(base + (slot * p.per) as u64, from, self.copy.cu_stream()) }?;
                    stats.host_to_vram_bytes += (p.per * count) as u64;
                }
            }
            Backing::File { file, .. } => self.read_runs(file, host, &[(entry, slot, count)], stats)?,
        }
        Ok(())
    }

    fn fence(&self, compute: &Arc<CudaStream>) -> Result<()> {
        self.copied.record(&self.copy)?;
        compute.wait(&self.copied)?;
        Ok(())
    }

    pub(in crate::runtime) fn stage(&self, compute: &Arc<CudaStream>, layer: usize, host: &HostExperts, wanted: &[usize], with_shared: bool, bulk: bool) -> Result<Staged> {
        let mut g = self.state.lock().unwrap();
        let g = &mut *g;
        if bulk && wanted.len() >= BULK_MIN_EXPERTS && g.book.misses(wanted.iter().map(|&e| self.page(layer, e))) * 2 > wanted.len() {
            g.book.clear_unpinned();
        }
        self.mark.record(compute)?;
        self.copy.wait(&self.mark)?;
        let mut stats = OffloadStats::default();
        let mut mapping = Vec::with_capacity(wanted.len());
        let mut copies = Vec::new();
        for &e in wanted {
            let Some(Acquired { slot, copy }) = g.book.acquire(self.page(layer, e), true, false) else {
                anyhow::bail!("expert cache of {} slots cannot hold {} experts at once", self.slots, wanted.len());
            };
            if copy {
                copies.push((e, slot));
            }
            mapping.push((e, slot));
        }
        let queued = self.enqueue_all(host, &mut copies, &mut stats).and_then(|_| {
            if with_shared {
                for s in 0..self.shared {
                    self.enqueue(host, self.local + s, self.slots + s, &mut stats)?;
                }
            }
            Ok(())
        });
        g.book.stats.absorb_io(&stats);
        queued?;
        self.fence(compute)?;
        Ok(Staged { mapping })
    }

    pub(in crate::runtime) fn table_for(&self, compute: &Arc<CudaStream>, layer: usize, host: &HostExperts, tokens: usize) -> Result<Option<&CudaSlice<i32>>> {
        let Some(tables) = &self.tables else { return Ok(None) };
        let mut g = self.state.lock().unwrap();
        let g = &mut *g;
        let (table, full) = crate::offload::pointer_table(&g.book, |e| self.page(layer, e), self.local, self.shared, self.slots);
        if crate::offload::expert_route(true, full, tokens, TABLE_MAX_TOKENS) == crate::offload::ExpertRoute::Staged {
            return Ok(None);
        }
        self.table_launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for e in 0..self.local {
            g.book.touch(self.page(layer, e));
        }
        if g.mirror[layer] != table {
            let (ptr, _sync) = tables[layer].device_ptr(compute);
            unsafe { cudarc::driver::result::memcpy_htod_async(ptr, &table, compute.cu_stream()) }?;
            g.mirror[layer] = table;
        }
        if self.shared > 0 {
            self.mark.record(compute)?;
            self.copy.wait(&self.mark)?;
            let mut stats = OffloadStats::default();
            let queued = (0..self.shared).try_for_each(|s| self.enqueue(host, self.local + s, self.slots + s, &mut stats));
            g.book.stats.absorb_io(&stats);
            queued?;
        }
        self.fence(compute)?;
        Ok(Some(&tables[layer]))
    }

    pub(in crate::runtime) fn table_launches(&self) -> u64 {
        self.table_launches.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(in crate::runtime) fn evict(&self, layer: usize, expert: usize) -> bool {
        self.state.lock().unwrap().book.evict(self.page(layer, expert))
    }

    pub(in crate::runtime) fn promote(&self, compute: &Arc<CudaStream>, layer: usize, host: &HostExperts, expert: usize) -> Result<()> {
        self.prefetch(compute, layer, host, &[expert])?;
        self.fence(compute)
    }

    pub(in crate::runtime) fn release(&self) {
        self.state.lock().unwrap().book.unpin_all();
    }

    pub(in crate::runtime) fn observe(&self, layer: usize, used: &[usize]) {
        let mut g = self.state.lock().unwrap();
        for &e in used {
            g.seen.record(layer, e);
        }
        g.last[layer] = used.to_vec();
    }

    pub(in crate::runtime) fn predicted(&self, layer: usize, k: usize) -> Vec<usize> {
        let g = self.state.lock().unwrap();
        let mut v = g.last[layer].clone();
        if v.is_empty()
            && let Some(p) = &g.prior
        {
            v = p.top(layer, k);
        }
        v.sort_unstable();
        v
    }

    pub(in crate::runtime) fn prefetch(&self, compute: &Arc<CudaStream>, layer: usize, host: &HostExperts, experts: &[usize]) -> Result<()> {
        self.mark.record(compute)?;
        self.copy.wait(&self.mark)?;
        let mut g = self.state.lock().unwrap();
        let g = &mut *g;
        let mut stats = OffloadStats::default();
        let mut queued = Ok(());
        for &e in experts {
            if let Some(Acquired { slot, copy: true }) = g.book.acquire(self.page(layer, e), false, true)
                && let Err(err) = self.enqueue(host, e, slot, &mut stats)
            {
                queued = Err(err);
                break;
            }
        }
        g.book.stats.absorb_io(&stats);
        queued
    }

    pub(in crate::runtime) fn warm(&self, compute: &Arc<CudaStream>, hosts: &[(usize, &HostExperts)]) -> Result<()> {
        let hot: Vec<(usize, usize)> = {
            let g = self.state.lock().unwrap();
            if self.tables.is_some() && self.slots >= self.local * hosts.len() {
                hosts.iter().flat_map(|(l, _)| (0..self.local).map(move |e| (*l, e))).collect()
            } else {
                let Some(prior) = &g.prior else { return Ok(()) };
                let mut pairs = prior.hottest(self.slots);
                pairs.sort_unstable();
                pairs
            }
        };
        self.mark.record(compute)?;
        self.copy.wait(&self.mark)?;
        let mut g = self.state.lock().unwrap();
        let g = &mut *g;
        let mut stats = OffloadStats::default();
        let mut queued = Ok(());
        for (layer, e) in hot {
            let Some((_, host)) = hosts.iter().find(|(l, _)| *l == layer) else { continue };
            if let Some(Acquired { slot, copy: true }) = g.book.acquire(self.page(layer, e), false, true)
                && let Err(err) = self.enqueue(host, e, slot, &mut stats)
            {
                queued = Err(err);
                break;
            }
        }
        g.book.stats.absorb_io(&stats);
        queued?;
        self.fence(compute)?;
        Ok(())
    }

    pub(in crate::runtime) fn detach(&self, compute: &Arc<CudaStream>) {
        let _ = compute.synchronize();
        let _ = self.copy.synchronize();
    }
}

unsafe fn submit_counted(pool: &crate::diskio::IoPool, file: &Arc<dyn crate::diskio::ReadAt>, offset: u64, dst: *mut u8, len: usize, stats: &mut OffloadStats) -> crate::diskio::Ticket {
    stats.nvme_reads += 1;
    stats.nvme_read_bytes += len as u64;
    unsafe { pool.read(file, offset, dst, len) }
}

fn wait_counted(tickets: Vec<crate::diskio::Ticket>, stats: &mut OffloadStats) -> Result<()> {
    let started = std::time::Instant::now();
    let mut first = Ok(());
    for t in tickets {
        if let Err(e) = t.wait() {
            stats.nvme_read_failures += 1;
            if first.is_ok() {
                first = Err(e);
            }
        }
    }
    stats.nvme_wait_nanos += started.elapsed().as_nanos() as u64;
    first
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskio::{IoPool, ReadAt};
    use std::time::Duration;

    struct Throttled {
        bytes_per_sec: f64,
        fail: bool,
    }

    impl ReadAt for Throttled {
        fn read_at(&self, buf: &mut [u8], _: u64) -> std::io::Result<()> {
            std::thread::sleep(Duration::from_secs_f64(buf.len() as f64 / self.bytes_per_sec));
            if self.fail {
                return Err(std::io::Error::other("drive gone"));
            }
            buf.fill(7);
            Ok(())
        }
    }

    fn run(fail: bool, reads: usize, bytes: usize) -> (Result<()>, OffloadStats, Vec<u8>) {
        let pool = IoPool::new(2);
        let file: Arc<dyn ReadAt> = Arc::new(Throttled { bytes_per_sec: 200e6, fail });
        let mut dst = vec![0u8; reads * bytes];
        let mut stats = OffloadStats::default();
        let base = dst.as_mut_ptr();
        let tickets: Vec<_> = (0..reads).map(|i| unsafe { submit_counted(&pool, &file, (i * bytes) as u64, base.add(i * bytes), bytes, &mut stats) }).collect();
        let r = wait_counted(tickets, &mut stats);
        (r, stats, dst)
    }

    #[test]
    fn expert_reads_count_reads_bytes_and_wait_time() {
        let (r, stats, dst) = run(false, 6, 1 << 20);
        r.unwrap();
        assert!(dst.iter().all(|&b| b == 7));
        assert_eq!(stats.nvme_reads, 6);
        assert_eq!(stats.nvme_read_bytes, 6 << 20);
        assert_eq!(stats.nvme_read_failures, 0);
        assert!(stats.nvme_wait_nanos > 0);
    }

    #[test]
    fn failed_expert_reads_are_counted() {
        let (r, stats, _) = run(true, 3, 4096);
        assert!(r.is_err());
        assert_eq!(stats.nvme_reads, 3);
        assert_eq!(stats.nvme_read_failures, 3);
    }
}
