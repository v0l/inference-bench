use anyhow::{Context, Result, anyhow, ensure};
use cudarc::driver::{CudaContext, PinnedHostSlice};
use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub const ALIGN: usize = 4096;
const PIN_CLASS: usize = 64 << 20;
const PIN_LIMIT: usize = 512 << 20;
const POOL_KEEP: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoaderOptions {
    pub enabled: bool,
    pub direct: bool,
    pub chunk_mb: usize,
    pub readers: usize,
    pub ahead: usize,
    pub workers: usize,
    pub lanes: usize,
    pub skip_cached: bool,
    pub pinned: bool,
    pub host_budget_mb: usize,
}

impl Default for LoaderOptions {
    fn default() -> Self {
        LoaderOptions { enabled: true, direct: true, chunk_mb: 32, readers: 8, ahead: 2, workers: 4, lanes: 1, skip_cached: true, pinned: false, host_budget_mb: 3584 }
    }
}

impl LoaderOptions {
    pub fn is_default(&self) -> bool {
        *self == LoaderOptions::default()
    }

    pub fn host_budget_bytes(&self) -> u64 {
        (self.host_budget_mb as u64) << 20
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.host_budget_mb >= 64, "[runtime.loader] host_budget_mb needs at least 64");
        ensure!(self.chunk_mb >= 1, "[runtime.loader] chunk_mb needs at least 1");
        ensure!((1..=64).contains(&self.readers), "[runtime.loader] readers must be between 1 and 64");
        ensure!(self.ahead >= 1, "[runtime.loader] ahead needs at least 1");
        ensure!((1..=64).contains(&self.workers), "[runtime.loader] workers must be between 1 and 64");
        ensure!((1..=16).contains(&self.lanes), "[runtime.loader] lanes must be between 1 and 16");
        Ok(())
    }
}

pub struct Budget {
    limit: u64,
    used: Mutex<u64>,
    peak: AtomicU64,
    freed: std::sync::Condvar,
}

pub struct Lease {
    budget: Arc<Budget>,
    bytes: u64,
}

impl Budget {
    pub fn new(limit: u64) -> Arc<Budget> {
        Arc::new(Budget { limit, used: Mutex::new(0), peak: AtomicU64::new(0), freed: std::sync::Condvar::new() })
    }

    pub fn acquire(self: &Arc<Self>, bytes: u64) -> Lease {
        let mut used = self.used.lock().unwrap();
        while *used > 0 && *used + bytes > self.limit {
            used = self.freed.wait(used).unwrap();
        }
        *used += bytes;
        self.peak.fetch_max(*used, Ordering::Relaxed);
        Lease { budget: self.clone(), bytes }
    }

    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut used = self.budget.used.lock().unwrap();
        *used -= self.bytes;
        self.budget.freed.notify_all();
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub name: String,
    pub file: usize,
    pub offset: u64,
    pub len: u64,
}

fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

enum Backing {
    Pinned(PinnedHostSlice<u8>),
    Heap(Layout),
}

pub struct Host {
    ptr: *mut u8,
    cap: usize,
    backing: Backing,
}

unsafe impl Send for Host {}
unsafe impl Sync for Host {}

impl Host {
    pub fn pinned(&self) -> bool {
        matches!(self.backing, Backing::Pinned(_))
    }

    pub fn capacity(&self) -> usize {
        match &self.backing {
            Backing::Pinned(p) => p.num_bytes().min(self.cap),
            Backing::Heap(_) => self.cap,
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        if let Backing::Heap(layout) = &self.backing {
            unsafe { std::alloc::dealloc(self.ptr, *layout) };
        }
    }
}

pub struct Pool {
    ctx: Option<Arc<CudaContext>>,
    free: Mutex<Vec<Host>>,
    keep: usize,
}

impl Pool {
    pub fn new(ctx: Option<Arc<CudaContext>>) -> Arc<Pool> {
        Self::with_keep(ctx, POOL_KEEP)
    }

    pub fn with_keep(ctx: Option<Arc<CudaContext>>, keep: usize) -> Arc<Pool> {
        Arc::new(Pool { ctx, free: Mutex::new(Vec::new()), keep })
    }

    pub fn pinned_buffers(&self) -> usize {
        self.free.lock().unwrap().iter().filter(|h| h.pinned()).count()
    }

    fn take(&self, need: usize) -> Result<Host> {
        let need = round_up(need.max(1), ALIGN);
        {
            let mut free = self.free.lock().unwrap();
            let best = free.iter().enumerate().filter(|(_, h)| h.cap >= need).min_by_key(|(_, h)| h.cap).map(|(i, _)| i);
            if let Some(i) = best {
                return Ok(free.swap_remove(i));
            }
        }
        if let Some(ctx) = self.ctx.as_ref().filter(|_| need <= PIN_LIMIT) {
            let cap = round_up(need, PIN_CLASS);
            if let Ok(mut slice) = unsafe { ctx.alloc_pinned_with_flags::<u8>(cap, 0) } {
                let ptr = slice.as_mut_ptr().context("pinned host buffer")?;
                return Ok(Host { ptr, cap, backing: Backing::Pinned(slice) });
            }
        }
        let layout = Layout::from_size_align(need, ALIGN).context("host buffer layout")?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        ensure!(!ptr.is_null(), "out of host memory for a {need} byte read buffer");
        Ok(Host { ptr, cap: need, backing: Backing::Heap(layout) })
    }

    fn give(&self, host: Host) {
        if !host.pinned() {
            return;
        }
        let mut free = self.free.lock().unwrap();
        free.push(host);
        if free.len() > self.keep {
            let smallest = free.iter().enumerate().min_by_key(|(_, h)| h.cap).map(|(i, _)| i).unwrap_or(0);
            free.swap_remove(smallest);
        }
    }
}

pub struct Batch {
    lease: Option<Lease>,
    host: Option<Host>,
    pool: Arc<Pool>,
    spans: HashMap<String, (usize, usize)>,
    pub bytes: u64,
}

impl Batch {
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        let &(at, len) = self.spans.get(name)?;
        let host = self.host.as_ref()?;
        Some(unsafe { std::slice::from_raw_parts(host.ptr.add(at), len) })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.spans.keys()
    }

    pub fn pinned(&self) -> bool {
        self.host.as_ref().is_some_and(Host::pinned)
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if let Some(host) = self.host.take() {
            self.pool.give(host);
        }
    }
}

pub struct Source {
    plain: File,
    direct: Option<File>,
    direct_ok: AtomicBool,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn direct_flag() -> i32 {
    libc::O_DIRECT
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn direct_flag() -> i32 {
    0
}

impl Source {
    pub fn open(path: &Path, want_direct: bool) -> Result<Source> {
        let plain = File::open(path).with_context(|| path.display().to_string())?;
        let direct = if want_direct && cfg!(any(target_os = "linux", target_os = "android")) { std::fs::OpenOptions::new().read(true).custom_flags(direct_flag()).open(path).ok() } else { None };
        Ok(Source { direct_ok: AtomicBool::new(direct.is_some()), plain, direct })
    }

    pub fn is_direct(&self) -> bool {
        self.direct_ok.load(Ordering::Relaxed)
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        if self.is_direct() {
            if let Some(f) = &self.direct {
                match f.read_at(buf, offset) {
                    Err(e) if e.raw_os_error() == Some(libc::EINVAL) => self.direct_ok.store(false, Ordering::Relaxed),
                    other => return other,
                }
            }
        }
        self.plain.read_at(buf, offset)
    }

    fn fill(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        let mut got = 0;
        while got < buf.len() {
            match self.read_at(&mut buf[got..], offset + got as u64) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(got)
    }
}

struct Job {
    file: usize,
    from: u64,
    len: usize,
    to: usize,
    need: usize,
}

pub fn read_batch(sources: &[Source], items: &[Item], pool: &Arc<Pool>, opts: &LoaderOptions) -> Result<Batch> {
    let mut spans = HashMap::with_capacity(items.len());
    let mut regions = Vec::with_capacity(items.len());
    let mut total = 0usize;
    for it in items {
        let start = it.offset;
        let from = start & !(ALIGN as u64 - 1);
        let end = start + it.len;
        let rlen = round_up((end - from) as usize, ALIGN);
        spans.insert(it.name.clone(), (total + (start - from) as usize, it.len as usize));
        regions.push((it.file, from, rlen, total, (end - from) as usize));
        total += rlen;
    }
    let host = pool.take(total)?;
    let chunk = (opts.chunk_mb << 20).max(ALIGN) & !(ALIGN - 1);
    let mut jobs = Vec::new();
    for &(file, from, rlen, to, need) in &regions {
        let mut done = 0;
        while done < rlen {
            let len = chunk.min(rlen - done);
            jobs.push(Job { file, from: from + done as u64, len, to: to + done, need: need.saturating_sub(done).min(len) });
            done += len;
        }
    }
    let base = host.ptr as usize;
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..opts.readers.min(jobs.len()).max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    if failure.lock().unwrap().is_some() {
                        break;
                    }
                    let dst = unsafe { std::slice::from_raw_parts_mut((base + job.to) as *mut u8, job.len) };
                    let src = &sources[job.file];
                    match src.fill(dst, job.from) {
                        Ok(got) if got >= job.need => {}
                        Ok(got) => {
                            *failure.lock().unwrap() = Some(anyhow!("short read at offset {}: got {got} of {} bytes, the file is truncated", job.from, job.need));
                        }
                        Err(e) => *failure.lock().unwrap() = Some(anyhow!("read at offset {}: {e}", job.from)),
                    }
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        pool.give(host);
        return Err(e);
    }
    Ok(Batch { lease: None, host: Some(host), pool: pool.clone(), spans, bytes: items.iter().map(|i| i.len).sum() })
}

struct Queue {
    ready: std::collections::BTreeMap<usize, Result<Batch>>,
    taken: usize,
    lanes: usize,
    stop: bool,
}

pub struct Prefetcher {
    queue: Arc<(Mutex<Queue>, std::sync::Condvar)>,
    threads: Vec<JoinHandle<()>>,
    pub direct: bool,
    pub total: u64,
}

impl Prefetcher {
    pub fn start(files: &[PathBuf], groups: Vec<Vec<Item>>, opts: LoaderOptions, ctx: Option<Arc<CudaContext>>, budget: Arc<Budget>) -> Result<Prefetcher> {
        let sources = Arc::new(files.iter().map(|p| Source::open(p, opts.direct)).collect::<Result<Vec<_>>>()?);
        let direct = sources.iter().all(Source::is_direct) && !sources.is_empty();
        let total = groups.iter().flatten().map(|i| i.len).sum();
        let lanes = opts.lanes.min(groups.len()).max(1);
        let pool = Pool::with_keep(ctx.clone().filter(|_| opts.pinned), opts.ahead + opts.workers);
        let queue = Arc::new((Mutex::new(Queue { ready: Default::default(), taken: 0, lanes, stop: false }), std::sync::Condvar::new()));
        let groups = Arc::new(groups);
        let claim = Arc::new(AtomicUsize::new(0));
        let per_lane = LoaderOptions { readers: opts.readers.div_ceil(lanes).max(1), ..opts };
        let mut threads = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let (queue, groups, claim, sources, pool, ctx, budget) = (queue.clone(), groups.clone(), claim.clone(), sources.clone(), pool.clone(), ctx.clone(), budget.clone());
            let t = std::thread::Builder::new()
                .name(format!("specrt-read-{lane}"))
                .spawn(move || {
                    if let Some(ctx) = &ctx {
                        let _ = ctx.bind_to_thread();
                    }
                    let (lock, cv) = &*queue;
                    loop {
                        let g = claim.fetch_add(1, Ordering::Relaxed);
                        if g >= groups.len() {
                            break;
                        }
                        {
                            let mut q = lock.lock().unwrap();
                            while !q.stop && g >= q.taken + opts.ahead {
                                q = cv.wait(q).unwrap();
                            }
                            if q.stop {
                                break;
                            }
                        }
                        let lease = budget.acquire(groups[g].iter().map(|i| i.len).sum());
                        let batch = read_batch(&sources, &groups[g], &pool, &per_lane).map(|mut b| {
                            b.lease = Some(lease);
                            b
                        });
                        let failed = batch.is_err();
                        let mut q = lock.lock().unwrap();
                        q.ready.insert(g, batch);
                        cv.notify_all();
                        if failed || q.stop {
                            break;
                        }
                    }
                    let mut q = lock.lock().unwrap();
                    q.lanes -= 1;
                    cv.notify_all();
                })
                .context("reader thread")?;
            threads.push(t);
        }
        Ok(Prefetcher { queue, threads, direct, total })
    }

    pub fn take(&self, group: usize) -> Result<Batch> {
        let (lock, cv) = &*self.queue;
        let mut q = lock.lock().unwrap();
        loop {
            if let Some(b) = q.ready.remove(&group) {
                q.taken += 1;
                cv.notify_all();
                return b;
            }
            if q.lanes == 0 {
                return Err(anyhow!("the reader threads stopped before weight group {group} was read"));
            }
            q = cv.wait(q).unwrap();
        }
    }

}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        {
            let (lock, cv) = &*self.queue;
            let mut q = lock.lock().unwrap();
            q.stop = true;
            q.ready.clear();
            cv.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        self.queue.0.lock().unwrap().ready.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file_with(dir: &Path, name: &str, len: usize) -> (PathBuf, Vec<u8>) {
        let data: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) ^ (i >> 7)) as u8).collect();
        let path = dir.join(name);
        std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
        (path, data)
    }

    fn items(file: usize, spans: &[(u64, u64)]) -> Vec<Item> {
        spans.iter().enumerate().map(|(i, &(offset, len))| Item { name: format!("t{file}.{i}"), file, offset, len }).collect()
    }

    fn check(dir: &Path, direct: bool) -> bool {
        let (p, data) = file_with(dir, "a.bin", 3 * (1 << 20) + 1234);
        let opts = LoaderOptions { enabled: true, direct, chunk_mb: 1, readers: 3, ahead: 2, workers: 2, lanes: 2, skip_cached: false, pinned: false, host_budget_mb: 3584 };
        let spans = [(5u64, 100u64), (4096, 4096), (10_000, 1_500_000), (3 * (1 << 20) + 1000, 234)];
        let src = Source::open(&p, direct).unwrap();
        let used_direct = src.is_direct();
        let pool = Pool::new(None);
        let it = items(0, &spans);
        let batch = read_batch(&[src], &it, &pool, &opts).unwrap();
        for (i, &(off, len)) in spans.iter().enumerate() {
            assert_eq!(batch.get(&format!("t0.{i}")).unwrap(), &data[off as usize..(off + len) as usize], "span {i}");
        }
        used_direct
    }

    #[test]
    fn reads_unaligned_extents_exactly_with_direct_io_or_buffered() {
        let dir = std::env::temp_dir().join(format!("specrt-loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        check(&dir, true);
        check(&dir, false);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_filesystem_that_refuses_direct_io_falls_back_to_buffered_reads() {
        let dir = PathBuf::from("/dev/shm").join(format!("specrt-loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        check(&dir, true);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_buffered_option_never_opens_direct() {
        let dir = std::env::temp_dir().join(format!("specrt-loader-buf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!check(&dir, false));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_budget_holds_back_a_second_lease_until_the_first_is_dropped_but_never_blocks_a_lone_oversized_one() {
        let b = Budget::new(100);
        let big = b.acquire(500);
        let (tx, rx) = std::sync::mpsc::channel();
        let b2 = b.clone();
        let t = std::thread::spawn(move || {
            let l = b2.acquire(60);
            tx.send(()).unwrap();
            drop(l);
        });
        assert!(rx.recv_timeout(std::time::Duration::from_millis(100)).is_err());
        drop(big);
        rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        t.join().unwrap();
        let x = b.acquire(40);
        let y = b.acquire(60);
        assert_eq!(b.peak(), 500);
        drop((x, y));
    }

    #[test]
    fn a_read_past_the_end_of_a_truncated_file_is_an_error() {
        let dir = std::env::temp_dir().join(format!("specrt-loader-short-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, _) = file_with(&dir, "short.bin", 10_000);
        let src = Source::open(&p, true).unwrap();
        let pool = Pool::new(None);
        let it = items(0, &[(9_000, 5_000)]);
        let err = read_batch(&[src], &it, &pool, &LoaderOptions::default()).err().expect("short read");
        assert!(err.to_string().contains("truncated"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_prefetcher_delivers_groups_in_order_and_drops_cleanly() {
        let dir = std::env::temp_dir().join(format!("specrt-loader-order-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, data) = file_with(&dir, "b.bin", 1 << 20);
        let groups: Vec<Vec<Item>> = (0..6).map(|g| vec![Item { name: "x".into(), file: 0, offset: g * 1000, len: 700 }]).collect();
        let pf = Prefetcher::start(&[p], groups, LoaderOptions::default(), None, Budget::new(1 << 30)).unwrap();
        assert_eq!(pf.total, 6 * 700);
        for g in [1usize, 0, 2] {
            let b = pf.take(g).unwrap();
            assert_eq!(b.get("x").unwrap(), &data[g * 1000..g * 1000 + 700]);
        }
        drop(pf);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
