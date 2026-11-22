use anyhow::{Result, anyhow};
use std::fs::File;
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

pub const BLOCK: usize = 4096;
pub const CHUNK: usize = 4 << 20;
pub const THREADS: usize = 4;

static SHARED: std::sync::OnceLock<Arc<IoPool>> = std::sync::OnceLock::new();

pub fn shared_pool() -> Arc<IoPool> {
    SHARED.get_or_init(|| Arc::new(IoPool::new(THREADS))).clone()
}

pub fn unique_path(dir: &Path, prefix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
    dir.join(format!("{prefix}-{}-{:08x}{:04x}.pages", std::process::id(), nanos, NEXT.fetch_add(1, Ordering::Relaxed) & 0xffff))
}

pub fn remove_stale(dir: &Path, prefix: &str) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    let mut removed = 0;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(prefix).and_then(|r| r.strip_prefix('-')) else { continue };
        if !name.ends_with(".pages") {
            continue;
        }
        let Some(pid) = rest.split('-').next().and_then(|p| p.parse::<u32>().ok()) else { continue };
        if !Path::new(&format!("/proc/{pid}")).exists() && std::fs::remove_file(e.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

pub fn round_up(n: usize) -> usize {
    n.div_ceil(BLOCK) * BLOCK
}

pub trait ReadAt: Send + Sync {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
}

pub struct DirectFile {
    file: File,
    direct: bool,
}

impl DirectFile {
    pub fn open(path: &Path) -> Result<Self> {
        match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) {
            Ok(file) => Ok(DirectFile { file, direct: true }),
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("diskio: O_DIRECT unavailable for {} ({e}), using buffered reads", path.display()));
                Ok(DirectFile { file: File::open(path)?, direct: false })
            }
        }
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    pub fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut b = vec![0u8; len];
        self.read_at(&mut b, offset)?;
        Ok(b)
    }
}

pub fn read_whole(source: &Arc<DirectFile>) -> Result<Vec<u8>> {
    let len = source.len()? as usize;
    let mut raw = vec![0u8; round_up(len) + BLOCK];
    let skew = raw.as_ptr() as usize % BLOCK;
    let start = if skew == 0 { 0 } else { BLOCK - skew };
    if len > 0 {
        let dyn_src: Arc<dyn ReadAt> = source.clone();
        unsafe { shared_pool().read(&dyn_src, 0, raw.as_mut_ptr().add(start), round_up(len)) }.wait()?;
    }
    raw.drain(..start);
    raw.truncate(len);
    Ok(raw)
}

impl ReadAt for DirectFile {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if self.direct && (!(buf.as_ptr() as usize).is_multiple_of(BLOCK) || !buf.len().is_multiple_of(BLOCK) || !(offset as usize).is_multiple_of(BLOCK)) {
            let aligned_off = offset - offset % BLOCK as u64;
            let head = (offset - aligned_off) as usize;
            let len = round_up(head + buf.len());
            let mut tmp = vec![0u8; len + BLOCK];
            let skew = tmp.as_ptr() as usize % BLOCK;
            let start = if skew == 0 { 0 } else { BLOCK - skew };
            let n = self.file.read_at(&mut tmp[start..start + len], aligned_off)?;
            if n < head + buf.len() {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            buf.copy_from_slice(&tmp[start + head..start + head + buf.len()]);
            return Ok(());
        }
        let mut done = 0;
        while done < buf.len() {
            let n = self.file.read_at(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                if self.direct && done > 0 && done % BLOCK != 0 {
                    return Ok(());
                }
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

struct Pending {
    left: usize,
    error: Option<String>,
}

pub struct Ticket {
    state: Arc<(Mutex<Pending>, Condvar)>,
}

impl Ticket {
    pub fn wait(self) -> Result<()> {
        let (m, cv) = &*self.state;
        let mut g = m.lock().unwrap();
        while g.left > 0 {
            g = cv.wait(g).unwrap();
        }
        match g.error.take() {
            Some(e) => Err(anyhow!("disk read failed: {e}")),
            None => Ok(()),
        }
    }

    pub fn is_done(&self) -> bool {
        self.state.0.lock().unwrap().left == 0
    }
}

struct Job {
    source: Arc<dyn ReadAt>,
    offset: u64,
    dst: usize,
    len: usize,
    state: Arc<(Mutex<Pending>, Condvar)>,
}

pub struct IoPool {
    jobs: Mutex<Option<Sender<Job>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl IoPool {
    pub fn new(threads: usize) -> Self {
        let (tx, rx) = channel::<Job>();
        let rx: Arc<Mutex<Receiver<Job>>> = Arc::new(Mutex::new(rx));
        let workers = (0..threads.max(1))
            .map(|_| {
                let rx = rx.clone();
                std::thread::spawn(move || {
                    loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        };
                        let buf = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, job.len) };
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.source.read_at(buf, job.offset))).unwrap_or_else(|_| Err(io::Error::other("reader panicked")));
                        let (m, cv) = &*job.state;
                        let mut g = m.lock().unwrap();
                        if let Err(e) = r {
                            g.error.get_or_insert(e.to_string());
                        }
                        g.left -= 1;
                        if g.left == 0 {
                            cv.notify_all();
                        }
                    }
                })
            })
            .collect();
        IoPool { jobs: Mutex::new(Some(tx)), workers: Mutex::new(workers) }
    }

    /// # Safety
    /// `dst` must stay valid and unaliased for `len` bytes until the ticket is waited on.
    pub unsafe fn read(&self, source: &Arc<dyn ReadAt>, offset: u64, dst: *mut u8, len: usize) -> Ticket {
        unsafe { self.read_chunked(source, offset, dst, len, CHUNK) }
    }

    /// # Safety
    /// Same as `read`.
    pub unsafe fn read_chunked(&self, source: &Arc<dyn ReadAt>, offset: u64, dst: *mut u8, len: usize, chunk: usize) -> Ticket {
        let chunks = len.div_ceil(chunk).max(1);
        let state = Arc::new((Mutex::new(Pending { left: chunks, error: None }), Condvar::new()));
        let guard = self.jobs.lock().unwrap();
        let tx = guard.as_ref().expect("pool is running");
        for c in 0..chunks {
            let at = c * chunk;
            let n = chunk.min(len - at);
            tx.send(Job { source: source.clone(), offset: offset + at as u64, dst: dst as usize + at, len: n, state: state.clone() }).expect("workers run");
        }
        Ticket { state }
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        self.jobs.lock().unwrap().take();
        for w in self.workers.lock().unwrap().drain(..) {
            let _ = w.join();
        }
    }
}

pub struct Appender {
    file: File,
    path: PathBuf,
    at: u64,
}

impl Appender {
    pub fn create(path: PathBuf) -> Result<Self> {
        let file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path)?;
        Ok(Appender { file, path, at: 0 })
    }

    pub fn finish_exact(self, len: u64) -> Result<PathBuf> {
        self.file.set_len(len)?;
        self.file.sync_all()?;
        Ok(self.path)
    }


    pub fn append(&mut self, bytes: &[u8]) -> Result<u64> {
        use std::io::Write;
        let start = self.at;
        self.file.write_all(bytes)?;
        let pad = round_up(bytes.len()) - bytes.len();
        self.file.write_all(&vec![0u8; pad])?;
        self.at += round_up(bytes.len()) as u64;
        Ok(start)
    }

    pub fn finish(self) -> Result<(PathBuf, u64)> {
        self.file.sync_all()?;
        Ok((self.path, self.at))
    }
}

pub struct SwapFile {
    path: PathBuf,
    source: Arc<dyn ReadAt>,
    len: usize,
}

impl SwapFile {
    pub fn write(dir: &Path, bytes: &[u8]) -> Result<Self> {
        static SWEPT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
        std::fs::create_dir_all(dir)?;
        {
            let mut swept = SWEPT.lock().unwrap();
            if !swept.iter().any(|d| d == dir) {
                remove_stale(dir, "swap");
                swept.push(dir.to_path_buf());
            }
        }
        let mut w = Appender::create(unique_path(dir, "swap"))?;
        if let Err(e) = w.append(bytes) {
            let _ = std::fs::remove_file(&w.path);
            return Err(e);
        }
        let (path, _) = w.finish()?;
        let source: Arc<dyn ReadAt> = Arc::new(DirectFile::open(&path)?);
        Ok(SwapFile { path, source, len: bytes.len() })
    }

    pub fn with_source(mut self, wrap: impl FnOnce(Arc<dyn ReadAt>) -> Arc<dyn ReadAt>) -> Self {
        self.source = wrap(self.source.clone());
        self
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn padded_len(&self) -> usize {
        round_up(self.len)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn read_into(&self, dst: &mut [u8]) -> Result<()> {
        anyhow::ensure!(dst.len() >= self.len, "swap read buffer of {} bytes for {}", dst.len(), self.len);
        let padded = self.padded_len();
        let n = if dst.len() >= padded && (dst.as_ptr() as usize).is_multiple_of(BLOCK) { padded } else { self.len };
        if n == 0 {
            return Ok(());
        }
        unsafe { shared_pool().read(&self.source, 0, dst.as_mut_ptr(), n) }.wait()
    }
}

impl Drop for SwapFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    struct Throttled {
        data: Vec<u8>,
        bytes_per_sec: f64,
        calls: AtomicUsize,
    }

    impl ReadAt for Throttled {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_secs_f64(buf.len() as f64 / self.bytes_per_sec));
            buf.copy_from_slice(&self.data[offset as usize..offset as usize + buf.len()]);
            Ok(())
        }
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 + i / 4096) as u8).collect()
    }

    #[test]
    fn chunked_read_is_identical_and_split_in_four_megabyte_pieces() {
        let data = pattern(10 << 20);
        let src = Arc::new(Throttled { data: data.clone(), bytes_per_sec: 1e12, calls: AtomicUsize::new(0) });
        let pool = IoPool::new(3);
        let mut out = vec![0u8; 9 << 20];
        let dyn_src: Arc<dyn ReadAt> = src.clone();
        unsafe { pool.read(&dyn_src, 4096, out.as_mut_ptr(), out.len()) }.wait().unwrap();
        assert_eq!(out, data[4096..4096 + (9 << 20)]);
        assert_eq!(src.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn parallel_workers_overlap_a_throttled_drive() {
        let data = pattern(16 << 20);
        let src: Arc<dyn ReadAt> = Arc::new(Throttled { data, bytes_per_sec: 400e6, calls: AtomicUsize::new(0) });
        let pool = IoPool::new(4);
        let mut out = vec![0u8; 16 << 20];
        let t = Instant::now();
        unsafe { pool.read(&src, 0, out.as_mut_ptr(), out.len()) }.wait().unwrap();
        let serial = (16 << 20) as f64 / 400e6;
        assert!(t.elapsed().as_secs_f64() < serial * 0.6, "{:?}", t.elapsed());
    }

    #[test]
    #[ignore = "needs DISKIO_BENCH_DIR on the drive to measure"]
    fn drive_bandwidth_by_threads_and_chunk() {
        let dir = PathBuf::from(std::env::var("DISKIO_BENCH_DIR").unwrap());
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diskio-bench.pages");
        let total: usize = 4 << 30;
        let block = pattern(64 << 20);
        let mut w = Appender::create(path.clone()).unwrap();
        for _ in 0..total / block.len() {
            w.append(&block).unwrap();
        }
        w.finish().unwrap();
        let f = DirectFile::open(&path).unwrap();
        eprintln!("O_DIRECT in use: {}", f.is_direct());
        let src: Arc<dyn ReadAt> = Arc::new(f);
        let mut out = vec![0u8; 256 << 20];
        let ptr = ((out.as_mut_ptr() as usize).div_ceil(BLOCK) * BLOCK) as *mut u8;
        let len = 128 << 20;
        for threads in [1, 2, 4, 8] {
            let pool = IoPool::new(threads);
            for chunk_mb in [1usize, 2, 4] {
                let t = Instant::now();
                let mut read = 0usize;
                while read < total {
                    unsafe { pool.read_chunked(&src, read as u64, ptr, len, chunk_mb << 20) }.wait().unwrap();
                    read += len;
                }
                eprintln!("threads {threads} chunk {chunk_mb} MB: {:.2} GB/s", total as f64 / t.elapsed().as_secs_f64() / 1e9);
            }
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn unaligned_direct_reads_near_block_boundaries_are_identical() {
        let dir = std::env::temp_dir().join(format!("diskio-unal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blobs");
        let data = pattern(40000);
        let mut w = Appender::create(path.clone()).unwrap();
        w.append(&data).unwrap();
        w.finish().unwrap();
        let f = DirectFile::open(&path).unwrap();
        for (off, len) in [(1usize, 4095usize), (4095, 4097), (100, 8000), (4000, 12000), (39000, 1000)] {
            let mut got = vec![0u8; len];
            f.read_at(&mut got, off as u64).unwrap();
            assert_eq!(got, data[off..off + len], "{off}+{len}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stale_files_of_dead_pids_are_removed_and_live_ones_kept() {
        let dir = std::env::temp_dir().join(format!("diskio-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dead = dir.join("layers-4294967290-deadbeef0001.pages");
        let live = unique_path(&dir, "layers");
        let other = dir.join("experts-4294967290-deadbeef0001.pages");
        for p in [&dead, &live, &other] {
            std::fs::write(p, b"x").unwrap();
        }
        assert_eq!(remove_stale(&dir, "layers"), 1);
        assert!(!dead.exists() && live.exists() && other.exists());
        assert_ne!(unique_path(&dir, "layers"), unique_path(&dir, "layers"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn swap_file_round_trips_through_a_throttled_drive_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("diskio-swap-{}", std::process::id()));
        let stale = dir.join("swap-4294967290-deadbeef0001.pages");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&stale, b"x").unwrap();
        let data = pattern(9 << 20 | 123);
        let calls = Arc::new(AtomicUsize::new(0));
        let file = SwapFile::write(&dir, &data).unwrap();
        assert!(!stale.exists());
        let throttle = calls.clone();
        let file = file.with_source(move |inner| {
            struct Slow(Arc<dyn ReadAt>, Arc<AtomicUsize>);
            impl ReadAt for Slow {
                fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
                    self.1.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_secs_f64(buf.len() as f64 / 20e6));
                    self.0.read_at(buf, offset)
                }
            }
            Arc::new(Slow(inner, throttle))
        });
        let mut raw = vec![0u8; file.padded_len() + BLOCK];
        let start = (BLOCK - raw.as_ptr() as usize % BLOCK) % BLOCK;
        let t = Instant::now();
        file.read_into(&mut raw[start..]).unwrap();
        let serial = file.padded_len() as f64 / 20e6;
        assert_eq!(raw[start..start + data.len()], data[..]);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(t.elapsed().as_secs_f64() < serial * 0.8, "{:?}", t.elapsed());
        let mut small = vec![0u8; data.len()];
        file.read_into(&mut small).unwrap();
        assert_eq!(small, data);
        let path = file.path().to_path_buf();
        assert!(path.exists());
        drop(file);
        assert!(!path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_whole_returns_exactly_the_appended_bytes() {
        let dir = std::env::temp_dir().join(format!("diskio-whole-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        let data = pattern(10000);
        let mut w = Appender::create(path.clone()).unwrap();
        w.append(&data).unwrap();
        w.finish_exact(data.len() as u64).unwrap();
        assert_eq!(read_whole(&Arc::new(DirectFile::open(&path).unwrap())).unwrap(), data);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_chunk_is_an_error() {
        struct Bad;
        impl ReadAt for Bad {
            fn read_at(&self, _: &mut [u8], _: u64) -> io::Result<()> {
                Err(io::Error::other("boom"))
            }
        }
        let src: Arc<dyn ReadAt> = Arc::new(Bad);
        let pool = IoPool::new(1);
        let mut out = vec![0u8; 8192];
        assert!(unsafe { pool.read(&src, 0, out.as_mut_ptr(), out.len()) }.wait().is_err());
    }

    #[test]
    fn direct_file_round_trips_padded_blobs() {
        let dir = std::env::temp_dir().join(format!("diskio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blobs");
        let a = pattern(5000);
        let b = pattern(9000);
        let mut w = Appender::create(path.clone()).unwrap();
        assert_eq!(w.append(&a).unwrap(), 0);
        assert_eq!(w.append(&b).unwrap(), 8192);
        w.finish().unwrap();
        let f = DirectFile::open(&path).unwrap();
        let mut got = vec![0u8; 9000];
        f.read_at(&mut got, 8192).unwrap();
        assert_eq!(got, b);
        let mut got = vec![0u8; 5000];
        f.read_at(&mut got, 0).unwrap();
        assert_eq!(got, a);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
