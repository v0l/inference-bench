use anyhow::{Result, anyhow};
use std::fs::File;
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

pub const BLOCK: usize = 4096;
pub const CHUNK: usize = 4 << 20;

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
            Err(_) => Ok(DirectFile { file: File::open(path)?, direct: false }),
        }
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }
}

impl ReadAt for DirectFile {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if self.direct && (buf.as_ptr() as usize % BLOCK != 0 || buf.len() % BLOCK != 0 || offset as usize % BLOCK != 0) {
            let mut tmp = vec![0u8; round_up(buf.len()) + BLOCK];
            let skew = tmp.as_ptr() as usize % BLOCK;
            let start = if skew == 0 { 0 } else { BLOCK - skew };
            let aligned_off = offset - offset % BLOCK as u64;
            let head = (offset - aligned_off) as usize;
            let len = round_up(head + buf.len());
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
                        let r = job.source.read_at(buf, job.offset);
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
        let chunks = len.div_ceil(CHUNK).max(1);
        let state = Arc::new((Mutex::new(Pending { left: chunks, error: None }), Condvar::new()));
        let guard = self.jobs.lock().unwrap();
        let tx = guard.as_ref().expect("pool is running");
        for c in 0..chunks {
            let at = c * CHUNK;
            let n = CHUNK.min(len - at);
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
        Ok(Appender { file: File::create(&path)?, path, at: 0 })
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
