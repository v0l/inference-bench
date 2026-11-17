use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const BLOCK: u64 = 4 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wanted {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub label: &'static str,
    pub url: String,
    pub files: Vec<Wanted>,
}

impl Source {
    pub fn nvidia_cuda_13() -> Source {
        Source {
            label: "13.0.88",
            url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-13.0.88-archive.zip".into(),
            files: vec![
                Wanted { name: "nvrtc-builtins64_130.dll", size: 4_471_328, sha256: "cd4ce3e0351b0d10f43b07c667782028420e03858999fab65207c40a28546d19" },
                Wanted { name: "nvrtc64_130_0.dll", size: 90_965_024, sha256: "28d1021a82003f4aed742ea36ece1d595e32a107f915dfbd07d3ee8764858bce" },
            ],
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

pub fn install_dir(cache_dir: &Path, source: &Source) -> PathBuf {
    cache_dir.join(format!("nvrtc-{}", source.label))
}

pub fn is_installed(dir: &Path, source: &Source) -> bool {
    source.files.iter().all(|f| std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.len() == f.size))
}

struct Ranged {
    url: String,
    len: u64,
    pos: u64,
    block: Vec<u8>,
    block_at: u64,
    served: u64,
}

fn http_range(url: &str, from: u64, to_inclusive: u64) -> Result<(Vec<u8>, Option<u64>)> {
    let resp = ureq::get(url).header("Range", &format!("bytes={from}-{to_inclusive}")).call().map_err(|e| anyhow::anyhow!("{url}: {e}"))?;
    ensure!(resp.status().as_u16() == 206, "{url}: the server ignored the byte range (status {})", resp.status());
    let total = resp.headers().get("content-range").and_then(|v| v.to_str().ok()).and_then(|v| v.rsplit('/').next()).and_then(|t| t.parse::<u64>().ok());
    let mut out = Vec::new();
    resp.into_body().into_reader().take(to_inclusive - from + 1).read_to_end(&mut out).with_context(|| url.to_string())?;
    Ok((out, total))
}

impl Ranged {
    fn open(url: &str) -> Result<Ranged> {
        let (_, total) = http_range(url, 0, 0)?;
        let len = total.with_context(|| format!("{url}: no content-range in the reply"))?;
        Ok(Ranged { url: url.to_string(), len, pos: 0, block: Vec::new(), block_at: 0, served: 1 })
    }
}

impl Read for Ranged {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let inside = self.pos >= self.block_at && self.pos < self.block_at + self.block.len() as u64;
        if !inside {
            let end = (self.pos + BLOCK - 1).min(self.len - 1);
            let (bytes, _) = http_range(&self.url, self.pos, end).map_err(std::io::Error::other)?;
            self.served += bytes.len() as u64;
            self.block = bytes;
            self.block_at = self.pos;
        }
        let start = (self.pos - self.block_at) as usize;
        let n = buf.len().min(self.block.len() - start);
        buf[..n].copy_from_slice(&self.block[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Ranged {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let next = match to {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if next < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before the start"));
        }
        self.pos = next as u64;
        Ok(self.pos)
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn fetch_entry(entry: &mut impl Read, want: &Wanted, dir: &Path, done: &mut u64, total: u64) -> Result<()> {
    let part = dir.join(format!("{}.part", want.name));
    let mut out = std::fs::File::create(&part).with_context(|| part.display().to_string())?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut written = 0u64;
    loop {
        let n = entry.read(&mut buf).with_context(|| format!("downloading {}", want.name))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        written += n as u64;
        *done += n as u64;
        crate::progress::report(crate::progress::Stage::CudaCompiler, *done, total);
    }
    drop(out);
    let got = hex(&hash.finalize());
    if written != want.size || got != want.sha256 {
        let _ = std::fs::remove_file(&part);
        bail!("{} did not match the pinned checksum (got {got}, {written} bytes): refusing to use it", want.name);
    }
    std::fs::rename(&part, dir.join(want.name))?;
    Ok(())
}

pub fn download(cache_dir: &Path, source: &Source) -> Result<PathBuf> {
    let dir = install_dir(cache_dir, source);
    if is_installed(&dir, source) {
        return Ok(dir);
    }
    std::fs::create_dir_all(&dir)?;
    let total = source.total_bytes();
    crate::progress::report(crate::progress::Stage::CudaCompiler, 0, total);
    let mut archive = zip::ZipArchive::new(Ranged::open(&source.url)?).with_context(|| format!("{}: not a readable zip", source.url))?;
    let mut done = 0u64;
    for want in &source.files {
        if std::fs::metadata(dir.join(want.name)).is_ok_and(|m| m.len() == want.size) {
            done += want.size;
            continue;
        }
        let index = (0..archive.len())
            .find(|&i| archive.name_for_index(i).is_some_and(|n| n.rsplit('/').next() == Some(want.name) && n.contains("/bin/")))
            .with_context(|| format!("{} is not in {}", want.name, source.url))?;
        let mut entry = archive.by_index(index)?;
        fetch_entry(&mut entry, want, &dir, &mut done, total)?;
    }
    Ok(dir)
}

#[cfg(windows)]
pub fn load_from(dir: &Path, source: &Source) -> Result<()> {
    use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library};
    for want in &source.files {
        let lib = unsafe { Library::load_with_flags(dir.join(want.name), LOAD_WITH_ALTERED_SEARCH_PATH) }.with_context(|| format!("loading {}", dir.join(want.name).display()))?;
        std::mem::forget(lib);
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn load_from(_dir: &Path, _source: &Source) -> Result<()> {
    Ok(())
}

pub fn provide(cache_dir: &Path, source: &Source) -> Result<()> {
    let dir = download(cache_dir, source)?;
    load_from(&dir, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn filler(seed: u8, n: usize) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
    }

    fn archive(parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        let mut w = zip::ZipWriter::new(&mut out);
        let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, body) in parts {
            w.start_file(*name, options).unwrap();
            w.write_all(body).unwrap();
        }
        w.finish().unwrap();
        out.into_inner()
    }

    struct Server {
        url: String,
        sent: Arc<AtomicU64>,
        requests: Arc<AtomicU64>,
    }

    fn serve(zip_bytes: Vec<u8>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/pkg.zip", listener.local_addr().unwrap());
        let (sent, requests) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let (s, r) = (sent.clone(), requests.clone());
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut range = None;
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("range: bytes=") {
                        let (a, b) = v.trim().split_once('-').unwrap();
                        range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()));
                    }
                }
                r.fetch_add(1, Ordering::SeqCst);
                let (a, b) = range.unwrap();
                let b = b.min(zip_bytes.len() - 1);
                let body = &zip_bytes[a..=b];
                s.fetch_add(body.len() as u64, Ordering::SeqCst);
                let head = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {a}-{b}/{}\r\nConnection: close\r\n\r\n", body.len(), zip_bytes.len());
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });
        Server { url, sent, requests }
    }

    fn sample() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (filler(1, 300_000), filler(2, 40_000), filler(3, 6_000_000))
    }

    fn source(url: &str, big: &[u8], small: &[u8]) -> Source {
        let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
        Source {
            label: "test",
            url: url.to_string(),
            files: vec![
                Wanted { name: "nvrtc-builtins64_130.dll", size: small.len() as u64, sha256: leak(hex(&Sha256::digest(small))) },
                Wanted { name: "nvrtc64_130_0.dll", size: big.len() as u64, sha256: leak(hex(&Sha256::digest(big))) },
            ],
        }
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nvrtc-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn only_the_two_wanted_files_are_downloaded_from_a_large_archive() {
        let (big, small, unrelated) = sample();
        let zip_bytes = archive(&[("p/LICENSE", b"license"), ("p/bin/x64/nvrtc64_130_0.alt.dll", &unrelated), ("p/bin/x64/nvrtc64_130_0.dll", &big), ("p/bin/x64/nvrtc-builtins64_130.dll", &small)]);
        let total = zip_bytes.len() as u64;
        let server = serve(zip_bytes);
        let cache = temp("partial");
        let src = source(&server.url, &big, &small);
        let dir = download(&cache, &src).unwrap();
        assert_eq!(std::fs::read(dir.join("nvrtc64_130_0.dll")).unwrap(), big);
        assert_eq!(std::fs::read(dir.join("nvrtc-builtins64_130.dll")).unwrap(), small);
        assert!(!dir.join("nvrtc64_130_0.alt.dll").exists());
        let sent = server.sent.load(Ordering::SeqCst);
        assert!(sent < total - unrelated.len() as u64 / 2, "sent {sent} of {total}");
        let _ = std::fs::remove_dir_all(cache);
    }

    #[test]
    fn a_second_call_reuses_the_install_and_makes_no_request() {
        let (big, small, _) = sample();
        let zip_bytes = archive(&[("p/bin/x64/nvrtc64_130_0.dll", &big), ("p/bin/x64/nvrtc-builtins64_130.dll", &small)]);
        let server = serve(zip_bytes);
        let cache = temp("reuse");
        let src = source(&server.url, &big, &small);
        download(&cache, &src).unwrap();
        let after_first = server.requests.load(Ordering::SeqCst);
        assert!(after_first > 0 && is_installed(&install_dir(&cache, &src), &src));
        download(&cache, &src).unwrap();
        assert_eq!(server.requests.load(Ordering::SeqCst), after_first);
        let _ = std::fs::remove_dir_all(cache);
    }

    #[test]
    fn a_file_with_the_wrong_checksum_is_refused_and_not_kept() {
        let (big, small, _) = sample();
        let zip_bytes = archive(&[("p/bin/x64/nvrtc64_130_0.dll", &big), ("p/bin/x64/nvrtc-builtins64_130.dll", &small)]);
        let server = serve(zip_bytes);
        let cache = temp("bad");
        let mut src = source(&server.url, &big, &small);
        src.files[1].sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = download(&cache, &src).unwrap_err().to_string();
        assert!(err.contains("pinned checksum"), "{err}");
        let dir = install_dir(&cache, &src);
        assert!(!dir.join("nvrtc64_130_0.dll").exists() && !dir.join("nvrtc64_130_0.dll.part").exists());
        assert!(!is_installed(&dir, &src));
        let _ = std::fs::remove_dir_all(cache);
    }

    #[test]
    fn progress_counts_the_bytes_of_the_wanted_files_only() {
        let (big, small, _) = sample();
        let zip_bytes = archive(&[("p/bin/x64/nvrtc64_130_0.dll", &big), ("p/bin/x64/nvrtc-builtins64_130.dll", &small)]);
        let server = serve(zip_bytes);
        let cache = temp("progress");
        let src = source(&server.url, &big, &small);
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let s = seen.clone();
        crate::progress::with_sink(Some(crate::progress::Sink::new(move |p| s.lock().unwrap().push(p))), || download(&cache, &src).unwrap());
        let seen = seen.lock().unwrap();
        assert!(seen.iter().all(|p| p.stage == crate::progress::Stage::CudaCompiler && p.total == src.total_bytes()));
        assert_eq!(seen.last().unwrap().done, src.total_bytes());
        let _ = std::fs::remove_dir_all(cache);
    }

    #[test]
    fn the_pinned_nvidia_source_names_a_windows_zip_and_both_libraries() {
        let s = Source::nvidia_cuda_13();
        assert!(s.url.starts_with("https://developer.download.nvidia.com/") && s.url.ends_with("windows-x86_64-13.0.88-archive.zip"));
        assert_eq!(s.files.iter().map(|f| f.name).collect::<Vec<_>>(), ["nvrtc-builtins64_130.dll", "nvrtc64_130_0.dll"]);
        assert!(s.files.iter().all(|f| f.sha256.len() == 64));
    }
}
