use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"SPKVS\0\0\x01";
const FOOTER: &[u8; 8] = b"SPKVSEND";
const HEADER_LEN: usize = 8 + 4 * 4 + 8 * 3 + 32 * 3 + 4 * 2 + 4;
const LAST_USED_AT: usize = 8 + 4 * 4 + 8 * 2;
const SUFFIX: &str = "kvs";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionOptions {
    pub enabled: bool,
    pub dir: String,
    pub idle_seconds: u64,
    pub ttl_hours: u64,
    pub min_tokens: usize,
    pub max_bytes: u64,
    pub keep_after_restore: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        SessionOptions { enabled: false, dir: String::new(), idle_seconds: 300, ttl_hours: 8, min_tokens: 4096, max_bytes: 0, keep_after_restore: false }
    }
}

impl SessionOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.idle_seconds > 0, "[sessions] idle_seconds must be positive");
        ensure!(self.ttl_hours > 0, "[sessions] ttl_hours must be positive");
        Ok(())
    }

    pub fn ttl_seconds(&self) -> u64 {
        self.ttl_hours * 3600
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compat {
    pub model_hash: [u8; 32],
    pub spec_hash: [u8; 32],
    pub kv_dtype: u32,
    pub layers: u32,
    pub block_size: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    pub scope: [u8; 32],
    pub tokens: Vec<u32>,
    pub sections: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    pub scope: [u8; 32],
    pub tokens: usize,
    pub bytes: u64,
    pub created: u64,
    pub last_used: u64,
    pub stripped: bool,
}

impl Entry {
    pub fn expires(&self, ttl: u64) -> u64 {
        self.last_used + ttl
    }
}

#[derive(Default)]
pub struct Metrics {
    pub snapshots: AtomicU64,
    pub snapshot_bytes: AtomicU64,
    pub snapshot_failures: AtomicU64,
    pub restores_ok: AtomicU64,
    pub restores_miss: AtomicU64,
    pub restores_corrupt: AtomicU64,
    pub restore_micros: AtomicU64,
    pub prefill_tokens_saved: AtomicU64,
}

pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    });
    let mut c = !crc;
    for &b in data {
        c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

pub fn session_id(compat: &Compat, scope: &[u8; 32], tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(compat.model_hash);
    h.update(compat.spec_hash);
    h.update(scope);
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    h.finalize()[..16].iter().map(|b| format!("{b:02x}")).collect()
}

struct Header {
    compat: Compat,
    scope: [u8; 32],
    tokens: u64,
    created: u64,
    last_used: u64,
    sections: u32,
    stripped: bool,
}

fn encode_header(h: &Header) -> Vec<u8> {
    let mut b = Vec::with_capacity(HEADER_LEN);
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&h.compat.kv_dtype.to_le_bytes());
    b.extend_from_slice(&h.compat.layers.to_le_bytes());
    b.extend_from_slice(&h.compat.block_size.to_le_bytes());
    b.extend_from_slice(&h.tokens.to_le_bytes());
    b.extend_from_slice(&h.created.to_le_bytes());
    b.extend_from_slice(&h.last_used.to_le_bytes());
    b.extend_from_slice(&h.compat.model_hash);
    b.extend_from_slice(&h.compat.spec_hash);
    b.extend_from_slice(&h.scope);
    b.extend_from_slice(&h.sections.to_le_bytes());
    b.extend_from_slice(&(h.stripped as u32).to_le_bytes());
    let crc = crc32(&b);
    b.extend_from_slice(&crc.to_le_bytes());
    b
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

fn decode_header(b: &[u8]) -> Result<Header> {
    ensure!(b.len() == HEADER_LEN, "short header");
    ensure!(&b[..8] == MAGIC, "bad magic");
    ensure!(u32_at(b, 8) == VERSION, "format version {} is not {VERSION}", u32_at(b, 8));
    ensure!(crc32(&b[..HEADER_LEN - 4]) == u32_at(b, HEADER_LEN - 4), "header checksum mismatch");
    let mut model_hash = [0u8; 32];
    let mut spec_hash = [0u8; 32];
    let mut scope = [0u8; 32];
    model_hash.copy_from_slice(&b[48..80]);
    spec_hash.copy_from_slice(&b[80..112]);
    scope.copy_from_slice(&b[112..144]);
    Ok(Header {
        scope,
        compat: Compat { model_hash, spec_hash, kv_dtype: u32_at(b, 12), layers: u32_at(b, 16), block_size: u32_at(b, 20) },
        tokens: u64_at(b, 24),
        created: u64_at(b, 32),
        last_used: u64_at(b, 40),
        sections: u32_at(b, 144),
        stripped: u32_at(b, 148) != 0,
    })
}

fn encode(compat: &Compat, session: &Session, created: u64, last_used: u64, stripped: bool) -> Vec<u8> {
    let sections = if stripped { &[][..] } else { &session.sections[..] };
    let header = Header { compat: compat.clone(), scope: session.scope, tokens: session.tokens.len() as u64, created, last_used, sections: sections.len() as u32, stripped };
    let mut out = encode_header(&header);
    let mut tok = Vec::with_capacity(session.tokens.len() * 4);
    for t in &session.tokens {
        tok.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&tok);
    out.extend_from_slice(&crc32(&tok).to_le_bytes());
    for s in sections {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s);
        out.extend_from_slice(&crc32(s).to_le_bytes());
    }
    out.extend_from_slice(FOOTER);
    let total = out.len() as u64 + 8;
    out.extend_from_slice(&total.to_le_bytes());
    out
}

fn read_exact_at(f: &mut impl Read, len: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; len];
    f.read_exact(&mut b)?;
    Ok(b)
}

fn read_tokens(f: &mut impl Read, count: u64, file_len: u64) -> Result<Vec<u32>> {
    ensure!(count.saturating_mul(4) < file_len, "token count exceeds file size");
    let raw = read_exact_at(f, count as usize * 4)?;
    let crc = read_exact_at(f, 4)?;
    ensure!(crc32(&raw) == u32_at(&crc, 0), "token checksum mismatch");
    Ok(raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
}

fn read_full(path: &Path) -> Result<(Header, Session)> {
    let file = std::sync::Arc::new(crate::diskio::DirectFile::open(path)?);
    let raw = crate::diskio::read_whole(&file)?;
    let file_len = raw.len() as u64;
    ensure!(file_len >= (HEADER_LEN + 4 + 16) as u64, "file too short");
    let mut f = std::io::Cursor::new(raw);
    let header = decode_header(&read_exact_at(&mut f, HEADER_LEN)?)?;
    let tokens = read_tokens(&mut f, header.tokens, file_len)?;
    let mut sections = Vec::new();
    let mut pos = f.position();
    for i in 0..header.sections {
        let len = u64::from_le_bytes(read_exact_at(&mut f, 8)?.try_into().expect("8 bytes"));
        ensure!(pos + 8 + len + 4 <= file_len, "section {i} runs past the end of the file");
        let body = read_exact_at(&mut f, len as usize)?;
        let crc = read_exact_at(&mut f, 4)?;
        ensure!(crc32(&body) == u32_at(&crc, 0), "section {i} checksum mismatch");
        pos += 8 + len + 4;
        sections.push(body);
    }
    let footer = read_exact_at(&mut f, 16)?;
    ensure!(&footer[..8] == FOOTER && u64_at(&footer, 8) == file_len, "bad footer");
    ensure!(f.position() == file_len, "trailing bytes");
    let scope = header.scope;
    Ok((header, Session { scope, tokens, sections }))
}

fn read_light(path: &Path) -> Result<(Header, Vec<u32>, u64)> {
    let mut f = File::open(path)?;
    let file_len = f.metadata()?.len();
    ensure!(file_len >= (HEADER_LEN + 4 + 16) as u64, "file too short");
    let header = decode_header(&read_exact_at(&mut f, HEADER_LEN)?)?;
    let tokens = read_tokens(&mut f, header.tokens, file_len)?;
    f.seek(SeekFrom::End(-16))?;
    let footer = read_exact_at(&mut f, 16)?;
    ensure!(&footer[..8] == FOOTER && u64_at(&footer, 8) == file_len, "bad footer");
    Ok((header, tokens, file_len))
}

struct Indexed {
    entry: Entry,
    scope: [u8; 32],
    tokens: Vec<u32>,
}

pub struct SessionStore {
    dir: PathBuf,
    compat: Compat,
    index: Mutex<HashMap<String, Indexed>>,
    pub metrics: Metrics,
}

impl SessionStore {
    pub fn open(root: &Path, compat: Compat) -> Result<Self> {
        let model: String = compat.model_hash.iter().take(8).map(|b| format!("{b:02x}")).collect();
        let dir = root.join("sessions").join(model);
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        restrict(&dir, 0o700);
        let store = SessionStore { dir, compat, index: Mutex::new(HashMap::new()), metrics: Metrics::default() };
        store.rescan()?;
        Ok(store)
    }

    pub fn compat(&self) -> &Compat {
        &self.compat
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.{SUFFIX}"))
    }

    pub fn rescan(&self) -> Result<()> {
        let mut index = HashMap::new();
        for item in fs::read_dir(&self.dir)? {
            let path = item?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
            if name.starts_with(".tmp-") {
                let _ = fs::remove_file(&path);
                continue;
            }
            let Some(id) = name.strip_suffix(&format!(".{SUFFIX}")) else { continue };
            match read_light(&path) {
                Ok((h, tokens, bytes)) if self.accepts(&h) => {
                    let entry = Entry { id: id.to_string(), scope: h.scope, tokens: tokens.len(), bytes, created: h.created, last_used: h.last_used, stripped: h.stripped };
                    index.insert(id.to_string(), Indexed { entry, scope: h.scope, tokens });
                }
                Ok(_) => {
                    tracing::info!(file = %path.display(), "session snapshot is for another spec or dtype, removing");
                    let _ = fs::remove_file(&path);
                }
                Err(e) => {
                    tracing::warn!(file = %path.display(), error = %format!("{e:#}"), "unreadable session snapshot, removing");
                    let _ = fs::remove_file(&path);
                }
            }
        }
        *self.index.lock().expect("index") = index;
        Ok(())
    }

    fn accepts(&self, h: &Header) -> bool {
        h.compat == self.compat
    }

    pub fn list(&self) -> Vec<Entry> {
        let mut v: Vec<Entry> = self.index.lock().expect("index").values().map(|i| i.entry.clone()).collect();
        v.sort_by(|a, b| b.last_used.cmp(&a.last_used).then(a.id.cmp(&b.id)));
        v
    }

    pub fn total_bytes(&self) -> u64 {
        self.index.lock().expect("index").values().map(|i| i.entry.bytes).sum()
    }

    pub fn write(&self, session: &Session, now: u64) -> Result<Entry> {
        let result = self.write_inner(session, now, false);
        match &result {
            Ok(e) => {
                self.metrics.snapshots.fetch_add(1, Ordering::Relaxed);
                self.metrics.snapshot_bytes.fetch_add(e.bytes, Ordering::Relaxed);
            }
            Err(e) => {
                self.metrics.snapshot_failures.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(error = %format!("{e:#}"), "session snapshot not written, continuing without it");
            }
        }
        result
    }

    fn write_inner(&self, session: &Session, now: u64, stripped: bool) -> Result<Entry> {
        let id = session_id(&self.compat, &session.scope, &session.tokens);
        let created = self.index.lock().expect("index").get(&id).map(|i| i.entry.created).unwrap_or(now);
        let bytes = encode(&self.compat, session, created, now, stripped);
        let tmp = crate::diskio::unique_path(&self.dir, ".tmp");
        let written = (|| -> Result<()> {
            let mut w = crate::diskio::Appender::create(tmp.clone())?;
            w.append(&bytes)?;
            w.finish_exact(bytes.len() as u64)?;
            fs::rename(&tmp, self.path(&id))?;
            File::open(&self.dir).and_then(|d| d.sync_all())?;
            Ok(())
        })();
        if let Err(e) = written {
            let _ = fs::remove_file(&tmp);
            return Err(e).context("write session snapshot");
        }
        let entry = Entry { id: id.clone(), scope: session.scope, tokens: session.tokens.len(), bytes: bytes.len() as u64, created, last_used: now, stripped };
        self.index.lock().expect("index").insert(id, Indexed { entry: entry.clone(), scope: session.scope, tokens: session.tokens.clone() });
        Ok(entry)
    }

    pub fn find_prefix(&self, scope: &[u8; 32], prompt: &[u32]) -> Option<Entry> {
        let index = self.index.lock().expect("index");
        index
            .values()
            .filter(|i| &i.scope == scope && !i.entry.stripped && i.tokens.len() < prompt.len() && prompt.starts_with(&i.tokens))
            .max_by_key(|i| i.tokens.len())
            .map(|i| i.entry.clone())
    }

    pub fn load(&self, id: &str, now: u64) -> Result<Session> {
        let started = std::time::Instant::now();
        let outcome = self.load_inner(id, now);
        self.metrics.restore_micros.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        match &outcome {
            Ok(s) => {
                self.metrics.restores_ok.fetch_add(1, Ordering::Relaxed);
                self.metrics.prefill_tokens_saved.fetch_add(s.tokens.len() as u64, Ordering::Relaxed);
            }
            Err(e) => {
                let corrupt = self.index.lock().expect("index").contains_key(id);
                if corrupt {
                    self.metrics.restores_corrupt.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(session = id, error = %format!("{e:#}"), "session snapshot failed verification, dropped, prefill instead");
                    let _ = self.delete(id);
                } else {
                    self.metrics.restores_miss.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        outcome
    }

    fn load_inner(&self, id: &str, now: u64) -> Result<Session> {
        ensure!(self.index.lock().expect("index").contains_key(id), "no session {id}");
        let path = self.path(id);
        let (header, session) = read_full(&path)?;
        ensure!(self.accepts(&header), "snapshot is for another model, spec or dtype");
        ensure!(!header.stripped, "session {id} is stripped, needs a prefill");
        ensure!(session_id(&self.compat, &session.scope, &session.tokens) == id, "token list does not match the session id");
        self.touch(id, now)?;
        Ok(session)
    }

    pub fn verify(&self, id: &str) -> Result<Entry> {
        let entry = self.list().into_iter().find(|e| e.id == id).with_context(|| format!("no session {id}"))?;
        ensure!(!entry.stripped, "session {id} is stripped, restoring it needs a prefill");
        let checked = read_full(&self.path(id)).and_then(|(header, session)| {
            ensure!(self.accepts(&header), "snapshot is for another model, spec or dtype");
            ensure!(session_id(&self.compat, &session.scope, &session.tokens) == id, "token list does not match the session id");
            Ok(())
        });
        if let Err(e) = checked {
            self.metrics.restores_corrupt.fetch_add(1, Ordering::Relaxed);
            let _ = self.delete(id);
            return Err(e);
        }
        Ok(entry)
    }

    pub fn touch(&self, id: &str, now: u64) -> Result<()> {
        let path = self.path(id);
        let mut f = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut head = read_exact_at(&mut f, HEADER_LEN)?;
        head[LAST_USED_AT..LAST_USED_AT + 8].copy_from_slice(&now.to_le_bytes());
        let crc = crc32(&head[..HEADER_LEN - 4]);
        head[HEADER_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&head)?;
        f.sync_data()?;
        if let Some(i) = self.index.lock().expect("index").get_mut(id) {
            i.entry.last_used = now;
        }
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        self.index.lock().expect("index").remove(id);
        match fs::remove_file(self.path(id)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn strip(&self, id: &str, now: u64) -> Result<Entry> {
        let (header, session) = read_full(&self.path(id))?;
        ensure!(self.accepts(&header), "snapshot is for another model, spec or dtype");
        let last = self.index.lock().expect("index").get(id).map(|i| i.entry.last_used).unwrap_or(now);
        let mut e = self.write_inner(&session, last, true)?;
        e.last_used = last;
        Ok(e)
    }

    pub fn purge(&self) -> Result<usize> {
        let ids: Vec<String> = self.index.lock().expect("index").keys().cloned().collect();
        for id in &ids {
            self.delete(id)?;
        }
        Ok(ids.len())
    }

    pub fn expire(&self, now: u64, ttl: u64) -> Vec<String> {
        let dead: Vec<String> = self.list().into_iter().filter(|e| now >= e.expires(ttl)).map(|e| e.id).collect();
        for id in &dead {
            let _ = self.delete(id);
        }
        dead
    }

    pub fn enforce_budget(&self, max_bytes: u64) -> Vec<String> {
        if max_bytes == 0 {
            return Vec::new();
        }
        let mut by_age = self.list();
        by_age.reverse();
        let mut total = self.total_bytes();
        let mut evicted = Vec::new();
        for e in by_age {
            if total <= max_bytes {
                break;
            }
            let _ = self.delete(&e.id);
            total -= e.bytes;
            evicted.push(e.id);
        }
        evicted
    }
}

pub fn purge_root(root: &Path) -> Result<usize> {
    let dir = root.join("sessions");
    let mut n = 0;
    if !dir.exists() {
        return Ok(0);
    }
    for model in fs::read_dir(&dir)? {
        let model = model?.path();
        if !model.is_dir() {
            continue;
        }
        for f in fs::read_dir(&model)? {
            let p = f?.path();
            if p.is_file() {
                fs::remove_file(&p)?;
                n += 1;
            }
        }
    }
    Ok(n)
}

#[cfg(unix)]
fn restrict(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn restrict(_: &Path, _: u32) {}

pub const SECTION_META: usize = 0;
pub const SECTION_K: usize = 1;
pub const SECTION_V: usize = 2;
pub const SECTION_STATE: usize = 3;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect()
}

pub fn pack_sections(k: Vec<u8>, v: Vec<u8>, state: Option<&crate::prefix::StateSnapshot>) -> Vec<Vec<u8>> {
    let (meta, body) = match state {
        Some(s) => {
            let mut body = f32_bytes(s.conv());
            body.extend(f32_bytes(s.ssm()));
            body.extend(f32_bytes(s.pending()));
            ([1u64, s.conv().len() as u64, s.ssm().len() as u64, s.pending().len() as u64], body)
        }
        None => ([0u64; 4], Vec::new()),
    };
    vec![meta.iter().flat_map(|m| m.to_le_bytes()).collect(), k, v, body]
}

pub fn unpack_state(sections: &[Vec<u8>]) -> Result<Option<crate::prefix::StateSnapshot>> {
    ensure!(sections.len() == 4 && sections[SECTION_META].len() == 32, "session payload has {} sections, expected 4", sections.len());
    let m: Vec<u64> = sections[SECTION_META].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
    if m[0] == 0 {
        return Ok(None);
    }
    let all = bytes_f32(&sections[SECTION_STATE]);
    let (c, s, p) = (m[1] as usize, m[2] as usize, m[3] as usize);
    ensure!(all.len() == c + s + p, "recurrent state section is {} values, expected {}", all.len(), c + s + p);
    Ok(Some(crate::prefix::StateSnapshot::new(all[..c].to_vec(), all[c..c + s].to_vec(), all[c + s..].to_vec())))
}

pub trait Capture {
    fn capture(&mut self, key: &str) -> Result<Session>;
    fn release(&mut self, key: &str);
}

#[derive(Default)]
pub struct Tracker {
    live: HashMap<String, Live>,
}

struct Live {
    last_request: u64,
    tokens: usize,
    saved: bool,
}

impl Tracker {
    pub fn touch(&mut self, key: &str, now: u64, tokens: usize) {
        self.live.insert(key.to_string(), Live { last_request: now, tokens, saved: false });
    }

    pub fn forget(&mut self, key: &str) {
        self.live.remove(key);
    }

    pub fn due(&self, now: u64, idle: u64, min_tokens: usize) -> Vec<String> {
        let mut v: Vec<String> = self
            .live
            .iter()
            .filter(|(_, l)| !l.saved && l.tokens >= min_tokens && now >= l.last_request + idle)
            .map(|(k, _)| k.clone())
            .collect();
        v.sort();
        v
    }

    pub fn last_request(&self, key: &str) -> Option<(u64, usize)> {
        self.live.get(key).map(|l| (l.last_request, l.tokens))
    }

    fn mark_saved(&mut self, key: &str) {
        if let Some(l) = self.live.get_mut(key) {
            l.saved = true;
        }
    }
}

pub struct Janitor {
    pub options: SessionOptions,
    pub tracker: Tracker,
}

impl Janitor {
    pub fn new(options: SessionOptions) -> Self {
        Janitor { options, tracker: Tracker::default() }
    }

    pub fn tick(&mut self, now: u64, store: &SessionStore, source: &mut dyn Capture) -> Result<Vec<Entry>> {
        store.expire(now, self.options.ttl_seconds());
        let due = self.tracker.due(now, self.options.idle_seconds, self.options.min_tokens);
        let written = self.save_keys(due, now, store, source);
        store.enforce_budget(self.options.max_bytes);
        Ok(written)
    }

    pub fn save_keys(&mut self, keys: Vec<String>, now: u64, store: &SessionStore, source: &mut dyn Capture) -> Vec<Entry> {
        let mut written = Vec::new();
        for key in keys {
            let session = match source.capture(&key) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(session = %key, error = %format!("{e:#}"), "capture failed");
                    continue;
                }
            };
            match store.write(&session, now) {
                Ok(entry) => {
                    source.release(&key);
                    self.tracker.mark_saved(&key);
                    written.push(entry);
                }
                Err(_) => {
                    self.tracker.mark_saved(&key);
                }
            }
        }
        written
    }
}

pub struct Shared {
    pub store: SessionStore,
    pub options: SessionOptions,
    pub force: Mutex<Vec<String>>,
    pub resident: Mutex<Vec<Resident>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resident {
    pub id: String,
    pub tokens: usize,
    pub last_used: u64,
}

impl Shared {
    pub fn new(store: SessionStore, options: SessionOptions) -> Self {
        Shared { store, options, force: Mutex::new(Vec::new()), resident: Mutex::new(Vec::new()) }
    }
}

pub fn compat_for(spec: &crate::spec::Spec, kv_planes: usize) -> Compat {
    let mut h = Sha256::new();
    h.update(spec.name.as_bytes());
    h.update(spec.repo.as_bytes());
    Compat {
        model_hash: h.finalize().into(),
        spec_hash: crate::prefix::namespace(spec, kv_planes, ""),
        kv_dtype: spec.runtime.kv_dtype as u32,
        layers: spec.kv_layers() as u32,
        block_size: crate::codegen::PAGE as u32,
    }
}

pub fn default_dir() -> PathBuf {
    dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("inference-bench")
}

#[cfg(unix)]
pub fn available_bytes(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return 0 };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return 0;
    }
    st.f_bavail as u64 * st.f_frsize as u64
}

#[cfg(not(unix))]
pub fn available_bytes(_: &Path) -> u64 {
    0
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Stats {
    pub snapshots: u64,
    pub snapshot_bytes: u64,
    pub snapshot_failures: u64,
    pub restores_ok: u64,
    pub restores_miss: u64,
    pub restores_corrupt: u64,
    pub restore_seconds: f64,
    pub prefill_tokens_saved: u64,
    pub stored: usize,
    pub stored_bytes: u64,
}

impl Shared {
    pub fn stats(&self) -> Stats {
        let m = &self.store.metrics;
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        Stats {
            snapshots: get(&m.snapshots),
            snapshot_bytes: get(&m.snapshot_bytes),
            snapshot_failures: get(&m.snapshot_failures),
            restores_ok: get(&m.restores_ok),
            restores_miss: get(&m.restores_miss),
            restores_corrupt: get(&m.restores_corrupt),
            restore_seconds: get(&m.restore_micros) as f64 / 1e6,
            prefill_tokens_saved: get(&m.prefill_tokens_saved),
            stored: self.store.list().len(),
            stored_bytes: self.store.total_bytes(),
        }
    }

    pub fn request_save(&self, id: &str) -> bool {
        if !self.resident.lock().expect("residents").iter().any(|r| r.id == id) {
            return false;
        }
        self.force.lock().expect("force").push(id.to_string());
        true
    }
}

pub fn now_seconds() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn free_space_budget(free_bytes: u64, percent: u64) -> u64 {
    free_bytes / 100 * percent.min(100)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compat() -> Compat {
        Compat { model_hash: [1; 32], spec_hash: [2; 32], kv_dtype: 0, layers: 4, block_size: 16 }
    }

    fn session(n: u32, fill: u8) -> Session {
        Session { scope: [0; 32], tokens: (0..n).collect(), sections: vec![vec![fill; 1000], vec![fill.wrapping_add(1); 300], vec![7; 50]] }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sessions-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    struct Fake(HashMap<String, Session>, Vec<String>);

    impl Capture for Fake {
        fn capture(&mut self, key: &str) -> Result<Session> {
            self.0.get(key).cloned().context("gone")
        }
        fn release(&mut self, key: &str) {
            self.1.push(key.to_string());
        }
    }

    #[test]
    fn roundtrip_is_exact_and_file_is_private() {
        let root = tmp("rt");
        let store = SessionStore::open(&root, compat()).unwrap();
        let s = session(100, 9);
        let e = store.write(&s, 10).unwrap();
        assert_eq!(store.load(&e.id, 20).unwrap(), s);
        assert_eq!(store.list()[0].last_used, 20);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(store.path(&e.id)).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        let reopened = SessionStore::open(&root, compat()).unwrap();
        assert_eq!(reopened.list(), store.list());
        assert_eq!(reopened.load(&e.id, 30).unwrap(), s);
    }

    #[test]
    fn longest_prefix_is_found_and_never_the_whole_prompt() {
        let store = SessionStore::open(&tmp("prefix"), compat()).unwrap();
        let short = store.write(&session(50, 1), 0).unwrap();
        let long = store.write(&session(80, 2), 0).unwrap();
        let prompt: Vec<u32> = (0..120).collect();
        assert_eq!(store.find_prefix(&[0; 32], &prompt).unwrap().id, long.id);
        let exact: Vec<u32> = (0..80).collect();
        assert_eq!(store.find_prefix(&[0; 32], &exact).unwrap().id, short.id);
        let other: Vec<u32> = (1..200).collect();
        assert!(store.find_prefix(&[0; 32], &other).is_none());
    }

    #[test]
    fn a_flipped_byte_in_any_region_drops_the_snapshot() {
        let root = tmp("flip");
        let store = SessionStore::open(&root, compat()).unwrap();
        let e = store.write(&session(64, 3), 0).unwrap();
        let path = store.path(&e.id);
        let original = fs::read(&path).unwrap();
        let tokens_at = HEADER_LEN + 5;
        let section_at = HEADER_LEN + 64 * 4 + 4 + 8 + 10;
        let last_section_at = original.len() - 16 - 4 - 20;
        for at in [tokens_at, section_at, last_section_at, original.len() - 3] {
            fs::write(&path, &original).unwrap();
            let store = SessionStore::open(&root, compat()).unwrap();
            let mut bytes = original.clone();
            bytes[at] ^= 0x40;
            fs::write(&path, &bytes).unwrap();
            assert!(store.load(&e.id, 1).is_err(), "flip at {at}");
            assert!(!path.exists(), "corrupt file kept after flip at {at}");
            assert_eq!(store.metrics.restores_corrupt.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn header_flips_and_truncation_are_rejected_at_open() {
        let root = tmp("trunc");
        let store = SessionStore::open(&root, compat()).unwrap();
        let e = store.write(&session(64, 3), 0).unwrap();
        let path = store.path(&e.id);
        let original = fs::read(&path).unwrap();
        for cut in [original.len() - 1, original.len() / 2, HEADER_LEN, 10] {
            fs::write(&path, &original[..cut]).unwrap();
            let reopened = SessionStore::open(&root, compat()).unwrap();
            assert!(reopened.list().is_empty(), "cut {cut}");
            assert!(!path.exists());
        }
        let mut bytes = original.clone();
        bytes[30] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(SessionStore::open(&root, compat()).unwrap().list().is_empty());
    }

    #[test]
    fn another_model_spec_dtype_or_version_is_never_loaded() {
        let root = tmp("compat");
        let store = SessionStore::open(&root, compat()).unwrap();
        let e = store.write(&session(64, 3), 0).unwrap();
        for other in [
            Compat { spec_hash: [9; 32], ..compat() },
            Compat { kv_dtype: 1, ..compat() },
            Compat { layers: 5, ..compat() },
            Compat { block_size: 32, ..compat() },
        ] {
            let dir = store.dir().to_path_buf();
            let foreign = SessionStore { dir, compat: other, index: Mutex::new(HashMap::new()), metrics: Metrics::default() };
            foreign.rescan().unwrap();
            assert!(foreign.list().is_empty());
            let copy = store.write(&session(64, 3), 0).unwrap();
            assert_eq!(copy.id, e.id);
        }
        let other_model = SessionStore::open(&root, Compat { model_hash: [5; 32], ..compat() }).unwrap();
        assert!(other_model.list().is_empty());
        let path = store.path(&e.id);
        let mut bytes = fs::read(&path).unwrap();
        bytes[8] = 2;
        let crc = crc32(&bytes[..HEADER_LEN - 4]);
        bytes[HEADER_LEN - 4..HEADER_LEN].copy_from_slice(&crc.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(SessionStore::open(&root, compat()).unwrap().list().is_empty());
        assert!(!path.exists());
    }

    #[test]
    fn a_partial_temp_file_from_a_crash_is_never_picked_up() {
        let root = tmp("crash");
        let store = SessionStore::open(&root, compat()).unwrap();
        let s = session(64, 3);
        let id = session_id(&compat(), &s.scope, &s.tokens);
        let full = encode(&compat(), &s, 0, 0, false);
        fs::write(store.dir().join(".tmp-999-0"), &full[..full.len() / 2]).unwrap();
        fs::write(store.dir().join(format!("{id}.{SUFFIX}.partial")), &full[..10]).unwrap();
        let reopened = SessionStore::open(&root, compat()).unwrap();
        assert!(reopened.list().is_empty());
        assert!(!store.dir().join(".tmp-999-0").exists());
    }

    #[test]
    fn a_killed_writer_leaves_no_visible_session() {
        let root = tmp("kill");
        let store = SessionStore::open(&root, compat()).unwrap();
        let dir = store.dir().to_path_buf();
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("head -c 5000 /dev/zero > {}/.tmp-1-1; sleep 30", dir.display()))
            .spawn()
            .unwrap();
        for _ in 0..50 {
            if dir.join(".tmp-1-1").exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(SessionStore::open(&root, compat()).unwrap().list().is_empty());
    }

    #[test]
    fn concurrent_writers_and_readers_of_one_session_never_see_a_partial_file() {
        let store = std::sync::Arc::new(SessionStore::open(&tmp("conc"), compat()).unwrap());
        let s = session(200, 4);
        let id = store.write(&s, 0).unwrap().id;
        let mut threads = Vec::new();
        for t in 0..4u64 {
            let store = store.clone();
            let s = s.clone();
            let id = id.clone();
            threads.push(std::thread::spawn(move || {
                for i in 0..25u64 {
                    if t % 2 == 0 {
                        store.write(&s, t * 100 + i).unwrap();
                    } else {
                        assert_eq!(store.load(&id, i).unwrap(), s);
                    }
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(store.list().len(), 1);
        assert_eq!(store.metrics.restores_corrupt.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn idle_policy_writes_at_five_minutes_and_deletes_after_eight_hours() {
        let store = SessionStore::open(&tmp("idle"), compat()).unwrap();
        let mut janitor = Janitor::new(SessionOptions { min_tokens: 64, ..SessionOptions::default() });
        let mut source = Fake(HashMap::new(), Vec::new());
        source.0.insert("a".into(), session(100, 1));
        janitor.tracker.touch("a", 1000, 100);
        assert!(janitor.tick(1000 + 299, &store, &mut source).unwrap().is_empty());
        assert!(store.list().is_empty());
        let written = janitor.tick(1000 + 301, &store, &mut source).unwrap();
        assert_eq!(written.len(), 1);
        assert_eq!(source.1, vec!["a".to_string()]);
        assert!(janitor.tick(1000 + 400, &store, &mut source).unwrap().is_empty());
        let t0 = 1000 + 301;
        janitor.tick(t0 + 8 * 3600 - 1, &store, &mut source).unwrap();
        assert_eq!(store.list().len(), 1);
        janitor.tick(t0 + 8 * 3600 + 1, &store, &mut source).unwrap();
        assert!(store.list().is_empty());
        assert!(!store.dir().join(format!("{}.{SUFFIX}", written[0].id)).exists());
    }

    #[test]
    fn restore_counts_from_last_use_for_the_ttl() {
        let store = SessionStore::open(&tmp("ttl"), compat()).unwrap();
        let e = store.write(&session(100, 1), 0).unwrap();
        store.load(&e.id, 7 * 3600).unwrap();
        assert!(store.expire(8 * 3600 + 1, 8 * 3600).is_empty());
        assert_eq!(store.expire(15 * 3600 + 1, 8 * 3600).len(), 1);
    }

    #[test]
    fn short_sessions_are_not_snapshotted() {
        let store = SessionStore::open(&tmp("short"), compat()).unwrap();
        let mut janitor = Janitor::new(SessionOptions::default());
        let mut source = Fake(HashMap::new(), Vec::new());
        source.0.insert("a".into(), session(100, 1));
        janitor.tracker.touch("a", 0, 100);
        assert!(janitor.tick(10_000, &store, &mut source).unwrap().is_empty());
    }

    #[test]
    fn budget_evicts_least_recently_used_first() {
        let store = SessionStore::open(&tmp("budget"), compat()).unwrap();
        let a = store.write(&session(100, 1), 10).unwrap();
        let b = store.write(&session(101, 2), 20).unwrap();
        let c = store.write(&session(102, 3), 30).unwrap();
        store.touch(&a.id, 40).unwrap();
        let budget = store.total_bytes() - 1;
        assert_eq!(store.enforce_budget(budget), vec![b.id.clone()]);
        let budget = a.bytes + 10;
        assert_eq!(store.enforce_budget(budget), vec![c.id.clone()]);
        assert_eq!(store.list().len(), 1);
        assert_eq!(store.list()[0].id, a.id);
    }

    #[test]
    fn a_failed_write_degrades_to_no_snapshot() {
        let root = tmp("full");
        let store = SessionStore::open(&root, compat()).unwrap();
        fs::remove_dir_all(store.dir()).unwrap();
        assert!(store.write(&session(100, 1), 0).is_err());
        assert_eq!(store.metrics.snapshot_failures.load(Ordering::Relaxed), 1);
        assert!(store.list().is_empty());
    }

    #[test]
    fn strip_keeps_tokens_and_drops_the_payload() {
        let store = SessionStore::open(&tmp("strip"), compat()).unwrap();
        let e = store.write(&session(100, 1), 5).unwrap();
        let stripped = store.strip(&e.id, 9).unwrap();
        assert!(stripped.stripped && stripped.bytes < e.bytes && stripped.tokens == 100 && stripped.last_used == 5);
        assert!(store.find_prefix(&[0; 32], &(0..200).collect::<Vec<u32>>()).is_none());
        assert!(store.load(&e.id, 10).is_err());
    }

    #[test]
    fn verify_reads_every_checksum_without_counting_a_restore() {
        let store = SessionStore::open(&tmp("verify"), compat()).unwrap();
        let e = store.write(&session(100, 1), 0).unwrap();
        assert_eq!(store.verify(&e.id).unwrap().tokens, 100);
        assert_eq!(store.metrics.restores_ok.load(Ordering::Relaxed), 0);
        let mut bytes = fs::read(store.path(&e.id)).unwrap();
        let at = bytes.len() / 2;
        bytes[at] ^= 1;
        fs::write(store.path(&e.id), bytes).unwrap();
        assert!(store.verify(&e.id).is_err());
        assert!(store.verify("nope").is_err());
    }

    #[test]
    fn purge_removes_every_file() {
        let root = tmp("purge");
        let store = SessionStore::open(&root, compat()).unwrap();
        store.write(&session(100, 1), 0).unwrap();
        store.write(&session(101, 1), 0).unwrap();
        assert_eq!(purge_root(&root).unwrap(), 2);
        assert!(SessionStore::open(&root, compat()).unwrap().list().is_empty());
    }

    #[test]
    fn sections_roundtrip_the_recurrent_state_exactly() {
        let state = crate::prefix::StateSnapshot::new(vec![1.0, -2.5], vec![f32::MIN_POSITIVE, 3.0, 4.0], vec![9.0]);
        let packed = pack_sections(vec![1, 2, 3], vec![4, 5], Some(&state));
        assert_eq!(unpack_state(&packed).unwrap().unwrap(), state);
        assert!(unpack_state(&pack_sections(vec![], vec![], None)).unwrap().is_none());
        assert!(unpack_state(&packed[..2]).is_err());
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
