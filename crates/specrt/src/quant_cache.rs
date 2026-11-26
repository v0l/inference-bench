use crate::nvfp4::Fp4;
use crate::spec::{Quant, Spec};
use crate::weights::Matrix;
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::tensor::{Metadata, TensorView};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const CACHE_VERSION: u32 = 1;
pub const MANIFEST: &str = "quant.toml";
pub const HUB_FOLDER: &str = "specrt-quant";

thread_local! {
    static SEEN: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

pub fn note_read(name: &str) {
    SEEN.with(|s| {
        if let Some(v) = s.borrow_mut().as_mut() {
            v.push(name.to_string());
        }
    });
}

pub fn recording<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    SEEN.with(|s| *s.borrow_mut() = Some(Vec::new()));
    let out = f();
    (out, SEEN.with(|s| s.borrow_mut().take()).unwrap_or_default())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub key: String,
    pub model: String,
    pub format_version: u32,
    pub bytes: u64,
    pub tensors: usize,
    pub files: Vec<String>,
    pub sources: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    pub dir: PathBuf,
    pub bytes: u64,
    pub hit: bool,
}

#[derive(Serialize)]
struct KeyParts<'a> {
    version: u32,
    format_version: u32,
    nvfp4_steps: &'a [i32],
    fp8_factors: &'a [f32],
    spec: Spec,
    sources: Vec<String>,
}

fn source_id(path: &Path) -> Result<String> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Ok(target) = std::fs::read_link(path)
        && let Some(blob) = target.file_name().and_then(|n| n.to_str())
        && blob.len() == 64
        && blob.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Ok(format!("{name}={blob}"));
    }
    let m = std::fs::metadata(path).with_context(|| path.display().to_string())?;
    let mtime = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
    Ok(format!("{name}:{}:{mtime}", m.len()))
}

pub fn key(spec: &Spec, files: &[PathBuf]) -> Result<String> {
    use sha2::Digest;
    let mut weights_only = spec.clone();
    weights_only.name = String::new();
    weights_only.repo = String::new();
    weights_only.runtime = crate::spec::RuntimeOptions { quantize: spec.runtime.quantize, ..Default::default() };
    let parts = KeyParts {
        version: CACHE_VERSION,
        format_version: crate::quantize::FORMAT_VERSION,
        nvfp4_steps: &crate::quantize::NVFP4_SCALE_STEPS,
        fp8_factors: &crate::quantize::FP8_SCALE_FACTORS,
        spec: weights_only,
        sources: files.iter().map(|f| source_id(f)).collect::<Result<_>>()?,
    };
    let text = serde_json::to_string(&parts)?;
    Ok(sha2::Sha256::digest(text.as_bytes()).iter().take(12).map(|b| format!("{b:02x}")).collect())
}

pub fn root_for(spec: &Spec, files: &[PathBuf]) -> PathBuf {
    let repo_dir = files
        .first()
        .and_then(|f| f.parent())
        .and_then(|snapshot| snapshot.parent())
        .filter(|s| s.file_name().is_some_and(|n| n == "snapshots"))
        .and_then(|s| s.parent())
        .filter(|r| r.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("models--")));
    match repo_dir {
        Some(r) => r.join(HUB_FOLDER),
        None => spec.runtime.cache_dir.clone().unwrap_or_else(crate::jit::default_cache_dir).join("quant").join(crate::quantcheck::model_name(&spec.repo)),
    }
}

pub fn dir_for(spec: &Spec, files: &[PathBuf]) -> Result<PathBuf> {
    Ok(root_for(spec, files).join(key(spec, files)?))
}

pub struct Store {
    pub dir: PathBuf,
    pub manifest: Manifest,
    shards: Vec<(Mmap, usize, Metadata)>,
    index: HashMap<String, usize>,
    sources: BTreeSet<String>,
}

impl Store {
    pub fn open(dir: &Path, key: &str) -> Result<Option<Store>> {
        let path = dir.join(MANIFEST);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
        let manifest: Manifest = toml::from_str(&text).with_context(|| path.display().to_string())?;
        ensure!(manifest.version == CACHE_VERSION && manifest.key == key, "{} is not the cache for key {key}", path.display());
        let mut shards = Vec::with_capacity(manifest.files.len());
        let mut index = HashMap::new();
        for (i, f) in manifest.files.iter().enumerate() {
            let p = dir.join(f);
            let file = std::fs::File::open(&p).with_context(|| p.display().to_string())?;
            let map = unsafe { Mmap::map(&file)? };
            let (header, meta) = SafeTensors::read_metadata(&map).with_context(|| p.display().to_string())?;
            for name in meta.tensors().into_keys() {
                index.insert(name, i);
            }
            shards.push((map, 8 + header, meta));
        }
        let sources = manifest.sources.iter().cloned().collect();
        Ok(Some(Store { dir: dir.to_path_buf(), manifest, shards, index, sources }))
    }

    pub fn covers(&self, source: &str) -> bool {
        self.sources.contains(source)
    }

    fn tensor(&self, name: &str) -> Option<(Dtype, &[usize], &[u8])> {
        let (map, start, meta) = &self.shards[*self.index.get(name)?];
        let info = meta.info(name)?;
        let (from, to) = info.data_offsets;
        Some((info.dtype, &info.shape, &map[start + from..start + to]))
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _, data) = self.tensor(name).with_context(|| format!("{name} is not in the quantised weight cache"))?;
        ensure!(dtype == Dtype::F32, "{name} is {dtype:?} in the quantised weight cache");
        Ok(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    pub fn matrix(&self, what: &str, quant: Quant) -> Result<Option<Matrix<'_>>> {
        match quant {
            Quant::Nvfp4G16 => {
                let Some((_, shape, packed)) = self.tensor(&format!("{what}.fp4.packed")) else { return Ok(None) };
                let (rows, cols) = (shape[0], shape[1] * 2);
                let (_, _, scales) = self.tensor(&format!("{what}.fp4.scales")).with_context(|| format!("{what}.fp4.scales is missing"))?;
                let globals = self.floats(&format!("{what}.fp4.global"))?;
                let mut f = Fp4::new(Cow::Borrowed(packed), Cow::Borrowed(scales), globals[0], rows, cols)?;
                f.input_global = globals[1];
                Ok(Some(Matrix::Fp4(f)))
            }
            Quant::Fp8Channel => {
                let Some((_, _, values)) = self.tensor(&format!("{what}.fp8.values")) else { return Ok(None) };
                Ok(Some(Matrix::Fp8 { values: Cow::Borrowed(values), scales: self.floats(&format!("{what}.fp8.scales"))? }))
            }
            _ => Ok(None),
        }
    }
}

type Staged = (String, Dtype, Vec<usize>, Vec<u8>);

#[derive(Default)]
struct Written {
    files: Vec<String>,
    sources: BTreeSet<String>,
    bytes: u64,
    tensors: usize,
    failed: Option<String>,
}

pub struct Writer {
    dir: PathBuf,
    partial: PathBuf,
    key: String,
    model: String,
    pending: Mutex<HashMap<std::thread::ThreadId, (Vec<Staged>, Vec<String>)>>,
    written: Mutex<Written>,
    finished: std::sync::atomic::AtomicBool,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Writer {
    pub fn new(dir: PathBuf, key: String, model: String) -> Result<Writer> {
        let partial = dir.with_file_name(format!("{key}.partial-{}", std::process::id()));
        if partial.exists() {
            std::fs::remove_dir_all(&partial).ok();
        }
        std::fs::create_dir_all(&partial).with_context(|| partial.display().to_string())?;
        Ok(Writer { dir, partial, key, model, pending: Default::default(), written: Default::default(), finished: Default::default() })
    }

    pub fn stage(&self, what: &str, m: &Matrix, sources: Vec<String>) {
        let rows_of = |len: usize, cols: usize| len / cols.max(1);
        let tensors: Vec<Staged> = match m {
            Matrix::Fp4(f) => vec![
                (format!("{what}.fp4.packed"), Dtype::U8, vec![rows_of(f.packed.len() * 2, f.cols), f.cols / 2], f.packed.to_vec()),
                (format!("{what}.fp4.scales"), Dtype::U8, vec![f.scales.len()], f.scales.to_vec()),
                (format!("{what}.fp4.global"), Dtype::F32, vec![2], f32_bytes(&[f.global, f.input_global])),
            ],
            Matrix::Fp8 { values, scales } => vec![(format!("{what}.fp8.values"), Dtype::U8, vec![values.len()], values.to_vec()), (format!("{what}.fp8.scales"), Dtype::F32, vec![scales.len()], f32_bytes(scales))],
            _ => return,
        };
        let mut pending = self.pending.lock().unwrap();
        let entry = pending.entry(std::thread::current().id()).or_default();
        entry.0.extend(tensors);
        entry.1.extend(sources);
    }

    pub fn flush(&self, tag: &str) {
        let Some((tensors, sources)) = self.pending.lock().unwrap().remove(&std::thread::current().id()) else { return };
        if tensors.is_empty() {
            return;
        }
        let file = format!("{tag}.safetensors");
        let path = self.partial.join(&file);
        let views: Result<Vec<(String, TensorView<'_>)>, _> = tensors.iter().map(|(n, d, s, b)| TensorView::new(*d, s.clone(), b).map(|v| (n.clone(), v))).collect();
        let outcome = views.map_err(|e| anyhow::anyhow!("{e:?}")).and_then(|v| safetensors::serialize_to_file(v, None, &path).map_err(|e| anyhow::anyhow!("{}: {e:?}", path.display())));
        let mut w = self.written.lock().unwrap();
        match outcome {
            Ok(()) => {
                w.bytes += std::fs::metadata(&path).map_or(0, |m| m.len());
                w.tensors += tensors.len();
                w.files.push(file);
                w.sources.extend(sources);
            }
            Err(e) => w.failed = Some(format!("{e:#}")),
        }
    }

    pub fn finish(&self) -> Result<Option<Usage>> {
        let leftover: Vec<std::thread::ThreadId> = self.pending.lock().unwrap().keys().copied().collect();
        ensure!(leftover.is_empty(), "{} load threads left quantised weights unwritten", leftover.len());
        let w = std::mem::take(&mut *self.written.lock().unwrap());
        self.finished.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(e) = w.failed {
            std::fs::remove_dir_all(&self.partial).ok();
            anyhow::bail!("writing the quantised weight cache failed: {e}");
        }
        if w.files.is_empty() {
            std::fs::remove_dir_all(&self.partial).ok();
            return Ok(None);
        }
        let mut files = w.files;
        files.sort();
        let manifest = Manifest { version: CACHE_VERSION, key: self.key.clone(), model: self.model.clone(), format_version: crate::quantize::FORMAT_VERSION, bytes: w.bytes, tensors: w.tensors, files, sources: w.sources.into_iter().collect() };
        std::fs::write(self.partial.join(MANIFEST), toml::to_string(&manifest)?).with_context(|| self.partial.display().to_string())?;
        if self.dir.exists() {
            std::fs::remove_dir_all(&self.partial).ok();
        } else {
            std::fs::rename(&self.partial, &self.dir).with_context(|| format!("{} to {}", self.partial.display(), self.dir.display()))?;
        }
        Ok(Some(Usage { dir: self.dir.clone(), bytes: manifest.bytes, hit: false }))
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.finished.load(std::sync::atomic::Ordering::Relaxed) {
            std::fs::remove_dir_all(&self.partial).ok();
        }
    }
}

pub enum Attached {
    Off,
    Hit(Store),
    Miss(Writer),
}

pub fn attach(spec: &Spec, files: &[PathBuf], safetensors: bool, whole: bool) -> Result<Attached> {
    if !spec.runtime.quantize_cache || !spec.quantized_at_load() {
        return Ok(Attached::Off);
    }
    if !safetensors || !whole {
        tracing::info!(repo = %spec.repo, "quantize_cache covers whole model safetensors loads on one device only; converting at load");
        return Ok(Attached::Off);
    }
    let key = key(spec, files)?;
    let dir = root_for(spec, files).join(&key);
    if let Some(store) = Store::open(&dir, &key)? {
        return Ok(Attached::Hit(store));
    }
    std::fs::create_dir_all(dir.parent().context("cache dir has no parent")?).with_context(|| dir.display().to_string())?;
    Ok(Attached::Miss(Writer::new(dir, key, spec.repo.clone())?))
}
