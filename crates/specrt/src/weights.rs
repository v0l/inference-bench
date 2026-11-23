use crate::codegen::{FP8_BLOCK, Q8_GROUP};
use crate::gguf::{DeviceRows, GgmlType, Gguf};
use crate::spec::{Block, Quant, Spec};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
#[cfg(unix)]
use memmap2::UncheckedAdvice;
use safetensors::tensor::Metadata;
use safetensors::{Dtype, SafeTensors};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub enum Matrix<'a> {
    Bf16(Cow<'a, [u16]>),
    Q8 { values: Vec<i8>, scales: Vec<f32> },
    Fp8 { values: Cow<'a, [u8]>, scales: Vec<f32> },
    Fp4(crate::nvfp4::Fp4<'a>),
    Ggml(DeviceRows<'a>),
}

pub struct Attn<'a> {
    pub qkv: Matrix<'a>,
    pub qkv_bias: Option<Vec<f32>>,
    pub o_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o: Matrix<'a>,
    pub extra: Vec<(String, Vec<f32>)>,
}

pub struct Gdn<'a> {
    pub in_proj: Matrix<'a>,
    pub ba: Matrix<'a>,
    pub conv: Vec<f32>,
    pub neg_a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: Matrix<'a>,
}

pub struct Described<'a> {
    pub in_proj: Matrix<'a>,
    pub out: Matrix<'a>,
    pub extra: Vec<(String, Vec<f32>)>,
}

pub enum Mixer<'a> {
    Attention(Attn<'a>),
    Linear(Gdn<'a>),
    Described(Described<'a>),
    None,
}

fn split_gate_up<T: Copy>(fused: &[T], row_len: usize, experts: usize, inter: usize, interleaved: bool) -> (Vec<T>, Vec<T>) {
    let (mut gate, mut up) = (Vec::with_capacity(fused.len() / 2), Vec::with_capacity(fused.len() / 2));
    for e in 0..experts {
        let rows = &fused[e * 2 * inter * row_len..(e + 1) * 2 * inter * row_len];
        for r in 0..2 * inter {
            let into = if interleaved { if r % 2 == 0 { &mut gate } else { &mut up } } else if r < inter { &mut gate } else { &mut up };
            into.extend_from_slice(&rows[r * row_len..(r + 1) * row_len]);
        }
    }
    (gate, up)
}

pub enum Stacked {
    Bf16(Vec<u16>),
    Fp4 { packed: Vec<u8>, scales: Vec<u8>, inv_global: Vec<f32>, input_global: f32 },
    Fp8 { values: Vec<u8>, grid: Vec<f32> },
    Block { packed: Vec<u8>, scales: Vec<u8> },
}

pub struct MoeWeights {
    pub router: Vec<u16>,
    pub shared_gate: Option<Vec<f32>>,
    pub score_bias: Option<Vec<f32>>,
    pub router_bias: Option<Vec<f32>>,
    pub gate_up_bias: Option<Vec<f32>>,
    pub down_bias: Option<Vec<f32>>,
    pub hash_table: Option<Vec<i32>>,
    pub gate: Stacked,
    pub up: Stacked,
    pub down: Stacked,
}

pub struct Layer<'a> {
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub ln_pre_mlp: Option<Vec<f32>>,
    pub ln_post_mlp: Option<Vec<f32>>,
    pub mixer: Mixer<'a>,
    pub gate_up: Matrix<'a>,
    pub down: Matrix<'a>,
    pub moe: Option<MoeWeights>,
}

pub struct Outside<'a> {
    pub embed: Cow<'a, [u16]>,
    pub norm: Vec<f32>,
    pub lm_head: Matrix<'a>,
}

pub struct MtpWeights<'a> {
    pub fc: Matrix<'a>,
    pub norm_embed: Vec<f32>,
    pub norm_hidden: Vec<f32>,
    pub layer: Layer<'a>,
    pub norm: Vec<f32>,
}

pub fn hub_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    dirs::home_dir().unwrap_or_default().join(".cache/huggingface/hub")
}

const INDEX: &str = "model.safetensors.index.json";
const SINGLE: &str = "model.safetensors";

pub fn find_checkpoint(repo: &str) -> Result<PathBuf> {
    let has = |dir: &Path| dir.join(INDEX).exists() || dir.join(SINGLE).exists();
    let local = Path::new(repo);
    if local.is_dir() {
        ensure!(has(local), "no {SINGLE} or {INDEX} in {repo}");
        return Ok(local.to_path_buf());
    }
    let snapshots =
        hub_dir().join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    for entry in std::fs::read_dir(&snapshots).with_context(|| snapshots.display().to_string())? {
        let dir = entry?.path();
        if has(&dir) {
            return Ok(dir);
        }
    }
    bail!("no {SINGLE} or {INDEX} for {repo} under {}", snapshots.display())
}

struct Shard {
    map: Mmap,
    data_start: usize,
    meta: Metadata,
}

pub struct Checkpoint {
    gguf: Option<std::sync::Arc<Gguf>>,
    shards: std::sync::Arc<Vec<Shard>>,
    names: std::sync::Arc<HashMap<String, (usize, String)>>,
    staged: Option<crate::loader::Batch>,
    cut: Option<std::sync::Arc<crate::tp::RankCut>>,
    owned: std::sync::Arc<std::sync::Mutex<Vec<Box<[u8]>>>>,
    pub files: Vec<PathBuf>,
}

pub struct View<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

fn shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join(INDEX);
    if !index.exists() {
        return Ok(vec![dir.join(SINGLE)]);
    }
    let text = std::fs::read_to_string(&index).with_context(|| index.display().to_string())?;
    let json: serde_json::Value = serde_json::from_str(&text).with_context(|| index.display().to_string())?;
    let map = json["weight_map"].as_object().with_context(|| format!("{} has no weight_map", index.display()))?;
    let mut files: Vec<&str> = map.values().filter_map(|v| v.as_str()).collect();
    files.sort_unstable();
    files.dedup();
    ensure!(!files.is_empty(), "{} names no shards", index.display());
    Ok(files.into_iter().map(|f| dir.join(f)).collect())
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        if dir.extension().is_some_and(|e| e == "gguf") {
            return Ok(Checkpoint { gguf: Some(std::sync::Arc::new(Gguf::open(dir)?)), shards: Default::default(), names: Default::default(), staged: None, cut: None, owned: Default::default(), files: vec![dir.to_path_buf()] });
        }
        let files = shard_files(dir)?;
        let mut shards = Vec::with_capacity(files.len());
        let mut names = HashMap::new();
        for (i, path) in files.iter().enumerate() {
            let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
            let map = unsafe { Mmap::map(&file)? };
            let (header, meta) =
                SafeTensors::read_metadata(&map).with_context(|| path.display().to_string())?;
            for name in meta.tensors().into_keys() {
                if let Some((prev, _)) = names.insert(name.clone(), (i, name.clone())) {
                    bail!("{name} is in both {} and {}", files[prev].display(), path.display());
                }
            }
            shards.push(Shard { map, data_start: 8 + header, meta });
        }
        Ok(Checkpoint { gguf: None, shards: std::sync::Arc::new(shards), names: std::sync::Arc::new(names), staged: None, cut: None, owned: Default::default(), files })
    }

    pub fn for_rank(&self, cut: crate::tp::RankCut) -> Result<Checkpoint> {
        ensure!(self.gguf.is_none(), "tp does not slice GGUF checkpoints");
        Ok(Checkpoint { cut: Some(std::sync::Arc::new(cut)), owned: Default::default(), ..self.with_staged(None) })
    }

    pub fn rank_cut(&self) -> Option<&crate::tp::RankCut> {
        self.cut.as_deref()
    }

    fn keep(&self, bytes: Vec<u8>) -> &[u8] {
        let boxed = bytes.into_boxed_slice();
        let (ptr, len) = (boxed.as_ptr(), boxed.len());
        self.owned.lock().unwrap().push(boxed);
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn for_repo(repo: &str) -> Result<Self> {
        if repo.ends_with(".gguf") {
            return Self::open(Path::new(repo));
        }
        Self::open(&find_checkpoint(repo)?)
    }

    pub fn renamed(mut self, rules: &std::collections::BTreeMap<String, String>) -> Checkpoint {
        if rules.is_empty() || self.gguf.is_some() {
            return self;
        }
        let mut names = HashMap::new();
        for (name, at) in self.names.iter() {
            let mut public = name.clone();
            for (from, to) in rules {
                public = public.replace(from.as_str(), to);
            }
            names.insert(public, at.clone());
        }
        self.names = std::sync::Arc::new(names);
        self
    }

    pub fn has(&self, name: &str) -> bool {
        match &self.gguf {
            Some(g) => g.tensor_for_hf(name).is_some_and(|n| g.tensors.contains_key(&n)),
            None => self.names.contains_key(name),
        }
    }

    pub fn view(&self, name: &str) -> Result<View<'_>> {
        let full = self.full_view(name)?;
        let Some(cut) = self.cut.as_deref() else { return Ok(full) };
        let Some(rule) = cut.rule(name, &full.shape)? else { return Ok(full) };
        let elem = full.dtype.bitsize() / 8;
        let (shape, bytes) = crate::tp::cut_bytes(full.data, &full.shape, elem, &rule).with_context(|| format!("slicing {name} for rank {}", cut.rank))?;
        Ok(View { dtype: full.dtype, shape, data: match bytes { std::borrow::Cow::Borrowed(b) => b, std::borrow::Cow::Owned(v) => self.keep(v) } })
    }

    pub fn shape_of(&self, name: &str) -> Result<Vec<usize>> {
        let (i, stored) = self.names.get(name).with_context(|| format!("{name} is not in the checkpoint"))?;
        Ok(self.shards[*i].meta.info(stored).context("tensor info")?.shape.clone())
    }

    fn full_view(&self, name: &str) -> Result<View<'_>> {
        let (i, stored) = self.names.get(name).with_context(|| format!("{name} is not in the checkpoint"))?;
        let shard = &self.shards[*i];
        let info = shard.meta.info(stored).context("tensor info")?;
        let (from, to) = info.data_offsets;
        let data = match self.staged.as_ref().and_then(|b| b.get(name)) {
            Some(staged) => staged,
            None => &shard.map[shard.data_start + from..shard.data_start + to],
        };
        Ok(View { dtype: info.dtype, shape: info.shape.clone(), data })
    }

    pub fn extents(&self, names: &[String]) -> Result<Vec<crate::loader::Item>> {
        let mut items = Vec::with_capacity(names.len());
        for name in names {
            let (file, stored) = self.names.get(name).with_context(|| format!("{name} is not in the checkpoint"))?;
            let file = *file;
            let shard = &self.shards[file];
            let (from, to) = shard.meta.info(stored).context("tensor info")?.data_offsets;
            items.push(crate::loader::Item { name: name.clone(), file, offset: (shard.data_start + from) as u64, len: (to - from) as u64 });
        }
        items.sort_by_key(|i| (i.file, i.offset));
        Ok(items)
    }

    pub fn names_under(&self, prefix: &str) -> Vec<String> {
        self.names.keys().filter(|n| n.starts_with(prefix)).cloned().collect()
    }

    pub fn tensor_names(&self, wanted: &[String]) -> Vec<String> {
        wanted.iter().filter(|n| self.names.contains_key(*n)).cloned().collect()
    }

    pub fn is_safetensors(&self) -> bool {
        self.gguf.is_none()
    }

    pub fn has_staged(&self) -> bool {
        self.staged.is_some()
    }

    pub fn release_names(&self, names: &[String]) {
        let page = 4096usize;
        for name in names {
            let Some((i, stored)) = self.names.get(name) else { continue };
            let sh = &self.shards[*i];
            let Ok(info) = sh.meta.info(stored).context("tensor info") else { continue };
            let (from, to) = info.data_offsets;
            let start = (sh.data_start + from).div_ceil(page) * page;
            let end = (sh.data_start + to) / page * page;
            #[cfg(unix)]
            if end > start {
                let _ = unsafe { sh.map.unchecked_advise_range(UncheckedAdvice::DontNeed, start, end - start) };
            }
        }
    }

    #[cfg(not(unix))]
    pub fn resident_fraction(&self) -> f64 {
        0.0
    }

    #[cfg(unix)]
    pub fn resident_fraction(&self) -> f64 {
        let page = 4096usize;
        let (mut resident, mut total) = (0usize, 0usize);
        for sh in self.shards.iter() {
            let len = sh.map.len();
            if len == 0 {
                continue;
            }
            let mut vec = vec![0u8; len.div_ceil(page)];
            let ok = unsafe { libc::mincore(sh.map.as_ptr() as *mut libc::c_void, len, vec.as_mut_ptr() as *mut _) } == 0;
            if !ok {
                return 0.0;
            }
            resident += vec.iter().filter(|&&b| b & 1 == 1).count();
            total += vec.len();
        }
        if total == 0 { 0.0 } else { resident as f64 / total as f64 }
    }

    pub fn cached(&self, opts: &crate::loader::LoaderOptions) -> bool {
        opts.skip_cached && self.is_safetensors() && self.resident_fraction() >= 0.8
    }

    pub fn with_staged(&self, batch: Option<crate::loader::Batch>) -> Checkpoint {
        Checkpoint { gguf: self.gguf.clone(), shards: self.shards.clone(), names: self.names.clone(), staged: batch, cut: self.cut.clone(), owned: Default::default(), files: self.files.clone() }
    }

    pub fn release(&self) {
        if let Some(g) = &self.gguf {
            g.release();
        }
        for s in self.shards.iter() {
            #[cfg(unix)]
            let _ = unsafe { s.map.unchecked_advise(UncheckedAdvice::DontNeed) };
            #[cfg(not(unix))]
            let _ = &s.map;
        }
    }

    pub fn root(&self) -> &'static str {
        if self.gguf.is_some() {
            "model"
        } else if self.has("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else if self.has("language_model.model.embed_tokens.weight") {
            "language_model.model"
        } else {
            "model"
        }
    }

    fn shaped(&self, name: &str, shape: &[usize]) -> Result<View<'_>> {
        let v = self.view(name)?;
        ensure!(v.shape == shape, "{name} has shape {:?}, expected {shape:?}", v.shape);
        Ok(v)
    }

    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Cow<'_, [u16]>> {
        if let Some(g) = &self.gguf {
            return Ok(Cow::Owned(self.gguf_f32(g, name, shape)?.into_iter().map(f32_to_bf16).collect()));
        }
        let v = self.shaped(name, shape)?;
        let pairs = || v.data.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes([b[0], b[1]]));
        Ok(match v.dtype {
            Dtype::BF16 if cfg!(target_endian = "little") && v.data.as_ptr().align_offset(2) == 0 => {
                Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.data.as_ptr().cast(), v.data.len() / 2) })
            }
            Dtype::BF16 => Cow::Owned(pairs().collect()),
            Dtype::F16 => Cow::Owned(pairs().map(|h| f32_to_bf16(f16_to_f32(h))).collect()),
            Dtype::F32 => Cow::Owned(
                v.data.as_chunks::<4>().0.iter().map(|b| f32_to_bf16(f32::from_le_bytes(*b))).collect(),
            ),
            Dtype::F8_E4M3 => bail!("{name} is FP8; use quant = \"fp8_block128\""),
            other => bail!("{name} is {other:?}, expected bf16, f16 or f32"),
        })
    }

    fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        if let Some(g) = &self.gguf {
            return self.gguf_f32(g, name, shape);
        }
        let v = self.shaped(name, shape)?;
        match v.dtype {
            Dtype::F32 => Ok(v.data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
            _ => Ok(self.bf16(name, shape)?.iter().map(|&b| bf16_to_f32(b)).collect()),
        }
    }

    fn f32(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.floats(name, &[len])
    }

    fn flat(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let shape = match &self.gguf {
            Some(_) => vec![len],
            None => self.view(name)?.shape.clone(),
        };
        ensure!(shape.iter().product::<usize>() == len, "{name} has shape {shape:?}, expected {len} values");
        self.floats(name, &shape)
    }

    fn raw(&self, weight: &str, rows: usize, cols: usize) -> Result<Raw<'_>> {
        if let Some(g) = &self.gguf {
            return self.gguf_raw(g, weight, rows, cols);
        }
        if let Some(module) = weight.strip_suffix(".weight")
            && self.is_fp4(module)
        {
            let fp4 = self.fp4(module, rows, cols)?;
            let w = (0..rows).flat_map(|r| fp4.dequant_row(r)).map(f32_to_bf16).collect();
            return Ok(Raw::Bf16 { w: Cow::Owned(w), cols });
        }
        let v = self.view(weight)?;
        if v.dtype != Dtype::F8_E4M3 {
            return Ok(Raw::Bf16 { w: self.bf16(weight, &[rows, cols])?, cols });
        }
        ensure!(v.shape == [rows, cols], "{weight} has shape {:?}, expected {:?}", v.shape, [rows, cols]);
        let kb = cols.div_ceil(FP8_BLOCK);
        let channel = format!("{weight}_scale");
        if !self.has(&format!("{weight}_scale_inv")) && self.has(&channel) {
            let per_row = if self.view(&channel)?.shape.iter().product::<usize>() == 1 {
                vec![self.floats(&channel, &self.view(&channel)?.shape)?[0]; rows]
            } else {
                self.floats(&channel, &[rows, 1])?
            };
            let grid = per_row.iter().flat_map(|&s| std::iter::repeat_n(s, kb)).collect();
            let map = (0..rows).map(|r| (r * kb) as u32).collect();
            return Ok(Raw::Fp8 { w: Cow::Borrowed(v.data), grid, map, cols });
        }
        let grid = self.floats(&format!("{weight}_scale_inv"), &[rows.div_ceil(FP8_BLOCK), kb])?;
        let map = (0..rows).map(|r| (r / FP8_BLOCK * kb) as u32).collect();
        Ok(Raw::Fp8 { w: Cow::Borrowed(v.data), grid, map, cols })
    }
}

fn unpermute_rows(data: &[u8], heads: usize, rows: usize) -> Vec<u8> {
    let row_bytes = data.len() / rows;
    let head_dim = rows / heads;
    let half = head_dim / 2;
    let mut out = Vec::with_capacity(data.len());
    for h in 0..heads {
        for t in 0..2 {
            for j in 0..half {
                let src = h * head_dim + j * 2 + t;
                out.extend_from_slice(&data[src * row_bytes..(src + 1) * row_bytes]);
            }
        }
    }
    out
}

impl Checkpoint {
    fn gguf_tensor<'g>(&self, g: &'g Gguf, name: &str, shape: &[usize]) -> Result<(&'g crate::gguf::TensorInfo, Cow<'g, [u8]>)> {
        let mapped = g.tensor_for_hf(name).with_context(|| format!("{name} has no GGUF tensor name"))?;
        let info = g.tensor(&mapped).with_context(|| format!("{mapped} is not in the GGUF file"))?;
        let squeeze = |s: &[usize]| s.iter().copied().filter(|&d| d != 1).collect::<Vec<_>>();
        ensure!(squeeze(&info.shape) == squeeze(shape), "{mapped} has shape {:?}, expected {shape:?}", info.shape);
        let data = g.data(info);
        let naming = g.naming().context("no family for the GGUF architecture")?;
        let arch = g.architecture()?;
        let heads = match mapped.rsplit_once('.') {
            Some((m, "weight")) if naming.family.permuted_qk && m.ends_with(".attn_q") => g.count(&naming.family.key("head_count", arch)?),
            Some((m, "weight")) if naming.family.permuted_qk && m.ends_with(".attn_k") => g.count(&naming.family.key("head_count_kv", arch)?).or(g.count(&naming.family.key("head_count", arch)?)),
            _ => None,
        };
        Ok(match heads {
            Some(h) => (info, Cow::Owned(unpermute_rows(data, h, info.rows()))),
            None => (info, Cow::Borrowed(data)),
        })
    }

    fn gguf_f32(&self, g: &Gguf, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (info, data) = self.gguf_tensor(g, name, shape)?;
        let mut out = vec![0f32; shape.iter().product()];
        crate::gguf::dequantize(info.kind, &data, &mut out).with_context(|| name.to_string())?;
        Ok(out)
    }

    fn gguf_raw<'s>(&'s self, g: &'s Gguf, weight: &str, rows: usize, cols: usize) -> Result<Raw<'s>> {
        let (info, data) = self.gguf_tensor(g, weight, &[rows, cols])?;
        if g.weight_quant() == Quant::Gguf {
            let packed = match data {
                Cow::Borrowed(d) => DeviceRows::pack(info.kind, d, rows, cols),
                Cow::Owned(d) => DeviceRows::pack(info.kind, &d, rows, cols).map(|p| DeviceRows { parts: p.parts.into_iter().map(|c| Cow::Owned(c.into_owned())).collect(), rows: p.rows, cols: p.cols }),
            };
            return Ok(Raw::Ggml(packed.with_context(|| weight.to_string())?));
        }
        if info.kind == GgmlType::Q8_0 {
            let (values, scales) = crate::gguf::q8_0_split(&data);
            return Ok(Raw::Q8 { values, scales, cols });
        }
        let mut out = vec![0f32; rows * cols];
        crate::gguf::dequantize(info.kind, &data, &mut out).with_context(|| weight.to_string())?;
        Ok(Raw::Bf16 { w: Cow::Owned(out.into_iter().map(f32_to_bf16).collect()), cols })
    }
}

pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

enum Raw<'a> {
    Bf16 { w: Cow<'a, [u16]>, cols: usize },
    Q8 { values: Vec<i8>, scales: Vec<f32>, cols: usize },
    Ggml(DeviceRows<'a>),
    Fp8 { w: Cow<'a, [u8]>, grid: Vec<f32>, map: Vec<u32>, cols: usize },
}

pub fn fp8_scales(grid: &[f32], map: &[u32]) -> Vec<f32> {
    let rows = map.len() as u32;
    map.iter().map(|&m| f32::from_bits(rows + m)).chain(grid.iter().copied()).collect()
}

impl<'a> Raw<'a> {
    fn cols(&self) -> usize {
        match self {
            Raw::Bf16 { cols, .. } | Raw::Q8 { cols, .. } | Raw::Fp8 { cols, .. } => *cols,
            Raw::Ggml(d) => d.cols,
        }
    }

    fn rows(&self) -> usize {
        match self {
            Raw::Bf16 { w, cols } => w.len() / cols,
            Raw::Q8 { values, cols, .. } => values.len() / cols,
            Raw::Ggml(d) => d.rows.len(),
            Raw::Fp8 { w, cols, .. } => w.len() / cols,
        }
    }

    fn slice_rows(&self, from: usize, to: usize) -> Raw<'a> {
        match self {
            Raw::Bf16 { w, cols } => Raw::Bf16 { w: Cow::Owned(w[from * cols..to * cols].to_vec()), cols: *cols },
            Raw::Q8 { values, scales, cols } => {
                let groups = cols / Q8_GROUP;
                Raw::Q8 { values: values[from * cols..to * cols].to_vec(), scales: scales[from * groups..to * groups].to_vec(), cols: *cols }
            }
            Raw::Ggml(d) => Raw::Ggml(DeviceRows { parts: d.parts.clone(), rows: d.rows[from..to].to_vec(), cols: d.cols }),
            Raw::Fp8 { w, grid, map, cols } => {
                let kb = cols.div_ceil(FP8_BLOCK) as u32;
                let map = &map[from..to];
                let lo = map.iter().min().map_or(0, |m| m / kb * kb);
                let hi = map.iter().max().map_or(0, |m| m / kb * kb + kb);
                Raw::Fp8 {
                    w: Cow::Owned(w[from * cols..to * cols].to_vec()),
                    grid: grid[lo as usize..hi as usize].to_vec(),
                    map: map.iter().map(|m| m - lo).collect(),
                    cols: *cols,
                }
            }
        }
    }

    fn split_groups(&self, groups: usize, widths: &[usize]) -> Result<Vec<Raw<'a>>> {
        let stride: usize = widths.iter().sum();
        ensure!(self.rows() == groups * stride, "fused rows {} do not split into {groups} groups of {stride}", self.rows());
        let mut out = Vec::new();
        let mut offset = 0;
        for &width in widths {
            let parts = (0..groups).map(|g| self.slice_rows(g * stride + offset, g * stride + offset + width)).collect();
            out.push(Raw::concat(parts)?);
            offset += width;
        }
        Ok(out)
    }

    fn dequantized(self) -> Raw<'a> {
        match self {
            Raw::Fp8 { w, grid, map, cols } => {
                let kb = cols.div_ceil(FP8_BLOCK);
                let out = w
                    .chunks(cols)
                    .zip(&map)
                    .flat_map(|(row, &m)| {
                        let scales = &grid[m as usize..m as usize + kb];
                        row.iter().enumerate().map(move |(c, &b)| f32_to_bf16(e4m3_to_f32(b) * scales[c / FP8_BLOCK]))
                    })
                    .collect();
                Raw::Bf16 { w: Cow::Owned(out), cols }
            }
            Raw::Q8 { values, scales, cols } => {
                let out = values
                    .chunks(Q8_GROUP)
                    .zip(&scales)
                    .flat_map(|(g, &s)| g.iter().map(move |&q| f32_to_bf16(q as f32 * s)))
                    .collect();
                Raw::Bf16 { w: Cow::Owned(out), cols }
            }
            Raw::Ggml(d) => {
                let out = (0..d.rows.len()).flat_map(|r| d.row_f32(r)).map(f32_to_bf16).collect();
                Raw::Bf16 { w: Cow::Owned(out), cols: d.cols }
            }
            bf16 => bf16,
        }
    }

    fn into_ggml(self) -> Result<DeviceRows<'a>> {
        match self {
            Raw::Ggml(d) => Ok(d),
            Raw::Bf16 { w, cols } => DeviceRows::from_bf16(&w, cols),
            other => other.dequantized().into_ggml(),
        }
    }

    fn concat(parts: Vec<Raw<'a>>) -> Result<Raw<'a>> {
        let mut parts = parts;
        if parts.iter().any(|p| matches!(p, Raw::Ggml(_))) {
            let mut rows = parts.into_iter().map(Raw::into_ggml);
            let mut out = rows.next().context("nothing to concatenate")??;
            for more in rows {
                let more = more?;
                ensure!(more.cols == out.cols, "fused rows have different widths");
                let table = out.append(more);
                out.rows.extend(table);
            }
            return Ok(Raw::Ggml(out));
        }
        if parts.iter().any(|p| matches!(p, Raw::Q8 { .. })) && !parts.iter().all(|p| matches!(p, Raw::Q8 { .. })) {
            parts = parts.into_iter().map(Raw::dequantized).collect();
        }
        let mut parts = parts.into_iter();
        let mut out = parts.next().context("nothing to concatenate")?;
        for p in parts {
            ensure!(p.cols() == out.cols(), "fused rows have different widths");
            match (&mut out, p) {
                (Raw::Q8 { values, scales, .. }, Raw::Q8 { values: v, scales: s, .. }) => {
                    values.extend(v);
                    scales.extend(s);
                }
                (Raw::Bf16 { w, .. }, Raw::Bf16 { w: more, .. }) => w.to_mut().extend_from_slice(&more),
                (Raw::Fp8 { w, grid, map, .. }, Raw::Fp8 { w: more, grid: g, map: m, .. }) => {
                    w.to_mut().extend_from_slice(&more);
                    let base = grid.len() as u32;
                    map.extend(m.iter().map(|r| r + base));
                    grid.extend(g);
                }
                _ => bail!("fused projections mix FP8 and bf16"),
            }
        }
        Ok(out)
    }

    fn interleave(a: Raw<'a>, b: Raw<'a>) -> Result<Raw<'a>> {
        ensure!(a.rows() == b.rows() && a.cols() == b.cols(), "gate and up differ in shape");
        if matches!(a, Raw::Ggml(_)) || matches!(b, Raw::Ggml(_)) {
            let mut out = a.into_ggml()?;
            let gate = std::mem::take(&mut out.rows);
            let up = out.append(b.into_ggml()?);
            out.rows = interleave_rows(&gate, &up, 1);
            return Ok(Raw::Ggml(out));
        }
        Ok(match (a, b) {
            (Raw::Q8 { values: va, scales: sa, cols }, Raw::Q8 { values: vb, scales: sb, .. }) => Raw::Q8 {
                values: interleave_rows(&va, &vb, cols),
                scales: interleave_rows(&sa, &sb, cols / Q8_GROUP),
                cols,
            },
            (a @ Raw::Q8 { .. }, b) | (a, b @ Raw::Q8 { .. }) => return Raw::interleave(a.dequantized(), b.dequantized()),
            (Raw::Bf16 { w: a, cols }, Raw::Bf16 { w: b, .. }) => {
                Raw::Bf16 { w: Cow::Owned(interleave_rows(&a, &b, cols)), cols }
            }
            (Raw::Fp8 { w: a, grid: ga, map: ma, cols }, Raw::Fp8 { w: b, grid: gb, map: mb, .. }) => {
                let base = ga.len() as u32;
                let mb: Vec<u32> = mb.iter().map(|r| r + base).collect();
                Raw::Fp8 {
                    w: Cow::Owned(interleave_rows(&a, &b, cols)),
                    grid: [ga, gb].concat(),
                    map: interleave_rows(&ma, &mb, 1),
                    cols,
                }
            }
            _ => bail!("gate and up mix FP8 and bf16"),
        })
    }

    fn into_matrix(self, quant: Quant, what: &str) -> Result<Matrix<'a>> {
        if matches!(self, Raw::Q8 { .. } | Raw::Ggml(_)) && !matches!(quant, Quant::Q8 | Quant::Gguf) {
            return self.dequantized().into_matrix(quant, what);
        }
        if quant == Quant::Gguf {
            return Ok(Matrix::Ggml(self.into_ggml()?));
        }
        if matches!(self, Raw::Ggml(_)) {
            return self.dequantized().into_matrix(quant, what);
        }
        Ok(match (self, quant) {
            (Raw::Q8 { values, scales, .. }, _) => Matrix::Q8 { values, scales },
            (Raw::Bf16 { w, .. }, Quant::Bf16) => Matrix::Bf16(w),
            (Raw::Bf16 { w, cols }, Quant::Q8) => {
                let (values, scales) = quantize_q8(&w, cols);
                Matrix::Q8 { values, scales }
            }
            (raw @ Raw::Fp8 { .. }, Quant::Nvfp4G16) => return raw.dequantized().into_matrix(quant, what),
            (Raw::Bf16 { w, cols }, Quant::Fp8Channel) => {
                let (values, scales) = quantize_fp8_channel(&w, cols);
                Matrix::Fp8 { values: Cow::Owned(values), scales }
            }
            (Raw::Bf16 { w, cols }, Quant::Nvfp4G16) => Matrix::Fp4(quantize_nvfp4(&w, cols)),
            (Raw::Fp8 { w, grid, map, .. }, q) if q.is_fp8() => {
                Matrix::Fp8 { values: w, scales: fp8_scales(&grid, &map) }
            }
            (Raw::Bf16 { .. }, q) => bail!("{what} is bf16 but the spec says {q:?}"),
            (Raw::Ggml(_), _) => unreachable!("ggml rows are handled above"),
            (Raw::Fp8 { .. }, _) => bail!("{what} is FP8; use quant = \"fp8_block128\" or \"fp8_channel\""),
        })
    }
}

pub fn f16_to_f32(v: u16) -> f32 {
    let sign = if v & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((v >> 10) & 0x1f) as i32;
    let frac = (v & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

pub fn e4m3_to_f32(b: u8) -> f32 {
    let mag = b & 0x7f;
    let exp = (mag >> 3) as i32;
    let man = (mag & 7) as f32;
    let v = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    if b & 0x80 != 0 { -v } else { v }
}

pub fn f32_to_e4m3(v: f32) -> u8 {
    let a = v.abs();
    let code = if a < 0.015625 {
        (a * 512.0).round_ties_even() as u32
    } else {
        let bits = a.to_bits().min(0x43f0_0000);
        let bits = bits + 0x7ffff + ((bits >> 20) & 1);
        ((((bits >> 23) as i32 - 120) as u32) << 3) | ((bits >> 20) & 7)
    };
    code.min(0x7e) as u8 | if v.is_sign_negative() { 0x80 } else { 0 }
}

fn par_rows<A: Send, B: Send>(
    w: &[u16],
    cols: usize,
    (a, a_cols): (&mut [A], usize),
    (b, b_cols): (&mut [B], usize),
    f: impl Fn(&[u16], &mut [A], &mut [B]) + Sync,
) {
    let rows = w.len() / cols;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for ((src, da), db) in w.chunks(per * cols).zip(a.chunks_mut(per * a_cols)).zip(b.chunks_mut(per * b_cols)) {
            let f = &f;
            s.spawn(move || {
                for ((row, ra), rb) in src.chunks_exact(cols).zip(da.chunks_exact_mut(a_cols)).zip(db.chunks_exact_mut(b_cols)) {
                    f(row, ra, rb);
                }
            });
        }
    });
}

pub fn quantize_fp8_channel(w: &[u16], cols: usize) -> (Vec<u8>, Vec<f32>) {
    let rows = w.len() / cols;
    let kb = cols.div_ceil(FP8_BLOCK);
    let mut values = vec![0u8; w.len()];
    let mut row_scales = vec![0f32; rows];
    par_rows(w, cols, (&mut values, cols), (&mut row_scales, 1), |row, out, scale| {
        let max = row.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max);
        scale[0] = if max == 0.0 { 1.0 } else { max / 448.0 };
        for (o, &v) in out.iter_mut().zip(row) {
            *o = f32_to_e4m3(bf16_to_f32(v) / scale[0]);
        }
    });
    let grid = row_scales.iter().flat_map(|&s| std::iter::repeat_n(s, kb)).collect::<Vec<_>>();
    let map = (0..rows).map(|r| (r * kb) as u32).collect::<Vec<_>>();
    (values, fp8_scales(&grid, &map))
}

pub fn nvfp4_global_scale(w: &[u16]) -> f32 {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = w.len().div_ceil(threads).max(1 << 16);
    let amax = std::thread::scope(|s| {
        let parts: Vec<_> = w.chunks(per).map(|c| s.spawn(move || c.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max))).collect();
        parts.into_iter().map(|h| h.join().unwrap_or(f32::NAN)).fold(0f32, f32::max)
    });
    if amax == 0.0 { 1.0 } else { 448.0 * 6.0 / amax }
}

pub fn quantize_nvfp4(w: &[u16], cols: usize) -> crate::nvfp4::Fp4<'static> {
    quantize_nvfp4_with(w, cols, nvfp4_global_scale(w))
}

fn head_rows_matrix<'a>(raw: Raw<'a>, quant: Quant, rows: std::ops::Range<usize>, what: &str) -> Result<Matrix<'a>> {
    if rows.start == 0 && rows.end == raw.rows() {
        return raw.into_matrix(quant, what);
    }
    ensure!(rows.end <= raw.rows(), "{what}: rows {rows:?} past its {} rows", raw.rows());
    match raw {
        Raw::Bf16 { w, cols } if quant == Quant::Nvfp4G16 => {
            let global = nvfp4_global_scale(&w);
            Ok(Matrix::Fp4(quantize_nvfp4_with(&w[rows.start * cols..rows.end * cols], cols, global)))
        }
        Raw::Bf16 { w: Cow::Borrowed(w), cols } => Raw::Bf16 { w: Cow::Borrowed(&w[rows.start * cols..rows.end * cols]), cols }.into_matrix(quant, what),
        other => other.slice_rows(rows.start, rows.end).into_matrix(quant, what),
    }
}

pub fn quantize_nvfp4_with(w: &[u16], cols: usize, global: f32) -> crate::nvfp4::Fp4<'static> {
    use crate::nvfp4::{Fp4, GROUP, f32_to_e2m1};
    let mut packed = vec![0u8; w.len() / 2];
    let mut scales = vec![0u8; w.len() / GROUP];
    par_rows(w, cols, (&mut packed, cols / 2), (&mut scales, cols / GROUP), |row, out, row_scales| {
        for ((g, group), code) in row.chunks_exact(GROUP).enumerate().zip(row_scales.iter_mut()) {
            let amax = group.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max);
            *code = f32_to_e4m3((amax / 6.0 * global).min(448.0));
            let s = e4m3_to_f32(*code) / global;
            for (i, &v) in group.iter().enumerate() {
                let q = if s > 0.0 { f32_to_e2m1(bf16_to_f32(v) / s) } else { 0 };
                let at = g * GROUP + i;
                out[at / 2] |= if at.is_multiple_of(2) { q } else { q << 4 };
            }
        }
    });
    let rows = w.len() / cols;
    Fp4::new(Cow::Owned(packed), Cow::Owned(scales), global, rows, cols).expect("nvfp4 head shape")
}

pub fn quantize_q8(w: &[u16], cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut values = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(w.len() / Q8_GROUP);
    for row in w.chunks_exact(cols) {
        for group in row.as_chunks::<Q8_GROUP>().0 {
            let max = group.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max);
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            scales.push(scale);
            values.extend(
                group.iter().map(|&v| (bf16_to_f32(v) / scale).round().clamp(-127.0, 127.0) as i8),
            );
        }
    }
    (values, scales)
}

pub fn interleave_rows<T: Copy>(a: &[T], b: &[T], cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    for (ra, rb) in a.chunks_exact(cols).zip(b.chunks_exact(cols)) {
        out.extend_from_slice(ra);
        out.extend_from_slice(rb);
    }
    out
}

impl Checkpoint {
    fn norm(&self, spec: &Spec, name: &str, len: usize) -> Result<Vec<f32>> {
        let offset = if spec.norm.unit_offset { 1.0 } else { 0.0 };
        Ok(self.f32(name, len)?.into_iter().map(|w| w + offset).collect())
    }

    fn attention(&self, spec: &Spec, attn: &str, block: Block) -> Result<Attn<'_>> {
        self.attention_as(spec, attn, false, block)
    }

    fn attention_as(&self, spec: &Spec, attn: &str, bf16: bool, block: Block) -> Result<Attn<'_>> {
        let a = &spec.attention;
        let quant = if bf16 { Quant::Bf16 } else { spec.quant };
        let (qkv_quant, o_quant) = if bf16 { (Quant::Bf16, Quant::Bf16) } else { (spec.requanted("qkv", quant), spec.requanted("o", quant)) };
        let (h, q, kv) = (spec.dims.hidden, spec.q_dim(), spec.kv_dim());
        let w = |p: &str, rows: usize, cols: usize| {
            self.raw(&format!("{attn}.{p}.weight"), rows, cols).map(|r| if bf16 { r.dequantized() } else { r })
        };
        let mut parts = Vec::new();
        let mut gates = Vec::new();
        if !a.projection.is_empty() {
            for p in &a.projection {
                let rows = crate::state::dim(spec, &p.rows)?;
                parts.push(if p.present_in(block) { w(&p.tensor, rows, h)? } else { Raw::Bf16 { w: Cow::Owned(vec![0; rows * h]), cols: h } });
            }
        } else if a.fused_qkv {
            ensure!(!a.output_gate, "fused qkv with an output gate is not supported");
            parts.push(w("qkv_proj", q + 2 * kv, h)?);
        } else {
            if a.output_gate {
                let both = w("q_proj", 2 * q, h)?;
                let hd = a.head_dim;
                for head in 0..a.heads {
                    parts.push(both.slice_rows(2 * head * hd, (2 * head + 1) * hd));
                    gates.push(both.slice_rows((2 * head + 1) * hd, (2 * head + 2) * hd));
                }
            } else {
                parts.push(w("q_proj", q, h)?);
            }
            parts.push(w("k_proj", kv, h)?);
            parts.push(w("v_proj", kv, h)?);
        }
        parts.extend(gates);
        let qkv_bias = if a.qkv_bias {
            ensure!(!a.output_gate, "qkv bias with an output gate is not supported");
            let mut b = self.f32(&format!("{attn}.q_proj.bias"), q)?;
            b.extend(self.f32(&format!("{attn}.k_proj.bias"), kv)?);
            b.extend(self.f32(&format!("{attn}.v_proj.bias"), kv)?);
            Some(b)
        } else {
            None
        };
        let (q_len, k_len) = if a.qk_norm_full { (q, kv) } else { (a.head_dim, a.head_dim) };
        let (q_norm, k_norm) = if a.qk_norm && !a.qk_norm_unit {
            (
                Some(self.norm(spec, &format!("{attn}.q_norm.weight"), q_len)?),
                Some(self.norm(spec, &format!("{attn}.k_norm.weight"), k_len)?),
            )
        } else {
            (None, None)
        };
        let mut extra = Vec::new();
        for bind in spec.attention.node.iter().filter(|n| n.present_in(block)).flat_map(|n| n.bind.values()) {
            if let crate::graph::Bind::Weight { weight, len } = bind
                && !extra.iter().any(|(n, _): &(String, Vec<f32>)| n == weight)
            {
                extra.push((weight.clone(), self.flat(&format!("{attn}.{weight}"), spec.weight_len(len)?)?));
            }
        }
        Ok(Attn {
            extra,
            qkv: Raw::concat(parts)?.into_matrix(qkv_quant, &format!("{attn}.qkv"))?,
            qkv_bias,
            o_bias: if a.o_bias { Some(self.f32(&format!("{attn}.o_proj.bias"), h)?) } else { None },
            q_norm,
            k_norm,
            o: w(a.o_tensor.as_deref().unwrap_or("o_proj"), h, spec.o_dim())?.into_matrix(o_quant, &format!("{attn}.{}", a.o_tensor.as_deref().unwrap_or("o_proj")))?,
        })
    }

    fn neg_a(&self, name: &str, heads: usize) -> Result<Vec<f32>> {
        let stored = self.f32(name, heads)?;
        let negated = self.gguf.as_ref().and_then(|g| g.naming()).is_some_and(|n| n.family.a_stored_negated);
        Ok(if negated { stored } else { stored.into_iter().map(|a| -a.exp()).collect() })
    }

    fn gated_delta_net(&self, spec: &Spec, p: &str) -> Result<Gdn<'_>> {
        let la = spec.linear_attention.as_ref().context("no [linear_attention]")?;
        let h = spec.dims.hidden;
        let (conv_dim, value_dim, heads) = (la.conv_dim(), la.value_dim(), la.value_heads);
        let w = |n: &str, rows: usize, cols: usize| self.raw(&format!("{p}.{n}.weight"), rows, cols);
        let (in_proj, ba) = if la.fused_interleaved {
            let key_dim = la.key_head_dim;
            let per_group = heads / la.key_heads;
            let value_per_group = per_group * la.value_head_dim;
            let mut pieces = w("in_proj_qkvz", la.in_proj_rows(), h)?.split_groups(la.key_heads, &[key_dim, key_dim, value_per_group, value_per_group])?;
            let z = pieces.pop().context("no z rows")?;
            let in_proj = Raw::concat(vec![pieces.remove(0), pieces.remove(0), pieces.remove(0), z])?;
            let pair = w("in_proj_ba", la.ba_rows(), h)?.split_groups(la.key_heads, &[per_group, per_group])?;
            let Raw::Bf16 { w: b, .. } = Raw::concat(pair)?.dequantized() else { bail!("{p}.in_proj_ba is not bf16") };
            (in_proj, b.into_owned())
        } else {
            let in_proj = Raw::concat(vec![w("in_proj_qkv", conv_dim, h)?, w("in_proj_z", value_dim, h)?])?;
            let mut ba = self.bf16(&format!("{p}.in_proj_b.weight"), &[heads, h])?.into_owned();
            ba.extend_from_slice(&self.bf16(&format!("{p}.in_proj_a.weight"), &[heads, h])?);
            (in_proj, ba)
        };
        Ok(Gdn {
            in_proj: in_proj.into_matrix(spec.requanted("lin_in", spec.quant), &format!("{p}.in_proj"))?,
            ba: Matrix::Bf16(Cow::Owned(ba)),
            conv: self.floats(&format!("{p}.conv1d.weight"), &[conv_dim, 1, la.conv_kernel])?,
            neg_a: self.neg_a(&format!("{p}.A_log"), heads)?,
            dt_bias: self.f32(&format!("{p}.dt_bias"), heads)?,
            norm: self.f32(&format!("{p}.norm.weight"), la.value_head_dim)?,
            out: w("out_proj", h, value_dim)?.into_matrix(spec.requanted("lin_out", spec.quant), &format!("{p}.out_proj"))?,
        })
    }

    fn described_mixer(&self, spec: &Spec, p: &str) -> Result<Described<'_>> {
        let m = spec.mixer.as_ref().context("no [mixer]")?;
        let h = spec.dims.hidden;
        let module = format!("{p}.{}", m.module);
        let mut parts = Vec::new();
        for proj in &m.projection {
            parts.push(self.raw(&format!("{module}.{}.weight", proj.tensor), crate::state::dim(spec, &proj.rows)?, h)?);
        }
        let mut extra: Vec<(String, Vec<f32>)> = Vec::new();
        for bind in m.node.iter().flat_map(|n| n.bind.values()) {
            if let crate::graph::Bind::Weight { weight, len } = bind
                && !extra.iter().any(|(n, _)| n == weight)
            {
                extra.push((weight.clone(), self.flat(&format!("{module}.{weight}"), spec.weight_len(len)?)?));
            }
        }
        Ok(Described {
            in_proj: Raw::concat(parts)?.into_matrix(spec.quant, &format!("{module}.in_proj"))?,
            out: self.raw(&format!("{module}.{}.weight", m.out_tensor), h, spec.mix_out_cols())?.into_matrix(spec.quant, &format!("{module}.out_proj"))?,
            extra,
        })
    }

    pub fn layer(&self, spec: &Spec, i: usize) -> Result<Layer<'_>> {
        self.layer_at(spec, i, i)
    }

    pub fn layer_at(&self, spec: &Spec, i: usize, global: usize) -> Result<Layer<'_>> {
        let (h, inter) = (spec.dims.hidden, spec.dims.intermediate);
        let p = format!("{}.layers.{global}", self.root());
        self.require_activation_scales(spec, i, global, &p)?;
        let mixer = if !spec.has_mixer(i) {
            Mixer::None
        } else if spec.block(i).is_linear() && spec.mixer.is_some() {
            Mixer::Described(self.described_mixer(spec, &p)?)
        } else if spec.block(i).is_linear() {
            Mixer::Linear(self.gated_delta_net(spec, &format!("{p}.linear_attn"))?)
        } else {
            Mixer::Attention(self.attention(spec, &format!("{p}.self_attn"), spec.block(i))?)
        };
        let mlp = spec.mlp_quant(i);
        let served = spec.served_mlp_quant(i);
        let sparse = spec.moe.as_ref().is_some_and(|m| m.is_sparse(global)) && spec.has_mlp(i);
        let moe = if sparse { Some(self.moe_layer(spec, &p, mlp)?) } else { None };
        let none = || (Matrix::Bf16(Cow::Owned(vec![0; 8])), Matrix::Bf16(Cow::Owned(vec![0; 8])));
        let (gate_up, down) = if sparse || !spec.has_mlp(i) {
            none()
        } else if spec.mlp.mapped.is_some() {
            let up = self.raw(&format!("{p}.mlp.up_proj.weight"), inter, h)?;
            let down = self.raw(&format!("{p}.mlp.down_proj.weight"), h, inter)?;
            (up.into_matrix(mlp, &format!("{p}.mlp.up_proj"))?, down.into_matrix(mlp, &format!("{p}.mlp.down_proj"))?)
        } else if mlp == Quant::Nvfp4G16 {
            let gate = self.fp4(&format!("{p}.mlp.gate_proj"), inter, h)?;
            let up = self.fp4(&format!("{p}.mlp.up_proj"), inter, h)?;
            (Matrix::Fp4(crate::nvfp4::Fp4::interleave(gate, up)?), Matrix::Fp4(self.fp4(&format!("{p}.mlp.down_proj"), h, inter)?))
        } else {
            let (gate, up) = if spec.mlp.fused_gate_up {
                let both = self.raw(&format!("{p}.mlp.gate_up_proj.weight"), 2 * inter, h)?;
                (both.slice_rows(0, inter), both.slice_rows(inter, 2 * inter))
            } else {
                (self.raw(&format!("{p}.mlp.gate_proj.weight"), inter, h)?, self.raw(&format!("{p}.mlp.up_proj.weight"), inter, h)?)
            };
            let down = self.raw(&format!("{p}.mlp.down_proj.weight"), h, inter)?;
            (
                Raw::interleave(gate, up)?.into_matrix(served, &format!("{p}.mlp.gate_up"))?,
                down.into_matrix(served, &format!("{p}.mlp.down_proj"))?,
            )
        };
        let ln1 = if spec.norm.pre { self.norm(spec, &format!("{p}.input_layernorm.weight"), h)? } else { vec![0.0; 1] };
        let ln2 = if spec.layers.split { ln1.clone() } else { self.norm(spec, &format!("{p}.post_attention_layernorm.weight"), h)? };
        let branch = |w: Vec<f32>| match spec.norm.residual_scale.filter(|_| spec.norm.sandwich) {
            Some(s) => w.into_iter().map(|v| (v as f64 * s) as f32).collect(),
            None => w,
        };
        Ok(Layer {
            ln1,
            ln2: if spec.norm.sandwich { branch(ln2) } else { ln2 },
            ln_pre_mlp: if spec.norm.sandwich && spec.norm.pre { Some(self.norm(spec, &format!("{p}.pre_feedforward_layernorm.weight"), h)?) } else { None },
            ln_post_mlp: if spec.norm.sandwich { Some(branch(self.norm(spec, &format!("{p}.post_feedforward_layernorm.weight"), h)?)) } else { None },
            mixer,
            gate_up,
            down,
            moe,
        })
    }

    fn gguf_stack(&self, g: &Gguf, name: &str, shape: &[usize], out: &mut Vec<u16>) -> Result<()> {
        let mapped = name.to_string();
        let info = g.tensor(&mapped).with_context(|| format!("{mapped} is not in the GGUF file"))?;
        ensure!(info.shape == shape, "{mapped} has shape {:?}, expected {shape:?}", info.shape);
        let mut values = vec![0f32; shape.iter().product()];
        crate::gguf::dequantize(info.kind, g.data(info), &mut values).with_context(|| mapped.clone())?;
        out.extend(values.into_iter().map(f32_to_bf16));
        Ok(())
    }

    fn gguf_moe_layer(&self, g: &Gguf, spec: &Spec, global: usize) -> Result<MoeWeights> {
        let m = spec.moe.as_ref().context("no [moe]")?;
        let (h, inter, e) = (spec.dims.hidden, m.expert_intermediate, m.experts);
        let shared = m.shared_intermediate > 0;
        ensure!(!shared || m.shared_intermediate == inter, "the shared expert must be as wide as the routed experts");
        let blk = format!("blk.{global}");
        let stack = |which: &str, rows: usize, cols: usize| -> Result<Stacked> {
            let mut w = Vec::with_capacity((e + 1) * rows * cols);
            self.gguf_stack(g, &format!("{blk}.ffn_{which}_exps.weight"), &[e, rows, cols], &mut w)?;
            if shared {
                self.gguf_stack(g, &format!("{blk}.ffn_{which}_shexp.weight"), &[rows, cols], &mut w)?;
            }
            Ok(Stacked::Bf16(w))
        };
        let mut router = Vec::with_capacity(e * h);
        self.gguf_stack(g, &format!("{blk}.ffn_gate_inp.weight"), &[e, h], &mut router)?;
        let shared_gate = if m.shared_gate {
            let name = format!("{blk}.ffn_gate_inp_shexp.weight");
            let info = g.tensor(&name).with_context(|| format!("{name} is not in the GGUF file"))?;
            let mut v = vec![0f32; h];
            crate::gguf::dequantize(info.kind, g.data(info), &mut v)?;
            Some(v)
        } else {
            None
        };
        Ok(MoeWeights { router, shared_gate, score_bias: None, router_bias: None, gate_up_bias: None, down_bias: None, hash_table: None, gate: stack("gate", inter, h)?, up: stack("up", inter, h)?, down: stack("down", h, inter)? })
    }

    fn moe_layer(&self, spec: &Spec, p: &str, quant: Quant) -> Result<MoeWeights> {
        if let Some(g) = &self.gguf {
            let global: usize = p.rsplit('.').next().and_then(|n| n.parse().ok()).context("GGUF layer index")?;
            return self.gguf_moe_layer(g, spec, global);
        }
        let m = spec.moe.as_ref().context("no [moe]")?;
        let (h, inter, e) = (spec.dims.hidden, m.expert_intermediate, m.experts);
        let (local, first) = (m.local_experts(), m.first_expert());
        let shared_slots = m.shared_slots();
        let total = local + shared_slots;
        let shared_module = format!("{p}.mlp.{}", m.shared_module);
        let module = |i: usize, which: &str| {
            if i < local { format!("{p}.mlp.experts.{}.{which}_proj", first + i) } else { format!("{shared_module}.{which}_proj") }
        };
        let stack = |which: &str, rows: usize, cols: usize| -> Result<Stacked> {
            if quant == Quant::Fp8Block128 {
                let mut values = Vec::with_capacity(total * rows * cols);
                let mut grid = Vec::new();
                for i in 0..total {
                    let Raw::Fp8 { w, grid: g, .. } = self.raw(&format!("{}.weight", module(i, which)), rows, cols)? else {
                        bail!("{} is not FP8", module(i, which));
                    };
                    values.extend_from_slice(&w);
                    grid.extend_from_slice(&g);
                }
                Ok(Stacked::Fp8 { values, grid })
            } else if quant == Quant::Nvfp4G16 {
                let mut packed = Vec::with_capacity(total * rows * cols / 2);
                let mut scales = Vec::with_capacity(total * rows * cols / crate::nvfp4::GROUP);
                let mut inv_global = Vec::with_capacity(total);
                let mut input_global = f32::MAX;
                for i in 0..total {
                    let f = self.fp4(&module(i, which), rows, cols)?;
                    packed.extend_from_slice(&f.packed);
                    scales.extend_from_slice(&f.scales);
                    inv_global.push(1.0 / f.global);
                    if f.input_global > 0.0 {
                        input_global = input_global.min(f.input_global);
                    }
                }
                Ok(Stacked::Fp4 { packed, scales, inv_global, input_global })
            } else {
                let mut w = Vec::with_capacity(total * rows * cols);
                for i in 0..local {
                    w.extend_from_slice(&self.bf16(&format!("{}.weight", module(i, which)), &[rows, cols])?);
                }
                if shared_slots == 1 {
                    w.extend_from_slice(&self.bf16(&format!("{}.weight", module(local, which)), &[rows, cols])?);
                } else if shared_slots > 1 {
                    let name = format!("{}.weight", module(local, which));
                    if which == "down" {
                        let all = self.bf16(&name, &[rows, cols * shared_slots])?;
                        for s in 0..shared_slots {
                            for r in 0..rows {
                                let start = r * cols * shared_slots + s * cols;
                                w.extend_from_slice(&all[start..start + cols]);
                            }
                        }
                    } else {
                        w.extend_from_slice(&self.bf16(&name, &[rows * shared_slots, cols])?);
                    }
                }
                Ok(Stacked::Bf16(w))
            }
        };
        let shared_gate = if m.shared_gate { Some(self.floats(&format!("{shared_module}_gate.weight"), &[1, h])?) } else { None };
        let (gate, up, down) = match &m.expert_block {
            Some(format) => {
                ensure!(shared_slots == 0 && local == e, "block quantised experts support neither shared experts nor an expert split yet");
                let (gate, up) = self.block_gate_up(format, &format!("{p}.mlp.experts.gate_up_proj"), e, inter, h, m.gate_up_interleaved)?;
                (gate, up, self.block_stack(format, &format!("{p}.mlp.experts.down_proj"), e, h, inter)?)
            }
            None if m.experts_transposed => {
                ensure!(quant == Quant::Bf16, "transposed fused experts are bf16 only");
                ensure!(shared_slots <= 1, "transposed fused experts take at most one shared expert");
                let fused = self.bf16(&format!("{p}.mlp.experts.gate_up_proj"), &[e, h, 2 * inter])?;
                let stored_down = self.bf16(&format!("{p}.mlp.experts.down_proj"), &[e, inter, h])?;
                let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
                for x in 0..e {
                    let (g, d) = (&fused[x * h * 2 * inter..(x + 1) * h * 2 * inter], &stored_down[x * inter * h..(x + 1) * inter * h]);
                    for r in 0..inter {
                        gate.extend((0..h).map(|c| g[c * 2 * inter + r]));
                        up.extend((0..h).map(|c| g[c * 2 * inter + inter + r]));
                    }
                    for r in 0..h {
                        down.extend((0..inter).map(|c| d[c * h + r]));
                    }
                }
                if shared_slots == 1 {
                    gate.extend_from_slice(&self.bf16(&format!("{shared_module}.gate_proj.weight"), &[inter, h])?);
                    up.extend_from_slice(&self.bf16(&format!("{shared_module}.up_proj.weight"), &[inter, h])?);
                    down.extend_from_slice(&self.bf16(&format!("{shared_module}.down_proj.weight"), &[h, inter])?);
                }
                (Stacked::Bf16(gate), Stacked::Bf16(up), Stacked::Bf16(down))
            }
            None if m.fused_experts => {
                ensure!(quant == Quant::Bf16, "fused experts are bf16 only");
                let shared = m.shared_intermediate;
                let fused = self.bf16(&format!("{p}.mlp.experts.gate_up_proj.weight"), &[e, 2 * inter, h])?;
                let (mut gate, mut up) = split_gate_up(&fused, h, e, inter, false);
                let mut down = self.bf16(&format!("{p}.mlp.experts.down_proj.weight"), &[e, h, inter])?.into_owned();
                if shared > 0 {
                    let both = self.bf16(&format!("{shared_module}.input_linear.weight"), &[2 * shared, h])?;
                    gate.extend_from_slice(&both[..shared * h]);
                    up.extend_from_slice(&both[shared * h..]);
                    let all = self.bf16(&format!("{shared_module}.output_linear.weight"), &[h, shared])?;
                    for s in 0..shared_slots {
                        for r in 0..h {
                            let start = r * shared + s * inter;
                            down.extend_from_slice(&all[start..start + inter]);
                        }
                    }
                }
                (Stacked::Bf16(gate), Stacked::Bf16(up), Stacked::Bf16(down))
            }
            None if spec.mlp.mapped.is_some() => (Stacked::Bf16(vec![0; 8]), stack("up", inter, h)?, stack("down", h, inter)?),
            None => (stack("gate", inter, h)?, stack("up", inter, h)?, stack("down", h, inter)?),
        };
        Ok(MoeWeights {
            router: self.bf16(&format!("{p}.mlp.{}.weight", m.router_module), &[e, h])?.into_owned(),
            shared_gate,
            score_bias: if m.score_bias { Some(self.floats(&format!("{p}.mlp.{}.e_score_correction_bias", m.router_module), &[e])?) } else { None },
            router_bias: if m.router_bias { Some(self.floats(&format!("{p}.mlp.{}.bias", m.router_module), &[e])?) } else { None },
            gate_up_bias: if m.expert_bias { Some(self.expert_gate_up_bias(&format!("{p}.mlp.experts.gate_up_proj_bias"), e, inter, m.gate_up_interleaved)?) } else { None },
            down_bias: if m.expert_bias { Some(self.floats(&format!("{p}.mlp.experts.down_proj_bias"), &[e, h])?) } else { None },
            hash_table: match (&m.hash_table, m.is_hashed(layer_index(p))) {
                (Some(t), true) => Some(self.int_table(&format!("{p}.mlp.{}.{t}", m.router_module), m.top_k())?),
                _ => None,
            },
            gate,
            up,
            down,
        })
    }

    fn expert_gate_up_bias(&self, name: &str, experts: usize, inter: usize, interleaved: bool) -> Result<Vec<f32>> {
        let fused = self.floats(name, &[experts, 2 * inter])?;
        let mut out = Vec::with_capacity(fused.len());
        for e in 0..experts {
            let row = &fused[e * 2 * inter..(e + 1) * 2 * inter];
            let part = |which: usize| (0..inter).map(move |r| row[if interleaved { 2 * r + which } else { which * inter + r }]);
            out.extend(part(0));
            out.extend(part(1));
        }
        Ok(out)
    }

    fn int_table(&self, name: &str, width: usize) -> Result<Vec<i32>> {
        let v = self.view(name)?;
        ensure!(v.shape.len() == 2 && v.shape[1] == width, "{name} has shape {:?}, expected [vocab, {width}]", v.shape);
        match v.dtype {
            Dtype::I64 => Ok(v.data.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b) as i32).collect()),
            Dtype::I32 => Ok(v.data.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect()),
            other => bail!("{name} is {other:?}, expected an integer table"),
        }
    }

    fn block_stack(&self, format: &crate::dequant::BlockFormat, name: &str, experts: usize, rows: usize, cols: usize) -> Result<Stacked> {
        let blocks = self.view(&format!("{name}_blocks"))?;
        let scales = self.view(&format!("{name}_scales"))?;
        ensure!(blocks.dtype == Dtype::U8 && scales.dtype == Dtype::U8, "{name} blocks and scales must be U8");
        ensure!(blocks.shape.first() == Some(&experts) && scales.shape.first() == Some(&experts), "{name} is not stacked over {experts} experts");
        ensure!(blocks.data.len() == format.packed_len(experts * rows * cols), "{name}_blocks holds {} bytes, expected {} for {experts}x{rows}x{cols}", blocks.data.len(), format.packed_len(experts * rows * cols));
        if format.device_packed() {
            ensure!(cols.is_multiple_of(format.group), "{name} has {cols} columns, not a whole number of groups of {}", format.group);
            ensure!(scales.data.len() == experts * rows * cols / format.group, "{name}_scales holds {} bytes, expected {}", scales.data.len(), experts * rows * cols / format.group);
            let padded = format.padded_cols(cols);
            let packed = format.pad_rows(blocks.data, cols / 2, padded / 2);
            let scales = format.pad_rows(scales.data, cols / format.group, padded / format.group);
            return Ok(Stacked::Block { packed, scales });
        }
        Ok(Stacked::Bf16(format.decode_bf16(blocks.data, scales.data).with_context(|| name.to_string())?))
    }

    fn block_gate_up(&self, format: &crate::dequant::BlockFormat, name: &str, experts: usize, inter: usize, h: usize, interleaved: bool) -> Result<(Stacked, Stacked)> {
        Ok(match self.block_stack(format, name, experts, 2 * inter, h)? {
            Stacked::Block { packed, scales } => {
                let padded = format.padded_cols(h);
                let (gp, up) = split_gate_up(&packed, padded / 2, experts, inter, interleaved);
                let (gs, us) = split_gate_up(&scales, padded / format.group, experts, inter, interleaved);
                (Stacked::Block { packed: gp, scales: gs }, Stacked::Block { packed: up, scales: us })
            }
            Stacked::Bf16(fused) => {
                let (gate, up) = split_gate_up(&fused, h, experts, inter, interleaved);
                (Stacked::Bf16(gate), Stacked::Bf16(up))
            }
            _ => unreachable!("block stacks are packed or bf16"),
        })
    }

    fn has_input_scale(&self, module: &str) -> bool {
        self.has(&format!("{module}.input_scale")) || self.has(&format!("{module}.input_global_scale"))
    }

    fn require_activation_scales(&self, spec: &Spec, i: usize, global: usize, p: &str) -> Result<()> {
        use crate::spec::ActQuant;
        let mut modules: Vec<String> = Vec::new();
        if spec.mlp.act_quant == ActQuant::Fp4 && spec.mlp_quant(i) == Quant::Nvfp4G16 && !spec.moe.as_ref().is_some_and(|m| m.is_sparse(global)) {
            modules.extend(["gate_proj", "up_proj", "down_proj"].map(|m| format!("{p}.mlp.{m}")));
        }
        if let Some(missing) = modules.iter().find(|m| !self.has_input_scale(m)) {
            bail!("layer {global}: {missing} has no input_scale, so this checkpoint is weight-only for those layers; remove act_quant (and [mlp] act_quant) from the spec");
        }
        Ok(())
    }

    fn is_fp4(&self, module: &str) -> bool {
        self.has(&format!("{module}.weight_packed"))
            || (self.has(&format!("{module}.weight_scale_2")) && self.view(&format!("{module}.weight")).is_ok_and(|v| v.dtype == Dtype::U8))
    }

    fn scalar(&self, name: &str) -> Result<f32> {
        let shape = self.view(name)?.shape;
        ensure!(shape.iter().product::<usize>() == 1, "{name} has shape {shape:?}, expected one value");
        Ok(self.floats(name, &shape)?[0])
    }

    fn global_scale(&self, module: &str, divisor: &str, multiplier: &str) -> Result<Option<f32>> {
        if self.has(&format!("{module}.{divisor}")) {
            return Ok(Some(self.scalar(&format!("{module}.{divisor}"))?));
        }
        if self.has(&format!("{module}.{multiplier}")) {
            return Ok(Some(1.0 / self.scalar(&format!("{module}.{multiplier}"))?));
        }
        Ok(None)
    }

    pub fn fp4(&self, module: &str, rows: usize, cols: usize) -> Result<crate::nvfp4::Fp4<'_>> {
        use crate::nvfp4::GROUP;
        let weight = if self.has(&format!("{module}.weight_packed")) { format!("{module}.weight_packed") } else { format!("{module}.weight") };
        let packed = self.shaped(&weight, &[rows, cols / 2])?;
        ensure!(packed.dtype == Dtype::U8, "{weight} is {:?}, expected U8", packed.dtype);
        let scales = self.shaped(&format!("{module}.weight_scale"), &[rows, cols / GROUP])?;
        ensure!(scales.dtype == Dtype::F8_E4M3, "{module}.weight_scale is {:?}, expected F8_E4M3", scales.dtype);
        let global = self.global_scale(module, "weight_global_scale", "weight_scale_2")?.with_context(|| format!("{module} has no weight_global_scale or weight_scale_2"))?;
        let mut m = crate::nvfp4::Fp4::new(Cow::Borrowed(packed.data), Cow::Borrowed(scales.data), global, rows, cols)?;
        m.input_global = self.global_scale(module, "input_global_scale", "input_scale")?.unwrap_or(0.0);
        Ok(m)
    }

    pub fn mtp<'a>(&'a self, spec: &Spec) -> Result<MtpWeights<'a>> {
        let (h, inter) = (spec.dims.hidden, spec.dims.intermediate);
        let lay = spec.mtp_layout();
        ensure!(self.has(&lay.fc), "the checkpoint has no {}", lay.fc);
        let bf16 = |r: Raw<'a>, what: &str| -> Result<Matrix<'a>> { r.dequantized().into_matrix(Quant::Bf16, what) };
        let p = lay.layer.as_str();
        let attn = self.attention_as(spec, &format!("{p}.self_attn"), true, Block::Full)?;
        let sparse = spec.moe.is_some();
        let (gate_up, down, moe) = if sparse {
            let dummy = || Matrix::Bf16(Cow::Owned(vec![0; 8]));
            (dummy(), dummy(), Some(self.moe_layer(spec, p, if spec.quant == Quant::Fp8Block128 { Quant::Fp8Block128 } else { Quant::Bf16 })?))
        } else {
            let gate = self.raw(&format!("{p}.mlp.gate_proj.weight"), inter, h)?.dequantized();
            let up = self.raw(&format!("{p}.mlp.up_proj.weight"), inter, h)?.dequantized();
            let down = self.raw(&format!("{p}.mlp.down_proj.weight"), h, inter)?;
            (Raw::interleave(gate, up)?.into_matrix(Quant::Bf16, "mtp gate_up")?, bf16(down, "mtp down_proj")?, None)
        };
        Ok(MtpWeights {
            fc: bf16(self.raw(&lay.fc, h, 2 * h)?, "mtp.fc")?,
            norm_embed: self.norm(spec, &lay.norm_embed, h)?,
            norm_hidden: self.norm(spec, &lay.norm_hidden, h)?,
            layer: Layer {
                ln1: self.norm(spec, &format!("{p}.input_layernorm.weight"), h)?,
                ln2: self.norm(spec, &format!("{p}.post_attention_layernorm.weight"), h)?,
                ln_pre_mlp: None,
                ln_post_mlp: None,
                mixer: Mixer::Attention(attn),
                gate_up,
                down,
                moe,
            },
            norm: self.norm(spec, &lay.norm, h)?,
        })
    }

    pub fn outside(&self, spec: &Spec) -> Result<Outside<'_>> {
        let embed = self.embedding(spec)?;
        let (norm, lm_head) = self.head_from(spec, &embed)?;
        Ok(Outside { norm, lm_head, embed })
    }

    pub fn embedding(&self, spec: &Spec) -> Result<Cow<'_, [u16]>> {
        let d = &spec.dims;
        self.bf16(&format!("{}.embed_tokens.weight", self.root()), &[d.vocab, d.hidden])
    }

    pub fn head(&self, spec: &Spec) -> Result<(Vec<f32>, Matrix<'_>)> {
        let embed = if self.gguf.is_some() || !spec.dims.tie_embeddings { Cow::Borrowed(&[][..]) } else { self.embedding(spec)? };
        self.head_from(spec, &embed)
    }

    pub fn embedding_rows(&self, spec: &Spec) -> Result<DeviceRows<'_>> {
        let d = &spec.dims;
        if self.gguf.is_some()
            && let Raw::Ggml(rows) = self.raw(&format!("{}.embed_tokens.weight", self.root()), d.vocab, d.hidden)?
        {
            return Ok(rows);
        }
        DeviceRows::from_bf16(&self.embedding(spec)?, d.hidden)
    }

    fn head_from<'s>(&'s self, spec: &Spec, embed: &Cow<'s, [u16]>) -> Result<(Vec<f32>, Matrix<'s>)> {
        let d = &spec.dims;
        let norm = self.norm(spec, &format!("{}.norm.weight", self.root()), d.hidden)?;
        let lm_head = head_rows_matrix(self.head_raw(spec, embed)?, spec.lm_head_quant(), spec.head_rows(), "lm_head")?;
        Ok((norm, lm_head))
    }

    fn head_name(&self) -> String {
        let nested = self.root().strip_suffix(".model").map(|p| format!("{p}.lm_head.weight"));
        match nested {
            Some(n) if self.has(&n) => n,
            _ => "lm_head.weight".to_string(),
        }
    }

    fn head_raw<'s>(&'s self, spec: &Spec, embed: &Cow<'s, [u16]>) -> Result<Raw<'s>> {
        let d = &spec.dims;
        if d.tie_embeddings && self.gguf.is_some() {
            self.raw(&format!("{}.embed_tokens.weight", self.root()), d.vocab, d.hidden)
        } else if d.tie_embeddings {
            Ok(Raw::Bf16 { w: embed.clone(), cols: d.hidden })
        } else {
            self.raw(&self.head_name(), d.vocab, d.hidden)
        }
    }

    pub fn draft_head_bf16(&self, spec: &Spec) -> Result<Option<(Cow<'_, [u16]>, usize)>> {
        let embed = self.bf16(&format!("{}.embed_tokens.weight", self.root()), &[spec.dims.vocab, spec.dims.hidden])?;
        Ok(match self.head_raw(spec, &embed)? {
            Raw::Bf16 { w, cols } => Some((w, cols)),
            _ => None,
        })
    }

    pub fn draft_head(&self, spec: &Spec, quant: Quant) -> Result<Matrix<'_>> {
        if let Some(name) = spec.mtp_layout().head {
            return head_rows_matrix(self.raw(&name, spec.dims.vocab, spec.dims.hidden)?, quant, spec.head_rows(), "mtp lm_head");
        }
        let embed = self.bf16(&format!("{}.embed_tokens.weight", self.root()), &[spec.dims.vocab, spec.dims.hidden])?;
        head_rows_matrix(self.head_raw(spec, &embed)?, quant, spec.head_rows(), "mtp lm_head")
    }
}

pub fn find_in_snapshot(repo: &str, file: &str) -> Result<PathBuf> {
    let local = Path::new(repo).join(file);
    if Path::new(repo).is_dir() && local.exists() {
        return Ok(local);
    }
    let snapshots =
        hub_dir().join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    for entry in std::fs::read_dir(&snapshots).with_context(|| snapshots.display().to_string())? {
        let path = entry?.path().join(file);
        if path.exists() {
            return Ok(path);
        }
    }
    bail!("no {file} for {repo} under {}", snapshots.display())
}

pub struct Feed {
    done: u64,
    total: u64,
    sizes: Vec<u64>,
    reader: Option<std::sync::Arc<crate::loader::Prefetcher>>,
}

impl Feed {
    pub fn new(
        ck: &Checkpoint,
        groups: Vec<Vec<String>>,
        opts: &crate::loader::LoaderOptions,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        budget: std::sync::Arc<crate::loader::Budget>,
    ) -> Result<Feed> {
        let mut items = Vec::with_capacity(groups.len());
        for g in &groups {
            items.push(ck.extents(g)?);
        }
        let sizes: Vec<u64> = items.iter().map(|g| g.iter().map(|i| i.len).sum()).collect();
        let total = sizes.iter().sum();
        let reader = if opts.enabled && ck.is_safetensors() && !groups.is_empty() && !ck.cached(opts) {
            Some(std::sync::Arc::new(crate::loader::Prefetcher::start(&ck.files, items, *opts, Some(ctx.clone()), budget)?))
        } else {
            None
        };
        crate::progress::report(crate::progress::Stage::Weights, 0, total);
        Ok(Feed { done: 0, total, sizes, reader })
    }

    pub fn direct(&self) -> bool {
        self.reader.as_ref().is_some_and(|r| r.direct)
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn reader(&self) -> Option<std::sync::Arc<crate::loader::Prefetcher>> {
        self.reader.clone()
    }

    pub fn size(&self, group: usize) -> u64 {
        self.sizes.get(group).copied().unwrap_or(0)
    }

    pub fn finished(&mut self, group: usize) {
        if let Some(&size) = self.sizes.get(group) {
            self.done += size;
            crate::progress::report(crate::progress::Stage::Weights, self.done, self.total);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_in_snapshot_reads_a_local_checkpoint_directory() {
        let dir = std::env::temp_dir().join(format!("specrt-local-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        let repo = dir.display().to_string();
        assert_eq!(find_in_snapshot(&repo, "tokenizer.json").unwrap(), dir.join("tokenizer.json"));
        assert!(find_in_snapshot(&repo, "missing.json").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fused_interleaved_rows_split_by_key_group() {
        let rows: Vec<u16> = (0..12).collect();
        let fused = Raw::Bf16 { w: Cow::Owned(rows), cols: 1 };
        let parts = fused.split_groups(2, &[1, 2, 3]).unwrap();
        let flat = |r: &Raw| match r {
            Raw::Bf16 { w, .. } => w.to_vec(),
            _ => unreachable!(),
        };
        assert_eq!(flat(&parts[0]), vec![0, 6]);
        assert_eq!(flat(&parts[1]), vec![1, 2, 7, 8]);
        assert_eq!(flat(&parts[2]), vec![3, 4, 5, 9, 10, 11]);
        assert!(Raw::Bf16 { w: Cow::Owned(vec![0; 5]), cols: 1 }.split_groups(2, &[1, 2]).is_err());
    }

    #[test]
    fn f16_converts_to_bf16() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(f32_to_bf16(f16_to_f32(0x3555))), 0.333_984_38);
    }

    #[test]
    fn gguf_expert_tensors_become_stacked_experts_with_the_shared_expert_last() {
        use crate::gguf::writer::{Kv, build};
        let (h, inter, e) = (256usize, 16usize, 3usize);
        let f32s = |n: usize, base: f32| -> Vec<u8> { (0..n).flat_map(|i| (base + i as f32).to_le_bytes()).collect() };
        let gate = f32s(e * inter * h, 0.0);
        let shared_gate = f32s(inter * h, 1000.0);
        let router = f32s(e * h, 0.0);
        let gate_inp_shexp = f32s(h, 5.0);
        let up = f32s(e * inter * h, 0.0);
        let shared_up = f32s(inter * h, 0.0);
        let down = f32s(e * h * inter, 0.0);
        let shared_down = f32s(h * inter, 0.0);
        let embd = f32s(8 * h, 0.0);
        let bytes = build(
            3,
            &[
                ("general.architecture", Kv::Str("qwen3moe")),
                ("qwen3moe.block_count", Kv::U32(1)),
                ("qwen3moe.embedding_length", Kv::U32(h as u32)),
                ("qwen3moe.attention.head_count", Kv::U32(2)),
                ("qwen3moe.attention.head_count_kv", Kv::U32(1)),
                ("qwen3moe.attention.key_length", Kv::U32(4)),
                ("qwen3moe.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
                ("qwen3moe.expert_count", Kv::U32(e as u32)),
                ("qwen3moe.expert_used_count", Kv::U32(2)),
                ("qwen3moe.expert_feed_forward_length", Kv::U32(inter as u32)),
                ("qwen3moe.expert_shared_feed_forward_length", Kv::U32(inter as u32)),
            ],
            &[
                ("token_embd.weight", &[h as u64, 8], 0, &embd),
                ("blk.0.ffn_gate_inp.weight", &[h as u64, e as u64], 0, &router),
                ("blk.0.ffn_gate_inp_shexp.weight", &[h as u64], 0, &gate_inp_shexp),
                ("blk.0.ffn_gate_exps.weight", &[h as u64, inter as u64, e as u64], 0, &gate),
                ("blk.0.ffn_up_exps.weight", &[h as u64, inter as u64, e as u64], 0, &up),
                ("blk.0.ffn_down_exps.weight", &[inter as u64, h as u64, e as u64], 0, &down),
                ("blk.0.ffn_gate_shexp.weight", &[h as u64, inter as u64], 0, &shared_gate),
                ("blk.0.ffn_up_shexp.weight", &[h as u64, inter as u64], 0, &shared_up),
                ("blk.0.ffn_down_shexp.weight", &[inter as u64, h as u64], 0, &shared_down),
            ],
        );
        let g = Gguf::from_bytes(bytes).unwrap();
        let spec = g.spec().unwrap();
        let m = spec.moe.as_ref().unwrap();
        assert_eq!((m.experts, m.top_k(), m.expert_intermediate, m.shared_intermediate, m.shared_gate), (e, 2, inter, inter, true));
        spec.check_moe_runnable().unwrap();
        let ck = Checkpoint { gguf: Some(std::sync::Arc::new(g)), shards: Default::default(), names: Default::default(), staged: None, cut: None, owned: Default::default(), files: Vec::new() };
        let w = ck.moe_layer(&spec, "model.layers.0", Quant::Bf16).unwrap();
        let Stacked::Bf16(gate) = &w.gate else { panic!("bf16 stack") };
        assert_eq!(gate.len(), (e + 1) * inter * h);
        assert_eq!(bf16_to_f32(gate[0]), 0.0);
        assert_eq!(bf16_to_f32(gate[inter * h]), 4096.0);
        assert_eq!(bf16_to_f32(gate[e * inter * h]), 1000.0);
        assert_eq!(w.router.len(), e * h);
        assert_eq!(w.shared_gate.as_deref(), Some(&(0..h).map(|v| 5.0 + v as f32).collect::<Vec<_>>()[..]));
    }

    #[test]
    fn fp8_scale_map_follows_rows_through_split_concat_and_interleave() {
        let (rows, cols, kb) = (384usize, 256usize, 2usize);
        let source = |tag: f32| {
            let grid: Vec<f32> = (0..rows / FP8_BLOCK * kb).map(|i| tag + i as f32).collect();
            let map = (0..rows).map(|r| (r / FP8_BLOCK * kb) as u32).collect();
            Raw::Fp8 { w: Cow::Owned(vec![0; rows * cols]), grid, map, cols }
        };
        let expect = |tag: f32, r: usize, b: usize| tag + (r / FP8_BLOCK * kb + b) as f32;
        let lookup = |raw: &Raw, r: usize, b: usize| {
            let Raw::Fp8 { grid, map, .. } = raw else { unreachable!() };
            let scales = fp8_scales(grid, map);
            scales[scales[r].to_bits() as usize + b]
        };
        let a = source(1000.0);
        let fused = Raw::concat(vec![a.slice_rows(200, 300), source(2000.0), a.slice_rows(10, 140)]).unwrap();
        assert_eq!(fused.rows(), 100 + rows + 130);
        let Raw::Fp8 { grid, .. } = &fused else { unreachable!() };
        assert_eq!(grid.len(), 2 * kb + rows / FP8_BLOCK * kb + 2 * kb);
        for b in 0..kb {
            assert_eq!(lookup(&fused, 0, b), expect(1000.0, 200, b));
            assert_eq!(lookup(&fused, 99, b), expect(1000.0, 299, b));
            assert_eq!(lookup(&fused, 100 + 130, b), expect(2000.0, 130, b));
            assert_eq!(lookup(&fused, 100 + rows + 120, b), expect(1000.0, 130, b));
        }
        let both = Raw::interleave(source(1000.0), source(2000.0)).unwrap();
        for r in [0, 127, 128, 383] {
            for b in 0..kb {
                assert_eq!(lookup(&both, 2 * r, b), expect(1000.0, r, b));
                assert_eq!(lookup(&both, 2 * r + 1, b), expect(2000.0, r, b));
            }
        }
    }

    fn exhaustive_e4m3(v: f32) -> u8 {
        let sign = if v.is_sign_negative() { 0x80 } else { 0 };
        let a = v.abs().min(448.0);
        let mut best = 0u8;
        for code in 1..=0x7eu8 {
            let (d, b) = ((e4m3_to_f32(code) - a).abs(), (e4m3_to_f32(best) - a).abs());
            if d < b || (d == b && code & 1 == 0) {
                best = code;
            }
        }
        best | sign
    }

    #[test]
    fn e4m3_encode_matches_the_nearest_even_search_on_every_bf16_and_random_f32() {
        for bits in 0..=u16::MAX {
            let v = bf16_to_f32(bits);
            if v.is_finite() {
                assert_eq!(f32_to_e4m3(v), exhaustive_e4m3(v), "{v} ({bits:#06x})");
            }
        }
        let mut x = 0x2545f491u32;
        for i in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = (x as f32 / u32::MAX as f32 - 0.5) * if i % 2 == 0 { 1000.0 } else { 0.05 };
            assert_eq!(f32_to_e4m3(v), exhaustive_e4m3(v), "{v}");
        }
        assert_eq!(f32_to_e4m3(f32::MAX), 0x7e);
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
    }

    fn head_rows(rows: usize, cols: usize) -> Vec<u16> {
        (0..rows * cols)
            .map(|i| {
                let r = i / cols;
                let scale = [0.001, 0.7, 3.0, 0.0][r % 4];
                f32_to_bf16(((i * 37 % 101) as f32 - 50.0) * scale * if i % 29 == 0 { 9.0 } else { 1.0 })
            })
            .collect()
    }

    #[test]
    fn fp8_channel_head_scales_each_row_by_its_absmax_and_rounds_to_nearest() {
        let (rows, cols) = (5, 256);
        let w = head_rows(rows, cols);
        let (values, scales) = quantize_fp8_channel(&w, cols);
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let amax = row.iter().map(|&v| bf16_to_f32(v).abs()).fold(0f32, f32::max);
            let s = scales[scales[r].to_bits() as usize];
            assert_eq!(s, if amax == 0.0 { 1.0 } else { amax / 448.0 });
            for b in 0..cols / FP8_BLOCK {
                assert_eq!(scales[scales[r].to_bits() as usize + b], s);
            }
            for c in 0..cols {
                assert_eq!(values[r * cols + c], exhaustive_e4m3(bf16_to_f32(row[c]) / s), "{r} {c}");
            }
        }
    }

    #[test]
    fn nvfp4_head_matches_the_reference_quantiser_with_a_global_scale_from_the_absmax() {
        let (rows, cols) = (6, 64);
        let w = head_rows(rows, cols);
        let m = quantize_nvfp4(&w, cols);
        let f: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
        let amax = f.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(m.global, 448.0 * 6.0 / amax);
        let (packed, scales) = crate::nvfp4::quantize(&f, cols, m.global);
        assert_eq!(m.packed.as_ref(), packed.as_slice());
        assert_eq!(m.scales.as_ref(), scales.as_slice());
        assert_eq!(m.rows(), rows);
        assert!(m.scales.contains(&0x7e), "the largest group uses the top e4m3 scale");
    }

    #[test]
    fn e4m3_encode_inverts_decode_and_saturates() {
        for code in 0..=0x7eu8 {
            assert_eq!(f32_to_e4m3(e4m3_to_f32(code)), code, "{code:#x}");
            assert_eq!(f32_to_e4m3(-e4m3_to_f32(code)) & 0x7f, code);
        }
        assert_eq!(f32_to_e4m3(1e6), 0x7e);
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
    }

    #[test]
    fn e4m3_decodes_normals_subnormals_and_sign() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0xfe), -448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0x07), 7.0 * 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(e4m3_to_f32(0x80), -0.0);
    }
}

#[cfg(test)]
mod block_expert_tests {
    use super::*;
    use crate::dequant::MXFP4;
    use safetensors::tensor::TensorView;

    fn checkpoint(dir: &Path, experts: usize, inter: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
        let mut blocks = vec![0u8; experts * 2 * inter * h / 2];
        let mut scales = vec![127u8; experts * 2 * inter * h / 32];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        scales[1] = 128;
        let down_blocks = vec![0x22u8; experts * h * inter / 2];
        let down_scales = vec![127u8; experts * h * inter / 32];
        let views = vec![
            ("model.layers.0.mlp.experts.gate_up_proj_blocks".to_string(), TensorView::new(Dtype::U8, vec![experts, 2 * inter, h / 32, 16], &blocks).unwrap()),
            ("model.layers.0.mlp.experts.gate_up_proj_scales".to_string(), TensorView::new(Dtype::U8, vec![experts, 2 * inter, h / 32], &scales).unwrap()),
            ("model.layers.0.mlp.experts.down_proj_blocks".to_string(), TensorView::new(Dtype::U8, vec![experts, h, inter / 32, 16], &down_blocks).unwrap()),
            ("model.layers.0.mlp.experts.down_proj_scales".to_string(), TensorView::new(Dtype::U8, vec![experts, h, inter / 32], &down_scales).unwrap()),
        ];
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
        (blocks, scales)
    }

    #[test]
    fn mxfp4_expert_blocks_decode_into_gate_and_up_with_the_declared_row_order() {
        let (experts, inter, h) = (2usize, 32usize, 160usize);
        let dir = std::env::temp_dir().join(format!("mxfp4-experts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (blocks, scales) = checkpoint(&dir, experts, inter, h);
        let ck = Checkpoint::open(&dir).unwrap();
        let name = "model.layers.0.mlp.experts.gate_up_proj";
        let mut all = Vec::new();
        MXFP4.decode(&blocks, &scales, &mut all).unwrap();
        let want = |r: usize, e: usize| -> Vec<u16> { all[(e * 2 * inter + r) * h..(e * 2 * inter + r + 1) * h].iter().map(|v| f32_to_bf16(*v)).collect() };
        let row = |s: &Stacked, r: usize| -> Vec<u16> {
            let Stacked::Block { packed, scales } = s else { panic!("packed") };
            let mut f = Vec::new();
            let ph = MXFP4.padded_cols(h);
            MXFP4.decode(&packed[r * ph / 2..r * ph / 2 + h / 2], &scales[r * ph / 32..r * ph / 32 + h / 32], &mut f).unwrap();
            assert!(packed[r * ph / 2 + h / 2..(r + 1) * ph / 2].iter().all(|b| *b == 0) && scales[r * ph / 32 + h / 32..(r + 1) * ph / 32].iter().all(|b| *b == 0));
            f.into_iter().map(f32_to_bf16).collect()
        };
        let (g, u) = ck.block_gate_up(&MXFP4, name, experts, inter, h, true).unwrap();
        assert_eq!(row(&g, 3), want(6, 0));
        assert_eq!(row(&u, 3), want(7, 0));
        assert_eq!(row(&g, inter + 1), want(2, 1));
        let (g, u) = ck.block_gate_up(&MXFP4, name, experts, inter, h, false).unwrap();
        assert_eq!(row(&g, 3), want(3, 0));
        assert_eq!(row(&u, 3), want(inter + 3, 0));
        let Stacked::Block { packed, scales } = ck.block_stack(&MXFP4, "model.layers.0.mlp.experts.down_proj", experts, h, inter).unwrap() else { panic!("packed") };
        let pi = MXFP4.padded_cols(inter);
        assert_eq!((packed.len(), scales.len()), (experts * h * pi / 2, experts * h * pi / 32));
        assert!(packed.chunks(pi / 2).all(|r| r[..inter / 2].iter().all(|b| *b == 0x22) && r[inter / 2..].iter().all(|b| *b == 0)));
        assert!(scales.chunks(pi / 32).all(|r| r[..inter / 32].iter().all(|s| *s == 127) && r[inter / 32..].iter().all(|s| *s == 0)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_blocks_tensor_of_the_wrong_size_is_an_error() {
        let dir = std::env::temp_dir().join(format!("mxfp4-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        checkpoint(&dir, 2, 32, 160);
        let ck = Checkpoint::open(&dir).unwrap();
        assert!(ck.block_stack(&MXFP4, "model.layers.0.mlp.experts.gate_up_proj", 2, 32, 160).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}

fn layer_index(prefix: &str) -> usize {
    prefix.rsplit('.').next().and_then(|n| n.parse().ok()).unwrap_or(usize::MAX)
}
