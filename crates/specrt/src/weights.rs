use crate::codegen::{FP8_BLOCK, Q8_GROUP};
use crate::spec::{Quant, Spec};
use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, UncheckedAdvice};
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
}

pub struct Attn<'a> {
    pub qkv: Matrix<'a>,
    pub qkv_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o: Matrix<'a>,
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

pub enum Mixer<'a> {
    Attention(Attn<'a>),
    Linear(Gdn<'a>),
}

pub struct Layer<'a> {
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub mixer: Mixer<'a>,
    pub gate_up: Matrix<'a>,
    pub down: Matrix<'a>,
}

pub struct Outside<'a> {
    pub embed: Cow<'a, [u16]>,
    pub norm: Vec<f32>,
    pub lm_head: Matrix<'a>,
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
    shards: Vec<Shard>,
    names: HashMap<String, usize>,
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
        let files = shard_files(dir)?;
        let mut shards = Vec::with_capacity(files.len());
        let mut names = HashMap::new();
        for (i, path) in files.iter().enumerate() {
            let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
            let map = unsafe { Mmap::map(&file)? };
            let (header, meta) =
                SafeTensors::read_metadata(&map).with_context(|| path.display().to_string())?;
            for name in meta.tensors().into_keys() {
                if let Some(prev) = names.insert(name.clone(), i) {
                    bail!("{name} is in both {} and {}", files[prev].display(), path.display());
                }
            }
            shards.push(Shard { map, data_start: 8 + header, meta });
        }
        Ok(Checkpoint { shards, names, files })
    }

    pub fn for_repo(repo: &str) -> Result<Self> {
        Self::open(&find_checkpoint(repo)?)
    }

    pub fn has(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }

    pub fn view(&self, name: &str) -> Result<View<'_>> {
        let &i = self.names.get(name).with_context(|| format!("{name} is not in the checkpoint"))?;
        let shard = &self.shards[i];
        let info = shard.meta.info(name).context("tensor info")?;
        let (from, to) = info.data_offsets;
        Ok(View {
            dtype: info.dtype,
            shape: info.shape.clone(),
            data: &shard.map[shard.data_start + from..shard.data_start + to],
        })
    }

    pub fn release(&self) {
        for s in &self.shards {
            let _ = unsafe { s.map.unchecked_advise(UncheckedAdvice::DontNeed) };
        }
    }

    pub fn root(&self) -> &'static str {
        if self.has("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" }
    }

    fn shaped(&self, name: &str, shape: &[usize]) -> Result<View<'_>> {
        let v = self.view(name)?;
        ensure!(v.shape == shape, "{name} has shape {:?}, expected {shape:?}", v.shape);
        Ok(v)
    }

    fn bf16(&self, name: &str, shape: &[usize]) -> Result<Cow<'_, [u16]>> {
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
        let v = self.shaped(name, shape)?;
        match v.dtype {
            Dtype::F32 => Ok(v.data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
            _ => Ok(self.bf16(name, shape)?.iter().map(|&b| bf16_to_f32(b)).collect()),
        }
    }

    fn f32(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.floats(name, &[len])
    }

    fn raw(&self, weight: &str, rows: usize, cols: usize) -> Result<Raw<'_>> {
        let v = self.view(weight)?;
        if v.dtype != Dtype::F8_E4M3 {
            return Ok(Raw::Bf16 { w: self.bf16(weight, &[rows, cols])?, cols });
        }
        ensure!(v.shape == [rows, cols], "{weight} has shape {:?}, expected {:?}", v.shape, [rows, cols]);
        let kb = cols.div_ceil(FP8_BLOCK);
        let channel = format!("{weight}_scale");
        if !self.has(&format!("{weight}_scale_inv")) && self.has(&channel) {
            let per_row = self.floats(&channel, &[rows, 1])?;
            let grid = per_row.iter().flat_map(|&s| std::iter::repeat_n(s, kb)).collect();
            let map = (0..rows).map(|r| (r * kb) as u32).collect();
            return Ok(Raw::Fp8 { w: Cow::Borrowed(v.data), grid, map, cols });
        }
        let grid = self.floats(&format!("{weight}_scale_inv"), &[rows.div_ceil(FP8_BLOCK), kb])?;
        let map = (0..rows).map(|r| (r / FP8_BLOCK * kb) as u32).collect();
        Ok(Raw::Fp8 { w: Cow::Borrowed(v.data), grid, map, cols })
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
    Fp8 { w: Cow<'a, [u8]>, grid: Vec<f32>, map: Vec<u32>, cols: usize },
}

pub fn fp8_scales(grid: &[f32], map: &[u32]) -> Vec<f32> {
    let rows = map.len() as u32;
    map.iter().map(|&m| f32::from_bits(rows + m)).chain(grid.iter().copied()).collect()
}

impl<'a> Raw<'a> {
    fn cols(&self) -> usize {
        match self {
            Raw::Bf16 { cols, .. } | Raw::Fp8 { cols, .. } => *cols,
        }
    }

    fn rows(&self) -> usize {
        match self {
            Raw::Bf16 { w, cols } => w.len() / cols,
            Raw::Fp8 { w, cols, .. } => w.len() / cols,
        }
    }

    fn slice_rows(&self, from: usize, to: usize) -> Raw<'a> {
        match self {
            Raw::Bf16 { w, cols } => Raw::Bf16 { w: Cow::Owned(w[from * cols..to * cols].to_vec()), cols: *cols },
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

    fn concat(parts: Vec<Raw<'a>>) -> Result<Raw<'a>> {
        let mut parts = parts.into_iter();
        let mut out = parts.next().context("nothing to concatenate")?;
        for p in parts {
            ensure!(p.cols() == out.cols(), "fused rows have different widths");
            match (&mut out, p) {
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
        Ok(match (a, b) {
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
        Ok(match (self, quant) {
            (Raw::Bf16 { w, .. }, Quant::Bf16) => Matrix::Bf16(w),
            (Raw::Bf16 { w, cols }, Quant::Q8) => {
                let (values, scales) = quantize_q8(&w, cols);
                Matrix::Q8 { values, scales }
            }
            (Raw::Fp8 { w, grid, map, .. }, q) if q.is_fp8() => {
                Matrix::Fp8 { values: w, scales: fp8_scales(&grid, &map) }
            }
            (Raw::Bf16 { .. }, q) => bail!("{what} is bf16 but the spec says {q:?}"),
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

    fn attention(&self, spec: &Spec, attn: &str) -> Result<Attn<'_>> {
        let a = &spec.attention;
        let (h, q, kv) = (spec.dims.hidden, spec.q_dim(), spec.kv_dim());
        let w = |p: &str, rows: usize, cols: usize| self.raw(&format!("{attn}.{p}.weight"), rows, cols);
        let mut parts = Vec::new();
        let mut gates = Vec::new();
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
        let (q_norm, k_norm) = if a.qk_norm {
            (
                Some(self.norm(spec, &format!("{attn}.q_norm.weight"), a.head_dim)?),
                Some(self.norm(spec, &format!("{attn}.k_norm.weight"), a.head_dim)?),
            )
        } else {
            (None, None)
        };
        Ok(Attn {
            qkv: Raw::concat(parts)?.into_matrix(spec.quant, &format!("{attn}.qkv"))?,
            qkv_bias,
            q_norm,
            k_norm,
            o: w("o_proj", h, q)?.into_matrix(spec.quant, &format!("{attn}.o_proj"))?,
        })
    }

    fn gated_delta_net(&self, spec: &Spec, p: &str) -> Result<Gdn<'_>> {
        let la = spec.linear_attention.as_ref().context("no [linear_attention]")?;
        let h = spec.dims.hidden;
        let (conv_dim, value_dim, heads) = (la.conv_dim(), la.value_dim(), la.value_heads);
        let w = |n: &str, rows: usize, cols: usize| self.raw(&format!("{p}.{n}.weight"), rows, cols);
        let in_proj = Raw::concat(vec![w("in_proj_qkv", conv_dim, h)?, w("in_proj_z", value_dim, h)?])?;
        let mut ba = self.bf16(&format!("{p}.in_proj_b.weight"), &[heads, h])?.into_owned();
        ba.extend_from_slice(&self.bf16(&format!("{p}.in_proj_a.weight"), &[heads, h])?);
        Ok(Gdn {
            in_proj: in_proj.into_matrix(spec.quant, &format!("{p}.in_proj"))?,
            ba: Matrix::Bf16(Cow::Owned(ba)),
            conv: self.floats(&format!("{p}.conv1d.weight"), &[conv_dim, 1, la.conv_kernel])?,
            neg_a: self.f32(&format!("{p}.A_log"), heads)?.into_iter().map(|a| -a.exp()).collect(),
            dt_bias: self.f32(&format!("{p}.dt_bias"), heads)?,
            norm: self.f32(&format!("{p}.norm.weight"), la.value_head_dim)?,
            out: w("out_proj", h, value_dim)?.into_matrix(spec.quant, &format!("{p}.out_proj"))?,
        })
    }

    pub fn layer(&self, spec: &Spec, i: usize) -> Result<Layer<'_>> {
        let (h, inter) = (spec.dims.hidden, spec.dims.intermediate);
        let p = format!("{}.layers.{i}", self.root());
        let mixer = if spec.block(i).is_linear() {
            Mixer::Linear(self.gated_delta_net(spec, &format!("{p}.linear_attn"))?)
        } else {
            Mixer::Attention(self.attention(spec, &format!("{p}.self_attn"))?)
        };
        let mlp = spec.mlp_quant(i);
        let (gate_up, down) = if mlp == Quant::Nvfp4G16 {
            let gate = self.fp4(&format!("{p}.mlp.gate_proj"), inter, h)?;
            let up = self.fp4(&format!("{p}.mlp.up_proj"), inter, h)?;
            (Matrix::Fp4(crate::nvfp4::Fp4::interleave(gate, up)?), Matrix::Fp4(self.fp4(&format!("{p}.mlp.down_proj"), h, inter)?))
        } else {
            let gate = self.raw(&format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            let up = self.raw(&format!("{p}.mlp.up_proj.weight"), inter, h)?;
            let down = self.raw(&format!("{p}.mlp.down_proj.weight"), h, inter)?;
            (
                Raw::interleave(gate, up)?.into_matrix(mlp, &format!("{p}.mlp.gate_up"))?,
                down.into_matrix(mlp, &format!("{p}.mlp.down_proj"))?,
            )
        };
        Ok(Layer {
            ln1: self.norm(spec, &format!("{p}.input_layernorm.weight"), h)?,
            ln2: self.norm(spec, &format!("{p}.post_attention_layernorm.weight"), h)?,
            mixer,
            gate_up,
            down,
        })
    }

    pub fn fp4(&self, module: &str, rows: usize, cols: usize) -> Result<crate::nvfp4::Fp4<'_>> {
        use crate::nvfp4::GROUP;
        let packed = self.shaped(&format!("{module}.weight_packed"), &[rows, cols / 2])?;
        ensure!(packed.dtype == Dtype::U8, "{module}.weight_packed is {:?}, expected U8", packed.dtype);
        let scales = self.shaped(&format!("{module}.weight_scale"), &[rows, cols / GROUP])?;
        ensure!(scales.dtype == Dtype::F8_E4M3, "{module}.weight_scale is {:?}, expected F8_E4M3", scales.dtype);
        let global = self.floats(&format!("{module}.weight_global_scale"), &[1])?[0];
        let mut m = crate::nvfp4::Fp4::new(Cow::Borrowed(packed.data), Cow::Borrowed(scales.data), global, rows, cols)?;
        m.input_global = self.floats(&format!("{module}.input_global_scale"), &[1])?[0];
        Ok(m)
    }

    pub fn outside(&self, spec: &Spec) -> Result<Outside<'_>> {
        let d = &spec.dims;
        let root = self.root();
        let embed = self.bf16(&format!("{root}.embed_tokens.weight"), &[d.vocab, d.hidden])?;
        let head_quant = if spec.quant.quantised_checkpoint() { Quant::Bf16 } else { spec.quant };
        let lm_head = if d.tie_embeddings {
            Raw::Bf16 { w: embed.clone(), cols: d.hidden }
        } else {
            self.raw("lm_head.weight", d.vocab, d.hidden)?
        };
        Ok(Outside {
            norm: self.norm(spec, &format!("{root}.norm.weight"), d.hidden)?,
            lm_head: lm_head.into_matrix(head_quant, "lm_head")?,
            embed,
        })
    }
}

pub fn find_in_snapshot(repo: &str, file: &str) -> Result<PathBuf> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_converts_to_bf16() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(f32_to_bf16(f16_to_f32(0x3555))), 0.333_984_38);
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
