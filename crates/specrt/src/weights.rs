use crate::codegen::Q8_GROUP;
use crate::spec::{Quant, Spec};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::path::{Path, PathBuf};

pub enum Matrix {
    Bf16(Vec<u16>),
    Q8 { values: Vec<i8>, scales: Vec<f32> },
}

pub struct Layer {
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub qkv: Matrix,
    pub qkv_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o: Matrix,
    pub gate_up: Matrix,
    pub down: Matrix,
}

pub struct Weights {
    pub embed: Vec<u16>,
    pub norm: Vec<f32>,
    pub lm_head: Matrix,
    pub layers: Vec<Layer>,
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

pub fn find_safetensors(repo: &str) -> Result<PathBuf> {
    let snapshots =
        hub_dir().join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    for entry in std::fs::read_dir(&snapshots).with_context(|| snapshots.display().to_string())? {
        let file = entry?.path().join("model.safetensors");
        if file.exists() {
            return Ok(file);
        }
    }
    bail!("no model.safetensors for {repo} under {}", snapshots.display())
}

struct Reader<'a> {
    st: SafeTensors<'a>,
}

impl Reader<'_> {
    fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let t = self.st.tensor(name).with_context(|| name.to_string())?;
        ensure!(t.shape() == shape, "{name} has shape {:?}, expected {shape:?}", t.shape());
        match t.dtype() {
            Dtype::BF16 => Ok(t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()),
            Dtype::F16 => Ok(t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32_to_bf16(f16_to_f32(u16::from_le_bytes([b[0], b[1]]))))
                .collect()),
            Dtype::F32 => Ok(t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32_to_bf16(f32::from_le_bytes(*b)))
                .collect()),
            other => bail!("{name} is {other:?}, expected bf16, f16 or f32"),
        }
    }

    fn f32(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        Ok(self.bf16(name, &[len])?.into_iter().map(bf16_to_f32).collect())
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

fn matrix(quant: Quant, w: Vec<u16>, cols: usize) -> Matrix {
    match quant {
        Quant::Bf16 => Matrix::Bf16(w),
        Quant::Q8 => {
            let (values, scales) = quantize_q8(&w, cols);
            Matrix::Q8 { values, scales }
        }
    }
}

impl Weights {
    pub fn load(spec: &Spec, path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
        let map = unsafe { Mmap::map(&file)? };
        let r = Reader { st: SafeTensors::deserialize(&map)? };
        let d = &spec.dims;
        let a = &spec.attention;
        let (h, q, kv, inter) = (d.hidden, spec.q_dim(), spec.kv_dim(), d.intermediate);
        let embed = r.bf16("model.embed_tokens.weight", &[d.vocab, h])?;
        let lm_head =
            if d.tie_embeddings { embed.clone() } else { r.bf16("lm_head.weight", &[d.vocab, h])? };
        let mut layers = Vec::with_capacity(spec.layers.count);
        for i in 0..spec.layers.count {
            let p = format!("model.layers.{i}");
            let attn = format!("{p}.self_attn");
            let mut qkv = r.bf16(&format!("{attn}.q_proj.weight"), &[q, h])?;
            qkv.extend(r.bf16(&format!("{attn}.k_proj.weight"), &[kv, h])?);
            qkv.extend(r.bf16(&format!("{attn}.v_proj.weight"), &[kv, h])?);
            let qkv_bias = if a.qkv_bias {
                let mut b = r.f32(&format!("{attn}.q_proj.bias"), q)?;
                b.extend(r.f32(&format!("{attn}.k_proj.bias"), kv)?);
                b.extend(r.f32(&format!("{attn}.v_proj.bias"), kv)?);
                Some(b)
            } else {
                None
            };
            let (q_norm, k_norm) = if a.qk_norm {
                (
                    Some(r.f32(&format!("{attn}.q_norm.weight"), a.head_dim)?),
                    Some(r.f32(&format!("{attn}.k_norm.weight"), a.head_dim)?),
                )
            } else {
                (None, None)
            };
            let gate = r.bf16(&format!("{p}.mlp.gate_proj.weight"), &[inter, h])?;
            let up = r.bf16(&format!("{p}.mlp.up_proj.weight"), &[inter, h])?;
            let gate_up = interleave_rows(&gate, &up, h);
            layers.push(Layer {
                ln1: r.f32(&format!("{p}.input_layernorm.weight"), h)?,
                ln2: r.f32(&format!("{p}.post_attention_layernorm.weight"), h)?,
                qkv: matrix(spec.quant, qkv, h),
                qkv_bias,
                q_norm,
                k_norm,
                o: matrix(spec.quant, r.bf16(&format!("{attn}.o_proj.weight"), &[h, q])?, q),
                gate_up: matrix(spec.quant, gate_up, h),
                down: matrix(
                    spec.quant,
                    r.bf16(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
                    inter,
                ),
            });
        }
        Ok(Weights {
            embed,
            norm: r.f32("model.norm.weight", h)?,
            lm_head: matrix(spec.quant, lm_head, h),
            layers,
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
}
