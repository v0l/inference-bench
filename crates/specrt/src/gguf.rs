use crate::spec::{
    Activation, ActQuant, Attention, AttentionKind, Block, Dims, Layers, Mlp, Norm, NormKind, Quant, Rope, RopeScaling,
    RopeScalingKind, RopeStyle, RuntimeOptions, Spec,
};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde_json::{Value as Json, json};
use std::collections::HashMap;
use std::path::Path;

pub const DEFAULT_ALIGNMENT: usize = 32;
const QK_K: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::Bf16,
            other => GgmlType::Other(other),
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Bf16 => "BF16".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q4K => "Q4_K".into(),
            GgmlType::Q5K => "Q5_K".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::Other(id) => match id {
                2 => "Q4_0".into(),
                3 => "Q4_1".into(),
                6 => "Q5_0".into(),
                7 => "Q5_1".into(),
                9 => "Q8_1".into(),
                10 => "Q2_K".into(),
                11 => "Q3_K".into(),
                15 => "Q8_K".into(),
                id => format!("type {id}"),
            },
        }
    }

    pub fn block(self) -> Option<(usize, usize)> {
        Some(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::Bf16 => (1, 2),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q4K => (QK_K, 144),
            GgmlType::Q5K => (QK_K, 176),
            GgmlType::Q6K => (QK_K, 210),
            GgmlType::Other(_) => return None,
        })
    }

    pub fn quantised(self) -> bool {
        !matches!(self, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16)
    }

    pub fn row_bytes(self, cols: usize) -> Result<usize> {
        let (elems, bytes) = self.block().with_context(|| format!("GGUF tensor type {} is not supported", self.name()))?;
        ensure!(cols.is_multiple_of(elems), "a {} row of {cols} is not a whole number of {elems}-element blocks", self.name());
        Ok(cols / elems * bytes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: GgmlType,
    pub offset: usize,
    pub bytes: usize,
}

impl TensorInfo {
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    pub fn rows(&self) -> usize {
        self.shape.iter().rev().skip(1).product()
    }
}

enum Backing {
    Map(Mmap),
    Owned(Vec<u8>),
}

pub struct Gguf {
    backing: Backing,
    pub version: u32,
    pub alignment: usize,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len()).context("the GGUF file ends early")?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("take returned N bytes"))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?)?;
        Ok(String::from_utf8(self.take(len)?.to_vec()).context("a GGUF string is not UTF-8")?)
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => match self.array::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => bail!("a GGUF bool is {b}"),
            },
            8 => Value::Str(self.string()?),
            9 => {
                let inner = self.u32()?;
                let count = usize::try_from(self.u64()?)?;
                ensure!(count <= self.bytes.len() - self.pos, "a GGUF array of {count} items does not fit in the file");
                Value::Array((0..count).map(|_| self.value(inner)).collect::<Result<_>>()?)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => bail!("GGUF metadata value type {other} is unknown"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
        let map = unsafe { Mmap::map(&file)? };
        Self::parse(Backing::Map(map)).with_context(|| path.display().to_string())
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Backing::Owned(bytes))
    }

    fn parse(backing: Backing) -> Result<Self> {
        let bytes: &[u8] = match &backing {
            Backing::Map(m) => m,
            Backing::Owned(v) => v,
        };
        let mut r = Reader { bytes, pos: 0 };
        ensure!(r.take(4).is_ok_and(|m| m == b"GGUF"), "not a GGUF file (bad magic)");
        let version = r.u32()?;
        ensure!(matches!(version, 2 | 3), "GGUF version {version} is not supported (2 and 3 are)");
        let tensor_count = usize::try_from(r.u64()?)?;
        let kv_count = usize::try_from(r.u64()?)?;
        ensure!(tensor_count <= bytes.len() && kv_count <= bytes.len(), "the GGUF counts are larger than the file");
        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let kind = r.u32()?;
            let value = r.value(kind).with_context(|| key.clone())?;
            ensure!(metadata.insert(key.clone(), value).is_none(), "GGUF key {key} appears twice");
        }
        let alignment = match metadata.get("general.alignment") {
            Some(v) => v.as_u64().context("general.alignment is not a number")? as usize,
            None => DEFAULT_ALIGNMENT,
        };
        ensure!(alignment.is_power_of_two(), "general.alignment {alignment} is not a power of two");
        let mut tensors = HashMap::with_capacity(tensor_count);
        let mut order = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let dims = r.u32()? as usize;
            ensure!((1..=4).contains(&dims), "tensor {name} has {dims} dimensions");
            let mut shape = (0..dims).map(|_| Ok(usize::try_from(r.u64()?)?)).collect::<Result<Vec<usize>>>()?;
            shape.reverse();
            let kind = GgmlType::from_id(r.u32()?);
            let offset = usize::try_from(r.u64()?)?;
            let cols = *shape.last().expect("at least one dimension");
            let rows: usize = shape.iter().rev().skip(1).product();
            let bytes = match kind.block() {
                Some(_) => kind.row_bytes(cols).with_context(|| name.clone())?.checked_mul(rows).context("tensor size overflows")?,
                None => 0,
            };
            order.push(name.clone());
            ensure!(tensors.insert(name.clone(), TensorInfo { name: name.clone(), shape, kind, offset, bytes }).is_none(), "tensor {name} appears twice");
        }
        let data_start = r.pos.next_multiple_of(alignment);
        ensure!(tensors.is_empty() || data_start <= bytes.len(), "the GGUF file ends before its tensor data");
        for info in tensors.values() {
            ensure!(
                info.offset.is_multiple_of(alignment) && data_start + info.offset + info.bytes <= bytes.len(),
                "tensor {} lies outside the file or is not aligned",
                info.name
            );
        }
        Ok(Gguf { backing, version, alignment, metadata, tensors, data_start })
    }

    fn bytes(&self) -> &[u8] {
        match &self.backing {
            Backing::Map(m) => m,
            Backing::Owned(v) => v,
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn data(&self, info: &TensorInfo) -> &[u8] {
        let from = self.data_start + info.offset;
        &self.bytes()[from..from + info.bytes]
    }

    pub fn release(&self) {
        if let Backing::Map(m) = &self.backing {
            let _ = unsafe { m.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        }
    }

    pub fn string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    pub fn number(&self, key: &str) -> Option<f64> {
        self.metadata.get(key)?.as_f64()
    }

    pub fn count(&self, key: &str) -> Option<usize> {
        self.metadata.get(key)?.as_u64().map(|v| v as usize)
    }

    pub fn architecture(&self) -> Result<&str> {
        self.string("general.architecture").context("the GGUF file has no general.architecture")
    }

    fn arch_key(&self, key: &str) -> Result<String> {
        Ok(format!("{}.{key}", self.architecture()?))
    }

    fn need_count(&self, key: &str) -> Result<usize> {
        let full = self.arch_key(key)?;
        self.count(&full).with_context(|| format!("the GGUF file has no {full}"))
    }

    pub fn weight_type_bytes(&self) -> Vec<(GgmlType, usize)> {
        let mut by: HashMap<String, (GgmlType, usize)> = HashMap::new();
        for t in self.tensors.values().filter(|t| t.shape.len() == 2) {
            let e = by.entry(t.kind.name()).or_insert((t.kind, 0));
            e.1 += t.bytes;
        }
        let mut out: Vec<_> = by.into_values().collect();
        out.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
        out
    }

    pub fn spec(&self) -> Result<Spec> {
        let arch = self.architecture()?;
        ensure!(
            matches!(arch, "llama" | "qwen2" | "qwen3"),
            "GGUF architecture {arch} is not supported (llama, qwen2 and qwen3 are)"
        );
        let hidden = self.need_count("embedding_length")?;
        let layers = self.need_count("block_count")?;
        let heads = self.need_count("attention.head_count")?;
        let kv_heads = self.count(&self.arch_key("attention.head_count_kv")?).unwrap_or(heads);
        let intermediate = self.need_count("feed_forward_length")?;
        let head_dim = self.count(&self.arch_key("attention.key_length")?).unwrap_or(hidden / heads);
        let value_dim = self.count(&self.arch_key("attention.value_length")?).unwrap_or(head_dim);
        ensure!(value_dim == head_dim, "GGUF key_length {head_dim} and value_length {value_dim} differ");
        let rotary = self.count(&self.arch_key("rope.dimension_count")?).filter(|&r| r != head_dim);
        let eps = self.number(&self.arch_key("attention.layer_norm_rms_epsilon")?).context("the GGUF file has no layer_norm_rms_epsilon")?;
        let theta = self.number(&self.arch_key("rope.freq_base")?).unwrap_or(10000.0);
        let embd = self.tensor("token_embd.weight").context("the GGUF file has no token_embd.weight")?;
        ensure!(embd.shape == [embd.rows(), hidden], "token_embd.weight has shape {:?}, expected [vocab, {hidden}]", embd.shape);
        let tied = !self.tensors.contains_key("output.weight");
        let quant = self.weight_quant();
        let spec = Spec {
            name: self.string("general.name").unwrap_or(arch).to_lowercase().replace(' ', "-"),
            min_engine: 0,
            repo: String::new(),
            quant,
            head_quant: None,
            act_quant: ActQuant::None,
            dims: Dims { hidden, intermediate, vocab: embd.rows(), tie_embeddings: tied },
            layers: Layers { count: layers, pattern: vec![Block::Dense], window: None },
            norm: Norm { kind: NormKind::Rmsnorm, eps, unit_offset: false },
            attention: Attention {
                kind: AttentionKind::Gqa,
                heads,
                kv_heads,
                head_dim,
                qkv_bias: self.tensors.contains_key("blk.0.attn_q.bias"),
                qk_norm: self.tensors.contains_key("blk.0.attn_q_norm.weight"),
                output_gate: false,
                rotary_dim: rotary,
            },
            rope: Rope { style: RopeStyle::Neox, theta, scaling: self.rope_scaling(head_dim)? },
            mlp: Mlp { activation: Activation::Swiglu, quant: None, quant_until: None, act_quant: ActQuant::None },
            moe: None,
            linear_attention: None,
            mtp: None,
            runtime: RuntimeOptions::default(),
        };
        spec.validate()?;
        Ok(spec)
    }

    fn weight_quant(&self) -> Quant {
        let q8 = self.tensor("blk.0.ffn_down.weight").is_some_and(|t| t.kind == GgmlType::Q8_0);
        if q8 { Quant::Q8 } else { Quant::Bf16 }
    }

    fn rope_scaling(&self, head_dim: usize) -> Result<Option<RopeScaling>> {
        let kind = self.string(&self.arch_key("rope.scaling.type")?).unwrap_or("none");
        let factor = self.number(&self.arch_key("rope.scaling.factor")?);
        match kind {
            "linear" => Ok(Some(RopeScaling {
                kind: RopeScalingKind::Linear,
                factor: factor.context("rope.scaling.type is linear but there is no rope.scaling.factor")?,
                low_freq_factor: None,
                high_freq_factor: None,
                original_max_positions: None,
            })),
            "none" | "" => match self.tensor("rope_freqs.weight") {
                Some(t) => Ok(Some(self.llama3_scaling(t, head_dim)?)),
                None => Ok(None),
            },
            other => bail!("GGUF rope scaling {other} is not supported"),
        }
    }

    fn llama3_scaling(&self, t: &TensorInfo, head_dim: usize) -> Result<RopeScaling> {
        ensure!(t.kind == GgmlType::F32 && t.shape == [head_dim / 2], "rope_freqs.weight has an unexpected shape or type");
        let freqs: Vec<f32> = self.data(t).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))).collect();
        let factor = freqs.iter().copied().fold(1.0f32, f32::max) as f64;
        let scaling = RopeScaling {
            kind: RopeScalingKind::Llama3,
            factor,
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            original_max_positions: Some(
                self.number(&self.arch_key("rope.scaling.original_context_length")?).unwrap_or(8192.0),
            ),
        };
        let theta = self.number(&self.arch_key("rope.freq_base")?).unwrap_or(10000.0);
        let expected = llama3_factors(theta, head_dim, &scaling);
        for (i, (a, b)) in freqs.iter().zip(&expected).enumerate() {
            ensure!(
                (*a as f64 - b).abs() <= 1e-5 * b.abs().max(1.0),
                "rope_freqs.weight[{i}] is {a} but llama3 scaling with factor {factor} gives {b}; this rope scaling is not supported"
            );
        }
        Ok(scaling)
    }

    pub fn vocab(&self) -> Result<Vec<&str>> {
        let tokens = self.metadata.get("tokenizer.ggml.tokens").and_then(Value::as_array).context("the GGUF file has no tokenizer.ggml.tokens")?;
        tokens.iter().map(|t| t.as_str().context("a token is not a string")).collect()
    }

    pub fn tokenizer_json(&self) -> Result<String> {
        let model = self.string("tokenizer.ggml.model").unwrap_or("");
        ensure!(model == "gpt2", "GGUF tokenizer model {model:?} is not supported (gpt2 byte-level BPE is)");
        let pre = self.string("tokenizer.ggml.pre").unwrap_or("default");
        let pattern = match pre {
            "qwen2" | "deepseek-r1-qwen" => QWEN2_PRE,
            "llama-bpe" | "llama3" | "smaug-bpe" => LLAMA3_PRE,
            other => bail!("GGUF pre-tokenizer {other:?} is not supported (qwen2 and llama-bpe are)"),
        };
        let tokens = self.vocab()?;
        let kinds = self.metadata.get("tokenizer.ggml.token_type").and_then(Value::as_array);
        let merges = self.metadata.get("tokenizer.ggml.merges").and_then(Value::as_array).context("the GGUF file has no tokenizer.ggml.merges")?;
        let mut vocab = serde_json::Map::with_capacity(tokens.len());
        let mut added = Vec::new();
        for (id, t) in tokens.iter().enumerate() {
            vocab.insert((*t).to_string(), json!(id));
            let kind = kinds.and_then(|k| k.get(id)).and_then(Value::as_u64).unwrap_or(1);
            if kind == 3 || kind == 4 {
                added.push(json!({"id": id, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": kind == 3}));
            }
        }
        let merges: Vec<Json> = merges.iter().map(|m| m.as_str().map(|s| json!(s)).context("a merge is not a string")).collect::<Result<_>>()?;
        Ok(json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges}
        })
        .to_string())
    }

    fn token(&self, key: &str) -> Option<String> {
        let id = self.count(key)?;
        self.vocab().ok()?.get(id).map(|t| t.to_string())
    }

    pub fn tokenizer_config_json(&self) -> Option<String> {
        let template = self.string("tokenizer.chat_template")?;
        Some(
            json!({
                "chat_template": template,
                "bos_token": self.token("tokenizer.ggml.bos_token_id"),
                "eos_token": self.token("tokenizer.ggml.eos_token_id"),
            })
            .to_string(),
        )
    }

    pub fn generation_config_json(&self) -> String {
        let mut ids: Vec<u64> = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"]
            .iter()
            .filter_map(|k| self.metadata.get(*k)?.as_u64())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        json!({"eos_token_id": ids}).to_string()
    }

    pub fn adds_bos(&self) -> Option<u32> {
        match self.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(Value::Bool(true)) => self.count("tokenizer.ggml.bos_token_id").map(|b| b as u32),
            _ => None,
        }
    }
}

const QWEN2_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub fn llama3_factors(theta: f64, head_dim: usize, s: &RopeScaling) -> Vec<f64> {
    let (low, high, orig) = (s.low_freq_factor.unwrap_or(1.0), s.high_freq_factor.unwrap_or(4.0), s.original_max_positions.unwrap_or(8192.0));
    let (low_wavelen, high_wavelen) = (orig / low, orig / high);
    (0..head_dim / 2)
        .map(|i| {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let wavelen = 2.0 * std::f64::consts::PI / freq;
            if wavelen < high_wavelen {
                1.0
            } else if wavelen > low_wavelen {
                s.factor
            } else {
                let smooth = (orig / wavelen - low) / (high - low);
                1.0 / ((1.0 - smooth) / s.factor + smooth)
            }
        })
        .collect()
}

fn f16(b: &[u8]) -> f32 {
    crate::weights::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

pub fn dequantize_q8_0(block: &[u8], out: &mut [f32]) {
    let d = f16(block);
    for (o, &q) in out.iter_mut().zip(&block[2..34]) {
        *o = d * (q as i8) as f32;
    }
}

pub fn dequantize_q4_k(block: &[u8], out: &mut [f32]) {
    let (d, min) = (f16(block), f16(&block[2..]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    for j in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * j, scales);
        let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        let q = &qs[32 * j..32 * j + 32];
        for l in 0..32 {
            out[64 * j + l] = d1 * (q[l] & 0xf) as f32 - mm1;
            out[64 * j + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
        }
    }
}

pub fn dequantize_q5_k(block: &[u8], out: &mut [f32]) {
    let (d, min) = (f16(block), f16(&block[2..]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let ql = &block[48..176];
    for j in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * j, scales);
        let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
        let q = &ql[32 * j..32 * j + 32];
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[64 * j + l] = d1 * ((q[l] & 0xf) + hi1) as f32 - mm1;
            out[64 * j + 32 + l] = d2 * ((q[l] >> 4) + hi2) as f32 - mm2;
        }
    }
}

pub fn dequantize_q6_k(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = f16(&block[208..]);
    for half in 0..2 {
        let (ql, qh, sc, y) = (&ql[64 * half..], &qh[32 * half..], &sc[8 * half..], &mut out[128 * half..]);
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

pub fn dequantize(kind: GgmlType, data: &[u8], out: &mut [f32]) -> Result<()> {
    let (elems, bytes) = kind.block().with_context(|| format!("GGUF tensor type {} is not supported", kind.name()))?;
    ensure!(data.len() / bytes * elems == out.len() && data.len().is_multiple_of(bytes), "{} data of {} bytes does not fill {} values", kind.name(), data.len(), out.len());
    match kind {
        GgmlType::F32 => out.iter_mut().zip(data.chunks_exact(4)).for_each(|(o, b)| *o = f32::from_le_bytes(b.try_into().expect("4 bytes"))),
        GgmlType::F16 => out.iter_mut().zip(data.chunks_exact(2)).for_each(|(o, b)| *o = f16(b)),
        GgmlType::Bf16 => out.iter_mut().zip(data.chunks_exact(2)).for_each(|(o, b)| *o = crate::weights::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
        _ => {
            let f: fn(&[u8], &mut [f32]) = match kind {
                GgmlType::Q8_0 => dequantize_q8_0,
                GgmlType::Q4K => dequantize_q4_k,
                GgmlType::Q5K => dequantize_q5_k,
                _ => dequantize_q6_k,
            };
            for (block, y) in data.chunks_exact(bytes).zip(out.chunks_exact_mut(elems)) {
                f(block, y);
            }
        }
    }
    Ok(())
}

pub fn q8_0_split(data: &[u8]) -> (Vec<i8>, Vec<f32>) {
    let blocks = data.len() / 34;
    let mut values = Vec::with_capacity(blocks * 32);
    let mut scales = Vec::with_capacity(blocks);
    for b in data.chunks_exact(34) {
        scales.push(f16(b));
        values.extend(b[2..].iter().map(|&q| q as i8));
    }
    (values, scales)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub enum Kv<'a> {
        U8(u8),
        I8(i8),
        U16(u16),
        I16(i16),
        U32(u32),
        I32(i32),
        F32(f32),
        Bool(bool),
        Str(&'a str),
        U64(u64),
        I64(i64),
        F64(f64),
        Strs(&'a [&'a str]),
        U32s(&'a [u32]),
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn put_kv(out: &mut Vec<u8>, kv: &Kv) {
        match kv {
            Kv::U8(v) => (out.extend(0u32.to_le_bytes()), out.push(*v)),
            Kv::I8(v) => (out.extend(1u32.to_le_bytes()), out.push(*v as u8)),
            Kv::U16(v) => (out.extend(2u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::I16(v) => (out.extend(3u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::U32(v) => (out.extend(4u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::I32(v) => (out.extend(5u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::F32(v) => (out.extend(6u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::Bool(v) => (out.extend(7u32.to_le_bytes()), out.push(*v as u8)),
            Kv::Str(s) => (out.extend(8u32.to_le_bytes()), put_str(out, s)),
            Kv::U64(v) => (out.extend(10u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::I64(v) => (out.extend(11u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::F64(v) => (out.extend(12u32.to_le_bytes()), out.extend(v.to_le_bytes())),
            Kv::Strs(list) => {
                out.extend(9u32.to_le_bytes());
                out.extend(8u32.to_le_bytes());
                out.extend((list.len() as u64).to_le_bytes());
                list.iter().for_each(|s| put_str(out, s));
                return;
            }
            Kv::U32s(list) => {
                out.extend(9u32.to_le_bytes());
                out.extend(4u32.to_le_bytes());
                out.extend((list.len() as u64).to_le_bytes());
                list.iter().for_each(|v| out.extend(v.to_le_bytes()));
                return;
            }
        };
    }

    pub fn build(version: u32, kvs: &[(&str, Kv)], tensors: &[(&str, &[u64], u32, &[u8])]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(version.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            put_str(&mut out, k);
            put_kv(&mut out, v);
        }
        let alignment = kvs
            .iter()
            .find_map(|(k, v)| match (*k, v) {
                ("general.alignment", Kv::U32(a)) => Some(*a as usize),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);
        let mut offset = 0usize;
        let mut placed = Vec::new();
        for (name, dims, ty, data) in tensors {
            put_str(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| out.extend(d.to_le_bytes()));
            out.extend(ty.to_le_bytes());
            out.extend((offset as u64).to_le_bytes());
            placed.push(offset);
            offset = (offset + data.len()).next_multiple_of(alignment);
        }
        out.resize(out.len().next_multiple_of(alignment), 0);
        let base = out.len();
        for ((_, _, _, data), at) in tensors.iter().zip(placed) {
            out.resize(base + at, 0);
            out.extend(*data);
        }
        out
    }

    #[test]
    fn every_metadata_type_and_array_round_trips() {
        let bytes = build(
            3,
            &[
                ("a.u8", Kv::U8(200)),
                ("a.i8", Kv::I8(-5)),
                ("a.u16", Kv::U16(60000)),
                ("a.i16", Kv::I16(-300)),
                ("a.u32", Kv::U32(4_000_000_000)),
                ("a.i32", Kv::I32(-7)),
                ("a.f32", Kv::F32(1.5)),
                ("a.bool", Kv::Bool(true)),
                ("a.str", Kv::Str("héllo")),
                ("a.u64", Kv::U64(1 << 40)),
                ("a.i64", Kv::I64(-(1 << 40))),
                ("a.f64", Kv::F64(0.25)),
                ("a.strs", Kv::Strs(&["x", "", "yz"])),
                ("a.u32s", Kv::U32s(&[1, 2, 3])),
            ],
            &[],
        );
        let g = Gguf::from_bytes(bytes).unwrap();
        let m = &g.metadata;
        assert_eq!(m["a.u8"], Value::U8(200));
        assert_eq!(m["a.i8"], Value::I8(-5));
        assert_eq!(m["a.u16"], Value::U16(60000));
        assert_eq!(m["a.i16"], Value::I16(-300));
        assert_eq!(m["a.u32"], Value::U32(4_000_000_000));
        assert_eq!(m["a.i32"], Value::I32(-7));
        assert_eq!(m["a.f32"], Value::F32(1.5));
        assert_eq!(m["a.bool"], Value::Bool(true));
        assert_eq!(m["a.str"], Value::Str("héllo".into()));
        assert_eq!(m["a.u64"], Value::U64(1 << 40));
        assert_eq!(m["a.i64"], Value::I64(-(1 << 40)));
        assert_eq!(m["a.f64"], Value::F64(0.25));
        assert_eq!(m["a.strs"], Value::Array(vec![Value::Str("x".into()), Value::Str("".into()), Value::Str("yz".into())]));
        assert_eq!(m["a.u32s"].as_array().unwrap().len(), 3);
        assert_eq!(g.version, 3);
        assert_eq!(g.alignment, 32);
    }

    #[test]
    fn tensor_data_starts_on_the_alignment_and_shapes_are_rows_first() {
        let a: Vec<u8> = (0..24).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let b: Vec<u8> = (0..8).flat_map(|v| (100.0 + v as f32).to_le_bytes()).collect();
        for alignment in [32u32, 64] {
            let bytes = build(3, &[("general.alignment", Kv::U32(alignment))], &[("a", &[6, 4], 0, &a), ("b", &[8], 0, &b)]);
            let g = Gguf::from_bytes(bytes).unwrap();
            assert_eq!(g.alignment, alignment as usize);
            let (ta, tb) = (g.tensor("a").unwrap(), g.tensor("b").unwrap());
            assert_eq!((ta.shape.clone(), ta.rows(), ta.cols()), (vec![4, 6], 4, 6));
            assert_eq!(tb.shape, vec![8]);
            assert_eq!(g.data(ta), a.as_slice());
            assert_eq!(g.data(tb), b.as_slice());
            assert_eq!(tb.offset % alignment as usize, 0);
        }
    }

    #[test]
    fn a_bad_magic_a_bad_version_and_a_truncated_file_are_refused() {
        let good = build(3, &[("a.u8", Kv::U8(1))], &[]);
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        let e = Gguf::from_bytes(bad_magic).err().unwrap().to_string();
        assert!(e.contains("bad magic"), "{e}");
        for v in [1u32, 4, 0] {
            let e = Gguf::from_bytes(build(v, &[], &[])).err().unwrap().to_string();
            assert!(e.contains(&format!("GGUF version {v} is not supported")), "{e}");
        }
        let e = Gguf::from_bytes(good[..20].to_vec()).err().unwrap().to_string();
        assert!(e.contains("ends early"), "{e}");
        let data = [0u8; 16];
        let mut short = build(3, &[], &[("t", &[4], 0, &data)]);
        short.truncate(short.len() - 4);
        let e = Gguf::from_bytes(short).err().unwrap().to_string();
        assert!(e.contains("outside the file"), "{e}");
    }

    #[test]
    fn an_unknown_tensor_type_parses_and_is_named_when_used() {
        let g = Gguf::from_bytes(build(3, &[], &[("t", &[32], 2, &[0u8; 18])])).unwrap();
        let t = g.tensor("t").unwrap();
        assert_eq!(t.kind.name(), "Q4_0");
        let e = dequantize(t.kind, g.data(t), &mut [0.0; 32]).err().unwrap().to_string();
        assert!(e.contains("Q4_0 is not supported"), "{e}");
    }

    #[test]
    fn an_unsupported_architecture_is_named() {
        let g = Gguf::from_bytes(build(3, &[("general.architecture", Kv::Str("gemma3"))], &[])).unwrap();
        let e = g.spec().err().unwrap().to_string();
        assert!(e.contains("architecture gemma3 is not supported"), "{e}");
    }

    fn tiny_qwen3() -> Vec<u8> {
        let norm = vec![0u8; 4 * 8];
        let embd = vec![0u8; 4 * 8 * 16];
        build(
            3,
            &[
                ("general.architecture", Kv::Str("qwen3")),
                ("general.name", Kv::Str("Tiny Q")),
                ("qwen3.block_count", Kv::U32(2)),
                ("qwen3.embedding_length", Kv::U32(8)),
                ("qwen3.feed_forward_length", Kv::U32(16)),
                ("qwen3.attention.head_count", Kv::U32(4)),
                ("qwen3.attention.head_count_kv", Kv::U32(2)),
                ("qwen3.attention.key_length", Kv::U32(4)),
                ("qwen3.attention.value_length", Kv::U32(4)),
                ("qwen3.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
                ("qwen3.rope.freq_base", Kv::F32(1000000.0)),
            ],
            &[("token_embd.weight", &[8, 16], 0, &embd), ("blk.0.attn_q_norm.weight", &[4], 0, &norm[..16])],
        )
    }

    #[test]
    fn metadata_becomes_a_spec() {
        let g = Gguf::from_bytes(tiny_qwen3()).unwrap();
        let s = g.spec().unwrap();
        assert_eq!(s.name, "tiny-q");
        assert_eq!((s.dims.hidden, s.dims.intermediate, s.dims.vocab, s.dims.tie_embeddings), (8, 16, 16, true));
        assert_eq!((s.layers.count, s.attention.heads, s.attention.kv_heads, s.attention.head_dim), (2, 4, 2, 4));
        assert!(s.attention.qk_norm && !s.attention.qkv_bias);
        assert_eq!((s.rope.theta, s.norm.eps as f32, s.quant), (1000000.0, 1e-6, Quant::Bf16));
        assert!(s.rope.scaling.is_none() && s.attention.rotary_dim.is_none());
    }

    #[test]
    fn the_llama3_scale_factors_match_the_converter_for_llama_3_2_1b() {
        let s = RopeScaling { kind: RopeScalingKind::Llama3, factor: 32.0, low_freq_factor: Some(1.0), high_freq_factor: Some(4.0), original_max_positions: Some(8192.0) };
        let f = llama3_factors(500000.0, 64, &s);
        assert_eq!(f.len(), 32);
        assert_eq!(f[0], 1.0);
        assert_eq!(f[31], 32.0);
        assert!(f.windows(2).all(|w| w[0] <= w[1]));
    }
}
