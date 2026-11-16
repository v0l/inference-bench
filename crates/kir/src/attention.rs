use crate::{Feature, Language, Target, fnv};
use std::fmt::{self, Write as _};

pub const PAGE: usize = 16;
pub const FP8_MAX: f32 = 448.0;
pub const DECODE_MAX_SPLITS: usize = 64;
pub const DECODE_CHUNK: usize = 256;
pub const DECODE_KEYS: usize = 16;
pub const DECODE_STAGES: usize = 2;
pub const DECODE_MAX_QLEN: usize = 4;
pub const FLASH_ROWS: usize = 128;
pub const SPLIT_BLOCK: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeShape {
    pub rotary_dim: usize,
    pub theta: f64,
    pub kind: u32,
    pub factor: f64,
    pub low: f64,
    pub high: f64,
    pub original: f64,
    pub mscale: f64,
    pub mrope: Option<MropeShape>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MropeShape {
    pub section: [usize; 3],
    pub interleaved: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub qkv_row: usize,
    pub output_gate: bool,
    pub qk_norm: bool,
    pub eps: f64,
    pub rope: RopeShape,
    pub kv_layers: usize,
    pub kv_planes: usize,
    pub kv_fp8: bool,
    pub kv_f16_layers: usize,
    pub query_scale: f64,
    pub softcap: f64,
    pub local_theta: f64,
    pub local_rope: Vec<bool>,
    pub bidirectional: bool,
    pub nope: Vec<bool>,
}

impl AttentionShape {
    pub fn group(&self) -> usize {
        self.heads / self.kv_heads
    }

    pub fn flash_keys(&self) -> usize {
        flash_keys(self.head_dim, self.kv_planes)
    }

    pub fn flash_stages(&self) -> usize {
        flash_stages(self.head_dim, self.kv_planes)
    }

    pub fn flash_warps(&self) -> usize {
        FLASH_ROWS / 16
    }

    pub fn flash_smem(&self) -> usize {
        let kv = self.flash_stages() * 2 * self.kv_planes * self.flash_keys() * self.head_dim * 2;
        kv.max(FLASH_ROWS * self.head_dim * 2)
    }

    pub fn legacy_flash_smem(&self) -> usize {
        ((16 + 32) * (self.head_dim + 1) + 32 * self.head_dim + 16 * 33) * 4
    }

    pub fn decode_smem(&self) -> usize {
        DECODE_STAGES * 2 * self.kv_planes * DECODE_KEYS * self.head_dim * 2
    }

    pub fn decode_warps(&self, qlen: usize) -> usize {
        (self.group() * qlen).div_ceil(16)
    }

    pub fn decode_max_warps(&self) -> usize {
        self.decode_warps(DECODE_MAX_QLEN)
    }
}

pub fn flash_keys(head_dim: usize, kv_planes: usize) -> usize {
    if head_dim > 128 && kv_planes == 2 { 16 } else { 32 }
}

pub fn flash_stages(head_dim: usize, kv_planes: usize) -> usize {
    if head_dim > 128 && kv_planes == 1 { 3 } else { 2 }
}

pub fn decode_splits(max_ctx: usize) -> usize {
    max_ctx.div_ceil(DECODE_CHUNK).clamp(1, DECODE_MAX_SPLITS)
}

pub fn decode_chunk(len: usize, max_ctx: usize) -> usize {
    let mut c = DECODE_CHUNK;
    while c * decode_splits(max_ctx) < len {
        c <<= 1;
    }
    c
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    RopeKv,
    SplitDecode,
    FlashLegacy,
    FlashMma,
    GroupedDecode,
}

impl Family {
    pub const ALL: [Family; 5] = [Family::RopeKv, Family::SplitDecode, Family::FlashLegacy, Family::FlashMma, Family::GroupedDecode];

    pub fn block(self) -> &'static str {
        match self {
            Family::RopeKv => "RopeKvWrite",
            Family::SplitDecode => "SplitDecode",
            Family::FlashLegacy => "FlashScalar",
            Family::FlashMma => "FlashMma",
            Family::GroupedDecode => "GroupedDecodeMma",
        }
    }

    pub fn kernels(self) -> &'static [&'static str] {
        match self {
            Family::RopeKv => &["qk_norm_rope", "kv_norm_rope"],
            Family::SplitDecode => &["attention_split", "attention_reduce"],
            Family::FlashLegacy => &["flash_attention"],
            Family::FlashMma => &["flash_mma"],
            Family::GroupedDecode => &["batch_decode_attention", "batch_decode_combine"],
        }
    }

    pub fn needs(self) -> &'static [Feature] {
        match self {
            Family::RopeKv | Family::FlashLegacy => &[Feature::WarpShuffle],
            Family::SplitDecode => &[Feature::WarpShuffle],
            Family::FlashMma | Family::GroupedDecode => &[Feature::Mma16816, Feature::CpAsync],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionKernel {
    pub family: Family,
    pub target: Target,
    pub shape: AttentionShape,
    pub threads: u32,
    pub smem_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionError(pub String);

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AttentionError {}

impl AttentionKernel {
    pub fn new(t: &Target, family: Family, shape: AttentionShape) -> Result<AttentionKernel, AttentionError> {
        let (threads, smem_bytes) = match family {
            Family::RopeKv => (shape.head_dim.max(32) as u32, 0),
            Family::SplitDecode => (SPLIT_BLOCK, 0),
            Family::FlashLegacy => (128, shape.legacy_flash_smem() as u32),
            Family::FlashMma => (32 * shape.flash_warps() as u32, shape.flash_smem() as u32),
            Family::GroupedDecode => (32 * shape.decode_max_warps() as u32, shape.decode_smem() as u32),
        };
        let k = AttentionKernel { family, target: *t, shape: shape.clone(), threads, smem_bytes };
        k.validate()?;
        Ok(k)
    }

    pub fn set(t: &Target, shape: AttentionShape) -> Result<Vec<AttentionKernel>, AttentionError> {
        Family::ALL.iter().map(|&f| AttentionKernel::new(t, f, shape.clone())).collect()
    }

    pub fn validate(&self) -> Result<(), AttentionError> {
        let t = &self.target;
        let bad = |s: String| Err(AttentionError(format!("{}: {s}", self.family.block())));
        if t.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", t.language));
        }
        for &f in self.family.needs() {
            if !t.has(f) {
                return bad(format!("needs {f:?}, which the target lacks"));
            }
        }
        if self.threads > t.max_threads {
            return bad(format!("{} threads exceed the target maximum {}", self.threads, t.max_threads));
        }
        if self.smem_bytes > t.smem_limit {
            return bad(format!("needs {} bytes of shared memory, the target allows {}", self.smem_bytes, t.smem_limit));
        }
        let s = &self.shape;
        if s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
            return bad("heads must be a multiple of kv_heads".into());
        }
        if !s.head_dim.is_multiple_of(16) || s.head_dim > 256 {
            return bad(format!("head_dim {} must be a multiple of 16 and at most 256", s.head_dim));
        }
        if s.kv_planes != 1 && s.kv_planes != 2 {
            return bad("kv planes must be 1 or 2".into());
        }
        if s.kv_fp8 && s.kv_planes != 1 {
            return bad("fp8 KV needs a single plane".into());
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!("{}\nattention {} threads={} smem={}\n  {:?}", self.target.describe(), self.family.block(), self.threads, self.smem_bytes, self.shape)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

pub fn attention_hash(kernels: &[AttentionKernel]) -> String {
    fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}

pub(crate) fn constants(s: &AttentionShape, families: &[Family]) -> String {
    let mut o = String::new();
    let f = |v: f64| format!("{v:e}f");
    let r = &s.rope;
    let m = r.mrope;
    let _ = writeln!(o, "constexpr int ROT = {};", r.rotary_dim);
    let _ = writeln!(o, "constexpr int ROT_HALF = ROT / 2;");
    let _ = writeln!(o, "constexpr bool QK_NORM = {};", s.qk_norm);
    let local: Vec<String> = s.local_rope.iter().map(|b| b.to_string()).collect();
    let _ = writeln!(o, "constexpr bool LOCAL_ROPE[] = {{ {} }};", if local.is_empty() { "false".into() } else { local.join(", ") });
    let nope: Vec<String> = s.nope.iter().map(|b| b.to_string()).collect();
    let _ = writeln!(o, "constexpr bool NOPE[] = {{ {} }};", if nope.is_empty() { "false".into() } else { nope.join(", ") });
    let _ = writeln!(o, "constexpr float LOCAL_THETA = {};", f(s.local_theta));
    let _ = writeln!(o, "constexpr float QUERY_SCALE = {};", f(s.query_scale));
    let _ = writeln!(o, "constexpr bool BIDI = {};", s.bidirectional);
    let _ = writeln!(o, "constexpr float SOFTCAP = {};", f(s.softcap));
    let _ = writeln!(o, "constexpr float THETA = {};", f(r.theta));
    let _ = writeln!(o, "constexpr int ROPE_KIND = {};", r.kind);
    let _ = writeln!(o, "constexpr bool MROPE = {};", m.is_some());
    let _ = writeln!(o, "constexpr bool MROPE_INTERLEAVED = {};", m.is_some_and(|m| m.interleaved));
    for i in 0..3 {
        let _ = writeln!(o, "constexpr int MROPE_S{i} = {};", m.map_or(0, |m| m.section[i]));
    }
    let _ = writeln!(o, "constexpr float ROPE_FACTOR = {};", f(r.factor));
    let _ = writeln!(o, "constexpr float ROPE_LOW_FREQ = {};", f(r.low));
    let _ = writeln!(o, "constexpr float ROPE_HIGH_FREQ = {};", f(r.high));
    let _ = writeln!(o, "constexpr float ROPE_ORIG_POS = {};", f(r.original));
    let _ = writeln!(o, "constexpr float ROPE_MSCALE = {};", f(r.mscale));
    let _ = writeln!(o, "constexpr int PAGE = {PAGE};");
    let _ = writeln!(o, "constexpr int ATTN_LAYERS = {};", s.kv_layers.max(1));
    let _ = writeln!(o, "constexpr int KV_PLANES = {};", s.kv_planes);
    let _ = writeln!(o, "constexpr bool KV_FP8 = {};", s.kv_fp8);
    let _ = writeln!(o, "constexpr int KV_F16_LAYERS = {};", if s.kv_fp8 { s.kv_f16_layers.min(s.kv_layers) } else { 0 });
    if families.contains(&Family::FlashMma) {
        let _ = writeln!(o, "constexpr int FM_WARPS = {};", s.flash_warps());
        let _ = writeln!(o, "constexpr int FM_BQ = 16 * FM_WARPS;");
        let _ = writeln!(o, "constexpr int FM_BK = {};", s.flash_keys());
        let _ = writeln!(o, "constexpr int FM_STAGES = {};", s.flash_stages());
        let _ = writeln!(o, "constexpr int FM_KB = HEAD_DIM / 16;");
        let _ = writeln!(o, "constexpr int FM_NB = HEAD_DIM / 8;");
        let _ = writeln!(o, "constexpr int FM_SMEM = {};", s.flash_smem());
    }
    if families.contains(&Family::GroupedDecode) {
        let _ = writeln!(o, "constexpr int DM_BK = {DECODE_KEYS};");
        let _ = writeln!(o, "constexpr int DM_STAGES = {DECODE_STAGES};");
        let _ = writeln!(o, "constexpr int DM_CHUNK = {DECODE_CHUNK};");
        let _ = writeln!(o, "constexpr bool DM_SPLIT = true;");
        let _ = writeln!(o, "constexpr int DM_MAX_WARPS = {};", s.decode_max_warps());
    }
    o
}
