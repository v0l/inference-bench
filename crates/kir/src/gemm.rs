use crate::{Feature, Language, Target};
use std::fmt::{self, Write as _};

pub const Q8_GROUP: usize = 32;
pub const FP8_BLOCK: usize = 128;
pub const FP4_GROUP: usize = 16;
pub const FP4_STAGES: usize = 3;
pub const F4W_BK: usize = 64;
pub const F4W_ROW: usize = 48;
pub const F4A_ROW: usize = 72;
pub const F4A_STAGES: usize = 4;
pub const F4A_SKINNY_STAGES: usize = 3;
pub const FP8_ACT_STAGES: usize = 2;
pub const BATCH_KC: usize = 128;
pub const BATCH_VARIANTS: [usize; 4] = [8, 16, 32, 64];
pub const BATCH_MAX_TOKENS: usize = 64;
pub const BATCH_REDUCE_BLOCK: usize = 256;
pub const BATCH_LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
pub const SKINNY_WIDE_MIN_ROWS: usize = 16384;
pub const ROWS_MAX_TOKENS: usize = 32;
pub const FP8_HUGE_MIN_TOKENS: usize = 256;
pub const GGB_MIN_TOKENS: usize = 32;
pub const GGML_VARIANTS: [usize; 3] = [8, 16, 32];
pub const GGML_GEMV_THREADS: u32 = 256;
pub const GGML_GEMV_ROWS: usize = 2;

pub fn ggml_gemv_rows(_n: usize) -> usize {
    GGML_GEMV_ROWS
}
pub const GGML_GEMV_MAX_K: usize = 8192;
const GGML_SUPER_BLOCK: usize = 256;
const GGML_WARPS: usize = 4;
const GGML_MIN_CHUNKS: usize = 2;
const GGML_BLOCKS_PER_SM: usize = 2;
pub const GGB_LINEARS: [&str; 4] = ["qkv", "o", "gate_up", "down"];
pub const FP4_GEMV_THREADS: u32 = 512;
pub const DEQUANT_TILE: u32 = 16;
const MAX_BLOCKS_PER_SM: usize = 4;
const BIG_DEFAULT: Tile = Tile { bm: 128, bn: 128, wm: 2, wn: 4 };
const BF16_BIG: Tile = Tile { bm: 256, bn: 128, wm: 4, wn: 2 };
const BF16_BIG_MIN_N: usize = 8192;
const TINY_WN: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeightFormat {
    Bf16,
    Q8,
    Fp8,
    Ggml,
    Fp4,
}

impl WeightFormat {
    pub fn decoder(self) -> &'static str {
        match self {
            WeightFormat::Bf16 => "DecBf16",
            WeightFormat::Q8 => "DecQ8",
            WeightFormat::Fp8 => "DecFp8",
            WeightFormat::Ggml => "DecGgml",
            WeightFormat::Fp4 => "DecFp4",
        }
    }

    pub fn from_decoder(decoder: &str) -> Option<WeightFormat> {
        [WeightFormat::Bf16, WeightFormat::Q8, WeightFormat::Fp8, WeightFormat::Ggml, WeightFormat::Fp4].into_iter().find(|w| w.decoder() == decoder)
    }

    pub fn gemv_rows(self) -> usize {
        match self {
            WeightFormat::Ggml => 2,
            _ => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WeightFormat::Bf16 => "bf16",
            WeightFormat::Q8 => "q8",
            WeightFormat::Fp8 => "fp8",
            WeightFormat::Ggml => "ggml",
            WeightFormat::Fp4 => "nvfp4",
        }
    }

    pub fn bytes(self) -> bool {
        matches!(self, WeightFormat::Q8 | WeightFormat::Fp8)
    }

    pub fn wrow(self, bk: usize) -> usize {
        match self {
            WeightFormat::Q8 | WeightFormat::Fp8 => bk + 16,
            WeightFormat::Fp4 => F4W_ROW,
            WeightFormat::Bf16 | WeightFormat::Ggml => bk * 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epilogue {
    Store,
    Residual,
    Act,
    Map,
}

impl Epilogue {
    pub fn cuda(self) -> &'static str {
        match self {
            Epilogue::Store => "EPI_STORE",
            Epilogue::Residual => "EPI_RESIDUAL",
            Epilogue::Act => "EPI_ACT",
            Epilogue::Map => "EPI_MAP",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub n: usize,
    pub k: usize,
    pub epi: Epilogue,
    pub bias: bool,
    pub act: crate::Activation,
}

impl GemmShape {
    pub fn new(n: usize, k: usize, epi: Epilogue, bias: bool) -> GemmShape {
        GemmShape { n, k, epi, bias, act: crate::Activation::Silu }
    }

    pub fn with_act(self, act: crate::Activation) -> GemmShape {
        GemmShape { act, ..self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub wm: usize,
    pub wn: usize,
}

impl Tile {
    pub const SMALL: Tile = Tile { bm: 64, bn: 64, wm: 2, wn: 2 };
    pub const FP4_BIG: Tile = Tile { bm: 128, bn: 256, wm: 4, wn: 4 };
    pub const FP8_HUGE: Tile = Tile { bm: 128, bn: 128, wm: 2, wn: 4 };
    pub const NARROW: Tile = Tile { bm: 64, bn: 32, wm: 2, wn: 2 };
    pub const SKINNY: Tile = Tile { bm: 32, bn: 32, wm: 2, wn: 2 };
    pub const SKINNY_WIDE: Tile = Tile { bm: 32, bn: 64, wm: 2, wn: 2 };
    pub const TINY: Tile = Tile { bm: 16, bn: 16 * TINY_WN, wm: 1, wn: TINY_WN };

    pub fn from_array([bm, bn, wm, wn]: [usize; 4]) -> Tile {
        Tile { bm, bn, wm, wn }
    }

    pub fn threads(&self) -> u32 {
        (self.wm * self.wn * 32) as u32
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TileChoice {
    #[default]
    Auto,
    Small,
    Big,
    Narrow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    Small,
    Big,
    Narrow,
    Tiny,
    Huge,
    Skinny,
    SkinnyWide,
}

impl Variant {
    pub const MMA: [Variant; 4] = [Variant::Small, Variant::Big, Variant::Narrow, Variant::Tiny];
    pub const FP4_ACT: [Variant; 5] = [Variant::Small, Variant::Big, Variant::Narrow, Variant::Skinny, Variant::SkinnyWide];
    pub const FP8_ACT: [Variant; 4] = [Variant::Small, Variant::Big, Variant::Narrow, Variant::Huge];

    pub fn suffix(self) -> &'static str {
        match self {
            Variant::Small => "",
            Variant::Big => "_big",
            Variant::Narrow => "_narrow",
            Variant::Tiny => "_tiny",
            Variant::Huge => "_huge",
            Variant::Skinny => "_skinny",
            Variant::SkinnyWide => "_skinnywide",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tuning {
    pub tile: TileChoice,
    pub big: Option<Tile>,
    pub fp4_big: Option<Tile>,
    pub fp8_huge: Option<Tile>,
    pub fp4_stages: Option<usize>,
}

impl Tuning {
    pub fn tile(&self, v: Variant) -> Tile {
        match v {
            Variant::Small => Tile::SMALL,
            Variant::Big => self.big.unwrap_or(BIG_DEFAULT),
            Variant::Narrow => Tile::NARROW,
            Variant::Tiny => Tile::TINY,
            Variant::Huge => self.fp8_huge.unwrap_or(Tile::FP8_HUGE),
            Variant::Skinny => Tile::SKINNY,
            Variant::SkinnyWide => Tile::SKINNY_WIDE,
        }
    }

    pub fn mma_tile(&self, v: Variant, wf: WeightFormat, n: usize) -> Tile {
        match (v, wf) {
            (Variant::Big, WeightFormat::Bf16) if n >= BF16_BIG_MIN_N => self.big.unwrap_or(BF16_BIG),
            _ => self.tile(v),
        }
    }

    pub fn fp4_tile(&self, v: Variant) -> Tile {
        match v {
            Variant::Big => self.fp4_big.unwrap_or(Tile::FP4_BIG),
            _ => self.tile(v),
        }
    }

    pub fn f4a_stages(&self, t: &Target, tile: Tile) -> usize {
        let fit = t.smem_limit as usize / ((tile.bm + tile.bn) * F4A_ROW);
        self.fp4_stages.unwrap_or(if tile.bm <= 32 { F4A_SKINNY_STAGES } else { F4A_STAGES }).min(fit).max(2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MmaShape {
    pub tile: Tile,
    pub bk: usize,
    pub stages: usize,
    pub split: bool,
    pub wf: WeightFormat,
}

impl MmaShape {
    pub fn new(t: &Target, tile: Tile, k: usize, wf: WeightFormat, split: bool) -> MmaShape {
        let bk = match wf {
            WeightFormat::Q8 => Q8_GROUP,
            WeightFormat::Fp4 => F4W_BK,
            WeightFormat::Fp8 if tile.bm == 16 => 128,
            WeightFormat::Fp8 => 64,
            WeightFormat::Bf16 | WeightFormat::Ggml if tile.bm == 16 && k.is_multiple_of(128) => 128,
            WeightFormat::Bf16 | WeightFormat::Ggml => 32,
        };
        let stages = match wf {
            WeightFormat::Fp4 => {
                let per_stage = tile.bm * bk * 2 * if split { 2 } else { 1 } + tile.bn * wf.wrow(bk);
                (t.smem_limit as usize / per_stage).clamp(2, FP4_STAGES)
            }
            _ => 2,
        };
        MmaShape { tile, bk, stages, split, wf }
    }

    pub fn smem_bytes(&self) -> u32 {
        let planes = if self.split { 2 } else { 1 };
        let a = self.tile.bm * self.bk * 2 * planes;
        ((a + self.tile.bn * self.wf.wrow(self.bk)) * self.stages) as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BatchShape {
    pub tokens: usize,
    pub warps: usize,
    pub stages: usize,
    pub mj: usize,
}

impl BatchShape {
    pub fn for_tokens(tokens: usize) -> BatchShape {
        let tokens = BATCH_VARIANTS.iter().copied().find(|&v| v >= tokens).unwrap_or(BATCH_MAX_TOKENS);
        match tokens {
            0..=16 => BatchShape { tokens, warps: 4, stages: 3, mj: 2 },
            17..=32 => BatchShape { tokens, warps: 8, stages: 3, mj: 1 },
            _ => BatchShape { tokens, warps: 8, stages: 2, mj: 1 },
        }
    }

    pub fn for_ggml(tokens: usize) -> BatchShape {
        let tokens = BATCH_VARIANTS.iter().copied().find(|&v| v >= tokens).unwrap_or(GGB_MIN_TOKENS).min(GGB_MIN_TOKENS);
        BatchShape { tokens, warps: GGML_WARPS, stages: 3, mj: 1 }
    }

    pub fn ggml_splits(&self, t: &Target, n: usize, k: usize) -> usize {
        let blocks = n.div_ceil(self.rows());
        let most = (k / BATCH_KC / GGML_MIN_CHUNKS).max(1);
        let want = t.sm_count as usize * GGML_BLOCKS_PER_SM;
        (1..=most).find(|&s| blocks * s >= want).unwrap_or(most)
    }

    pub fn rows(&self) -> usize {
        self.warps * self.mj * 16
    }

    pub fn threads(&self) -> u32 {
        (self.warps * 32) as u32
    }

    pub fn smem(&self) -> u32 {
        (self.stages * self.tokens * BATCH_KC * 2 * 2) as u32
    }

    pub fn blocks_per_sm(&self, t: &Target) -> usize {
        (t.smem_per_sm() as usize / (self.smem() as usize + t.smem_reserved_per_block() as usize)).clamp(1, MAX_BLOCKS_PER_SM)
    }

    pub fn splits(&self, t: &Target, n: usize, k: usize) -> usize {
        let blocks = n.div_ceil(self.rows());
        let most = (k / BATCH_KC / 4).max(1);
        let wave = self.blocks_per_sm(t) * t.sm_count as usize;
        let fill = |s: usize| (blocks * s) as f64 / ((blocks * s).div_ceil(wave) * wave) as f64;
        (1..=most).find(|&s| blocks * s >= wave && fill(s) >= 0.85).unwrap_or(most)
    }
}

pub fn batch_part_len(t: &Target, n: usize, k: usize) -> usize {
    BATCH_VARIANTS.iter().map(|&v| BatchShape::for_tokens(v)).map(|s| s.splits(t, n, k) * n * s.tokens).max().unwrap_or(0)
}

pub fn ggml_part_len(t: &Target, n: usize, k: usize) -> usize {
    GGML_VARIANTS.iter().map(|&v| BatchShape::for_ggml(v)).map(|s| s.ggml_splits(t, n, k) * n * s.tokens).max().unwrap_or(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp8Scale {
    Block,
    Channel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mainloop {
    Mma(MmaShape),
    DequantF16 { tile: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp4Gemv {
    GateUpAct,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Tile { shape: GemmShape, wf: WeightFormat, main: Mainloop, token_major: bool },
    Fp8Act { shape: GemmShape, tile: Tile, stages: usize, scale: Fp8Scale },
    Fp4Act { shape: GemmShape, tile: Tile, stages: usize },
    Batch { shape: GemmShape, wf: WeightFormat, batch: BatchShape, blocks_per_sm: usize },
    GgmlBatch { shape: GemmShape, batch: BatchShape, blocks_per_sm: usize, split: bool },
    GgmlGemv { shape: GemmShape, split: bool },
    BatchReduce { shape: GemmShape, wf: WeightFormat },
    QuantFp8 { scale: Fp8Scale },
    QuantFp4,
    Gemv { op: Fp4Gemv, n: usize, k: usize },
    HeadNormFp4 { n: usize, k: usize, rows: usize },
}

impl Kind {
    pub fn block(&self) -> &'static str {
        match self {
            Kind::Tile { main: Mainloop::Mma(_), .. } => "TileMma",
            Kind::Tile { main: Mainloop::DequantF16 { .. }, .. } => "DequantF16",
            Kind::Fp8Act { .. } => "Fp8ActMma",
            Kind::Fp4Act { .. } => "BlockScaledFp4Mma",
            Kind::Batch { .. } => "BatchRowsMma",
            Kind::GgmlBatch { .. } => "GgmlBatchRowsMma",
            Kind::GgmlGemv { .. } => "GgmlGemv",
            Kind::BatchReduce { .. } => "BatchReduce",
            Kind::QuantFp8 { .. } => "QuantRowsFp8",
            Kind::QuantFp4 => "QuantRowsFp4",
            Kind::Gemv { .. } => "Fp4Gemv",
            Kind::HeadNormFp4 { .. } => "Fp4HeadGemv",
        }
    }

    pub fn needs(&self) -> &'static [Feature] {
        match self {
            Kind::Tile { main: Mainloop::Mma(_), .. } | Kind::Batch { .. } | Kind::GgmlBatch { .. } => &[Feature::Mma16816, Feature::CpAsync],
            Kind::Tile { main: Mainloop::DequantF16 { .. }, .. } => &[Feature::ShaderF16],
            Kind::Fp8Act { .. } => &[Feature::Fp8Mma, Feature::CpAsync],
            Kind::Fp4Act { .. } => &[Feature::BlockScaledFp4, Feature::CpAsync],
            Kind::QuantFp8 { .. } => &[Feature::Fp8Mma],
            Kind::QuantFp4 => &[Feature::BlockScaledFp4],
            Kind::BatchReduce { .. } => &[],
            Kind::Gemv { .. } | Kind::HeadNormFp4 { .. } | Kind::GgmlGemv { .. } => &[Feature::WarpShuffle, Feature::CpAsync],
        }
    }

    pub fn languages(&self) -> &'static [Language] {
        match self {
            Kind::Tile { main: Mainloop::DequantF16 { .. }, .. } => &[Language::Wgsl],
            _ => &[Language::Cuda],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GemmError {
    Unsupported { kernel: String, block: &'static str, reason: String },
    Invalid(String),
}

impl GemmError {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, GemmError::Unsupported { .. })
    }
}

impl fmt::Display for GemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmError::Unsupported { kernel, block, reason } => write!(f, "{kernel}: block {block} {reason}"),
            GemmError::Invalid(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for GemmError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelInstance {
    pub name: String,
    pub target: Target,
    pub kind: Kind,
    pub threads: u32,
    pub smem_bytes: u32,
}

fn unsupported(name: &str, kind: &Kind, reason: String) -> GemmError {
    GemmError::Unsupported { kernel: name.to_string(), block: kind.block(), reason }
}

impl KernelInstance {
    fn build(t: &Target, name: String, kind: Kind, threads: u32, smem_bytes: u32) -> Result<KernelInstance, GemmError> {
        let k = KernelInstance { name, target: *t, kind, threads, smem_bytes };
        k.validate()?;
        Ok(k)
    }

    pub fn gemm(t: &Target, name: &str, shape: GemmShape, wf: WeightFormat, tile: Tile, split: bool, token_major: bool) -> Result<KernelInstance, GemmError> {
        match t.language {
            Language::Cuda => {
                let m = MmaShape::new(t, tile, shape.k, wf, split);
                let kind = Kind::Tile { shape, wf, main: Mainloop::Mma(m), token_major };
                KernelInstance::build(t, name.to_string(), kind, tile.threads(), m.smem_bytes())
            }
            Language::Wgsl => {
                let kind = Kind::Tile { shape, wf, main: Mainloop::DequantF16 { tile: DEQUANT_TILE }, token_major };
                if matches!(wf, WeightFormat::Q8 | WeightFormat::Ggml) {
                    return Err(unsupported(name, &kind, format!("has no {} decoder", wf.label())));
                }
                let smem = 2 * DEQUANT_TILE * DEQUANT_TILE * 2 + DEQUANT_TILE * DEQUANT_TILE * 4;
                KernelInstance::build(t, name.to_string(), kind, DEQUANT_TILE * DEQUANT_TILE, smem)
            }
        }
    }

    pub fn fp8_act(t: &Target, name: &str, shape: GemmShape, tile: Tile, scale: Fp8Scale) -> Result<KernelInstance, GemmError> {
        let smem = (FP8_ACT_STAGES * (tile.bm + tile.bn) * FP8_BLOCK) as u32;
        KernelInstance::build(t, name.to_string(), Kind::Fp8Act { shape, tile, stages: FP8_ACT_STAGES, scale }, tile.threads(), smem)
    }

    pub fn fp4_act(t: &Target, name: &str, shape: GemmShape, tile: Tile, tuning: &Tuning) -> Result<KernelInstance, GemmError> {
        let stages = tuning.f4a_stages(t, tile);
        let smem = (stages * (tile.bm + tile.bn) * F4A_ROW) as u32;
        KernelInstance::build(t, name.to_string(), Kind::Fp4Act { shape, tile, stages }, tile.threads(), smem)
    }

    pub fn batch(t: &Target, name: &str, shape: GemmShape, wf: WeightFormat, tokens: usize) -> Result<KernelInstance, GemmError> {
        let batch = BatchShape::for_tokens(tokens);
        let kind = Kind::Batch { shape, wf, batch, blocks_per_sm: batch.blocks_per_sm(t) };
        KernelInstance::build(t, name.to_string(), kind, batch.threads(), batch.smem())
    }

    pub fn ggml_batch(t: &Target, name: &str, shape: GemmShape, tokens: usize, split: bool) -> Result<KernelInstance, GemmError> {
        let batch = BatchShape::for_ggml(tokens);
        let kind = Kind::GgmlBatch { shape, batch, blocks_per_sm: batch.blocks_per_sm(t), split };
        KernelInstance::build(t, name.to_string(), kind, batch.threads(), batch.smem())
    }

    pub fn batch_reduce(t: &Target, name: &str, shape: GemmShape, wf: WeightFormat) -> Result<KernelInstance, GemmError> {
        KernelInstance::build(t, name.to_string(), Kind::BatchReduce { shape, wf }, BATCH_REDUCE_BLOCK as u32, 0)
    }

    pub fn quant_fp8(t: &Target, scale: Fp8Scale) -> Result<KernelInstance, GemmError> {
        let name = match scale {
            Fp8Scale::Block => "quant_rows_fp8",
            Fp8Scale::Channel => "quant_rows_fp8_token",
        };
        let threads = match scale {
            Fp8Scale::Block => (FP8_BLOCK / 2) as u32,
            Fp8Scale::Channel => 256,
        };
        KernelInstance::build(t, name.to_string(), Kind::QuantFp8 { scale }, threads, 0)
    }

    pub fn quant_fp4(t: &Target) -> Result<KernelInstance, GemmError> {
        KernelInstance::build(t, "quant_rows_fp4".to_string(), Kind::QuantFp4, 128, 0)
    }

    pub fn ggml_gemv(t: &Target, name: &str, shape: GemmShape, split: bool) -> Result<KernelInstance, GemmError> {
        KernelInstance::build(t, name.to_string(), Kind::GgmlGemv { shape, split }, GGML_GEMV_THREADS, 0)
    }

    pub fn fp4_gemv(t: &Target, name: &str, op: Fp4Gemv, n: usize, k: usize) -> Result<KernelInstance, GemmError> {
        KernelInstance::build(t, name.to_string(), Kind::Gemv { op, n, k }, FP4_GEMV_THREADS, 0)
    }

    pub fn fp4_head_norm(t: &Target, name: &str, n: usize, k: usize) -> Result<KernelInstance, GemmError> {
        let rows = if n.is_multiple_of(2) { 2 } else { 1 };
        KernelInstance::build(t, name.to_string(), Kind::HeadNormFp4 { n, k, rows }, FP4_GEMV_THREADS, 0)
    }

    pub fn tile(&self) -> Option<Tile> {
        match self.kind {
            Kind::Tile { main: Mainloop::Mma(m), .. } => Some(m.tile),
            Kind::Tile { main: Mainloop::DequantF16 { tile }, .. } => {
                let t = tile as usize;
                Some(Tile { bm: t, bn: t, wm: 1, wn: 1 })
            }
            Kind::Fp8Act { tile, .. } | Kind::Fp4Act { tile, .. } => Some(tile),
            _ => None,
        }
    }

    pub fn batch_shape(&self) -> Option<BatchShape> {
        match self.kind {
            Kind::Batch { batch, .. } | Kind::GgmlBatch { batch, .. } => Some(batch),
            _ => None,
        }
    }

    pub fn needs(&self) -> &'static [Feature] {
        self.kind.needs()
    }

    pub fn validate(&self) -> Result<(), GemmError> {
        let t = &self.target;
        let bad = |s: String| Err(GemmError::Invalid(format!("{}: {s}", self.name)));
        if self.name.is_empty() || !self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return bad("kernel name is not an identifier".into());
        }
        if !self.kind.languages().contains(&t.language) {
            return Err(unsupported(&self.name, &self.kind, format!("has no {:?} lowering", t.language)));
        }
        for &f in self.needs() {
            if !t.has(f) {
                return Err(unsupported(&self.name, &self.kind, format!("needs {f:?}, which the target lacks")));
            }
        }
        if self.threads == 0 || self.threads > t.max_threads || !self.threads.is_multiple_of(t.warp_size.max(1)) {
            return bad(format!("{} threads do not fit the target ({} max, warp {})", self.threads, t.max_threads, t.warp_size));
        }
        if self.smem_bytes > t.smem_limit {
            return bad(format!("{} bytes of shared memory exceeds the target limit {}", self.smem_bytes, t.smem_limit));
        }
        let check_tile = |tile: Tile| -> Result<(), GemmError> {
            if tile.wm == 0 || tile.wn == 0 || !tile.bm.is_multiple_of(16 * tile.wm) || !tile.bn.is_multiple_of(16 * tile.wn) {
                return bad(format!("tile {tile:?} needs bm a multiple of 16 * wm and bn a multiple of 16 * wn"));
            }
            Ok(())
        };
        let check_shape = |s: &GemmShape, step: usize| -> Result<(), GemmError> {
            if s.n == 0 || s.k == 0 || !s.k.is_multiple_of(step) {
                return bad(format!("shape {}x{} needs k a multiple of {step}", s.n, s.k));
            }
            if s.epi == Epilogue::Act && !s.n.is_multiple_of(4) {
                return bad("gate/up epilogue needs n a multiple of 4".into());
            }
            if s.epi == Epilogue::Map && !s.n.is_multiple_of(2) {
                return bad("the mapped epilogue needs n a multiple of 2".into());
            }
            Ok(())
        };
        match &self.kind {
            Kind::Tile { shape, main: Mainloop::Mma(m), .. } => {
                check_tile(m.tile)?;
                check_shape(shape, m.bk)?;
                if m.stages < 2 {
                    return bad("the pipeline needs at least 2 stages".into());
                }
            }
            Kind::Tile { shape, main: Mainloop::DequantF16 { tile }, wf, .. } => {
                let step = if *wf == WeightFormat::Fp8 { FP8_BLOCK } else { *tile as usize };
                check_shape(shape, step)?;
            }
            Kind::Fp8Act { shape, tile, .. } => {
                check_tile(*tile)?;
                check_shape(shape, FP8_BLOCK)?;
            }
            Kind::Fp4Act { shape, tile, stages } => {
                check_tile(*tile)?;
                check_shape(shape, 128)?;
                if *stages < 2 {
                    return bad("the pipeline needs at least 2 stages".into());
                }
            }
            Kind::Batch { shape, wf, .. } => {
                if !matches!(wf, WeightFormat::Fp4 | WeightFormat::Fp8) {
                    return bad(format!("batch rows have no {} decoder", wf.label()));
                }
                check_shape(shape, BATCH_KC)?;
                if *wf == WeightFormat::Fp4 && !shape.n.is_multiple_of(16) {
                    return bad("nvfp4 batch rows need n a multiple of 16".into());
                }
            }
            Kind::GgmlBatch { shape, .. } => check_shape(shape, BATCH_KC)?,
            Kind::GgmlGemv { shape, .. } => {
                check_shape(shape, GGML_SUPER_BLOCK)?;
                if shape.k > GGML_GEMV_MAX_K || !shape.n.is_multiple_of(ggml_gemv_rows(shape.n)) {
                    return bad(format!("shape {}x{} does not fit the ggml gemv", shape.n, shape.k));
                }
            }
            Kind::BatchReduce { wf, .. } => {
                if !matches!(wf, WeightFormat::Fp4 | WeightFormat::Fp8 | WeightFormat::Ggml) {
                    return bad(format!("batch rows have no {} decoder", wf.label()));
                }
            }
            Kind::Gemv { n, k, .. } | Kind::HeadNormFp4 { n, k, .. } => {
                if *n == 0 || !k.is_multiple_of(FP4_GROUP) {
                    return bad(format!("{n}x{k} needs k a multiple of the nvfp4 group"));
                }
            }
            Kind::QuantFp8 { .. } | Kind::QuantFp4 => {}
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        let _ = write!(s, "{}\ngemm {} block={} threads={} smem={}", self.target.describe(), self.name, self.kind.block(), self.threads, self.smem_bytes);
        let _ = write!(s, "\n  {:?}", self.kind);
        s
    }

    pub fn hash(&self) -> String {
        crate::fnv(self.describe().bytes())
    }
}

impl fmt::Display for KernelInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

pub fn instances_hash(kernels: &[KernelInstance]) -> String {
    crate::fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmPath {
    Mma,
    Ggb,
    F4w,
    F4a,
    F8a,
    F8b,
    F4b,
    Ggr,
    Ggv,
    Rows,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Policy {
    pub tuning: Tuning,
    pub fp8_weights: Option<Fp8Scale>,
    pub fp8_act: bool,
    pub fp4_act: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request<'a> {
    pub linear: &'a str,
    pub shape: GemmShape,
    pub tokens: usize,
    pub weight: WeightFormat,
    pub split: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub name: String,
    pub quant: Option<&'static str>,
    pub reduce: Option<String>,
    pub tile: Tile,
    pub path: GemmPath,
    pub decoder: &'static str,
    pub kernel: Option<KernelInstance>,
}

fn decoder_for(path: GemmPath, weight: WeightFormat) -> &'static str {
    match path {
        GemmPath::F8a => "fp8 x fp8 act",
        GemmPath::F4a => "nvfp4 x nvfp4 act",
        GemmPath::F8b => "fp8 batch rows",
        GemmPath::F4b => "nvfp4 batch rows",
        GemmPath::Ggr => "ggml batch rows",
        GemmPath::Ggv => "ggml gemv",
        GemmPath::Rows => "bf16 one block per row",
        GemmPath::Ggb => "ggml to bf16 scratch",
        GemmPath::Mma | GemmPath::F4w => weight.decoder(),
    }
}

pub fn ggr_suffix(tokens: usize) -> usize {
    BatchShape::for_ggml(tokens).tokens
}

pub fn batch_suffix(tokens: usize) -> usize {
    BatchShape::for_tokens(tokens).tokens
}

pub fn variant_for(t: &Target, tuning: &Tuning, linear: &str, rows: usize, tokens: usize) -> Variant {
    let sms = t.sm_count as usize;
    let blocks = |tile: Tile| rows.div_ceil(tile.bn) * tokens.div_ceil(tile.bm);
    match tuning.tile {
        TileChoice::Small => Variant::Small,
        TileChoice::Big => Variant::Big,
        TileChoice::Narrow => Variant::Narrow,
        _ if tokens <= 16 => Variant::Tiny,
        _ if linear == "lm_head" && tokens <= BATCH_MAX_TOKENS => Variant::Narrow,
        _ if blocks(tuning.tile(Variant::Big)) >= 2 * sms => Variant::Big,
        _ if blocks(Tile::SMALL) >= 2 * sms => Variant::Small,
        _ => Variant::Narrow,
    }
}

fn skinny_variant(r: &Request, tuning: &Tuning) -> Option<Variant> {
    if tuning.tile != TileChoice::Auto || r.tokens <= BATCH_VARIANTS[1] || r.tokens > ROWS_MAX_TOKENS {
        return None;
    }
    Some(if r.shape.n >= SKINNY_WIDE_MIN_ROWS { Variant::SkinnyWide } else { Variant::Skinny })
}

fn plain(r: &Request, name: String, path: GemmPath, tile: Tile, kernel: Option<KernelInstance>) -> Choice {
    Choice { name, quant: None, reduce: None, tile, path, decoder: decoder_for(path, r.weight), kernel }
}

fn fp8_act_choice(t: &Target, policy: &Policy, r: &Request, variant: Variant) -> Result<Choice, GemmError> {
    let scale = policy.fp8_weights.unwrap_or(Fp8Scale::Block);
    let tile = policy.tuning.tile(variant);
    let name = format!("{}_f8{}", r.linear, variant.suffix());
    let k = KernelInstance::fp8_act(t, &name, r.shape, tile, scale)?;
    let quant = match scale {
        Fp8Scale::Block => "quant_rows_fp8",
        Fp8Scale::Channel => "quant_rows_fp8_token",
    };
    Ok(Choice { quant: Some(quant), ..plain(r, name, GemmPath::F8a, tile, Some(k)) })
}

fn fp4_act_choice(t: &Target, policy: &Policy, r: &Request, variant: Variant) -> Result<Choice, GemmError> {
    let tile = policy.tuning.fp4_tile(variant);
    let name = format!("{}_f4a{}", r.linear, variant.suffix());
    let k = KernelInstance::fp4_act(t, &name, r.shape, tile, &policy.tuning)?;
    Ok(Choice { quant: Some("quant_rows_fp4"), ..plain(r, name, GemmPath::F4a, tile, Some(k)) })
}

fn batch_choice(t: &Target, r: &Request, rows: usize, tile: Tile) -> Result<Choice, GemmError> {
    let (stem, path) = if r.weight == WeightFormat::Fp8 { ("f8b", GemmPath::F8b) } else { ("f4b", GemmPath::F4b) };
    let name = format!("{}_{stem}{}", r.linear, batch_suffix(rows));
    let k = KernelInstance::batch(t, &name, r.shape, r.weight, rows)?;
    Ok(Choice { reduce: Some(format!("{}_{stem}_reduce", r.linear)), ..plain(r, name, path, tile, Some(k)) })
}

fn weight_tile_choice(t: &Target, policy: &Policy, r: &Request, variant: Variant) -> Result<Choice, GemmError> {
    let tuning = &policy.tuning;
    let suffix = variant.suffix();
    if r.weight == WeightFormat::Fp4 {
        let tile = tuning.tile(variant);
        let name = format!("{}_f4w{suffix}", r.linear);
        let k = KernelInstance::gemm(t, &name, r.shape, WeightFormat::Fp4, tile, true, false)?;
        return Ok(plain(r, name, GemmPath::F4w, tile, Some(k)));
    }
    let w = r.weight;
    let stem = if w == WeightFormat::Fp8 && policy.fp8_weights.is_none() && !r.linear.ends_with("head") { "f8w" } else { "mma" };
    let name = format!("{}_{stem}{suffix}", r.linear);
    let tile = tuning.mma_tile(variant, w, r.shape.n);
    let k = KernelInstance::gemm(t, &name, r.shape, w, tile, r.split, false)?;
    Ok(plain(r, name, GemmPath::Mma, tile, Some(k)))
}

pub fn candidates(t: &Target, policy: &Policy, r: &Request) -> Vec<Choice> {
    let head = r.linear.ends_with("head");
    let tiled = matches!(r.weight, WeightFormat::Fp8 | WeightFormat::Fp4);
    if policy.tuning.tile != TileChoice::Auto || head || !tiled || !t.has(Feature::Mma16816) {
        return select(t, policy, r).into_iter().collect();
    }
    let mut out = Vec::new();
    let fp8_act = policy.fp8_act && policy.fp8_weights.is_some() && t.has(Feature::Fp8Mma);
    let fp4_act = policy.fp4_act && t.has(Feature::BlockScaledFp4);
    match r.weight {
        WeightFormat::Fp8 if fp8_act => out.extend(Variant::FP8_ACT.into_iter().filter_map(|v| fp8_act_choice(t, policy, r, v).ok())),
        WeightFormat::Fp4 if fp4_act => out.extend(Variant::FP4_ACT.into_iter().filter_map(|v| fp4_act_choice(t, policy, r, v).ok())),
        _ => {}
    }
    let batch = r.tokens <= BATCH_MAX_TOKENS && (r.weight == WeightFormat::Fp4 || BATCH_LINEARS.contains(&r.linear));
    if batch {
        out.extend(BATCH_VARIANTS.into_iter().filter(|&v| v >= r.tokens).filter_map(|v| batch_choice(t, r, v, Tile::TINY).ok()));
    }
    out.extend(Variant::MMA.into_iter().filter_map(|v| weight_tile_choice(t, policy, r, v).ok()));
    out
}

pub fn select(t: &Target, policy: &Policy, r: &Request) -> Result<Choice, GemmError> {
    let tuning = &policy.tuning;
    let auto = tuning.tile == TileChoice::Auto;
    let variant = variant_for(t, tuning, r.linear, r.shape.n, r.tokens);
    let suffix = variant.suffix();
    let head = r.linear.ends_with("head");
    let huge = policy.fp8_weights.is_some() && r.weight == WeightFormat::Fp8 && auto && variant == Variant::Big && r.tokens > FP8_HUGE_MIN_TOKENS;
    let tile = tuning.tile(variant);
    let act_quant = !head && variant != Variant::Tiny;
    let fp8_act = policy.fp8_act && policy.fp8_weights.is_some() && t.has(Feature::Fp8Mma);
    let fp4_act = policy.fp4_act && t.has(Feature::BlockScaledFp4);
    let mma = t.has(Feature::Mma16816);
    let f8b = auto && mma && r.tokens <= BATCH_MAX_TOKENS && BATCH_LINEARS.contains(&r.linear);
    let ggr = auto && mma && r.tokens <= GGB_MIN_TOKENS && (GGB_LINEARS.contains(&r.linear) || r.linear == "lm_head");
    let ggv = ggr && r.tokens == 1 && r.shape.k.is_multiple_of(GGML_SUPER_BLOCK) && r.shape.k <= GGML_GEMV_MAX_K && r.shape.n.is_multiple_of(ggml_gemv_rows(r.shape.n));
    let f4b = auto && mma && r.tokens <= BATCH_MAX_TOKENS && (!head || r.linear == "mtp_head");
    let plain = |name: String, path: GemmPath, tile: Tile, kernel: Option<KernelInstance>| Choice {
        name,
        quant: None,
        reduce: None,
        tile,
        path,
        decoder: decoder_for(path, r.weight),
        kernel,
    };
    if auto && mma && r.linear == "lin_ba" && r.tokens <= BATCH_MAX_TOKENS && r.weight == WeightFormat::Bf16 {
        return Ok(plain("lin_ba_rows".to_string(), GemmPath::Rows, tile, None));
    }
    Ok(match r.weight {
        WeightFormat::Fp8 if act_quant && fp8_act => {
            let scale = policy.fp8_weights.unwrap_or(Fp8Scale::Block);
            let (name, tile) = if huge { (format!("{}_f8_huge", r.linear), tuning.tile(Variant::Huge)) } else { (format!("{}_f8{suffix}", r.linear), tile) };
            let k = KernelInstance::fp8_act(t, &name, r.shape, tile, scale)?;
            let quant = match scale {
                Fp8Scale::Block => "quant_rows_fp8",
                Fp8Scale::Channel => "quant_rows_fp8_token",
            };
            Choice { quant: Some(quant), ..plain(name, GemmPath::F8a, tile, Some(k)) }
        }
        WeightFormat::Fp4 if act_quant && fp4_act => {
            let variant = skinny_variant(r, tuning).unwrap_or(variant);
            let tile = tuning.fp4_tile(variant);
            let name = format!("{}_f4a{}", r.linear, variant.suffix());
            let k = KernelInstance::fp4_act(t, &name, r.shape, tile, tuning)?;
            Choice { quant: Some("quant_rows_fp4"), ..plain(name, GemmPath::F4a, tile, Some(k)) }
        }
        WeightFormat::Fp8 if f8b => {
            let name = format!("{}_f8b{}", r.linear, batch_suffix(r.tokens));
            let k = KernelInstance::batch(t, &name, r.shape, WeightFormat::Fp8, r.tokens)?;
            Choice { reduce: Some(format!("{}_f8b_reduce", r.linear)), ..plain(name, GemmPath::F8b, tile, Some(k)) }
        }
        WeightFormat::Fp4 if f4b => {
            let name = format!("{}_f4b{}", r.linear, batch_suffix(r.tokens));
            let k = KernelInstance::batch(t, &name, r.shape, WeightFormat::Fp4, r.tokens)?;
            Choice { reduce: Some(format!("{}_f4b_reduce", r.linear)), ..plain(name, GemmPath::F4b, tile, Some(k)) }
        }
        WeightFormat::Fp4 => {
            let name = format!("{}_f4w{suffix}", r.linear);
            let k = KernelInstance::gemm(t, &name, r.shape, WeightFormat::Fp4, tile, true, false)?;
            plain(name, GemmPath::F4w, tile, Some(k))
        }
        WeightFormat::Ggml if ggv => {
            let name = format!("{}_ggv", r.linear);
            let k = KernelInstance::ggml_gemv(t, &name, r.shape, r.split)?;
            plain(name, GemmPath::Ggv, tile, Some(k))
        }
        WeightFormat::Ggml if ggr => {
            let name = format!("{}_ggr{}", r.linear, ggr_suffix(r.tokens));
            let k = KernelInstance::ggml_batch(t, &name, r.shape, r.tokens, r.split)?;
            Choice { reduce: Some(format!("{}_ggr_reduce", r.linear)), ..plain(name, GemmPath::Ggr, tile, Some(k)) }
        }
        WeightFormat::Ggml if r.tokens > GGB_MIN_TOKENS && GGB_LINEARS.contains(&r.linear) => {
            let name = format!("{}_ggb{suffix}", r.linear);
            let k = KernelInstance::gemm(t, &name, r.shape, WeightFormat::Bf16, tile, r.split, true)?;
            Choice { quant: Some("gg_dequant_bf16"), ..plain(name, GemmPath::Ggb, tile, Some(k)) }
        }
        w => {
            let stem = if w == WeightFormat::Fp8 && policy.fp8_weights.is_none() && !head { "f8w" } else { "mma" };
            let name = format!("{}_{stem}{suffix}", r.linear);
            let tile = tuning.mma_tile(variant, w, r.shape.n);
            let k = KernelInstance::gemm(t, &name, r.shape, w, tile, r.split, false)?;
            plain(name, GemmPath::Mma, tile, Some(k))
        }
    })
}
