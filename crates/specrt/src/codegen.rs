use crate::spec::{Activation, Quant, RopeScalingKind, Spec};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub const OPS_TEMPLATE: &str = include_str!("../templates/ops.cu");
pub const MODEL_TEMPLATE: &str = include_str!("../templates/model.cu");
pub const GDN_TEMPLATE: &str = include_str!("../templates/gdn.cu");
pub const PAGE: usize = 16;
pub const Q8_GROUP: usize = 32;
pub const FP8_BLOCK: usize = 128;
pub const BATCH_DECODE_SPLITS: usize = 8;
const FLASH_SMEM_LIMIT: usize = 99 * 1024;

pub fn flash_rows(head_dim: usize) -> usize {
    if head_dim > 128 { 64 } else { 128 }
}

pub fn flash_keys(head_dim: usize) -> usize {
    if head_dim > 128 { 16 } else { 32 }
}

pub const LA_CHUNK: usize = 64;
pub const LA_KSPLIT: usize = 4;
pub const LA_COLS: usize = 16;
const FLASH_STAGES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub wm: usize,
    pub wn: usize,
}

pub const MMA_LINEARS: [&str; 5] = ["qkv", "o", "gate_up", "down", "lm_head"];
pub const GDN_LINEARS: [&str; 3] = ["lin_in", "lin_out", "lin_ba"];
pub const MMA_VARIANTS: [&str; 4] = ["", "_big", "_narrow", "_tiny"];
pub const FP8_ACT_LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
pub const FP8_ACT_VARIANTS: [&str; 3] = ["", "_big", "_narrow"];
pub const FP8_ACT_KERNELS: &[&str] = &[
    "quant_rows_fp8",
    "qkv_f8",
    "qkv_f8_big",
    "qkv_f8_narrow",
    "o_f8",
    "o_f8_big",
    "o_f8_narrow",
    "gate_up_f8",
    "gate_up_f8_big",
    "gate_up_f8_narrow",
    "down_f8",
    "down_f8_big",
    "down_f8_narrow",
    "lin_in_f8",
    "lin_in_f8_big",
    "lin_in_f8_narrow",
    "lin_out_f8",
    "lin_out_f8_big",
    "lin_out_f8_narrow",
];
const FP8_ACT_STAGES: usize = 2;

pub fn fp8_act_smem(tile: Tile) -> u32 {
    (FP8_ACT_STAGES * (tile.bm + tile.bn) * (FP8_BLOCK + 16)) as u32
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Tile {
    pub const SMALL: Tile = Tile { bm: 64, bn: 64, wm: 2, wn: 2 };
    pub const NARROW: Tile = Tile { bm: 64, bn: 32, wm: 2, wn: 2 };

    pub fn tiny() -> Tile {
        let wn = env_usize("SPECRT_TINY_WN", 2);
        Tile { bm: 16, bn: 16 * wn, wm: 1, wn }
    }

    pub fn threads(&self) -> u32 {
        (self.wm * self.wn * 32) as u32
    }

    pub fn big() -> Tile {
        match std::env::var("SPECRT_BIG_TILE").ok().and_then(|v| {
            let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 4).then(|| Tile { bm: p[0], bn: p[1], wm: p[2], wn: p[3] })
        }) {
            Some(t) => t,
            None => Tile { bm: 64, bn: 128, wm: 2, wn: 4 },
        }
    }

    pub fn variant(suffix: &str) -> Tile {
        match suffix {
            "_big" => Tile::big(),
            "_narrow" => Tile::NARROW,
            "_tiny" => Tile::tiny(),
            _ => Tile::SMALL,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    Bf16,
    Q8,
    Fp8,
}

impl WeightFormat {
    pub fn of(quant: Quant) -> WeightFormat {
        match quant {
            Quant::Bf16 => WeightFormat::Bf16,
            Quant::Q8 => WeightFormat::Q8,
            Quant::Fp8Block128 => WeightFormat::Fp8,
        }
    }

    pub fn head(quant: Quant) -> WeightFormat {
        match quant {
            Quant::Fp8Block128 => WeightFormat::Bf16,
            q => WeightFormat::of(q),
        }
    }

    pub fn code(self) -> u32 {
        match self {
            WeightFormat::Bf16 => 0,
            WeightFormat::Q8 => 1,
            WeightFormat::Fp8 => 2,
        }
    }

    pub fn bytes(self) -> bool {
        self != WeightFormat::Bf16
    }

    fn gemv(self) -> &'static str {
        match self {
            WeightFormat::Bf16 => "linear_bf16",
            WeightFormat::Q8 => "linear_q8",
            WeightFormat::Fp8 => "linear_fp8",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MmaShape {
    pub tile: Tile,
    pub bk: usize,
    pub stages: usize,
    pub split: bool,
    pub wf: WeightFormat,
}

impl MmaShape {
    pub fn new(tile: Tile, k: usize, wf: WeightFormat, split: bool) -> MmaShape {
        let bk = match wf {
            WeightFormat::Q8 => Q8_GROUP,
            WeightFormat::Fp8 if tile.bm == 16 => 128,
            WeightFormat::Fp8 => env_usize("SPECRT_FP8_BK", 64),
            WeightFormat::Bf16 if tile.bm == 16 && k % 128 == 0 => 128,
            WeightFormat::Bf16 => env_usize("SPECRT_BK", 32),
        };
        let stages = if tile.bm == 16 { 2 } else { env_usize("SPECRT_STAGES", 2) };
        MmaShape { tile, bk, stages, split, wf }
    }

    pub fn smem_bytes(&self) -> u32 {
        let planes = if self.split { 2 } else { 1 };
        let a = self.tile.bm * self.bk * 2 * planes;
        let wrow = if self.wf.bytes() { self.bk + 16 } else { self.bk * 2 };
        ((a + self.tile.bn * wrow) * self.stages) as u32
    }
}

pub const KERNELS: &[&str] = &[
    "embed",
    "rmsnorm",
    "qkv_proj",
    "qk_norm_rope",
    "attention",
    "attention_split",
    "attention_reduce",
    "o_proj",
    "gate_up_proj",
    "swiglu",
    "down_proj",
    "lm_head",
    "argmax_advance",
    "qkv_gemm",
    "o_gemm",
    "gate_up_gemm",
    "down_gemm",
    "lm_head_gemm",
    "qkv_proj_norm",
    "gate_up_act",
    "lm_head_norm",
    "qkv_mma",
    "o_mma",
    "gate_up_mma",
    "down_mma",
    "lm_head_mma",
    "qkv_mma_big",
    "o_mma_big",
    "gate_up_mma_big",
    "down_mma_big",
    "lm_head_mma_big",
    "rms_split",
    "qkv_mma_tiny",
    "o_mma_tiny",
    "gate_up_mma_tiny",
    "down_mma_tiny",
    "lm_head_mma_tiny",
    "kv_norm_rope",
    "argmax_rows",
    "bump",
    "qkv_mma_narrow",
    "o_mma_narrow",
    "gate_up_mma_narrow",
    "down_mma_narrow",
    "lm_head_mma_narrow",
    "flash_attention",
    "flash_mma",
    "batch_decode_attention",
    "batch_decode_combine",
    "copy_rows",
    "split_rows",
];

pub const GDN_KERNELS: &[&str] = &[
    "gdn_conv_step",
    "gdn_conv_prefill",
    "gdn_conv_keep",
    "gdn_step",
    "gdn_qk_prep",
    "gdn_gates",
    "gdn_chunk_prep",
    "gdn_chunk_scan",
    "gdn_gated_norm",
    "lin_in_norm",
    "lin_ba_norm",
    "lin_out_proj",
    "lin_in_mma",
    "lin_out_mma",
    "lin_in_mma_big",
    "lin_out_mma_big",
    "lin_in_mma_narrow",
    "lin_out_mma_narrow",
    "lin_in_mma_tiny",
    "lin_out_mma_tiny",
    "lin_ba_mma",
    "lin_ba_mma_big",
    "lin_ba_mma_narrow",
    "lin_ba_mma_tiny",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub source: String,
    pub kernels: Vec<&'static str>,
    pub smem: BTreeMap<&'static str, u32>,
}

pub fn render(template: &str, vars: &BTreeMap<&str, String>) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            bail!("unterminated slot");
        };
        let key = &after[..end];
        match vars.get(key) {
            Some(v) => out.push_str(v),
            None => bail!("template slot {key} has no value"),
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

fn float_literal(v: f64) -> String {
    format!("{v:e}f")
}

fn mma_kernel_name(linear: &str, suffix: &str) -> &'static str {
    let name = format!("{linear}_mma{suffix}");
    KERNELS
        .iter()
        .chain(GDN_KERNELS)
        .copied()
        .find(|k| *k == name)
        .expect("mma kernel listed in KERNELS")
}

fn linear_dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str, bool) {
    let d = &spec.dims;
    let la = spec.linear_attention.as_ref();
    match linear {
        "qkv" => (spec.qkv_rows(), d.hidden, "EPI_STORE", spec.attention.qkv_bias),
        "o" => (d.hidden, spec.q_dim(), "EPI_RESIDUAL", false),
        "gate_up" => (2 * d.intermediate, d.hidden, "EPI_ACT", false),
        "down" => (d.hidden, d.intermediate, "EPI_RESIDUAL", false),
        "lin_in" => (la.map_or(0, |l| l.in_proj_rows()), d.hidden, "EPI_STORE", false),
        "lin_ba" => (la.map_or(0, |l| l.ba_rows()), d.hidden, "EPI_STORE", false),
        "lin_out" => (d.hidden, la.map_or(0, |l| l.value_dim()), "EPI_RESIDUAL", false),
        _ => (d.vocab, d.hidden, "EPI_STORE", false),
    }
}

fn mma_kernels(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> String {
    let unsplit = std::env::var("SPECRT_UNSPLIT").unwrap_or_default();
    let mut out = String::new();
    let gdn: &[&str] = if spec.linear_attention.is_some() { &GDN_LINEARS } else { &[] };
    for &linear in MMA_LINEARS.iter().chain(gdn) {
        let wf = match linear {
            "lm_head" => WeightFormat::head(spec.quant),
            "lin_ba" => WeightFormat::Bf16,
            _ => WeightFormat::of(spec.quant),
        };
        let split = wf != WeightFormat::Q8 && !unsplit.split(',').any(|n| n == linear);
        let (n, k, epi, bias) = linear_dims(spec, linear);
        for suffix in MMA_VARIANTS {
            let shape = MmaShape::new(Tile::variant(suffix), k, wf, split);
            let t = shape.tile;
            let name = mma_kernel_name(linear, suffix);
            smem.insert(name, shape.smem_bytes());
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_tc<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {bk}, {stages}, {epi}, {bias}, {wf}, {split}>(w, s, b, xh, xl, out, oh, ol, tokens);\n}}\n\n",
                threads = t.threads(),
                bm = t.bm,
                bn = t.bn,
                wm = t.wm,
                wn = t.wn,
                bk = shape.bk,
                stages = shape.stages,
                wf = wf.code(),
            ));
        }
    }
    out
}

fn fp8_act_kernels(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::new();
    let la = spec.linear_attention.is_some();
    for &linear in FP8_ACT_LINEARS.iter().filter(|l| la || !l.starts_with("lin_")) {
        let (n, k, epi, bias) = linear_dims(spec, linear);
        for suffix in FP8_ACT_VARIANTS {
            let t = Tile::variant(suffix);
            let name = format!("{linear}_f8{suffix}");
            let name = FP8_ACT_KERNELS.iter().copied().find(|k| *k == name).expect("listed in FP8_ACT_KERNELS");
            smem.insert(name, fp8_act_smem(t));
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const float* s, const float* b, const unsigned char* xq, const float* xs, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_fp8<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {FP8_ACT_STAGES}, {epi}, {bias}>(w, s, b, xq, xs, out, oh, ol, tokens);\n}}\n\n",
                threads = t.threads(),
                bm = t.bm,
                bn = t.bn,
                wm = t.wm,
                wn = t.wn,
            ));
        }
    }
    out
}

impl Module {
    pub fn generate(spec: &Spec, max_ctx: usize) -> Result<Self> {
        let a = &spec.attention;
        let d = &spec.dims;
        let wf = WeightFormat::of(spec.quant);
        let head_wf = WeightFormat::head(spec.quant);
        let scaling = spec.rope.scaling.as_ref();
        let rope_kind = match scaling.map(|s| s.kind) {
            None => 0,
            Some(RopeScalingKind::Linear) => 1,
            Some(RopeScalingKind::Llama3) => 2,
        };
        let opt = |f: fn(&crate::spec::RopeScaling) -> Option<f64>| {
            float_literal(scaling.and_then(f).unwrap_or(1.0))
        };
        let act = match spec.mlp.activation {
            Activation::Swiglu => 0,
            Activation::Geglu => 1,
        };
        let mut smem = BTreeMap::new();
        let mma = mma_kernels(spec, &mut smem);
        let fp8 = spec.quant == Quant::Fp8Block128;
        let (flash_rows, flash_keys) = (flash_rows(a.head_dim), flash_keys(a.head_dim));
        let kv_bytes = FLASH_STAGES * 4 * flash_keys * a.head_dim * 2;
        let q_bytes = flash_rows * a.head_dim * 4;
        let flash_smem = kv_bytes.max(q_bytes);
        anyhow::ensure!(flash_smem <= FLASH_SMEM_LIMIT, "flash_mma needs {flash_smem} bytes of shared memory");
        smem.insert("flash_mma", flash_smem as u32);
        let la = spec.linear_attention.as_ref();
        let la_dim = |f: fn(&crate::spec::LinearAttention) -> usize| la.map_or("0".into(), |l| f(l).to_string());
        if let Some(l) = la {
            let floats = LA_CHUNK * (l.key_head_dim + 1) + LA_CHUNK * (LA_CHUNK + 1) + 2 * LA_CHUNK;
            smem.insert("gdn_chunk_prep", (floats * 4) as u32);
        }
        let vars: BTreeMap<&str, String> = [
            ("HIDDEN", d.hidden.to_string()),
            ("INTER", d.intermediate.to_string()),
            ("VOCAB", d.vocab.to_string()),
            ("HEADS", a.heads.to_string()),
            ("KV_HEADS", a.kv_heads.to_string()),
            ("HEAD_DIM", a.head_dim.to_string()),
            ("GROUP", (a.heads / a.kv_heads).to_string()),
            ("QK_NORM", a.qk_norm.to_string()),
            ("QKV_BIAS", a.qkv_bias.to_string()),
            ("EPS", float_literal(spec.norm.eps)),
            ("THETA", float_literal(spec.rope.theta)),
            ("ROPE_KIND", rope_kind.to_string()),
            ("ROPE_FACTOR", float_literal(scaling.map_or(1.0, |s| s.factor))),
            ("ROPE_LOW_FREQ", opt(|s| s.low_freq_factor)),
            ("ROPE_HIGH_FREQ", opt(|s| s.high_freq_factor)),
            ("ROPE_ORIG_POS", opt(|s| s.original_max_positions)),
            ("ACT_KIND", act.to_string()),
            ("MAX_CTX", max_ctx.to_string()),
            ("PAGE", PAGE.to_string()),
            ("Q8_GROUP", Q8_GROUP.to_string()),
            ("FP8_BLOCK", FP8_BLOCK.to_string()),
            ("BD_SPLITS", BATCH_DECODE_SPLITS.to_string()),
            ("Q_DIM", spec.q_dim().to_string()),
            ("QKV_N", spec.qkv_rows().to_string()),
            ("QKV_ROW", spec.qkv_rows().to_string()),
            ("OUT_GATE", a.output_gate.to_string()),
            ("ROT", spec.rotary_dim().to_string()),
            ("LA_KH", la_dim(|l| l.key_heads)),
            ("LA_VH", la_dim(|l| l.value_heads)),
            ("LA_D", la_dim(|l| l.key_head_dim)),
            ("LA_CONV_K", la_dim(|l| l.conv_kernel)),
            ("LA_CHUNK", LA_CHUNK.to_string()),
            ("LA_KSPLIT", LA_KSPLIT.to_string()),
            ("LA_COLS", LA_COLS.to_string()),
            ("GATE_UP_N", (2 * d.intermediate).to_string()),
            ("LINEAR", wf.gemv().to_string()),
            ("HEAD_LINEAR", head_wf.gemv().to_string()),
            ("WF", wf.code().to_string()),
            ("HEAD_WF", head_wf.code().to_string()),
            ("MMA_KERNELS", mma),
            ("FM_STAGES", FLASH_STAGES.to_string()),
            ("FM_BK", flash_keys.to_string()),
            ("FM_WARPS", (flash_rows / 16).to_string()),
            ("FM_SMEM", kv_bytes.max(q_bytes).to_string()),
        ]
        .into_iter()
        .collect();
        let mut source =
            format!("{}\n{}", render(OPS_TEMPLATE, &vars)?, render(MODEL_TEMPLATE, &vars)?);
        let mut kernels = KERNELS.to_vec();
        if la.is_some() {
            source.push('\n');
            source.push_str(&render(GDN_TEMPLATE, &vars)?);
            kernels.extend_from_slice(GDN_KERNELS);
        }
        if fp8 {
            source.push('\n');
            source.push_str(&fp8_act_kernels(spec, &mut smem));
            kernels.push("quant_rows_fp8");
            kernels.extend(smem.keys().filter(|k| k.contains("_f8")).copied().collect::<Vec<_>>());
        }
        Ok(Module { source, kernels, smem })
    }
}
