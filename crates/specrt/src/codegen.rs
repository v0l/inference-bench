use crate::spec::{Activation, ActQuant, Quant, RopeScalingKind, RuntimeOptions, Spec};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub const OPS_TEMPLATE: &str = include_str!("../templates/ops.cu");
pub const MODEL_TEMPLATE: &str = include_str!("../templates/model.cu");
pub const GDN_TEMPLATE: &str = include_str!("../templates/gdn.cu");
pub const FP4_TEMPLATE: &str = include_str!("../templates/fp4.cu");
pub const FP8_BATCH_TEMPLATE: &str = include_str!("../templates/fp8_batch.cu");
pub const FP4_ACT_TEMPLATE: &str = include_str!("../templates/fp4_act.cu");
pub const FP4_ACT_KERNELS: &[&str] = &[
    "quant_rows_fp4",
    "gate_up_f4a",
    "gate_up_f4a_big",
    "gate_up_f4a_narrow",
    "down_f4a",
    "down_f4a_big",
    "down_f4a_narrow",
];
const F4A_ROW: usize = 80;

pub fn fp4_act_enabled(spec: &Spec) -> bool {
    spec.mlp.act_quant == ActQuant::Fp4
}

pub fn fp8_act_enabled(spec: &Spec) -> bool {
    spec.act_quant == ActQuant::Fp8
}

pub fn f4a_smem(tile: Tile) -> u32 {
    (FP4_STAGES * (tile.bm + tile.bn) * F4A_ROW) as u32
}

fn fp4_act_kernels(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::from(FP4_ACT_TEMPLATE);
    for linear in ["gate_up", "down"] {
        let (n, k, epi, _) = linear_dims(spec, linear);
        for suffix in FP8_ACT_VARIANTS {
            let t = Tile::variant(suffix, &spec.runtime);
            let name = format!("{linear}_f4a{suffix}");
            let name = FP4_ACT_KERNELS.iter().copied().find(|k| *k == name).expect("listed in FP4_ACT_KERNELS");
            smem.insert(name, f4a_smem(t));
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const unsigned char* s, const unsigned char* xq, const unsigned char* xs, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_f4a<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {FP4_STAGES}, {epi}>(w, s, xq, xs, out, oh, ol, tokens);\n}}\n\n",
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
pub const FP4_KERNELS: &[&str] = &[
    "gate_up_act_f4",
    "down_proj_f4",
    "gate_up_f4w",
    "gate_up_f4w_big",
    "gate_up_f4w_narrow",
    "gate_up_f4w_tiny",
    "down_f4w",
    "down_f4w_big",
    "down_f4w_narrow",
    "down_f4w_tiny",
    "gate_up_f4b8",
    "gate_up_f4b16",
    "gate_up_f4b_reduce",
    "down_f4b8",
    "down_f4b16",
    "down_f4b_reduce",
];
const FP4_STAGES: usize = 3;
pub const FP4_BATCH_TEMPLATE: &str = include_str!("../templates/fp4_batch.cu");
pub const F4B_MAX_TOKENS: usize = 16;
pub const F4B_WARPS: usize = 4;
pub const F4B_MJ: usize = 2;
pub const F4B_ROWS: usize = F4B_WARPS * F4B_MJ * 16;
pub const F4B_KC: usize = 128;
const F4B_STAGES: usize = 3;
pub const F4B_REDUCE_BLOCK: usize = 256;
pub const F4B_BLOCKS_PER_SM: usize = 4;

pub fn f4b_tokens(tokens: usize) -> usize {
    if tokens <= 8 { 8 } else { 16 }
}

pub fn f4b_smem(tokens: usize) -> u32 {
    (F4B_STAGES * f4b_tokens(tokens) * F4B_KC * 2 * 2) as u32
}

pub fn f4b_splits(n: usize, k: usize, sms: usize) -> usize {
    let blocks = n.div_ceil(F4B_ROWS);
    let most = (k / F4B_KC / 4).max(1);
    let wave = F4B_BLOCKS_PER_SM * sms;
    let fill = |s: usize| (blocks * s) as f64 / ((blocks * s).div_ceil(wave) * wave) as f64;
    (1..=most).find(|&s| blocks * s >= wave && fill(s) >= 0.85).unwrap_or(most)
}

fn fp4_batch_kernels(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> Result<String> {
    let mut kernels = String::new();
    for linear in ["gate_up", "down"] {
        let (n, k, epi, _) = linear_dims(spec, linear);
        let find = |name: String| FP4_KERNELS.iter().copied().find(|x| *x == name).expect("listed in FP4_KERNELS");
        for tokens in [8, 16] {
            let name = find(format!("{linear}_f4b{tokens}"));
            smem.insert(name, f4b_smem(tokens));
            kernels.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}, {F4B_BLOCKS_PER_SM}) {name}(const void* w, const unsigned char* s, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {{\n    gemv_f4b<{n}, {k}, {epi}, {nt}>(w, s, xh, xl, out, oh, ol, part, tokens, splits);\n}}\n\n",
                threads = F4B_WARPS * 32,
                nt = tokens / 8,
            ));
        }
        let reduce = find(format!("{linear}_f4b_reduce"));
        kernels.push_str(&format!(
            "extern \"C\" __global__ void {reduce}(const float* part, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {{\n    f4b_reduce<{n}, {epi}>(part, out, oh, ol, tokens, splits);\n}}\n\n"
        ));
    }
    let vars: BTreeMap<&str, String> = [
        ("F4B_WARPS", F4B_WARPS.to_string()),
        ("F4B_MJ", F4B_MJ.to_string()),
        ("F4B_KC", F4B_KC.to_string()),
        ("F4B_STAGES", F4B_STAGES.to_string()),
        ("F4B_KERNELS", kernels),
    ]
    .into_iter()
    .collect();
    render(FP4_BATCH_TEMPLATE, &vars)
}
pub const F8B_LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
pub const F8B_KERNELS: &[&str] = &[
    "qkv_f8b8",
    "qkv_f8b16",
    "qkv_f8b_reduce",
    "o_f8b8",
    "o_f8b16",
    "o_f8b_reduce",
    "gate_up_f8b8",
    "gate_up_f8b16",
    "gate_up_f8b_reduce",
    "down_f8b8",
    "down_f8b16",
    "down_f8b_reduce",
    "lin_in_f8b8",
    "lin_in_f8b16",
    "lin_in_f8b_reduce",
    "lin_out_f8b8",
    "lin_out_f8b16",
    "lin_out_f8b_reduce",
];
pub const F8B_MAX_TOKENS: usize = 16;
pub const F8B_WARPS: usize = 4;
pub const F8B_MJ: usize = 2;
pub const F8B_ROWS: usize = F8B_WARPS * F8B_MJ * 16;
pub const F8B_KC: usize = 128;
const F8B_STAGES: usize = 3;
pub const F8B_REDUCE_BLOCK: usize = 256;
pub const F8B_BLOCKS_PER_SM: usize = 4;

pub fn f8b_tokens(tokens: usize) -> usize {
    if tokens <= 8 { 8 } else { 16 }
}

pub fn f8b_smem(tokens: usize) -> u32 {
    (F8B_STAGES * f8b_tokens(tokens) * F8B_KC * 2 * 2) as u32
}

pub fn f8b_splits(n: usize, k: usize, sms: usize) -> usize {
    let blocks = n.div_ceil(F8B_ROWS);
    let most = (k / F8B_KC / 4).max(1);
    let wave = F8B_BLOCKS_PER_SM * sms;
    let fill = |s: usize| (blocks * s) as f64 / ((blocks * s).div_ceil(wave) * wave) as f64;
    (1..=most).find(|&s| blocks * s >= wave && fill(s) >= 0.85).unwrap_or(most)
}

fn fp8_batch_kernels(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> Result<String> {
    let mut kernels = String::new();
    let la = spec.linear_attention.is_some();
    for &linear in F8B_LINEARS.iter().filter(|l| la || !l.starts_with("lin_")) {
        let (n, k, epi, bias) = linear_dims(spec, linear);
        let find = |name: String| F8B_KERNELS.iter().copied().find(|x| *x == name).expect("listed in F8B_KERNELS");
        for tokens in [8, 16] {
            let name = find(format!("{linear}_f8b{tokens}"));
            smem.insert(name, f8b_smem(tokens));
            kernels.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}, {F8B_BLOCKS_PER_SM}) {name}(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {{\n    gemv_f8b<{n}, {k}, {epi}, {bias}, {nt}>(w, s, b, xh, xl, out, oh, ol, part, tokens, splits);\n}}\n\n",
                threads = F8B_WARPS * 32,
                nt = tokens / 8,
            ));
        }
        let reduce = find(format!("{linear}_f8b_reduce"));
        kernels.push_str(&format!(
            "extern \"C\" __global__ void {reduce}(const float* part, const float* b, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {{\n    f8b_reduce<{n}, {epi}, {bias}>(part, b, out, oh, ol, tokens, splits);\n}}\n\n"
        ));
    }
    let vars: BTreeMap<&str, String> = [
        ("F8B_WARPS", F8B_WARPS.to_string()),
        ("F8B_MJ", F8B_MJ.to_string()),
        ("F8B_KC", F8B_KC.to_string()),
        ("F8B_STAGES", F8B_STAGES.to_string()),
        ("F8B_KERNELS", kernels),
    ]
    .into_iter()
    .collect();
    render(FP8_BATCH_TEMPLATE, &vars)
}
const F4W_BK: usize = 64;
const F4W_ROW: usize = 48;

pub fn f4w_smem(tile: Tile) -> u32 {
    (FP4_STAGES * (tile.bm * F4W_BK * 4 + tile.bn * F4W_ROW)) as u32
}

fn fp4_kernels(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::new();
    for linear in ["gate_up", "down"] {
        let (n, k, epi, _) = linear_dims(spec, linear);
        for suffix in MMA_VARIANTS {
            let t = Tile::variant(suffix, &spec.runtime);
            let name = format!("{linear}_f4w{suffix}");
            let name = FP4_KERNELS.iter().copied().find(|k| *k == name).expect("listed in FP4_KERNELS");
            smem.insert(name, f4w_smem(t));
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const unsigned char* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_f4w<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {FP4_STAGES}, {epi}>(w, s, xh, xl, out, oh, ol, tokens);\n}}\n\n",
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
pub const FP4_HEAD_TEMPLATE: &str = include_str!("../templates/fp4_head.cu");
pub const FP4_HEAD_KERNELS: &[&str] =
    &["lm_head_norm_f4", "lm_head_f4w", "lm_head_f4w_big", "lm_head_f4w_narrow", "lm_head_f4w_tiny"];

pub fn fp4_head_rows(vocab: usize) -> usize {
    if vocab.is_multiple_of(2) { 2 } else { 1 }
}

fn fp4_head_source(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> Result<String> {
    let (n, k) = (spec.dims.vocab, spec.dims.hidden);
    let mut gemms = String::new();
    for suffix in MMA_VARIANTS {
        let t = Tile::variant(suffix, &spec.runtime);
        let name = format!("lm_head_f4w{suffix}");
        let name = FP4_HEAD_KERNELS.iter().copied().find(|k| *k == name).expect("listed in FP4_HEAD_KERNELS");
        smem.insert(name, f4w_smem(t));
        gemms.push_str(&format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const unsigned char* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_f4w<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {FP4_STAGES}, EPI_STORE>(w, s, xh, xl, out, oh, ol, tokens);\n}}\n\n",
            threads = t.threads(),
            bm = t.bm,
            bn = t.bn,
            wm = t.wm,
            wn = t.wn,
        ));
    }
    let vars: BTreeMap<&str, String> =
        [("HEAD_F4_ROWS", fp4_head_rows(n).to_string()), ("HEAD_F4W_KERNELS", gemms)].into_iter().collect();
    render(FP4_HEAD_TEMPLATE, &vars)
}

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

    pub fn big(opts: &RuntimeOptions) -> Tile {
        match opts.big_tile {
            Some([bm, bn, wm, wn]) => Tile { bm, bn, wm, wn },
            None => Tile { bm: 64, bn: 128, wm: 2, wn: 4 },
        }
    }

    pub fn variant(suffix: &str, opts: &RuntimeOptions) -> Tile {
        match suffix {
            "_big" => Tile::big(opts),
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
            Quant::Fp8Block128 | Quant::Fp8Channel => WeightFormat::Fp8,
            Quant::Nvfp4G16 => unreachable!("nvfp4 is an MLP format with its own kernels"),
        }
    }

    pub fn head(spec: &Spec) -> WeightFormat {
        match spec.lm_head_quant() {
            Quant::Nvfp4G16 => WeightFormat::Bf16,
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
    "sample_rows",
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
    pub arch_features: bool,
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
    let unsplit = &spec.runtime.unsplit;
    let mut out = String::new();
    let gdn: &[&str] = if spec.linear_attention.is_some() { &GDN_LINEARS } else { &[] };
    for &linear in MMA_LINEARS.iter().chain(gdn) {
        let wf = match linear {
            "lm_head" => WeightFormat::head(spec),
            "lin_ba" => WeightFormat::Bf16,
            _ => WeightFormat::of(spec.quant),
        };
        let split = wf != WeightFormat::Q8 && !unsplit.iter().any(|n| n == linear);
        let (n, k, epi, bias) = linear_dims(spec, linear);
        for suffix in MMA_VARIANTS {
            let shape = MmaShape::new(Tile::variant(suffix, &spec.runtime), k, wf, split);
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
            let t = Tile::variant(suffix, &spec.runtime);
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
        Self::generate_kv(spec, max_ctx, 2)
    }

    pub fn generate_kv(spec: &Spec, max_ctx: usize, kv_planes: usize) -> Result<Self> {
        anyhow::ensure!(kv_planes == 1 || kv_planes == 2, "kv planes must be 1 or 2");
        let a = &spec.attention;
        let d = &spec.dims;
        let wf = WeightFormat::of(spec.quant);
        let head_wf = WeightFormat::head(spec);
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
        let fp8 = spec.quant.is_fp8();
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
            ("ATTN_LAYERS", spec.kv_layers().max(1).to_string()),
            ("KV_PLANES", kv_planes.to_string()),
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
        let mut arch_features = false;
        if la.is_some() {
            source.push('\n');
            source.push_str(&render(GDN_TEMPLATE, &vars)?);
            kernels.extend_from_slice(GDN_KERNELS);
        }
        if la.is_some() && spec.speculative().is_some() {
            source.push('\n');
            source.push_str(&crate::mtp::mtp_source(spec, &mut smem));
            kernels.extend_from_slice(crate::mtp::MTP_KERNELS);
            kernels.extend_from_slice(crate::mtp::MTP_MMA_KERNELS);
        }
        if fp8 {
            source.push('\n');
            source.push_str(&fp8_act_kernels(spec, &mut smem));
            kernels.push("quant_rows_fp8");
            source.push('\n');
            source.push_str(&fp8_batch_kernels(spec, &mut smem)?);
            kernels.extend(F8B_KERNELS.iter().filter(|k| la.is_some() || !k.starts_with("lin_")));
            kernels.extend(smem.keys().filter(|k| k.contains("_f8") && !k.contains("_f8b")).copied().collect::<Vec<_>>());
        }
        let fp4_mlp = spec.mlp.quant == Some(Quant::Nvfp4G16);
        let fp4_head = spec.lm_head_quant() == Quant::Nvfp4G16;
        if fp4_mlp || fp4_head {
            let mut fp4_vars = BTreeMap::new();
            fp4_vars.insert("F4W_KERNELS", if fp4_mlp { fp4_kernels(spec, &mut smem) } else { String::new() });
            let act = fp4_mlp && fp4_act_enabled(spec);
            fp4_vars.insert("F4A_SOURCE", if act { fp4_act_kernels(spec, &mut smem) } else { String::new() });
            source.push('\n');
            source.push_str(&render(FP4_TEMPLATE, &fp4_vars)?);
            if fp4_mlp {
                source.push('\n');
                source.push_str(&fp4_batch_kernels(spec, &mut smem)?);
                kernels.extend_from_slice(FP4_KERNELS);
            }
            if act {
                kernels.extend_from_slice(FP4_ACT_KERNELS);
                arch_features = true;
            }
        }
        if fp4_head {
            source.push('\n');
            source.push_str(&fp4_head_source(spec, &mut smem)?);
            kernels.extend_from_slice(FP4_HEAD_KERNELS);
        }
        Ok(Module { source, kernels, smem, arch_features })
    }
}
