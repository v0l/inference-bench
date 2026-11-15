use crate::spec::{Activation, Quant, Spec};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub use crate::gemm::{
    BATCH_MAX_TOKENS, BATCH_REDUCE_BLOCK, BATCH_VARIANTS, BatchShape, FP4_GEMV_THREADS, FP8_ACT_LINEARS, GDN_LINEARS, GGB_LINEARS,
    GGB_MIN_TOKENS, GemmPath, KernelChoice, MMA_LINEARS, MmaShape, ROWS_MAX_TOKENS, Tile, WeightFormat, WeightKind,
    fp4_act_enabled, fp8_act_enabled, kernel_for, linear_dims, linear_shape,
};

pub const OPS_TEMPLATE: &str = include_str!("../templates/ops.cu");
pub const MODEL_TEMPLATE: &str = include_str!("../templates/model.cu");
pub const GDN_TEMPLATE: &str = include_str!("../templates/gdn.cu");
pub const MOE_TEMPLATE: &str = include_str!("../templates/moe.cu");
pub const MOE_KERNELS: &[&str] = &["moe_route", "moe_up_bf16", "moe_up_fp4", "moe_down_bf16", "moe_down_fp4", "moe_combine", "moe_permute", "moe_group_up_bf16", "moe_group_up_fp4", "moe_group_down_bf16", "moe_group_down_fp4"];

pub const PAGE: usize = 16;
pub const Q8_GROUP: usize = 32;
pub const FP8_BLOCK: usize = 128;
pub const DECODE_MAX_SPLITS: usize = 64;
pub const DECODE_CHUNK: usize = 256;
pub const DECODE_KEYS: usize = 16;
pub const DECODE_STAGES: usize = 2;
pub const DECODE_SPLIT_PRECISION: bool = true;
pub const DECODE_MAX_QLEN: usize = crate::spec::MAX_DRAFT_TOKENS + 1;
const FLASH_SMEM_LIMIT: usize = 99 * 1024;

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

pub fn decode_smem(head_dim: usize, kv_planes: usize) -> usize {
    DECODE_STAGES * 2 * kv_planes * DECODE_KEYS * head_dim * 2
}

pub fn decode_warps(group: usize, qlen: usize) -> usize {
    (group * qlen).div_ceil(16)
}

pub fn flash_rows(_head_dim: usize) -> usize {
    128
}

pub fn flash_keys(head_dim: usize, kv_planes: usize) -> usize {
    if head_dim > 128 && kv_planes == 2 { 16 } else { 32 }
}

pub fn flash_stages(head_dim: usize, kv_planes: usize) -> usize {
    if head_dim > 128 && kv_planes == 1 { 3 } else { 2 }
}

pub const LA_CHUNK: usize = 64;
pub const LA_KSPLIT: usize = 4;
pub const LA_COLS: usize = 16;


pub const KERNELS: &[&str] = &[
    "embed",
    "splice_rows",
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
    "rms_split",
    "kv_norm_rope",
    "argmax_rows",
    "bump",
    "sample_rows",
    "apply_mask",
    "flash_attention",
    "flash_mma",
    "batch_decode_attention",
    "batch_decode_combine",
    "copy_rows",
    "split_rows",
    "gg_dequant_rows",
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
    "lin_ba_rows",
];


#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub source: String,
    pub kernels: Vec<&'static str>,
    pub smem: BTreeMap<&'static str, u32>,
    pub arch_features: bool,
    pub kir_hash: String,
}

pub fn kir_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let act = match spec.mlp.activation {
        Activation::Swiglu => kir::Activation::Silu,
        Activation::Geglu => kir::Activation::GeluTanh,
    };
    let hidden = spec.dims.hidden as u32;
    let inter = spec.dims.intermediate as u32;
    let norm = kir::Kernel::rmsnorm(target, hidden, spec.norm.eps as f32, 1024).map_err(anyhow::Error::msg)?;
    let gate = kir::Kernel::gated_activation(target, "swiglu", inter, act, 256).map_err(anyhow::Error::msg)?;
    Ok(vec![norm, gate])
}

pub fn cuda_target(ctx: &cudarc::driver::CudaContext) -> Result<kir::Target> {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let (major, minor) = ctx.compute_capability()?;
    let sms = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
    let smem = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as u32;
    Ok(kir::Target::cuda(major as u32, minor as u32, sms, smem))
}

pub fn default_target() -> kir::Target {
    kir::Target::sm_120()
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

pub fn ggb_scratch_len(spec: &Spec) -> usize {
    GGB_LINEARS.iter().map(|l| { let (n, k, _, _) = linear_dims(spec, l); n * k }).max().unwrap_or(0)
}

impl Module {
    pub fn generate(spec: &Spec, max_ctx: usize) -> Result<Self> {
        Self::generate_kv(spec, max_ctx, 2, &default_target())
    }

    pub fn generate_kv(spec: &Spec, max_ctx: usize, kv_planes: usize, target: &kir::Target) -> Result<Self> {
        anyhow::ensure!(kv_planes == 1 || kv_planes == 2, "kv planes must be 1 or 2");
        anyhow::ensure!(kv_planes == 1 || !spec.runtime.kv_dtype.is_fp8(), "fp8 KV needs a single plane");
        let a = &spec.attention;
        let d = &spec.dims;
        let wf = crate::gemm::weight_format(spec.quant);
        let head_wf = crate::gemm::head_format(spec);
        let rope = spec.rope_constants();
        let act = match spec.mlp.activation {
            Activation::Swiglu => 0,
            Activation::Geglu => 1,
        };
        let mut smem = BTreeMap::new();
        let (flash_rows, flash_keys) = (flash_rows(a.head_dim), flash_keys(a.head_dim, kv_planes));
        let flash_stages = flash_stages(a.head_dim, kv_planes);
        let kv_bytes = flash_stages * 2 * kv_planes * flash_keys * a.head_dim * 2;
        let q_bytes = flash_rows * a.head_dim * 2;
        let flash_smem = kv_bytes.max(q_bytes);
        anyhow::ensure!(flash_smem <= FLASH_SMEM_LIMIT, "flash_mma needs {flash_smem} bytes of shared memory");
        smem.insert("flash_mma", flash_smem as u32);
        let decode_smem = decode_smem(a.head_dim, kv_planes);
        anyhow::ensure!(decode_smem <= FLASH_SMEM_LIMIT, "batch_decode_attention needs {decode_smem} bytes of shared memory");
        smem.insert("batch_decode_attention", decode_smem as u32);
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
            ("ROPE_KIND", rope.kind.to_string()),
            ("MROPE", spec.rope.mrope.is_some().to_string()),
            ("MROPE_INTERLEAVED", spec.rope.mrope.as_ref().is_some_and(|m| m.interleaved).to_string()),
            ("MROPE_S0", spec.rope.mrope.as_ref().map_or(0, |m| m.section[0]).to_string()),
            ("MROPE_S1", spec.rope.mrope.as_ref().map_or(0, |m| m.section[1]).to_string()),
            ("MROPE_S2", spec.rope.mrope.as_ref().map_or(0, |m| m.section[2]).to_string()),
            ("ROPE_FACTOR", float_literal(rope.factor)),
            ("ROPE_LOW_FREQ", float_literal(rope.low)),
            ("ROPE_HIGH_FREQ", float_literal(rope.high)),
            ("ROPE_ORIG_POS", float_literal(rope.original)),
            ("ROPE_MSCALE", float_literal(rope.mscale)),
            ("ACT_KIND", act.to_string()),
            ("MAX_CTX", max_ctx.to_string()),
            ("PAGE", PAGE.to_string()),
            ("ATTN_LAYERS", spec.kv_layers().max(1).to_string()),
            ("KV_PLANES", kv_planes.to_string()),
            ("KV_FP8", spec.runtime.kv_dtype.is_fp8().to_string()),
            ("Q8_GROUP", Q8_GROUP.to_string()),
            ("FP8_BLOCK", FP8_BLOCK.to_string()),
            ("BD_SPLITS", decode_splits(max_ctx).to_string()),
            ("DM_BK", DECODE_KEYS.to_string()),
            ("DM_STAGES", DECODE_STAGES.to_string()),
            ("DM_CHUNK", DECODE_CHUNK.to_string()),
            ("DM_SPLIT", DECODE_SPLIT_PRECISION.to_string()),
            ("DM_MAX_QLEN", DECODE_MAX_QLEN.to_string()),
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
            ("WF", wf.decoder().to_string()),
            ("HEAD_WF", head_wf.decoder().to_string()),
            ("FM_STAGES", flash_stages.to_string()),
            ("FM_BK", flash_keys.to_string()),
            ("FM_WARPS", (flash_rows / 16).to_string()),
            ("FM_SMEM", kv_bytes.max(q_bytes).to_string()),
        ]
        .into_iter()
        .collect();
        let kir_kernels = kir_kernels(spec, target)?;
        let gemms = crate::gemm::gemm_kernels(spec, target)?;
        let kir_source = kir::cuda::emit_module(&kir_kernels).map_err(anyhow::Error::msg)?;
        let gemm_source = kir::cuda::emit_gemms(&gemms.kernels).map_err(anyhow::Error::msg)?;
        let kir_hash = kir::fold_hashes([kir::module_hash(&kir_kernels), kir::gemm::instances_hash(&gemms.kernels)]);
        let mut source = format!("{}\n{}\n{}\n{}", render(OPS_TEMPLATE, &vars)?, gemm_source, render(MODEL_TEMPLATE, &vars)?, kir_source);
        let mut kernels = KERNELS.to_vec();
        if la.is_some() {
            source.push('\n');
            source.push_str(&render(GDN_TEMPLATE, &vars)?);
            kernels.extend_from_slice(GDN_KERNELS);
            smem.insert("lin_ba_rows", 0);
        }
        if la.is_some() && spec.speculative().is_some() {
            source.push('\n');
            source.push_str(crate::mtp::MTP_TEMPLATE);
            kernels.extend_from_slice(crate::mtp::MTP_KERNELS);
        }
        if let Some(m) = &spec.moe {
            let vars: BTreeMap<&str, String> = [
                ("MOE_E", m.experts.to_string()),
                ("MOE_K", m.top_k().to_string()),
                ("MOE_PAIRS", m.pairs().to_string()),
                ("MOE_INTER", m.expert_intermediate.to_string()),
                ("MOE_SHARED", (m.shared_intermediate > 0).to_string()),
                ("MOE_NORM_TOPK", m.norm_topk.to_string()),
                ("MOE_SHARED_GATE", m.shared_gate.to_string()),
                ("MOE_SIGMOID", (m.router == crate::spec::RouterAct::Sigmoid).to_string()),
            ]
            .into_iter()
            .collect();
            source.push('\n');
            source.push_str(&render(MOE_TEMPLATE, &vars)?);
            kernels.extend_from_slice(MOE_KERNELS);
        }
        if spec.quant == Quant::Gguf {
            kernels.push("gg_dequant_bf16");
        }
        kernels.extend(gemms.names());
        smem.extend(gemms.smem());
        let arch_features = kir::cuda::needs_arch_features(&gemms.kernels);
        Ok(Module { source, kernels, smem, arch_features, kir_hash })
    }
}

pub fn live_target() -> Result<kir::Target> {
    let ctx = cudarc::driver::CudaContext::new(0).map_err(|e| anyhow::anyhow!("no CUDA device for the live target: {e:?}"))?;
    cuda_target(&ctx)
}

pub fn kernels_report(spec: &Spec, target: &kir::Target) -> Result<String> {
    let kernels = kir_kernels(spec, target)?;
    let gemms = crate::gemm::gemm_kernels(spec, target)?;
    let hash = kir::fold_hashes([kir::module_hash(&kernels), kir::gemm::instances_hash(&gemms.kernels)]);
    let mut out = String::new();
    out.push_str(&format!("spec {}\n{}\nmodule hash {hash}\n", spec.name, target.describe()));
    for k in &kernels {
        out.push_str(&format!("\n== {} ({})\n{}\n\n", k.name, k.hash(), k.describe()));
        let source = match target.language {
            kir::Language::Cuda => kir::cuda::emit_module(std::slice::from_ref(k)),
            kir::Language::Wgsl => kir::wgsl::emit_kernel(k),
        };
        out.push_str(&source.map_err(anyhow::Error::msg)?);
    }
    let mut families: BTreeMap<&str, Vec<&kir::gemm::KernelInstance>> = BTreeMap::new();
    for k in &gemms.kernels {
        families.entry(k.kind.block()).or_default().push(k);
    }
    for (block, ks) in families {
        out.push_str(&format!("\n== family {block} ({} kernels)\n", ks.len()));
        for k in ks {
            out.push_str(&format!("{} ({}) threads={} smem={} {:?}\n", k.name, k.hash(), k.threads, k.smem_bytes, k.kind));
        }
        let first = gemms.kernels.iter().find(|k| k.kind.block() == block).expect("family has a kernel");
        let source = match target.language {
            kir::Language::Cuda => kir::cuda::emit_gemm(first).map_err(|e| e.to_string()),
            kir::Language::Wgsl => kir::wgsl::emit_gemm(first),
        };
        out.push_str(&source.map_err(anyhow::Error::msg)?);
    }
    for r in &gemms.refused {
        out.push_str(&format!("\n== refused {r}\n"));
    }
    Ok(out)
}
