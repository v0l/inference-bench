use crate::spec::{Activation, Quant, Spec};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub use crate::gemm::{
    BATCH_MAX_TOKENS, BATCH_REDUCE_BLOCK, BATCH_VARIANTS, BatchShape, FP4_GEMV_THREADS, FP8_ACT_LINEARS, GDN_LINEARS, GGB_LINEARS,
    GGB_MIN_TOKENS, GemmPath, KernelChoice, MMA_LINEARS, MmaShape, ROWS_MAX_TOKENS, Tile, WeightFormat, WeightKind,
    fp4_act_enabled, fp8_act_enabled, select, linear_dims, linear_shape,
};

pub const OPS_TEMPLATE: &str = include_str!("../templates/ops.cu");
pub const MODEL_TEMPLATE: &str = include_str!("../templates/model.cu");
pub const GDN_TEMPLATE: &str = include_str!("../templates/gdn.cu");
pub const MOE_TEMPLATE: &str = include_str!("../templates/moe.cu");
pub const MOE_KERNELS: &[&str] = &["moe_route_logits", "moe_route", "moe_up_bf16", "moe_up_fp4", "moe_down_bf16", "moe_down_fp4", "moe_combine", "moe_permute", "moe_group_up_bf16", "moe_group_up_fp4", "moe_group_down_bf16", "moe_group_down_fp4", "moe_up_fp8", "moe_down_fp8", "moe_group_up_fp8", "moe_group_down_fp8"];
pub const MOE_MMA_TEMPLATE: &str = include_str!("../templates/moe_mma.cu");
pub const MOE_MMA_KERNELS: &[&str] = &["moe_quant_f32", "moe_tiles", "moe_mma_up", "moe_mma_up_s", "moe_mma_down", "moe_mma_down_s"];
pub const MOE_MMA_BM: usize = 64;
pub const MOE_MMA_BM_SMALL: usize = 16;
pub const MOE_MMA_STAGES: usize = 3;
pub const MOE_MMA_BN: usize = 64;
const MOE_MMA_ROW: usize = 72;

pub fn moe_mma_enabled(spec: &Spec, target: &kir::Target) -> bool {
    spec.moe.is_some()
        && spec.mlp.quant == Some(Quant::Nvfp4G16)
        && spec.mlp.act_quant == crate::spec::ActQuant::Fp4
        && target.compute_capability.0 >= 12
}

pub fn moe_mma_smem(rows_per_tile: usize, weights: usize) -> u32 {
    (MOE_MMA_STAGES * (rows_per_tile + weights * MOE_MMA_BN) * MOE_MMA_ROW) as u32
}

pub use kir::attention::PAGE;
pub const Q8_GROUP: usize = 32;
pub const FP8_BLOCK: usize = 128;
pub use kir::attention::{DECODE_CHUNK, DECODE_KEYS, DECODE_MAX_QLEN, DECODE_MAX_SPLITS, DECODE_STAGES, decode_chunk, decode_splits, flash_keys, flash_stages};

pub fn decode_warps(group: usize, qlen: usize) -> usize {
    (group * qlen).div_ceil(16)
}

pub fn flash_rows(_head_dim: usize) -> usize {
    kir::attention::FLASH_ROWS
}

pub fn attention_shape(spec: &Spec, kv_planes: usize) -> kir::attention::AttentionShape {
    let a = &spec.attention;
    let rope = spec.rope_constants();
    kir::attention::AttentionShape {
        heads: a.heads,
        kv_heads: a.kv_heads,
        head_dim: a.head_dim,
        qkv_row: spec.qkv_rows(),
        output_gate: a.output_gate,
        qk_norm: a.qk_norm,
        eps: spec.norm.eps,
        rope: kir::attention::RopeShape {
            rotary_dim: spec.rotary_dim(),
            theta: spec.rope.theta,
            kind: rope.kind as u32,
            factor: rope.factor,
            low: rope.low,
            high: rope.high,
            original: rope.original,
            mscale: rope.mscale,
            mrope: spec.rope.mrope.as_ref().map(|m| kir::attention::MropeShape { section: m.section, interleaved: m.interleaved }),
        },
        kv_layers: spec.kv_layers(),
        kv_planes,
        kv_fp8: spec.runtime.kv_dtype.is_fp8(),
        kv_f16_layers: spec.kv_f16_layers(),
        query_scale: spec.attention.query_scalar.map_or(1.0 / (a.head_dim as f64).sqrt(), |q| 1.0 / q.sqrt()),
        softcap: spec.attention.softcap.unwrap_or(0.0),
        bidirectional: spec.vision.as_ref().is_some_and(|v| v.config.bidirectional),
        local_theta: spec.rope.local_theta.unwrap_or(spec.rope.theta),
        local_rope: (0..spec.layers.count)
            .filter(|&i| !spec.block(i).is_linear())
            .map(|i| spec.block(i) == crate::spec::Block::Sliding && spec.rope.local_theta.is_some())
            .chain(std::iter::repeat_n(false, spec.speculative().map_or(0, |m| m.layers) + 1))
            .collect(),
    }
}

pub fn attention_kernels(spec: &Spec, kv_planes: usize, target: &kir::Target) -> Result<Vec<kir::attention::AttentionKernel>> {
    kir::attention::AttentionKernel::set(target, attention_shape(spec, kv_planes)).map_err(anyhow::Error::msg)
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
    let mut out = vec![norm, gate];
    let err = anyhow::Error::msg;
    if spec.norm.sandwich {
        out.push(kir::Kernel::norm_add(target, hidden, spec.norm.eps as f32, 0.0, 1024).map_err(err)?);
        out.push(kir::Kernel::pointwise(target, "clear_branch", hidden, kir::Pointwise::Fill(0.0), 256).map_err(err)?);
    }
    if let Some(cap) = spec.dims.logit_softcap {
        out.push(kir::Kernel::pointwise(target, "logit_softcap", spec.dims.vocab as u32, kir::Pointwise::SoftCap(cap as f32), 256).map_err(err)?);
    }
    Ok(out)
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

    pub fn generate_kv(spec: &Spec, _max_ctx: usize, kv_planes: usize, target: &kir::Target) -> Result<Self> {
        anyhow::ensure!(kv_planes == 1 || kv_planes == 2, "kv planes must be 1 or 2");
        anyhow::ensure!(kv_planes == 1 || !spec.runtime.kv_dtype.is_fp8(), "fp8 KV needs a single plane");
        let a = &spec.attention;
        let d = &spec.dims;
        let wf = crate::gemm::weight_format(spec.quant);
        let head_wf = crate::gemm::head_format(spec);
        let act = match spec.mlp.activation {
            Activation::Swiglu => 0,
            Activation::Geglu => 1,
        };
        let mut smem = BTreeMap::new();
        let attn = attention_kernels(spec, kv_planes, target)?;
        for k in &attn {
            match k.family {
                kir::attention::Family::FlashMma => smem.insert("flash_mma", k.smem_bytes),
                kir::attention::Family::GroupedDecode => smem.insert("batch_decode_attention", k.smem_bytes),
                kir::attention::Family::FlashLegacy => smem.insert("flash_attention", k.smem_bytes),
                _ => None,
            };
        }
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
            ("QKV_BIAS", a.qkv_bias.to_string()),
            ("EPS", float_literal(spec.norm.eps)),
            ("EMBED_SCALE", float_literal(spec.dims.embed_scale.unwrap_or(1.0))),
            ("THETA", float_literal(spec.rope.theta)),
            ("ACT_KIND", act.to_string()),
            ("Q8_GROUP", Q8_GROUP.to_string()),
            ("FP8_BLOCK", FP8_BLOCK.to_string()),
            ("Q_DIM", spec.q_dim().to_string()),
            ("QKV_N", spec.qkv_rows().to_string()),
            ("QKV_ROW", spec.qkv_rows().to_string()),
            ("OUT_GATE", a.output_gate.to_string()),
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
        ]
        .into_iter()
        .collect();
        let kir_kernels = kir_kernels(spec, target)?;
        let gemms = crate::gemm::gemm_kernels(spec, target)?;
        let kir_source = kir::cuda::emit_module(&kir_kernels).map_err(anyhow::Error::msg)?;
        let attn_source = kir::cuda::emit_attention(&attn).map_err(anyhow::Error::msg)?;
        let gemm_source = kir::cuda::emit_gemms(&gemms.kernels).map_err(anyhow::Error::msg)?;
        let kir_hash = kir::fold_hashes([kir::module_hash(&kir_kernels), kir::gemm::instances_hash(&gemms.kernels), kir::attention::attention_hash(&attn)]);
        let mut source = format!("{}\n{}\n{}\n{}\n{}", render(OPS_TEMPLATE, &vars)?, attn_source, gemm_source, render(MODEL_TEMPLATE, &vars)?, kir_source);
        let mut kernels = KERNELS.to_vec();
        for k in &kir_kernels {
            if !kernels.contains(&k.name) {
                kernels.push(k.name);
            }
        }
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
            if moe_mma_enabled(spec, target) {
                source.push('\n');
                source.push_str(MOE_MMA_TEMPLATE);
                kernels.extend_from_slice(MOE_MMA_KERNELS);
                smem.insert("moe_mma_up", moe_mma_smem(MOE_MMA_BM, 2));
                smem.insert("moe_mma_up_s", moe_mma_smem(MOE_MMA_BM_SMALL, 2));
                smem.insert("moe_mma_down", moe_mma_smem(MOE_MMA_BM, 1));
                smem.insert("moe_mma_down_s", moe_mma_smem(MOE_MMA_BM_SMALL, 1));
                arch_features = true;
            }
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
