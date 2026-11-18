use crate::spec::{Activation, Quant, Spec};
use anyhow::Result;
use std::collections::BTreeMap;

pub use crate::gemm::{
    BATCH_MAX_TOKENS, BATCH_REDUCE_BLOCK, BATCH_VARIANTS, BatchShape, FP4_GEMV_THREADS, FP8_ACT_LINEARS, GDN_LINEARS, GGB_LINEARS,
    GGB_MIN_TOKENS, GemmPath, KernelChoice, MMA_LINEARS, MmaShape, ROWS_MAX_TOKENS, Tile, WeightFormat, WeightKind,
    fp4_act_enabled, fp8_act_enabled, select, linear_dims, linear_shape,
};

pub const MOE_KERNELS: &[&str] = kir::moe::Family::Experts.kernels();
pub const MOE_MMA_KERNELS: &[&str] = kir::moe::Family::GroupedMma.kernels();
pub use kir::moe::{MMA_BM as MOE_MMA_BM, MMA_BM_SMALL as MOE_MMA_BM_SMALL, MMA_BN as MOE_MMA_BN, MMA_STAGES as MOE_MMA_STAGES, mma_smem as moe_mma_smem};

pub fn moe_mma_enabled(spec: &Spec, target: &kir::Target) -> bool {
    spec.moe.is_some()
        && spec.mlp.quant == Some(Quant::Nvfp4G16)
        && spec.mlp.act_quant == crate::spec::ActQuant::Fp4
        && target.compute_capability.0 >= 12
}

pub fn moe_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::moe::MoeKernel>> {
    let Some(m) = &spec.moe else { return Ok(Vec::new()) };
    let shape = kir::moe::MoeShape {
        experts: m.experts,
        top_k: m.top_k(),
        pairs: m.pairs(),
        inter: m.expert_intermediate,
        shared: m.shared_intermediate > 0,
        norm_topk: m.norm_topk,
        shared_gate: m.shared_gate,
        sigmoid: m.router == crate::spec::RouterAct::Sigmoid,
    };
    let mut out = vec![kir::moe::MoeKernel::new(target, kir::moe::Family::Experts, shape.clone()).map_err(anyhow::Error::msg)?];
    if moe_mma_enabled(spec, target) {
        out.push(kir::moe::MoeKernel::new(target, kir::moe::Family::GroupedMma, shape).map_err(anyhow::Error::msg)?);
    }
    Ok(out)
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

pub fn attention_shape(spec: &Spec, kv_planes: usize, max_ctx: usize) -> kir::attention::AttentionShape {
    let a = &spec.attention;
    let rope = spec.rope_constants();
    kir::attention::AttentionShape {
        heads: a.heads,
        kv_heads: a.kv_heads,
        head_dim: a.head_dim,
        qkv_row: spec.qkv_rows(),
        output_gate: a.output_gate,
        qk_norm: a.qk_norm && !a.qk_norm_full,
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
            freq_factors: spec.rope_freq_factors(max_ctx),
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
        nope: (0..spec.layers.count).filter(|&i| !spec.block(i).is_linear()).map(|i| spec.nope(i)).chain(std::iter::repeat_n(false, spec.speculative().map_or(0, |m| m.layers) + 1)).collect(),
        local_rope: (0..spec.layers.count)
            .filter(|&i| !spec.block(i).is_linear())
            .map(|i| spec.block(i) == crate::spec::Block::Sliding && spec.rope.local_theta.is_some())
            .chain(std::iter::repeat_n(false, spec.speculative().map_or(0, |m| m.layers) + 1))
            .collect(),
    }
}

pub fn attention_kernels(spec: &Spec, kv_planes: usize, max_ctx: usize, target: &kir::Target) -> Result<Vec<kir::attention::AttentionKernel>> {
    kir::attention::AttentionKernel::set(target, attention_shape(spec, kv_planes, max_ctx)).map_err(anyhow::Error::msg)
}

pub fn linear_kernel_shape(la: &crate::spec::LinearAttention) -> kir::linear::LinearShape {
    kir::linear::LinearShape { key_heads: la.key_heads, value_heads: la.value_heads, head_dim: la.key_head_dim, conv_kernel: la.conv_kernel, tiled_value_heads: la.v_heads == crate::spec::VHeadOrder::Tiled }
}

pub fn linear_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::linear::LinearKernel>> {
    let Some(la) = spec.linear_attention.as_ref() else { return Ok(Vec::new()) };
    let shape = linear_kernel_shape(la);
    let mut out = Vec::new();
    for f in kir::linear::Family::ALL {
        let needed = match f {
            kir::linear::Family::Verify | kir::linear::Family::Draft => spec.speculative().is_some(),
            _ => true,
        };
        if needed {
            out.push(kir::linear::LinearKernel::new(target, f, shape.clone()).map_err(anyhow::Error::msg)?);
        }
    }
    Ok(out)
}


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

pub const GDN_KERNELS: &[&str] = &["lin_in_norm", "lin_ba_norm", "lin_out_proj", "lin_ba_rows"];


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
    if spec.uses_branch() {
        out.push(kir::Kernel::norm_add(target, hidden, spec.norm.eps as f32, 0.0, 1024).map_err(err)?);
        out.push(kir::Kernel::pointwise(target, "clear_branch", hidden, kir::Pointwise::Fill(0.0), 256).map_err(err)?);
    }
    if let Some(scale) = spec.norm.residual_scale {
        out.push(kir::Kernel::add_scaled(target, "residual_add", hidden, scale as f32, 256).map_err(err)?);
    }
    if spec.attention.qk_norm_full {
        let (q, kv, row) = (spec.q_dim() as u32, spec.kv_dim() as u32, spec.qkv_rows() as u32);
        let eps = spec.norm.eps as f32;
        out.push(kir::Kernel::rmsnorm_segment(target, "q_norm_rows", q, row, 0, eps, 1024).map_err(err)?);
        out.push(kir::Kernel::rmsnorm_segment(target, "k_norm_rows", kv, row, q, eps, 1024).map_err(err)?);
    }
    if let Some(scale) = spec.dims.logit_scale {
        out.push(kir::Kernel::pointwise(target, "logit_scale", spec.dims.vocab as u32, kir::Pointwise::Scale(scale as f32), 256).map_err(err)?);
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

pub fn gemv_kernels(spec: &Spec, target: &kir::Target, ops: &kir::ops::OpsShape) -> Result<Vec<kir::gemv::GemvKernel>> {
    use kir::gemv::Gemv;
    let (wf, hwf) = (ops.weight_decoder.clone(), ops.head_decoder.clone());
    let (n, h, i, v, q) = (ops.qkv_row, ops.hidden, ops.inter, ops.vocab, ops.q_dim);
    let bias = ops.qkv_bias;
    let gu = 2 * ops.inter;
    let lin = |d: &str, n, k, bias, residual| Gemv::Linear { decoder: d.to_string(), n, k, bias, residual };
    let tiled = |d: &str, n, k, bias, residual| Gemv::Tiled { decoder: d.to_string(), n, k, bias, residual };
    let mut ops_list = vec![
        ("qkv_proj", lin(&wf, n, h, bias, false)),
        ("o_proj", lin(&wf, h, q, false, true)),
        ("gate_up_proj", lin(&wf, gu, h, false, false)),
        ("down_proj", lin(&wf, h, i, false, true)),
        ("lm_head", lin(&hwf, v, h, false, false)),
        ("qkv_gemm", tiled(&wf, n, h, bias, false)),
        ("o_gemm", tiled(&wf, h, q, false, true)),
        ("gate_up_gemm", tiled(&wf, gu, h, false, false)),
        ("down_gemm", tiled(&wf, h, i, false, true)),
        ("lm_head_gemm", tiled(&hwf, v, h, false, false)),
        ("qkv_proj_norm", Gemv::Norm { decoder: wf.clone(), n, bias }),
        ("gate_up_act", Gemv::GateUpAct { decoder: wf.clone() }),
        ("lm_head_norm", Gemv::Norm { decoder: hwf.clone(), n: v, bias: false }),
    ];
    if let Some(la) = spec.linear_attention.as_ref() {
        ops_list.push(("lin_in_norm", Gemv::Norm { decoder: wf.clone(), n: la.in_proj_rows(), bias: false }));
        ops_list.push(("lin_ba_norm", Gemv::Norm { decoder: "DecBf16".into(), n: la.ba_rows(), bias: false }));
        ops_list.push(("lin_ba_rows", Gemv::Bf16Rows { n: la.ba_rows(), k: h }));
        ops_list.push(("lin_out_proj", lin(&wf, h, la.value_dim(), false, true)));
    }
    ops_list.into_iter().map(|(name, op)| kir::gemv::GemvKernel::new(target, name, op).map_err(anyhow::Error::msg)).collect()
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
        let mut smem = BTreeMap::new();
        let attn = attention_kernels(spec, kv_planes, max_ctx, target)?;
        for k in &attn {
            match k.family {
                kir::attention::Family::FlashMma => smem.insert("flash_mma", k.smem_bytes),
                kir::attention::Family::GroupedDecode => smem.insert("batch_decode_attention", k.smem_bytes),
                kir::attention::Family::FlashLegacy => smem.insert("flash_attention", k.smem_bytes),
                _ => None,
            };
        }
        let la = spec.linear_attention.as_ref();
        let linear = linear_kernels(spec, target)?;
        for k in &linear {
            if k.family == kir::linear::Family::Chunk {
                smem.insert("gdn_chunk_prep", k.smem_bytes);
            }
        }
        let ops = kir::ops::OpsShape {
            hidden: d.hidden,
            inter: d.intermediate,
            vocab: d.vocab,
            heads: a.heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            qkv_row: spec.qkv_rows(),
            q_dim: spec.q_dim(),
            output_gate: a.output_gate,
            qkv_bias: a.qkv_bias,
            eps: spec.norm.eps,
            embed_scale: spec.dims.embed_scale.unwrap_or(1.0),
            gated_gelu: spec.mlp.activation == Activation::Geglu,
            q8_group: Q8_GROUP,
            fp8_block: FP8_BLOCK,
            weight_decoder: wf.decoder().to_string(),
            head_decoder: head_wf.decoder().to_string(),
        };
        let kir_kernels = kir_kernels(spec, target)?;
        let gemms = crate::gemm::gemm_kernels(spec, target)?;
        let kir_source = kir::cuda::emit_module(&kir_kernels).map_err(anyhow::Error::msg)?;
        let attn_source = kir::cuda::emit_attention(&attn).map_err(anyhow::Error::msg)?;
        let gemm_source = kir::cuda::emit_gemms(&gemms.kernels).map_err(anyhow::Error::msg)?;
        let linear_source = kir::cuda::emit_linear(&linear).map_err(anyhow::Error::msg)?;
        let kir_hash = kir::fold_hashes([
            kir::module_hash(&kir_kernels),
            kir::gemm::instances_hash(&gemms.kernels),
            kir::attention::attention_hash(&attn),
            kir::linear::linear_hash(&linear),
        ]);
        let ggml = kir::ggml::GgmlKernel::new(target);
        let prims = kir::prims::cuda_prims(target, d.hidden, d.vocab, spec.norm.eps, spec.dims.embed_scale.unwrap_or(1.0)).map_err(anyhow::Error::msg)?;
        let gemvs = gemv_kernels(spec, target, &ops)?;
        let kir_hash = kir::fold_hashes([kir_hash, ggml.hash(), kir::prims::prims_hash(&prims), kir::gemv::gemv_hash(&gemvs)]);
        let mut source = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            kir::cuda::emit_ops(&ops),
            kir::prims::emit_cuda(&prims).map_err(anyhow::Error::msg)?,
            ggml.emit().map_err(anyhow::Error::msg)?,
            attn_source,
            gemm_source,
            kir::gemv::emit_gemvs(&gemvs).map_err(anyhow::Error::msg)?,
            kir_source
        );
        let mut kernels = KERNELS.to_vec();
        for k in &kir_kernels {
            if !kernels.contains(&k.name) {
                kernels.push(k.name);
            }
        }
        if la.is_some() {
            source.push('\n');
            source.push_str(&linear_source);
            for k in &linear {
                kernels.extend_from_slice(k.family.kernels());
            }
            kernels.extend_from_slice(GDN_KERNELS);
            smem.insert("lin_ba_rows", 0);
        }
        let mut moe_arch = false;
        let moe = moe_kernels(spec, target)?;
        if !moe.is_empty() {
            source.push('\n');
            source.push_str(&kir::cuda::emit_moe(&moe).map_err(anyhow::Error::msg)?);
            for k in &moe {
                kernels.extend_from_slice(k.family.kernels());
                smem.extend(k.smem());
                moe_arch |= k.arch_features();
            }
        }
        let kir_hash = kir::fold_hashes([kir_hash, kir::moe::moe_hash(&moe)]);
        if spec.quant == Quant::Gguf {
            kernels.push("gg_dequant_bf16");
        }
        kernels.extend(gemms.names());
        smem.extend(gemms.smem());
        let arch_features = moe_arch || kir::cuda::needs_arch_features(&gemms.kernels);
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
