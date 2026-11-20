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
pub use kir::moe::{mma8_smem as moe_mma8_smem, mma_bf16_smem as moe_mma_bf16_smem, MMA_BM as MOE_MMA_BM, MMA_BM_SMALL as MOE_MMA_BM_SMALL, MMA_BN as MOE_MMA_BN, MMA_STAGES as MOE_MMA_STAGES, mma_smem as moe_mma_smem};

pub fn moe_mma_enabled(spec: &Spec, target: &kir::Target) -> bool {
    let Some(m) = &spec.moe else { return false };
    let weights = match &m.expert_block {
        Some(block) => block.native_mma(),
        None => spec.mlp.quant == Some(Quant::Nvfp4G16),
    };
    weights && spec.mlp.act_quant == crate::spec::ActQuant::Fp4 && target.compute_capability.0 >= 12
}

pub fn moe_mma_bf16_enabled(spec: &Spec, target: &kir::Target) -> bool {
    let Some(m) = &spec.moe else { return false };
    spec.mlp.quant.unwrap_or(spec.quant) == Quant::Bf16
        && m.expert_block.is_none()
        && target.has(kir::Feature::Mma16816)
        && target.has(kir::Feature::CpAsync)
        && m.expert_intermediate.is_multiple_of(64)
        && spec.dims.hidden.is_multiple_of(32)
}

pub fn moe_mma_fp8_enabled(spec: &Spec, target: &kir::Target) -> bool {
    let Some(m) = &spec.moe else { return false };
    spec.mlp.quant.unwrap_or(spec.quant) == Quant::Fp8Block128
        && spec.mlp.act_quant == crate::spec::ActQuant::Fp8
        && target.compute_capability >= (8, 9)
        && m.expert_block.is_none()
        && m.expert_intermediate.is_multiple_of(FP8_BLOCK)
        && spec.dims.hidden.is_multiple_of(FP8_BLOCK)
}

pub fn moe_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::moe::MoeKernel>> {
    let Some(m) = &spec.moe else { return Ok(Vec::new()) };
    let shape = kir::moe::MoeShape {
        experts: m.experts,
        local: m.local_experts(),
        top_k: m.top_k(),
        pairs: m.pairs(),
        inter: m.expert_intermediate,
        shared: m.shared_slots(),
        norm_topk: m.norm_topk,
        routed_scale: m.routed_scale,
        out_scale: m.out_scale,
        shared_gate: m.shared_gate,
        sigmoid: m.router == crate::spec::RouterAct::Sigmoid,
        groups: m.groups,
        topk_group: m.topk_group,
        score_bias: m.score_bias,
        router_bias: m.router_bias,
        expert_bias: m.expert_bias,
        hash: m.hash_table.is_some(),
        block_group: m.expert_block.map_or(0, |b| b.group),
    };
    let mut out = vec![kir::moe::MoeKernel::new(target, kir::moe::Family::Experts, shape.clone()).map_err(anyhow::Error::msg)?];
    if moe_mma_enabled(spec, target) {
        let family = if m.expert_block.is_some() { kir::moe::Family::GroupedMx } else { kir::moe::Family::GroupedMma };
        out.push(kir::moe::MoeKernel::new(target, family, shape.clone()).map_err(anyhow::Error::msg)?);
    }
    if moe_mma_bf16_enabled(spec, target) {
        out.push(kir::moe::MoeKernel::new(target, kir::moe::Family::GroupedMmaBf16, shape.clone()).map_err(anyhow::Error::msg)?);
    }
    if moe_mma_fp8_enabled(spec, target) {
        out.push(kir::moe::MoeKernel::new(target, kir::moe::Family::GroupedMmaFp8, shape).map_err(anyhow::Error::msg)?);
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

pub fn kir_shape(spec: &Spec) -> kir::desc::Shape {
    let act = match spec.mlp.activation {
        Activation::Swiglu | Activation::Mapped => "silu",
        Activation::Geglu => "gelu_tanh",
    };
    let mut shape = kir::desc::Shape::new()
        .num("hidden", spec.dims.hidden as f64)
        .num("inter", spec.dims.intermediate as f64)
        .num("vocab", spec.dims.vocab as f64)
        .num("norm_eps", spec.norm.eps)
        .num("q_dim", spec.q_dim() as f64)
        .num("kv_dim", spec.kv_dim() as f64)
        .num("qkv_row", spec.qkv_rows() as f64)
        .word("mlp_act", act);
    if let Some(v) = spec.norm.residual_scale {
        shape = shape.num("residual_scale", v);
    }
    if let Some(v) = spec.dims.logit_scale {
        shape = shape.num("logit_scale", v);
    }
    if let Some(v) = spec.dims.logit_softcap {
        shape = shape.num("logit_softcap", v);
    }
    shape
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
            theta: spec.theta(),
            kind: rope.kind,
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
        local_theta: spec.rope.local_theta.unwrap_or_else(|| spec.theta()),
        nope: (0..spec.layers.count).filter(|&i| spec.block(i).is_attention()).map(|i| spec.nope(i)).chain(std::iter::repeat_n(false, spec.speculative().map_or(0, |m| m.layers) + 1)).collect(),
        local_rope: (0..spec.layers.count)
            .filter(|&i| spec.block(i).is_attention())
            .map(|i| spec.block(i) == crate::spec::Block::Sliding && spec.rope.local_theta.is_some())
            .chain(std::iter::repeat_n(false, spec.speculative().map_or(0, |m| m.layers) + 1))
            .collect(),
    }
}

pub fn attention_symbols(spec: &Spec, kv_planes: usize, max_ctx: usize) -> kir::desc::Shape {
    let a = &spec.attention;
    let rope = spec.rope_constants();
    let shape = attention_shape(spec, kv_planes, max_ctx);
    let mut symbols = kir_shape(spec)
        .num("heads", a.heads as f64)
        .num("kv_heads", a.kv_heads as f64)
        .num("head_dim", a.head_dim as f64)
        .num("qkv_row", spec.qkv_rows() as f64)
        .flag("output_gate", a.output_gate)
        .flag("qk_norm", a.qk_norm && !a.qk_norm_full)
        .num("norm_eps", spec.norm.eps)
        .num("rotary_dim", spec.rotary_dim() as f64)
        .num("rope_theta", spec.theta())
        .num("rope_kind", rope.kind as f64)
        .num("rope_factor", rope.factor)
        .num("rope_low", rope.low)
        .num("rope_high", rope.high)
        .num("rope_original", rope.original)
        .num("rope_mscale", rope.mscale)
        .nums("rope_freq_factors", &shape.rope.freq_factors)
        .num("max_context", max_ctx as f64)
        .num("kv_layers", spec.kv_layers() as f64)
        .num("kv_planes", kv_planes as f64)
        .flag("kv_fp8", shape.kv_fp8)
        .num("kv_f16_layers", shape.kv_f16_layers as f64)
        .num("query_scale", shape.query_scale)
        .num("softcap", shape.softcap)
        .num("local_theta", shape.local_theta)
        .flag("bidirectional", shape.bidirectional)
        .flags("nope", &shape.nope)
        .flags("local_rope", &shape.local_rope);
    if let Some(m) = &shape.rope.mrope {
        symbols = symbols
            .nums("mrope_section", &m.section.iter().map(|s| *s as f64).collect::<Vec<_>>())
            .flag("mrope_interleaved", m.interleaved);
    }
    symbols
}

pub fn attention_kernels(spec: &Spec, kv_planes: usize, max_ctx: usize, target: &kir::Target) -> Result<Vec<kir::attention::AttentionKernel>> {
    let symbols = attention_symbols(spec, kv_planes, max_ctx);
    kir::desc::ATTENTION
        .iter()
        .map(|(n, _)| kir::desc::build_attention(n, target, &symbols).map_err(anyhow::Error::msg))
        .collect()
}

pub fn described_kernels(spec: &Spec, kv_planes: usize, max_ctx: usize, target: &kir::Target) -> Result<Vec<Option<kir::Kernel>>> {
    let mut symbols = attention_symbols(spec, kv_planes, max_ctx).num("o_dim", spec.o_dim() as f64);
    for (n, v) in &spec.attention.params {
        symbols = symbols.num(n, *v);
    }
    symbols = symbols.num("max_context", max_ctx as f64);
    for s in crate::state::positional(spec) {
        symbols = symbols.num(&format!("{}_layers", s.name), s.layer_count(spec) as f64);
    }
    let registry = crate::kir_kernel::Registry::active();
    let mut built: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, node) in spec.attention.node.iter().enumerate() {
        let Some(name) = node.kernel.as_deref() else {
            out.push(None);
            continue;
        };
        if !built.contains_key(name) {
            let d = registry.get(name).ok_or_else(|| anyhow::anyhow!("[[attention.node]] {i} names kernel {name:?}, which no kernel description provides"))?;
            built.insert(name.to_string(), d.build(target, &symbols).map_err(anyhow::Error::msg)?);
        }
        let kernel = &built[name];
        for p in kernel.params.iter().filter(|p| !kir::scan::LAUNCH.contains(&p.name)) {
            anyhow::ensure!(node.bind.contains_key(p.name), "[[attention.node]] {i} does not bind {name} param {:?}", p.name);
        }
        for (b, v) in &node.bind {
            let param = kernel.params.iter().find(|p| p.name == b).ok_or_else(|| anyhow::anyhow!("[[attention.node]] {i} binds {b:?}, which {name} has no param for"))?;
            match (param.role, v) {
                (kir::Role::Weight, crate::graph::Bind::Weight { .. }) => {}
                (kir::Role::Activation, crate::graph::Bind::Buffer { buffer }) => {
                    let ints = spec.attention.int_scratch.contains_key(buffer);
                    anyhow::ensure!(param.ty == if ints { kir::Scalar::I32 } else { kir::Scalar::F32 }, "[[attention.node]] {i} binds buffer {buffer:?} to {b:?}, which is not {}", if ints { "i32" } else { "f32" })
                }
                (kir::Role::Activation, crate::graph::Bind::State { .. }) => anyhow::ensure!(param.ty == kir::Scalar::Bf16, "[[attention.node]] {i} binds a state to {b:?}, which is not bf16"),
                (kir::Role::Value, crate::graph::Bind::Weight { .. } | crate::graph::Bind::Buffer { .. } | crate::graph::Bind::State { .. })
                | (kir::Role::Weight, _)
                | (kir::Role::Activation | kir::Role::Scratch, crate::graph::Bind::Weight { .. } | crate::graph::Bind::Number(_) | crate::graph::Bind::Name(_)) => {
                    anyhow::bail!("[[attention.node]] {i} binds {b:?} to the wrong kind of value")
                }
                (_, crate::graph::Bind::Name(n)) => anyhow::ensure!(matches!(n.as_str(), "window" | "nope") || spec.attention.params.contains_key(n), "[[attention.node]] {i} binds {b:?} to {n:?}, which is not window, nope or an [attention.params] entry"),
                _ => {}
            }
        }
        out.push(Some(kernel.clone()));
    }
    Ok(out)
}

pub fn mixer_symbols(spec: &Spec, kv_planes: usize, max_ctx: usize) -> kir::desc::Shape {
    let mut symbols = attention_symbols(spec, kv_planes, max_ctx).num("mix_in", spec.mix_in_rows() as f64).num("mix_out", spec.mix_out_cols() as f64).num("linear_layers", spec.linear_layers() as f64);
    for (n, v) in spec.mixer.iter().flat_map(|m| m.params.iter()) {
        symbols = symbols.num(n, *v);
    }
    symbols
}

pub fn mixer_kernels(spec: &Spec, kv_planes: usize, max_ctx: usize, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let Some(m) = &spec.mixer else { return Ok(Vec::new()) };
    let symbols = mixer_symbols(spec, kv_planes, max_ctx);
    let registry = crate::kir_kernel::Registry::active();
    let mut built: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, node) in m.node.iter().enumerate() {
        let name = node.kernel.as_deref().ok_or_else(|| anyhow::anyhow!("[[mixer.node]] {i} names no kernel"))?;
        if !built.contains_key(name) {
            let d = registry.get(name).ok_or_else(|| anyhow::anyhow!("[[mixer.node]] {i} names kernel {name:?}, which no kernel description provides"))?;
            built.insert(name.to_string(), d.build(target, &symbols).map_err(|e| anyhow::anyhow!("[[mixer.node]] {i}: {e}"))?);
        }
        let kernel = &built[name];
        for p in kernel.params.iter().filter(|p| !kir::recur::LAUNCH.contains(&p.name) && !matches!(p.name, "out_hi" | "out_lo")) {
            anyhow::ensure!(node.bind.contains_key(p.name), "[[mixer.node]] {i} does not bind {name} param {:?}", p.name);
        }
        for (b, v) in &node.bind {
            let param = kernel.params.iter().find(|p| p.name == b).ok_or_else(|| anyhow::anyhow!("[[mixer.node]] {i} binds {b:?}, which {name} has no param for"))?;
            let fits = match v {
                crate::graph::Bind::Weight { .. } => param.role == kir::Role::Weight,
                crate::graph::Bind::Buffer { .. } => param.role == kir::Role::Activation && param.ty == kir::Scalar::F32,
                crate::graph::Bind::State { .. } => param.role == kir::Role::Activation && param.ty == kir::Scalar::F32 && param.access == kir::Access::Write,
                crate::graph::Bind::Number(_) | crate::graph::Bind::Name(_) => param.role == kir::Role::Value,
            };
            anyhow::ensure!(fits, "[[mixer.node]] {i} binds {b:?} to the wrong kind of value");
        }
        out.push(kernel.clone());
    }
    Ok(out)
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
    let shape = kir_shape(spec);
    let mut names = vec!["rmsnorm", "swiglu"];
    if spec.uses_branch() {
        names.extend(["norm_add", "clear_branch"]);
    }
    if spec.norm.residual_scale.is_some() {
        names.push("residual_add");
    }
    if spec.attention.qk_norm_full {
        names.extend(["q_norm_rows", "k_norm_rows"]);
    }
    if spec.dims.logit_scale.is_some() {
        names.push("logit_scale");
    }
    if spec.dims.logit_softcap.is_some() {
        names.push("logit_softcap");
    }
    names.into_iter().map(|n| kir::desc::build_builtin(n, target, &shape).map_err(anyhow::Error::msg)).collect()
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
    let gu = if ops.mapped.is_some() { ops.inter } else { 2 * ops.inter };
    let lin = |d: &str, n, k, bias, residual| Gemv::Linear { decoder: d.to_string(), n, k, bias, residual };
    let tiled = |d: &str, n, k, bias, residual| Gemv::Tiled { decoder: d.to_string(), n, k, bias, residual };
    let mut ops_list = vec![
        ("qkv_proj", lin(&wf, n, h, bias, false)),
        ("o_proj", lin(&wf, h, q, ops.o_bias, true)),
        ("gate_up_proj", lin(&wf, gu, h, false, false)),
        ("down_proj", lin(&wf, h, i, false, true)),
        ("lm_head", lin(&hwf, v, h, false, false)),
        ("qkv_gemm", tiled(&wf, n, h, bias, false)),
        ("o_gemm", tiled(&wf, h, q, ops.o_bias, true)),
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
            o_bias: a.o_bias,
            eps: spec.norm.eps,
            embed_scale: spec.dims.embed_scale.unwrap_or(1.0),
            gated_gelu: spec.mlp.activation == Activation::Geglu,
            q8_group: Q8_GROUP,
            fp8_block: FP8_BLOCK,
            gated: spec.mlp.gated.as_ref().map(|g| kir::ops::GatedExpr { expr: g.expr.clone(), params: g.params.iter().map(|(k, v)| (k.clone(), *v)).collect() }),
            mapped: spec.mlp.mapped.as_ref().map(|g| kir::ops::GatedExpr { expr: g.expr.clone(), params: g.params.iter().map(|(k, v)| (k.clone(), *v)).collect() }),
            weight_decoder: wf.decoder().to_string(),
            head_decoder: head_wf.decoder().to_string(),
        };
        let kir_kernels = kir_kernels(spec, target)?;
        let gemms = crate::gemm::gemm_kernels(spec, target)?;
        let described = described_kernels(spec, kv_planes, max_ctx, target)?;
        let mut distinct: Vec<kir::Kernel> = Vec::new();
        for k in described.into_iter().flatten().chain(mixer_kernels(spec, kv_planes, max_ctx, target)?) {
            if !distinct.iter().any(|d| d.name == k.name) {
                distinct.push(k);
            }
        }
        let kir_source = kir::cuda::emit_module(&kir_kernels.iter().chain(&distinct).cloned().collect::<Vec<_>>()).map_err(anyhow::Error::msg)?;
        let attn_source = kir::cuda::emit_attention(&attn).map_err(anyhow::Error::msg)?;
        let gemm_source = kir::cuda::emit_gemms(&gemms.kernels).map_err(anyhow::Error::msg)?;
        let linear_source = kir::cuda::emit_linear(&linear).map_err(anyhow::Error::msg)?;
        let kir_hash = kir::fold_hashes([
            if distinct.is_empty() { kir::module_hash(&kir_kernels) } else { kir::fold_hashes([kir::module_hash(&kir_kernels), kir::module_hash(&distinct)]) },
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
            kir::cuda::emit_ops(&ops).map_err(anyhow::Error::msg)?,
            kir::prims::emit_cuda(&prims).map_err(anyhow::Error::msg)?,
            ggml.emit().map_err(anyhow::Error::msg)?,
            attn_source,
            gemm_source,
            kir::gemv::emit_gemvs(&gemvs).map_err(anyhow::Error::msg)?,
            kir_source
        );
        let mut kernels = KERNELS.to_vec();
        for k in kir_kernels.iter().chain(&distinct) {
            if !kernels.contains(&k.name) {
                kernels.push(k.name);
            }
            if let Some(f) = k.chunk_form() {
                kernels.extend([f.state, f.carry, f.out]);
                smem.insert(f.state, f.state_smem);
                smem.insert(f.out, f.out_smem);
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
