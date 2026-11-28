use crate::spec::{Activation, Quant, Spec};
use anyhow::Result;
use std::collections::BTreeMap;

pub use crate::gemm::{
    BATCH_MAX_TOKENS, BATCH_REDUCE_BLOCK, BATCH_VARIANTS, BatchShape, FP4_GEMV_THREADS, FP8_ACT_LINEARS, GDN_LINEARS, GGB_LINEARS,
    GGB_MIN_TOKENS, GGML_GEMV_THREADS, GemmPath, KernelChoice, MMA_LINEARS, MmaShape, ROWS_MAX_TOKENS, Tile, WeightFormat, WeightKind,
    fp4_act_enabled, fp8_act_enabled, select, select_act, linear_dims, linear_shape,
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
        && spec.dims.hidden.is_multiple_of(64)
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

pub fn moe_ggml_experts(spec: &Spec) -> bool {
    spec.moe.as_ref().is_some_and(|m| m.quantized_experts) && matches!(spec.mlp.quant.unwrap_or(spec.quant), Quant::Gguf | Quant::Q8)
}

pub fn moe_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::moe::MoeKernel>> {
    let Some(m) = &spec.moe else { return Ok(Vec::new()) };
    let shape = kir::moe::MoeShape {
        hidden: spec.dims.hidden,
        width: m.width(spec.dims.hidden),
        assign: m.latent.is_some(),
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
        mapped: spec.mlp.mapped.is_some(),
        input_scaled: m.input_scaled,
        input_norm: spec.gated_residual().is_none(),
        split_input: m.parallel_dense,
        expert_scale: m.expert_scale.is_some(),
    };
    let mut out = vec![kir::moe::MoeKernel::new(target, kir::moe::Family::Experts, shape.clone()).map_err(anyhow::Error::msg)?];
    if spec.runtime.offload.pointer_table && spec.runtime.offload.enabled() {
        out.push(kir::moe::MoeKernel::new(target, kir::moe::Family::ExpertTable, shape.clone()).map_err(anyhow::Error::msg)?);
    }
    if moe_ggml_experts(spec) {
        out.push(kir::moe::MoeKernel::new(target, kir::moe::Family::ExpertsGgml, shape.clone()).map_err(anyhow::Error::msg)?);
    }
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
pub use kir::attention::{flash_rows, DECODE_CHUNK, DECODE_KEYS, DECODE_MAX_QLEN, DECODE_MAX_SPLITS, DECODE_STAGES, decode_chunk, decode_splits, flash_keys, flash_stages};

pub fn decode_warps(group: usize, qlen: usize) -> usize {
    kir::attention::decode_warps(group, qlen)
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
    } else if spec.gated_residual().is_some() {
        shape = shape.num("residual_scale", 1.0);
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
        qk_norm_unit: a.qk_norm_unit,
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
            interleaved: spec.rope.style == crate::spec::RopeStyle::Pairs,
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
        .num("qkv_append", spec.appended_offset() as f64)
        .flag("output_gate", a.output_gate)
        .flag("qk_norm", a.qk_norm && !a.qk_norm_full)
        .flag("qk_norm_unit", a.qk_norm_unit)
        .flag("rope_interleaved", spec.rope.style == crate::spec::RopeStyle::Pairs)
        .num("norm_eps", spec.norm.eps)
        .flag("norm_unit_offset", spec.norm.unit_offset)
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
        .filter(|(n, _)| attention_available(n, target))
        .map(|(n, _)| kir::desc::build_attention(n, target, &symbols).map_err(anyhow::Error::msg))
        .collect()
}

pub fn attention_available(kernel: &str, target: &kir::Target) -> bool {
    let family = match kernel {
        "flash_mma" => kir::attention::Family::FlashMma,
        "batch_decode_attention" | "batch_decode_combine" => kir::attention::Family::GroupedDecode,
        _ => return true,
    };
    family.needs().iter().all(|&f| target.has(f))
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
                (_, crate::graph::Bind::Name(n)) => anyhow::ensure!(matches!(n.as_str(), "window" | "nope" | "kind_layer") || spec.attention.params.contains_key(n), "[[attention.node]] {i} binds {b:?} to {n:?}, which is not window, nope, kind_layer or an [attention.params] entry"),
                _ => {}
            }
        }
        out.push(Some(kernel.clone()));
    }
    Ok(out)
}

pub fn described_tiles(spec: &Spec, node: &crate::graph::Node) -> bool {
    let a = &spec.attention;
    !spec.runtime.kv_dtype.is_fp8() && node.grid.len() == 2 && node.grid[0] == "heads" && node.grid[1] == "rows" && a.heads / a.kv_heads <= 16
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

pub fn hyper_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let Some(h) = &spec.hyper else { return Ok(Vec::new()) };
    let symbols = kir_shape(spec).num("hc_streams", h.streams as f64).num("hc_sinkhorn_iters", h.sinkhorn_iters as f64).num("hc_eps", h.eps);
    let registry = crate::kir_kernel::Registry::active();
    ["hyper_expand", "hyper_pre", "hyper_post", "hyper_mean"]
        .iter()
        .map(|n| {
            let d = registry.get(n).ok_or_else(|| anyhow::anyhow!("no kernel description provides {n}"))?;
            d.build(target, &symbols).map_err(anyhow::Error::msg)
        })
        .collect()
}

pub fn residual_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let mut names: Vec<&str> = Vec::new();
    if spec.block_residual.is_some() {
        names.extend(["block_res_mix", "block_res_push", "block_res_add", "block_res_set", "block_res_out"]);
    }
    if spec.moe.as_ref().is_some_and(|m| m.latent.is_some()) {
        names.push("latent_norm");
    }
    let mut symbols = kir_shape(spec);
    if let Some(r) = &spec.block_residual {
        symbols = symbols.num("res_blocks", r.blocks(spec.layers.count) as f64);
    }
    if let Some(m) = &spec.moe {
        symbols = symbols.num("moe_width", m.width(spec.dims.hidden) as f64);
    }
    let registry = crate::kir_kernel::Registry::active();
    names
        .into_iter()
        .map(|n| {
            let d = registry.get(n).ok_or_else(|| anyhow::anyhow!("no kernel description provides {n}"))?;
            d.build(target, &symbols).map_err(anyhow::Error::msg)
        })
        .collect()
}

pub fn gated_symbols(spec: &Spec) -> kir::desc::Shape {
    let mut symbols = kir_shape(spec);
    if let Some(h) = spec.gated_residual() {
        symbols = symbols.num("hc_streams", h.streams as f64).num("hc_width", h.width(spec.dims.hidden) as f64).num("gate_up_rows", crate::gemm::linear_shape(spec, "gate_up").n as f64);
        for (n, v) in &h.params {
            symbols = symbols.num(n, *v);
        }
        if let Some(p) = spec.per_layer() {
            symbols = symbols.num("ple_dim", p.dim as f64).num("ple_width", (p.dim * spec.layers.count) as f64);
            for (n, v) in &p.params {
                symbols = symbols.num(n, *v);
            }
        }
    }
    symbols
}

pub fn gated_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let Some(h) = spec.gated_residual() else { return Ok(Vec::new()) };
    let symbols = gated_symbols(spec);
    let registry = crate::kir_kernel::Registry::active();
    let mut built: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, node) in h.node.iter().enumerate() {
        let name = node.kernel.as_deref().ok_or_else(|| anyhow::anyhow!("[[gated_residual.node]] {i} names no kernel"))?;
        if !built.contains_key(name) {
            let d = registry.get(name).ok_or_else(|| anyhow::anyhow!("[[gated_residual.node]] {i} names kernel {name:?}, which no kernel description provides"))?;
            built.insert(name.to_string(), d.build(target, &symbols).map_err(|e| anyhow::anyhow!("[[gated_residual.node]] {i}: {e}"))?);
        }
        let kernel = &built[name];
        for p in &kernel.params {
            anyhow::ensure!(node.bind.contains_key(p.name), "[[gated_residual.node]] {i} does not bind {name} param {:?}", p.name);
        }
        for (b, v) in &node.bind {
            let param = kernel.params.iter().find(|p| p.name == b).ok_or_else(|| anyhow::anyhow!("[[gated_residual.node]] {i} binds {b:?}, which {name} has no param for"))?;
            let fits = match v {
                crate::graph::Bind::Weight { .. } => param.role == kir::Role::Weight,
                crate::graph::Bind::Buffer { buffer } if matches!(buffer.as_str(), "act_hi" | "act_lo") => param.role == kir::Role::Activation && param.ty == kir::Scalar::Bf16,
                crate::graph::Bind::Buffer { .. } => param.role == kir::Role::Activation && param.ty == kir::Scalar::F32,
                crate::graph::Bind::Name(_) | crate::graph::Bind::Number(_) => param.role == kir::Role::Value,
                _ => false,
            };
            anyhow::ensure!(fits, "[[gated_residual.node]] {i} binds {b:?} to the wrong kind of value");
        }
        out.push(kernel.clone());
    }
    Ok(out)
}

pub fn per_layer_symbols(spec: &Spec) -> kir::desc::Shape {
    let mut symbols = gated_symbols(spec);
    if let Some(p) = spec.per_layer() {
        symbols = symbols.num("ple_dim", p.dim as f64).num("ple_width", (p.dim * spec.layers.count) as f64);
        for (n, v) in &p.params {
            symbols = symbols.num(n, *v);
        }
    }
    symbols
}

pub fn input_symbols(spec: &Spec) -> kir::desc::Shape {
    let mut symbols = per_layer_symbols(spec);
    if let Some(p) = &spec.inputs {
        for (n, v) in &p.params {
            symbols = symbols.num(n, *v);
        }
    }
    symbols
}

pub fn input_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let Some(p) = &spec.inputs else { return Ok(Vec::new()) };
    let symbols = input_symbols(spec);
    let registry = crate::kir_kernel::Registry::active();
    let mut built: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, node) in p.node.iter().enumerate() {
        let name = node.kernel.as_deref().ok_or_else(|| anyhow::anyhow!("[[inputs.node]] {i} names no kernel"))?;
        if !built.contains_key(name) {
            let d = registry.get(name).ok_or_else(|| anyhow::anyhow!("[[inputs.node]] {i} names kernel {name:?}, which no kernel description provides"))?;
            built.insert(name.to_string(), d.build(target, &symbols).map_err(|e| anyhow::anyhow!("[[inputs.node]] {i}: {e}"))?);
        }
        let kernel = &built[name];
        for param in &kernel.params {
            anyhow::ensure!(node.bind.contains_key(param.name), "[[inputs.node]] {i} does not bind {name} param {:?}", param.name);
        }
        for (b, v) in &node.bind {
            let param = kernel.params.iter().find(|p| p.name == b).ok_or_else(|| anyhow::anyhow!("[[inputs.node]] {i} binds {b:?}, which {name} has no param for"))?;
            let fits = match v {
                crate::graph::Bind::Weight { .. } => param.role == kir::Role::Weight,
                crate::graph::Bind::Buffer { buffer } if buffer == "tokens" => param.role == kir::Role::Activation && param.ty == kir::Scalar::I32,
                crate::graph::Bind::Buffer { .. } => param.role == kir::Role::Activation && param.ty == kir::Scalar::F32,
                crate::graph::Bind::Number(_) => param.role == kir::Role::Value,
                _ => false,
            };
            anyhow::ensure!(fits, "[[inputs.node]] {i} binds {b:?} to the wrong kind of value");
        }
        out.push(kernel.clone());
    }
    Ok(out)
}

pub fn per_layer_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let Some(p) = spec.per_layer() else { return Ok(Vec::new()) };
    let symbols = per_layer_symbols(spec);
    let registry = crate::kir_kernel::Registry::active();
    let mut built: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, node) in p.node.iter().enumerate() {
        let name = node.kernel.as_deref().ok_or_else(|| anyhow::anyhow!("[[per_layer.node]] {i} names no kernel"))?;
        if !built.contains_key(name) {
            let d = registry.get(name).ok_or_else(|| anyhow::anyhow!("[[per_layer.node]] {i} names kernel {name:?}, which no kernel description provides"))?;
            built.insert(name.to_string(), d.build(target, &symbols).map_err(|e| anyhow::anyhow!("[[per_layer.node]] {i}: {e}"))?);
        }
        let kernel = &built[name];
        for param in kernel.params.iter().filter(|p| !kir::scan::LAUNCH.contains(&p.name)) {
            anyhow::ensure!(node.bind.contains_key(param.name), "[[per_layer.node]] {i} does not bind {name} param {:?}", param.name);
        }
        for (b, v) in &node.bind {
            let param = kernel.params.iter().find(|p| p.name == b).ok_or_else(|| anyhow::anyhow!("[[per_layer.node]] {i} binds {b:?}, which {name} has no param for"))?;
            let fits = match v {
                crate::graph::Bind::Weight { .. } => param.role == kir::Role::Weight,
                crate::graph::Bind::Buffer { .. } => param.role == kir::Role::Activation && param.ty != kir::Scalar::Bf16,
                crate::graph::Bind::Name(n) => n == "layer" && param.role == kir::Role::Value,
                crate::graph::Bind::Number(_) => param.role == kir::Role::Value,
                crate::graph::Bind::State { .. } => false,
            };
            anyhow::ensure!(fits, "[[per_layer.node]] {i} binds {b:?} to the wrong kind of value");
        }
        out.push(kernel.clone());
    }
    Ok(out)
}

pub fn prelude_symbols(spec: &Spec, max_ctx: usize) -> kir::desc::Shape {
    let mut symbols = gated_symbols(spec).num("max_context", max_ctx as f64);
    if let Some(p) = &spec.prelude {
        for (n, v) in p.symbols(0) {
            symbols = symbols.num(&n, v);
        }
        for (n, w) in &p.words {
            symbols = symbols.word(n, w);
        }
    }
    symbols
}

pub fn prelude_kernels(spec: &Spec, max_ctx: usize, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let Some(p) = &spec.prelude else { return Ok(Vec::new()) };
    let symbols = prelude_symbols(spec, max_ctx);
    let registry = crate::kir_kernel::Registry::active();
    let mut built: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, node) in p.node.iter().enumerate() {
        let name = node.kernel.as_deref().ok_or_else(|| anyhow::anyhow!("[[prelude.node]] {i} names no kernel"))?;
        if !built.contains_key(name) {
            let d = registry.get(name).ok_or_else(|| anyhow::anyhow!("[[prelude.node]] {i} names kernel {name:?}, which no kernel description provides"))?;
            built.insert(name.to_string(), d.build(target, &symbols).map_err(|e| anyhow::anyhow!("[[prelude.node]] {i}: {e}"))?);
        }
        let kernel = &built[name];
        for q in kernel.params.iter().filter(|q| !kir::recur::LAUNCH.contains(&q.name)) {
            anyhow::ensure!(node.bind.contains_key(q.name), "[[prelude.node]] {i} does not bind {name} param {:?}", q.name);
        }
        for (b, v) in &node.bind {
            let param = kernel.params.iter().find(|q| q.name == b).ok_or_else(|| anyhow::anyhow!("[[prelude.node]] {i} binds {b:?}, which {name} has no param for"))?;
            let fits = match v {
                crate::graph::Bind::Weight { .. } => param.role == kir::Role::Weight,
                crate::graph::Bind::Buffer { buffer } => {
                    let ints = buffer == "tokens" || p.int_scratch.contains_key(buffer);
                    param.role == kir::Role::Activation && param.ty == if ints { kir::Scalar::I32 } else { kir::Scalar::F32 }
                }
                crate::graph::Bind::State { .. } => param.role == kir::Role::Activation && param.ty == kir::Scalar::F32 && param.access == kir::Access::Write,
                crate::graph::Bind::Number(_) | crate::graph::Bind::Name(_) => param.role == kir::Role::Value,
            };
            anyhow::ensure!(fits, "[[prelude.node]] {i} binds {b:?} to the wrong kind of value");
        }
        out.push(kernel.clone());
    }
    Ok(out)
}

pub fn linear_kernel_shape(la: &crate::spec::LinearAttention) -> kir::linear::LinearShape {
    kir::linear::LinearShape {
        key_heads: la.key_heads,
        value_heads: la.value_heads,
        head_dim: la.key_head_dim,
        conv_kernel: la.conv_kernel,
        tiled_value_heads: la.v_heads == crate::spec::VHeadOrder::Tiled,
        gate_sigmoid: la.gate_sigmoid,
        chunk_single_bf16: la.numerics == crate::spec::Numerics::Act,
    }
}

pub fn linear_kernels(spec: &Spec, target: &kir::Target) -> Result<Vec<kir::linear::LinearKernel>> {
    let Some(la) = spec.linear_attention.as_ref() else {
        if spec.speculative().is_none() {
            return Ok(Vec::new());
        }
        let unused = kir::linear::LinearShape { key_heads: 1, value_heads: 1, head_dim: 16, conv_kernel: 2, tiled_value_heads: false, gate_sigmoid: false, chunk_single_bf16: false };
        return Ok(vec![kir::linear::LinearKernel::new(target, kir::linear::Family::Draft, unused).map_err(anyhow::Error::msg)?]);
    };
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
    if spec.streamed_mtp() && spec.speculative().is_some() {
        out.push(kir::linear::LinearKernel::new(target, kir::linear::Family::DraftStreams, shape).map_err(anyhow::Error::msg)?);
    }
    Ok(out)
}


pub const KERNELS: &[&str] = &[
    "embed",
    "splice_rows",
    "add_rows",
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
    pub source: kir::KirSource,
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
    } else if spec.gated_residual().is_some() || spec.block_residual.is_some() {
        names.push("clear_branch");
    }
    if spec.norm.residual_scale.is_some() || spec.gated_residual().is_some() {
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

pub fn ops_shape(spec: &Spec) -> kir::ops::OpsShape {
    let (a, d) = (&spec.attention, &spec.dims);
    kir::ops::OpsShape {
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
        weight_decoder: crate::gemm::weight_format(spec.quant).decoder().to_string(),
        head_decoder: crate::gemm::head_format(spec).decoder().to_string(),
    }
}

pub fn gemv_kernels(spec: &Spec, target: &kir::Target, ops: &kir::ops::OpsShape) -> Result<Vec<kir::gemv::GemvKernel>> {
    use kir::gemv::Gemv;
    let (wf, hwf) = (ops.weight_decoder.clone(), ops.head_decoder.clone());
    let (n, h, i, v, q) = (ops.qkv_row, ops.hidden, ops.inter, ops.vocab, ops.q_dim);
    let bias = ops.qkv_bias;
    let gu = if ops.mapped.is_some() { ops.inter } else { 2 * ops.inter };
    let lin = |d: &str, n, k, bias, residual| Gemv::Linear { decoder: d.to_string(), n, k, bias, residual };
    let tiled = |d: &str, n, k, bias, residual| Gemv::Tiled { decoder: d.to_string(), n, k, bias, residual };
    let fused = |name: &'static str, decoder: &str, n: usize, bias: bool, gated: bool| -> Result<(&'static str, Gemv)> {
        let wf = kir::gemm::WeightFormat::from_decoder(decoder).ok_or_else(|| anyhow::anyhow!("{name}: unknown decoder {decoder}"))?;
        let k = crate::gemm::normed(target, name, kir::heavy::Family::Gemv { wf }, n, h, bias, gated)?.compile_gemv(target, crate::gemm::KERNEL_SELECT)?;
        Ok((name, k.op))
    };
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
        fused("qkv_proj_norm", &wf, n, bias, false)?,
        fused("gate_up_act", &wf, 2 * i, false, true)?,
        fused("lm_head_norm", &hwf, v, false, false)?,
    ];
    if let Some(la) = spec.linear_attention.as_ref() {
        ops_list.push(fused("lin_in_norm", &wf, la.in_proj_rows(), false, false)?);
        ops_list.push(fused("lin_ba_norm", "DecBf16", la.ba_rows(), false, false)?);
        ops_list.push(("lin_ba_rows", Gemv::Bf16Rows { n: la.ba_rows(), k: h }));
        ops_list.push(("lin_out_proj", lin(&wf, h, la.value_dim(), false, true)));
    }
    ops_list.into_iter().map(|(name, op)| kir::gemv::GemvKernel::new(target, name, op).map_err(anyhow::Error::msg)).collect()
}

pub const EPILOGUE_FILL: &str = "epilogue_fill";

pub const STORE_GEMVS: [&str; 2] = ["o_proj_store", "down_proj_store"];

pub fn store_gemvs(spec: &Spec, target: &kir::Target, ops: &kir::ops::OpsShape) -> Result<Vec<kir::gemv::GemvKernel>> {
    if !(spec.runtime.store_epilogue && spec.uses_branch()) {
        return Ok(Vec::new());
    }
    let wf = kir::gemm::WeightFormat::from_decoder(&ops.weight_decoder).ok_or_else(|| anyhow::anyhow!("unknown decoder {}", ops.weight_decoder))?;
    let (h, q, i) = (ops.hidden, ops.q_dim, ops.inter);
    [(STORE_GEMVS[0], q, ops.o_bias), (STORE_GEMVS[1], i, false)]
        .into_iter()
        .map(|(name, k, bias)| {
            let mut heavy = kir::heavy::Heavy::new(name, kir::heavy::Family::Gemv { wf }, kir::heavy::Problem { n: h, k, act: kir::Activation::Silu }, None);
            heavy.epilogue = kir::heavy::epilogue(kir::gemm::Epilogue::Residual, bias, h).map_err(anyhow::Error::msg)?;
            Ok(heavy.fill(target, EPILOGUE_FILL, kir::heavy::OUT, 0.0)?.compile_gemv(target, EPILOGUE_FILL)?)
        })
        .collect()
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
        let d = &spec.dims;
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
        let ops = ops_shape(spec);
        let kir_kernels = kir_kernels(spec, target)?;
        let gemms = crate::gemm::gemm_kernels(spec, target)?;
        let described = described_kernels(spec, kv_planes, max_ctx, target)?;
        let mut distinct: Vec<kir::Kernel> = Vec::new();
        for k in described.into_iter().flatten().chain(mixer_kernels(spec, kv_planes, max_ctx, target)?).chain(hyper_kernels(spec, target)?).chain(gated_kernels(spec, target)?).chain(per_layer_kernels(spec, target)?).chain(input_kernels(spec, target)?).chain(prelude_kernels(spec, max_ctx, target)?).chain(residual_kernels(spec, target)?) {
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
        let mut gemvs = gemv_kernels(spec, target, &ops)?;
        let stores = store_gemvs(spec, target, &ops)?;
        let store_names: Vec<&'static str> = stores.iter().map(|k| crate::gemm::intern(&k.name)).collect();
        gemvs.extend(stores);
        let kir_hash = kir::fold_hashes([kir_hash, ggml.hash(), kir::prims::prims_hash(&prims), kir::gemv::gemv_hash(&gemvs)]);
        let mut source = kir::KirSource::join(
            [
                &kir::cuda::emit_ops(&ops).map_err(anyhow::Error::msg)?,
                &kir::prims::emit_cuda(&prims).map_err(anyhow::Error::msg)?,
                &ggml.emit_cuda().map_err(anyhow::Error::msg)?,
                &attn_source,
                &gemm_source,
                &kir::gemv::emit_gemvs(&gemvs).map_err(anyhow::Error::msg)?,
                &kir_source,
            ],
            "\n",
        );
        let mut kernels: Vec<&'static str> = KERNELS.iter().copied().filter(|k| attention_available(k, target)).collect();
        kernels.extend(store_names);
        for k in kir_kernels.iter().chain(&distinct) {
            if !kernels.contains(&k.name) {
                kernels.push(k.name);
            }
            if k.tile_scan() && kernels.contains(&"flash_mma") {
                let tile_smem = smem.get("flash_mma").copied().unwrap_or(0);
                for n in [format!("{}_mma", k.name), format!("{}_mma_dec", k.name)] {
                    let n: &'static str = Box::leak(n.into_boxed_str());
                    kernels.push(n);
                    smem.insert(n, tile_smem);
                }
            }
            if let Some(f) = k.chunk_form() {
                kernels.extend([f.state, f.carry, f.out]);
                smem.insert(f.state, f.state_smem);
                smem.insert(f.out, f.out_smem);
            }
        }
        if la.is_some() || !linear.is_empty() {
            source.append("\n", &linear_source);
            for k in &linear {
                kernels.extend_from_slice(k.family.kernels());
            }
        }
        if la.is_some() {
            kernels.extend_from_slice(GDN_KERNELS);
            smem.insert("lin_ba_rows", 0);
        }
        let mut moe_arch = false;
        let moe = moe_kernels(spec, target)?;
        if !moe.is_empty() {
            source.append("\n", &kir::cuda::emit_moe(&moe).map_err(anyhow::Error::msg)?);
            for k in &moe {
                kernels.extend_from_slice(k.family.kernels());
                smem.extend(k.smem());
                moe_arch |= k.arch_features();
            }
        }
        let kir_hash = kir::fold_hashes([kir_hash, kir::moe::moe_hash(&moe)]);
        let kir_hash = match &spec.duplex {
            Some(d) => {
                let shape = kir::duplex::DuplexShape { hidden: spec.dims.hidden, vocab: spec.dims.vocab, latent: d.latent_dim };
                source.append("\n", &kir::cuda::emit_duplex(&shape).map_err(anyhow::Error::msg)?);
                kernels.extend_from_slice(kir::duplex::KERNELS);
                kir::fold_hashes([kir_hash, shape.hash()])
            }
            None => kir_hash,
        };
        if spec.quant == Quant::Gguf {
            kernels.push("gg_dequant_bf16");
        }
        kernels.extend(gemms.names());
        smem.extend(gemms.smem());
        let arch_features = moe_arch || kir::cuda::needs_arch_features(&gemms.kernels);
        let (source, kir_hash) = if spec.runtime.pdl && target.pdl { (kir::cuda::with_pdl(&source, target.sm_count, spec.runtime.pdl_hoist), kir::fold_hashes([kir_hash, if spec.runtime.pdl_hoist { "pdl-hoist" } else { "pdl" }.to_string()])) } else { (source, kir_hash) };
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
            kir::Language::Cuda => kir::cuda::emit_module(std::slice::from_ref(k)).map(kir::KirSource::into_string),
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
