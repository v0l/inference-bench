use crate::plan::{Op, Phase, Plan, Weight};
use crate::spec::{ActQuant, Quant, Spec};
use anyhow::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    Uniform,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WgslKernel {
    pub name: &'static str,
    pub source: String,
    pub bindings: Vec<Access>,
}

pub const LINEAR_TOKENS: usize = 8;
pub const LINEAR_WORKGROUP: usize = 64;
pub const FLASH_TOKENS: u32 = kir::wgsl::FLASH_TOKENS;
pub const FLASH_COOP_QUERIES: u32 = kir::wgsl::FLASH_COOP_QUERIES;
pub use kir::wgsl::SgVariant;
pub const SG_ROWS: u32 = kir::wgsl::SG_ROWS;
pub const TILE_MIN_TOKENS: usize = 32;
pub const TILE_TOKENS: usize = kir::wgsl::TILE_TOKENS as usize;
pub const TILE_ROWS: usize = kir::wgsl::TILE_ROWS as usize;
pub const ROW_SPAN: u32 = 32768;
pub const ELEMENT_SPAN: u32 = 8388608;


pub fn kernels_for(op: &Op, spec: &Spec, tokens: usize) -> Result<Vec<&'static str>> {
    Ok(match op {
        Op::Embed => vec!["embed"],
        Op::Splice => anyhow::bail!("vision is not yet on wgpu"),
        Op::Norm { .. } => vec!["rmsnorm"],
        Op::Linear { weight, .. } => {
            let mut k = vec![if tokens == 1 { "linear" } else if tokens >= TILE_MIN_TOKENS { "linear_tile" } else { "linear_t8" }];
            if *weight == Weight::Qkv && spec.attention.qkv_bias {
                k.push("bias");
            }
            k
        }
        Op::KvRope | Op::QkRope => vec!["rope"],
        Op::DecodeRope => anyhow::bail!("described kernels are not yet on wgpu"),
        Op::Flash | Op::FlashLegacy | Op::DecodeAttention => vec!["attention"],
        Op::Swiglu => vec!["swiglu"],
        Op::Kernel { .. } | Op::MixKernel { .. } | Op::Gated { .. } | Op::Prelude { .. } => anyhow::bail!("described kernels are not yet on wgpu"),
        Op::NormAdd { .. } | Op::LogitSoftcap | Op::LogitScale | Op::Split { .. } | Op::QkNorm | Op::ScaledAdd { .. } | Op::Clear { .. } => anyhow::bail!("this op is not yet on wgpu"),
        Op::GdnPrefill | Op::GdnDecode | Op::GdnVerify => anyhow::bail!("delta-net is not yet on wgpu"),
        Op::MtpIn | Op::MtpGather | Op::MtpNorm | Op::MtpDraft => anyhow::bail!("MTP is not yet on wgpu"),
        Op::MoeRoute | Op::MoeExperts | Op::MoeCombine => anyhow::bail!("mixture of experts is not yet on wgpu"),
        Op::AllReduce | Op::AllReduceNorm { .. } => anyhow::bail!("tensor parallel is not on wgpu"),
        Op::Gemv { .. } | Op::GemvStore { .. } | Op::StepAttention | Op::GdnStep => anyhow::bail!("the single row cuda plan is not on wgpu"),
        Op::HyperExpand | Op::HyperPre { .. } | Op::HyperPost | Op::HyperMean => anyhow::bail!("hyper-connections are not yet on wgpu"),
    })
}

pub fn plan(spec: &Spec, phase: Phase) -> Plan {
    Plan::build(spec, phase, false)
}

pub fn unsupported(spec: &Spec) -> Option<String> {
    if spec.iq_quants {
        return Some("IQ quant types need the CUDA engine".into());
    }
    if spec.linear_attention.is_some() {
        return Some("delta-net layers are not yet on wgpu".into());
    }
    if spec.gated_residual().is_some() {
        return Some("described hyper-connections are not yet on wgpu".into());
    }
    if spec.mixer.is_some() {
        return Some("described mixers are not yet on wgpu".into());
    }
    if spec.hyper.is_some() {
        return Some("hyper-connections are not yet on wgpu".into());
    }
    if spec.layers.split || spec.mlp.mapped.is_some() {
        return Some("mlp-only layers and non-gated mlps are not yet on wgpu".into());
    }
    if spec.speculative().is_some() {
        return Some("MTP is not yet on wgpu".into());
    }
    if !spec.attention.node.is_empty() {
        return Some("described attention kernels are not yet on wgpu".into());
    }
    if spec.attention.qk_norm_unit || spec.rope.style != crate::spec::RopeStyle::Neox {
        return Some("interleaved rope and weightless qk norm are not yet on wgpu".into());
    }
    if spec.attention.output_gate {
        return Some("output-gated attention is not yet on wgpu".into());
    }
    if spec.moe.is_some() {
        return Some("mixture of experts is not yet on wgpu".into());
    }
    if spec.layers.window.is_some() {
        return Some("sliding window attention is not yet on wgpu".into());
    }
    if spec.act_quant != ActQuant::None || spec.mlp.act_quant != ActQuant::None {
        return Some("activation quantisation is not yet on wgpu".into());
    }
    let mut quants = vec![spec.quant, spec.lm_head_quant()];
    quants.extend((0..spec.layers.count).map(|i| spec.mlp_quant(i)));
    for q in quants {
        match q {
            Quant::Bf16 | Quant::Gguf => {}
            Quant::Q8 => return Some("q8 safetensors weights are not yet on wgpu (GGUF Q8_0 is)".into()),
            Quant::Fp8Block128 | Quant::Fp8Channel => return Some("FP8 weights are not yet on wgpu".into()),
            Quant::Nvfp4G16 => return Some("NVFP4 weights are not yet on wgpu".into()),
        }
    }
    if spec.rope_constants().kind == 4 {
        return Some("longrope scaling is not yet on wgpu".into());
    }
    if spec.attention.head_dim > 256 {
        return Some("head_dim above 256 is not yet on wgpu".into());
    }
    if spec.rotary_dim() > spec.attention.head_dim || !spec.rotary_dim().is_multiple_of(2) {
        return Some("rotary_dim must be even and at most head_dim".into());
    }
    None
}

pub fn generate(spec: &Spec, max_ctx: usize, target: &kir::Target) -> Result<Vec<WgslKernel>> {
    let a = &spec.attention;
    let d = &spec.dims;
    let rope = spec.rope_constants();
    let shape = kir::ops::WgslShape {
        hidden: d.hidden,
        inter: d.intermediate,
        vocab: d.vocab,
        heads: a.heads,
        kv_heads: a.kv_heads,
        head_dim: a.head_dim,
        qk_norm: a.qk_norm,
        eps: spec.norm.eps,
        theta: spec.theta(),
        rope_kind: rope.kind,
        rope_factor: rope.factor,
        rope_low: rope.low,
        rope_high: rope.high,
        rope_original: rope.original,
        rope_mscale: rope.mscale,
        gated_gelu: spec.mlp.activation == crate::spec::Activation::Geglu,
        max_ctx,
        q_dim: spec.q_dim(),
        qkv_row: spec.qkv_rows(),
        rotary_dim: spec.rotary_dim(),
    };
    use Access::*;
    use kir::wgsl::OpKind;
    let op = |name: &'static str, kind: OpKind, bindings: &[Access]| WgslKernel { name, source: kir::wgsl::emit_op(kind, &shape), bindings: bindings.to_vec() };
    let mut out = vec![
        op("embed", OpKind::Embed, &[Read, Read, Read, Write, Uniform]),
        op("rope", OpKind::Rope, &[Write, Read, Read, Write, Write, Uniform]),
        op("attention", OpKind::Attention, &[Read, Read, Read, Write, Uniform]),
        op("bias", OpKind::Bias, &[Write, Read, Uniform]),
        op("argmax", OpKind::Argmax, &[Read, Write, Uniform]),
        op("dequant", OpKind::Dequant, &[Read, Read, Write, Uniform]),
    ];
    for k in crate::codegen::kir_kernels(spec, target)? {
        let bindings = kir::wgsl::bindings(&k)
            .into_iter()
            .map(|b| match b {
                kir::wgsl::Binding::Read => Read,
                kir::wgsl::Binding::Write => Write,
                kir::wgsl::Binding::Uniform => Uniform,
            })
            .collect();
        out.push(WgslKernel { name: k.name, source: kir::wgsl::emit_kernel(&k).map_err(anyhow::Error::msg)?, bindings });
    }
    for (name, tokens) in [("linear", 1), ("linear_t8", LINEAR_TOKENS)] {
        out.push(WgslKernel { name, source: kir::wgsl::emit_linear_op(&shape, tokens, LINEAR_WORKGROUP), bindings: vec![Read, Write, Read, Read, Uniform] });
    }
    out.push(WgslKernel { name: "linear_tile", source: kir::wgsl::emit_linear_tile_op(&shape), bindings: vec![Read, Write, Read, Read, Uniform] });
    if target.has(kir::Feature::Subgroups) && target.warp_size == 32 {
        out.push(WgslKernel { name: "flash", source: kir::wgsl::emit_flash_op(&shape), bindings: vec![Read, Read, Read, Write, Uniform] });
        out.push(WgslKernel { name: "flash_decode", source: kir::wgsl::emit_flash_decode_op(&shape), bindings: vec![Read, Read, Read, Write, Uniform] });
        for (name, variant) in [("linear_sg", SgVariant::Any), ("linear_sg_bf16", SgVariant::Bf16), ("linear_sg_q4k", SgVariant::Q4K), ("linear_sg_q6k", SgVariant::Q6K)] {
            out.push(WgslKernel { name, source: kir::wgsl::emit_linear_sg_op(&shape, variant), bindings: vec![Read, Write, Read, Read, Uniform] });
        }
    }
    if target.has(kir::Feature::CoopMatrix) {
        if spec.attention.head_dim.is_multiple_of(8) && spec.attention.head_dim <= 256 {
            out.push(WgslKernel { name: "flash_coop", source: kir::wgsl::emit_flash_coop_op(&shape), bindings: vec![Read, Read, Read, Write, Uniform] });
        }
        for (name, variant) in [("linear_coop", SgVariant::Any), ("linear_coop_bf16", SgVariant::Bf16), ("linear_coop_q4k", SgVariant::Q4K), ("linear_coop_q6k", SgVariant::Q6K)] {
            out.push(WgslKernel { name, source: kir::wgsl::emit_linear_coop_op(&shape, variant), bindings: vec![Read, Write, Read, Read, Uniform] });
        }
    }
    for k in crate::gemm::gemm_kernels(spec, target)?.kernels {
        let bindings = kir::wgsl::gemm_bindings(&k)
            .into_iter()
            .map(|b| match b {
                kir::wgsl::Binding::Read => Read,
                kir::wgsl::Binding::Write => Write,
                kir::wgsl::Binding::Uniform => Uniform,
            })
            .collect();
        let source = kir::wgsl::emit_gemm(&k).map_err(anyhow::Error::msg)?;
        out.push(WgslKernel { name: crate::gemm::intern(&k.name), source, bindings });
    }
    Ok(out)
}

pub fn coop_kernel_for(rows: &[u32]) -> &'static str {
    match sg_kernel_for(rows) {
        "linear_sg_bf16" => "linear_coop_bf16",
        "linear_sg_q4k" => "linear_coop_q4k",
        "linear_sg_q6k" => "linear_coop_q6k",
        _ => "linear_coop",
    }
}

pub fn sg_kernel_for(rows: &[u32]) -> &'static str {
    let first = rows.first().map_or(0, |r| r & 7);
    if rows.iter().any(|r| r & 7 != first) {
        return "linear_sg";
    }
    match first {
        4 => "linear_sg_bf16",
        0 => "linear_sg_q4k",
        2 => "linear_sg_q6k",
        _ => "linear_sg",
    }
}
