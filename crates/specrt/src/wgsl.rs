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
pub const ROW_SPAN: u32 = 32768;
pub const ELEMENT_SPAN: u32 = 8388608;


pub fn kernels_for(op: &Op, spec: &Spec, tokens: usize) -> Result<Vec<&'static str>> {
    Ok(match op {
        Op::Embed => vec!["embed"],
        Op::Splice => anyhow::bail!("vision is not yet on wgpu"),
        Op::Norm { .. } => vec!["rmsnorm"],
        Op::Linear { weight, .. } => {
            let mut k = vec![if tokens == 1 { "linear" } else { "linear_t8" }];
            if *weight == Weight::Qkv && spec.attention.qkv_bias {
                k.push("bias");
            }
            k
        }
        Op::KvRope | Op::QkRope => vec!["rope"],
        Op::Flash | Op::FlashLegacy | Op::DecodeAttention => vec!["attention"],
        Op::Swiglu => vec!["swiglu"],
        Op::NormAdd { .. } | Op::LogitSoftcap | Op::LogitScale | Op::Split { .. } | Op::QkNorm | Op::ScaledAdd { .. } | Op::Clear { .. } => anyhow::bail!("this op is not yet on wgpu"),
        Op::GdnPrefill | Op::GdnDecode | Op::GdnVerify => anyhow::bail!("delta-net is not yet on wgpu"),
        Op::MtpIn | Op::MtpGather | Op::MtpNorm | Op::MtpDraft => anyhow::bail!("MTP is not yet on wgpu"),
        Op::MoeRoute | Op::MoeExperts | Op::MoeCombine => anyhow::bail!("mixture of experts is not yet on wgpu"),
    })
}

pub fn plan(spec: &Spec, phase: Phase) -> Plan {
    Plan::build(spec, phase, false)
}

pub fn unsupported(spec: &Spec) -> Option<String> {
    if spec.linear_attention.is_some() {
        return Some("delta-net layers are not yet on wgpu".into());
    }
    if spec.speculative().is_some() {
        return Some("MTP is not yet on wgpu".into());
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
        theta: spec.rope.theta,
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
