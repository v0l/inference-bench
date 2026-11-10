use crate::codegen::render;
use crate::spec::{ActQuant, Quant, RopeScalingKind, Spec};
use anyhow::Result;
use std::collections::BTreeMap;

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

const COMMON: &str = include_str!("../templates/wgsl/common.wgsl");
const DECODE: &str = include_str!("../templates/wgsl/decode.wgsl");

fn float(v: f64) -> String {
    format!("{v:e}")
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
    if spec.attention.head_dim > 256 {
        return Some("head_dim above 256 is not yet on wgpu".into());
    }
    if spec.rotary_dim() > spec.attention.head_dim || !spec.rotary_dim().is_multiple_of(2) {
        return Some("rotary_dim must be even and at most head_dim".into());
    }
    None
}

pub fn generate(spec: &Spec, max_ctx: usize) -> Result<Vec<WgslKernel>> {
    let a = &spec.attention;
    let d = &spec.dims;
    let scaling = spec.rope.scaling.as_ref();
    let rope_kind = match scaling.map(|s| s.kind) {
        None => 0,
        Some(RopeScalingKind::Linear) => 1,
        Some(RopeScalingKind::Llama3) => 2,
    };
    let opt = |f: fn(&crate::spec::RopeScaling) -> Option<f64>| float(scaling.and_then(f).unwrap_or(1.0));
    let act = match spec.mlp.activation {
        crate::spec::Activation::Swiglu => 0,
        crate::spec::Activation::Geglu => 1,
    };
    let mut vars: BTreeMap<&str, String> = [
        ("HIDDEN", d.hidden.to_string()),
        ("INTER", d.intermediate.to_string()),
        ("VOCAB", d.vocab.to_string()),
        ("HEADS", a.heads.to_string()),
        ("KV_HEADS", a.kv_heads.to_string()),
        ("HEAD_DIM", a.head_dim.to_string()),
        ("GROUP", (a.heads / a.kv_heads).to_string()),
        ("QK_NORM", a.qk_norm.to_string()),
        ("EPS", float(spec.norm.eps)),
        ("THETA", float(spec.rope.theta)),
        ("ROPE_KIND", rope_kind.to_string()),
        ("ROPE_FACTOR", float(scaling.map_or(1.0, |s| s.factor))),
        ("ROPE_LOW_FREQ", opt(|s| s.low_freq_factor)),
        ("ROPE_HIGH_FREQ", opt(|s| s.high_freq_factor)),
        ("ROPE_ORIG_POS", opt(|s| s.original_max_positions)),
        ("ACT_KIND", act.to_string()),
        ("MAX_CTX", max_ctx.to_string()),
        ("Q_DIM", spec.q_dim().to_string()),
        ("QKV_ROW", spec.qkv_rows().to_string()),
        ("ROT", spec.rotary_dim().to_string()),
    ]
    .into_iter()
    .collect();
    use Access::*;
    let build = |name: &'static str, body: &str, decode: bool, bindings: &[Access], vars: &BTreeMap<&str, String>| -> Result<WgslKernel> {
        let mut source = render(COMMON, vars)?;
        source.push_str(&render(body, vars)?);
        if decode {
            source.push_str(DECODE);
        }
        Ok(WgslKernel { name, source, bindings: bindings.to_vec() })
    };
    let mut out = vec![
        build("embed", include_str!("../templates/wgsl/embed.wgsl"), true, &[Read, Read, Read, Write, Uniform], &vars)?,
        build("rmsnorm", include_str!("../templates/wgsl/rmsnorm.wgsl"), false, &[Read, Read, Write, Uniform], &vars)?,
        build("rope", include_str!("../templates/wgsl/rope.wgsl"), false, &[Write, Read, Read, Write, Write, Uniform], &vars)?,
        build("attention", include_str!("../templates/wgsl/attention.wgsl"), false, &[Read, Read, Read, Write, Uniform], &vars)?,
        build("swiglu", include_str!("../templates/wgsl/swiglu.wgsl"), false, &[Read, Write, Uniform], &vars)?,
        build("bias", include_str!("../templates/wgsl/bias.wgsl"), false, &[Write, Read, Uniform], &vars)?,
        build("argmax", include_str!("../templates/wgsl/argmax.wgsl"), false, &[Read, Write, Uniform], &vars)?,
        build("dequant", include_str!("../templates/wgsl/dequant.wgsl"), true, &[Read, Read, Write, Uniform], &vars)?,
    ];
    for (name, tokens) in [("linear", 1), ("linear_t8", LINEAR_TOKENS)] {
        vars.insert("TT", tokens.to_string());
        vars.insert("RED", (tokens * LINEAR_WORKGROUP).to_string());
        out.push(build(name, include_str!("../templates/wgsl/linear.wgsl"), true, &[Read, Write, Read, Read, Uniform], &vars)?);
    }
    Ok(out)
}
