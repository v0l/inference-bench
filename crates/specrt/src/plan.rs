use crate::codegen::{WeightKind, kernel_for};
use crate::spec::{Block, Quant, Spec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buf {
    X,
    Xn,
    XnPlanes,
    Qkv,
    Attn,
    AttnPlanes,
    GateUp,
    Act,
    ActPlanes,
    PreLin,
    PreBa,
    GdnPlanes,
    Logits,
}

impl Buf {
    pub fn label(self) -> &'static str {
        match self {
            Buf::X => "x",
            Buf::Xn => "xn",
            Buf::XnPlanes => "xn planes",
            Buf::Qkv => "qkv",
            Buf::Attn => "attn",
            Buf::AttnPlanes => "attn planes",
            Buf::GateUp => "gate_up",
            Buf::Act => "act",
            Buf::ActPlanes => "act planes",
            Buf::PreLin => "lin",
            Buf::PreBa => "ba",
            Buf::GdnPlanes => "gdn planes",
            Buf::Logits => "logits",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Qkv,
    O,
    GateUp,
    Down,
    LinIn,
    LinBa,
    LinOut,
    LmHead,
}

impl Weight {
    pub fn name(self) -> &'static str {
        match self {
            Weight::Qkv => "qkv",
            Weight::O => "o",
            Weight::GateUp => "gate_up",
            Weight::Down => "down",
            Weight::LinIn => "lin_in",
            Weight::LinBa => "lin_ba",
            Weight::LinOut => "lin_out",
            Weight::LmHead => "lm_head",
        }
    }

    pub fn rows(self, spec: &Spec) -> usize {
        let la = spec.linear_attention.as_ref();
        match self {
            Weight::Qkv => spec.qkv_rows(),
            Weight::GateUp => 2 * spec.dims.intermediate,
            Weight::O | Weight::Down | Weight::LinOut => spec.dims.hidden,
            Weight::LinIn => la.map_or(0, |l| l.in_proj_rows()),
            Weight::LinBa => la.map_or(0, |l| l.ba_rows()),
            Weight::LmHead => spec.dims.vocab,
        }
    }

    pub fn quant(self, spec: &Spec, layer: usize) -> Quant {
        match self {
            Weight::Qkv | Weight::O | Weight::LinIn | Weight::LinOut => spec.quant,
            Weight::GateUp | Weight::Down => spec.mlp_quant(layer),
            Weight::LinBa => Quant::Bf16,
            Weight::LmHead => spec.lm_head_quant(),
        }
    }

    pub fn tensors(self, spec: &Spec) -> Vec<String> {
        let w = |m: &str| format!("{m}.weight");
        match self {
            Weight::Qkv => {
                let mut t: Vec<String> = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"].iter().map(|m| w(m)).collect();
                if spec.attention.qkv_bias {
                    t.extend(["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"].map(String::from));
                }
                t
            }
            Weight::O => vec![w("self_attn.o_proj")],
            Weight::GateUp => vec![w("mlp.gate_proj"), w("mlp.up_proj")],
            Weight::Down => vec![w("mlp.down_proj")],
            Weight::LinIn => vec![w("linear_attn.in_proj_qkv"), w("linear_attn.in_proj_z")],
            Weight::LinBa => vec![w("linear_attn.in_proj_b"), w("linear_attn.in_proj_a")],
            Weight::LinOut => vec![w("linear_attn.out_proj")],
            Weight::LmHead if spec.dims.tie_embeddings => vec![w("embed_tokens")],
            Weight::LmHead => vec![w("lm_head")],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Input,
    PostAttention,
    Final,
}

impl NormKind {
    pub fn tensor(self) -> &'static str {
        match self {
            NormKind::Input => "input_layernorm.weight",
            NormKind::PostAttention => "post_attention_layernorm.weight",
            NormKind::Final => "norm.weight",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Embed,
    Norm { which: NormKind, into: Buf },
    Linear { weight: Weight, input: Buf, output: Buf },
    KvRope,
    QkRope,
    Flash,
    FlashLegacy,
    DecodeAttention,
    Swiglu,
    GdnPrefill,
    GdnDecode,
}

pub fn weight_kind(q: Quant) -> WeightKind {
    match q {
        Quant::Bf16 => WeightKind::Bf16,
        Quant::Q8 => WeightKind::Q8,
        Quant::Fp8Block128 | Quant::Fp8Channel => WeightKind::Fp8,
        Quant::Nvfp4G16 => WeightKind::Fp4,
        Quant::Gguf => WeightKind::Ggml,
    }
}

impl Op {
    pub fn label(&self) -> String {
        match self {
            Op::Embed => "embed".into(),
            Op::Norm { which, .. } => match which {
                NormKind::Input => "input norm".into(),
                NormKind::PostAttention => "post attention norm".into(),
                NormKind::Final => "final norm".into(),
            },
            Op::Linear { weight, .. } => weight.name().into(),
            Op::KvRope | Op::QkRope => "norm and rope".into(),
            Op::Flash | Op::FlashLegacy => "attention".into(),
            Op::DecodeAttention => "decode attention".into(),
            Op::Swiglu => "swiglu".into(),
            Op::GdnPrefill | Op::GdnDecode => "delta rule".into(),
        }
    }

    pub fn flow(&self) -> String {
        match self {
            Op::Embed => "tokens to x".into(),
            Op::Norm { into, .. } => format!("x to {}", into.label()),
            Op::Linear { input, output, .. } => format!("{} to {}", input.label(), output.label()),
            Op::KvRope | Op::QkRope => "qkv to kv cache".into(),
            Op::Flash | Op::FlashLegacy | Op::DecodeAttention => "qkv and kv cache to attn".into(),
            Op::Swiglu => "gate_up to act".into(),
            Op::GdnPrefill | Op::GdnDecode => "lin, ba and state to gdn".into(),
        }
    }

    pub fn tensors(&self, spec: &Spec) -> Vec<String> {
        match self {
            Op::Embed => vec!["embed_tokens.weight".into()],
            Op::Norm { which, .. } => vec![which.tensor().into()],
            Op::Linear { weight, .. } => weight.tensors(spec),
            Op::KvRope | Op::QkRope | Op::Flash | Op::DecodeAttention if spec.attention.qk_norm => {
                vec!["self_attn.q_norm.weight".into(), "self_attn.k_norm.weight".into()]
            }
            Op::GdnPrefill | Op::GdnDecode => ["conv1d.weight", "A_log", "dt_bias", "norm.weight"].iter().map(|t| format!("linear_attn.{t}")).collect(),
            _ => Vec::new(),
        }
    }

    pub fn checkpoint_tensors(&self, spec: &Spec, root: &str, layer: Option<usize>) -> Vec<String> {
        let tied_or_root = |t: &str| format!("{root}.{t}");
        self.tensors(spec)
            .into_iter()
            .map(|t| match (layer, &t) {
                (Some(i), _) => format!("{root}.layers.{i}.{t}"),
                (None, t) if t == "lm_head.weight" => t.clone(),
                (None, t) => tied_or_root(t),
            })
            .collect()
    }

    pub fn quant(&self, spec: &Spec, layer: usize) -> Option<Quant> {
        match self {
            Op::Linear { weight, .. } => Some(weight.quant(spec, layer)),
            _ => None,
        }
    }

    pub fn kernels(&self, spec: &Spec, layer: usize, tokens: usize, sms: usize) -> Vec<String> {
        let names = |k: &[&str]| k.iter().map(|s| s.to_string()).collect();
        match self {
            Op::Embed => names(&["embed"]),
            Op::Norm { into, .. } if matches!(into, Buf::XnPlanes) => names(&["rms_split"]),
            Op::Norm { .. } => names(&["rmsnorm"]),
            Op::Linear { weight, .. } => {
                let w = *weight;
                let kind = weight_kind(w.quant(spec, layer));
                let choice = kernel_for(spec, w.name(), w.rows(spec), tokens, sms, kind);
                let mut out: Vec<String> = choice.quant.into_iter().map(String::from).collect();
                out.push(choice.name);
                out.extend(choice.reduce);
                out
            }
            Op::KvRope => names(&["kv_norm_rope"]),
            Op::QkRope => names(&["qk_norm_rope"]),
            Op::Flash => names(&["flash_mma"]),
            Op::FlashLegacy => names(&["flash_attention"]),
            Op::DecodeAttention => names(&["qk_norm_rope", "batch_decode_attention", "batch_decode_combine"]),
            Op::Swiglu => names(&["swiglu"]),
            Op::GdnPrefill => names(&[
                "gdn_conv_prefill",
                "gdn_conv_keep",
                "gdn_qk_prep",
                "gdn_gates",
                "gdn_chunk_prep",
                "gdn_chunk_scan",
                "gdn_gated_norm",
            ]),
            Op::GdnDecode => names(&["gdn_conv_step", "gdn_step", "split_rows"]),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub phase: Phase,
    pub tensor_core: bool,
    pub head: Vec<Op>,
    pub attention: Vec<Op>,
    pub linear: Vec<Op>,
    pub tail: Vec<Op>,
}

impl Plan {
    pub fn build(spec: &Spec, phase: Phase, tensor_core: bool) -> Plan {
        let norm = |which, into| Op::Norm { which, into };
        let lin = |weight, input, output| Op::Linear { weight, input, output };
        if !tensor_core {
            return Plan {
                phase,
                tensor_core,
                head: vec![Op::Embed],
                attention: vec![
                    norm(NormKind::Input, Buf::Xn),
                    lin(Weight::Qkv, Buf::Xn, Buf::Qkv),
                    Op::QkRope,
                    Op::FlashLegacy,
                    lin(Weight::O, Buf::Attn, Buf::X),
                    norm(NormKind::PostAttention, Buf::Xn),
                    lin(Weight::GateUp, Buf::Xn, Buf::GateUp),
                    Op::Swiglu,
                    lin(Weight::Down, Buf::Act, Buf::X),
                ],
                linear: Vec::new(),
                tail: vec![norm(NormKind::Final, Buf::Xn), lin(Weight::LmHead, Buf::Xn, Buf::Logits)],
            };
        }
        let mlp = [
            norm(NormKind::PostAttention, Buf::XnPlanes),
            lin(Weight::GateUp, Buf::XnPlanes, Buf::ActPlanes),
            lin(Weight::Down, Buf::ActPlanes, Buf::X),
        ];
        let prefill = phase == Phase::Prefill;
        let mut attention = vec![norm(NormKind::Input, Buf::XnPlanes), lin(Weight::Qkv, Buf::XnPlanes, Buf::Qkv)];
        if prefill {
            attention.extend([Op::KvRope, Op::Flash]);
        } else {
            attention.push(Op::DecodeAttention);
        }
        attention.push(lin(Weight::O, Buf::AttnPlanes, Buf::X));
        attention.extend(mlp);
        let mut linear = vec![
            norm(NormKind::Input, Buf::XnPlanes),
            lin(Weight::LinIn, Buf::XnPlanes, Buf::PreLin),
            lin(Weight::LinBa, Buf::XnPlanes, Buf::PreBa),
            if prefill { Op::GdnPrefill } else { Op::GdnDecode },
            lin(Weight::LinOut, Buf::GdnPlanes, Buf::X),
        ];
        linear.extend(mlp);
        Plan {
            phase,
            tensor_core,
            head: vec![Op::Embed],
            attention,
            linear: if spec.linear_attention.is_some() { linear } else { Vec::new() },
            tail: vec![norm(NormKind::Final, Buf::XnPlanes), lin(Weight::LmHead, Buf::XnPlanes, Buf::Logits)],
        }
    }

    pub fn layer(&self, block: Block) -> &[Op] {
        if block.is_linear() { &self.linear } else { &self.attention }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::Structure;
    use std::path::Path;

    fn fixtures() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn described(config: &str, header: &str) -> (Spec, Structure) {
        let json = std::fs::read_to_string(fixtures().join(config)).unwrap();
        let mut spec = Spec::describe_hf_config(&json).unwrap();
        let structure = Structure::from_json(&std::fs::read_to_string(fixtures().join("headers").join(header)).unwrap()).unwrap();
        spec.apply_structure(&structure);
        (spec, structure)
    }

    fn assert_tensors_exist(spec: &Spec, structure: &Structure) {
        for tensor_core in [true, false] {
            for phase in [Phase::Prefill, Phase::Decode] {
                let plan = Plan::build(spec, phase, tensor_core);
                for op in plan.head.iter().chain(&plan.tail) {
                    for t in op.tensors(spec) {
                        assert!(structure.has_outside_tensor(&t), "{} reads {t}, which the checkpoint does not have", op.label());
                    }
                }
                let present: Vec<usize> = structure.main().unwrap().layers.iter().map(|l| l.index).filter(|&i| i < spec.layers.count).collect();
                assert!(!present.is_empty());
                for i in present {
                    for op in plan.layer(spec.block(i)) {
                        for t in op.tensors(spec) {
                            assert!(structure.layer_has_tensor(i, &t), "layer {i} {} reads {t}, which the checkpoint does not have", op.label());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_op_weight_exists_in_the_qwen3_checkpoint() {
        let (spec, structure) = described("qwen3-0.6b.config.json", "qwen3-0.6b.json");
        assert_tensors_exist(&spec, &structure);
    }

    #[test]
    fn every_op_weight_exists_in_the_hybrid_27b_checkpoint() {
        let (spec, structure) = described("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json");
        assert_tensors_exist(&spec, &structure);
    }

    #[test]
    fn every_op_weight_exists_in_the_llama_checkpoint() {
        let (spec, structure) = described("llama-3.2-1b.config.json", "llama-3.2-1b.json");
        assert_tensors_exist(&spec, &structure);
    }

    #[test]
    fn checkpoint_tensor_names_resolve_against_the_structure() {
        for (config, header) in [
            ("qwen3-0.6b.config.json", "qwen3-0.6b.json"),
            ("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json"),
            ("llama-3.2-1b.config.json", "llama-3.2-1b.json"),
        ] {
            let (spec, structure) = described(config, header);
            let root = structure.root().unwrap();
            let plan = Plan::build(&spec, Phase::Prefill, true);
            for op in plan.head.iter().chain(&plan.tail) {
                for t in op.checkpoint_tensors(&spec, root, None) {
                    assert!(structure.contains(&t), "{t} missing in {header}");
                }
            }
            let layers: Vec<usize> = structure.main().unwrap().layers.iter().map(|l| l.index).filter(|&i| i < spec.layers.count).collect();
            for i in layers {
                for op in plan.layer(spec.block(i)) {
                    for t in op.checkpoint_tensors(&spec, root, Some(i)) {
                        assert!(structure.contains(&t), "{t} missing in {header}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_missing_tensor_is_reported() {
        let (spec, structure) = described("qwen3-0.6b.config.json", "qwen3-0.6b.json");
        assert!(!structure.layer_has_tensor(0, "linear_attn.out_proj.weight"));
        assert!(!structure.has_outside_tensor("nothing.weight"));
        let _ = spec;
    }

    #[test]
    fn the_hybrid_plan_has_a_linear_and_an_attention_layer() {
        let (spec, _) = described("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json");
        let plan = Plan::build(&spec, Phase::Prefill, true);
        assert!(plan.layer(Block::Linear).contains(&Op::GdnPrefill));
        assert!(plan.layer(Block::Full).contains(&Op::Flash));
        let decode = Plan::build(&spec, Phase::Decode, true);
        assert!(decode.layer(Block::Linear).contains(&Op::GdnDecode));
        assert!(decode.layer(Block::Full).contains(&Op::DecodeAttention));
    }

    #[test]
    fn linear_kernels_come_from_kernel_for() {
        let (spec, _) = described("qwen3-0.6b.config.json", "qwen3-0.6b.json");
        let op = Op::Linear { weight: Weight::Qkv, input: Buf::XnPlanes, output: Buf::Qkv };
        let direct = kernel_for(&spec, "qkv", spec.qkv_rows(), 512, 84, WeightKind::Bf16);
        assert_eq!(op.kernels(&spec, 0, 512, 84), vec![direct.name]);
    }
}
