use crate::spec::Spec;
use crate::plan::{Buf, QuantSource, NormKind, Op, Phase, Plan, PlanBlock, Weight};
use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub op: String,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default, rename = "in")]
    pub input: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub when: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockGraph {
    pub name: String,
    pub layers: Vec<String>,
    #[serde(default)]
    pub mlp: Option<String>,
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub node: Vec<Node>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub head: Vec<Node>,
    #[serde(default)]
    pub block: Vec<BlockGraph>,
    #[serde(default)]
    pub tail: Vec<Node>,
}

pub const DECODER: &str = include_str!("../graphs/decoder.toml");

fn buffer(name: &str) -> Result<Buf> {
    Ok(match name {
        "x" => Buf::X,
        "xn" => Buf::Xn,
        "xn_planes" => Buf::XnPlanes,
        "qkv" => Buf::Qkv,
        "attn" => Buf::Attn,
        "attn_planes" => Buf::AttnPlanes,
        "gate_up" => Buf::GateUp,
        "act" => Buf::Act,
        "act_planes" => Buf::ActPlanes,
        "logits" => Buf::Logits,
        "lin" => Buf::PreLin,
        "ba" => Buf::PreBa,
        "gdn_planes" => Buf::GdnPlanes,
        "branch" => Buf::Branch,
        other => bail!("unknown buffer {other}"),
    })
}

fn weight(name: &str) -> Result<Weight> {
    Ok(match name {
        "qkv" => Weight::Qkv,
        "o" => Weight::O,
        "gate_up" => Weight::GateUp,
        "down" => Weight::Down,
        "lin_in" => Weight::LinIn,
        "lin_ba" => Weight::LinBa,
        "lin_out" => Weight::LinOut,
        "lm_head" => Weight::LmHead,
        other => bail!("unknown weight {other}"),
    })
}

fn quant_source(name: &str) -> Result<QuantSource> {
    Ok(match name {
        "attn" => QuantSource::Attn,
        "mlp" => QuantSource::Mlp,
        "bf16" => QuantSource::Bf16,
        "head" => QuantSource::Head,
        other => bail!("unknown quant source {other}"),
    })
}

fn norm(name: &str) -> Result<NormKind> {
    Ok(match name {
        "input_layernorm.weight" => NormKind::Input,
        "post_attention_layernorm.weight" => NormKind::PostAttention,
        "pre_feedforward_layernorm.weight" => NormKind::PreMlp,
        "post_feedforward_layernorm.weight" => NormKind::PostMlp,
        "norm.weight" => NormKind::Final,
        other => bail!("unknown norm weight {other}"),
    })
}

impl Node {
    fn runs_in(&self, phase: Phase, tensor_core: bool, spec: &Spec) -> bool {
        holds(self.when.as_deref(), tensor_core, spec) && match self.phase.as_deref() {
            None => true,
            Some("prefill") => phase == Phase::Prefill,
            Some("decode") => phase == Phase::Decode,
            Some(_) => false,
        }
    }

    fn lower(&self) -> Result<Op> {
        let field = |v: &Option<String>, what: &str| v.clone().with_context(|| format!("{} needs {what}", self.op));
        Ok(match self.op.as_str() {
            "embed" => Op::Embed,
            "rmsnorm" => Op::Norm { which: norm(&field(&self.weight, "weight")?)?, into: buffer(&field(&self.out, "out")?)? },
            "linear" => Op::Linear {
                weight: weight(&field(&self.weight, "weight")?)?,
                quant: quant_source(&field(&self.quant, "quant")?)?,
                input: buffer(&field(&self.input, "in")?)?,
                output: buffer(&field(&self.out, "out")?)?,
            },
            "norm_add" => Op::NormAdd { which: norm(&field(&self.weight, "weight")?)?, from: buffer(&field(&self.input, "in")?)? },
            "logit_softcap" => Op::LogitSoftcap,
            "clear" => Op::Clear { buf: buffer(&field(&self.out, "out")?)? },
            "kv_rope" => Op::KvRope,
            "qk_rope" => Op::QkRope,
            "flash_legacy" => Op::FlashLegacy,
            "swiglu" => Op::Swiglu,
            "gdn_prefill" => Op::GdnPrefill,
            "gdn_decode" => Op::GdnDecode,
            "moe_route" => Op::MoeRoute,
            "moe_experts" => Op::MoeExperts,
            "moe_combine" => Op::MoeCombine,
            "flash" => Op::Flash,
            "decode_attention" => Op::DecodeAttention,
            other => bail!("unknown op {other}"),
        })
    }
}

fn holds(when: Option<&str>, tensor_core: bool, spec: &Spec) -> bool {
    when.is_none_or(|w| {
        w.split('+').all(|c| match c {
            "tensor_core" => tensor_core,
            "legacy" => !tensor_core,
            "sandwich" => spec.norm.sandwich,
            "plain" => !spec.norm.sandwich,
            "logit_softcap" => spec.dims.logit_softcap.is_some(),
            _ => false,
        })
    })
}

fn lower(nodes: &[Node], phase: Phase, tensor_core: bool, spec: &Spec) -> Result<Vec<Op>> {
    nodes.iter().filter(|n| n.runs_in(phase, tensor_core, spec)).map(Node::lower).collect()
}

impl Graph {
    pub fn decoder() -> &'static Graph {
        static DECODER_GRAPH: std::sync::OnceLock<Graph> = std::sync::OnceLock::new();
        DECODER_GRAPH.get_or_init(|| Graph::parse(DECODER).expect("the embedded decoder graph parses"))
    }

    pub fn parse(text: &str) -> Result<Graph> {
        let graph: Graph = toml::from_str(text)?;
        if graph.kind != "graph" {
            bail!("{} is not a graph spec", graph.name);
        }
        Ok(graph)
    }

    pub fn plan(&self, spec: &Spec, phase: Phase, tensor_core: bool) -> Result<Plan> {
        let mut blocks = Vec::new();
        for b in self.block.iter().filter(|b| holds(b.when.as_deref(), tensor_core, spec)) {
            let sparse = match b.mlp.as_deref() {
                None | Some("dense") => false,
                Some("sparse") => true,
                Some(other) => bail!("block {} has unknown mlp {other}", b.name),
            };
            blocks.push(PlanBlock { layers: b.layers.clone(), sparse, ops: lower(&b.node, phase, tensor_core, spec)? });
        }
        Ok(Plan { phase, tensor_core, head: lower(&self.head, phase, tensor_core, spec)?, blocks, tail: lower(&self.tail, phase, tensor_core, spec)? })
    }
}

