use crate::plan::{Buf, NormKind, Op, Phase, Plan, PlanBlock, Weight};
use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub op: String,
    #[serde(default)]
    pub weight: Option<String>,
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

fn norm(name: &str) -> Result<NormKind> {
    Ok(match name {
        "input_layernorm.weight" => NormKind::Input,
        "post_attention_layernorm.weight" => NormKind::PostAttention,
        "norm.weight" => NormKind::Final,
        other => bail!("unknown norm weight {other}"),
    })
}

impl Node {
    fn runs_in(&self, phase: Phase, tensor_core: bool) -> bool {
        let target = match self.when.as_deref() {
            None => true,
            Some("tensor_core") => tensor_core,
            Some("legacy") => !tensor_core,
            Some(_) => false,
        };
        target && match self.phase.as_deref() {
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
                input: buffer(&field(&self.input, "in")?)?,
                output: buffer(&field(&self.out, "out")?)?,
            },
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

fn lower(nodes: &[Node], phase: Phase, tensor_core: bool) -> Result<Vec<Op>> {
    nodes.iter().filter(|n| n.runs_in(phase, tensor_core)).map(Node::lower).collect()
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

    pub fn plan(&self, phase: Phase, tensor_core: bool) -> Result<Plan> {
        let mut blocks = Vec::new();
        for b in &self.block {
            let sparse = match b.mlp.as_deref() {
                None | Some("dense") => false,
                Some("sparse") => true,
                Some(other) => bail!("block {} has unknown mlp {other}", b.name),
            };
            blocks.push(PlanBlock { layers: b.layers.clone(), sparse, ops: lower(&b.node, phase, tensor_core)? });
        }
        Ok(Plan { phase, tensor_core, head: lower(&self.head, phase, tensor_core)?, blocks, tail: lower(&self.tail, phase, tensor_core)? })
    }
}

