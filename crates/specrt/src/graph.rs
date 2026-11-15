use crate::plan::{Buf, NormKind, Op, Phase, Weight};
use crate::spec::{Block, Spec};
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
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockGraph {
    pub name: String,
    pub layers: Vec<String>,
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

pub const DENSE: &str = include_str!("../graphs/dense.toml");

fn block_name(block: Block) -> &'static str {
    match block {
        Block::Dense => "dense",
        Block::Full => "full",
        Block::Sliding => "sliding",
        Block::Linear => "linear",
    }
}

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
        other => bail!("unknown buffer {other}"),
    })
}

fn weight(name: &str) -> Result<Weight> {
    Ok(match name {
        "qkv" => Weight::Qkv,
        "o" => Weight::O,
        "gate_up" => Weight::GateUp,
        "down" => Weight::Down,
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
    fn runs_in(&self, phase: Phase) -> bool {
        match self.phase.as_deref() {
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
            "flash" => Op::Flash,
            "decode_attention" => Op::DecodeAttention,
            other => bail!("unknown op {other}"),
        })
    }
}

fn lower(nodes: &[Node], phase: Phase) -> Result<Vec<Op>> {
    nodes.iter().filter(|n| n.runs_in(phase)).map(Node::lower).collect()
}

impl Graph {
    pub fn parse(text: &str) -> Result<Graph> {
        let graph: Graph = toml::from_str(text)?;
        if graph.kind != "graph" {
            bail!("{} is not a graph spec", graph.name);
        }
        Ok(graph)
    }

    pub fn head(&self, phase: Phase) -> Result<Vec<Op>> {
        lower(&self.head, phase)
    }

    pub fn tail(&self, phase: Phase) -> Result<Vec<Op>> {
        lower(&self.tail, phase)
    }

    pub fn block_ops(&self, block: Block, phase: Phase) -> Result<Vec<Op>> {
        let name = block_name(block);
        let found = self.block.iter().find(|b| b.layers.iter().any(|l| l == name));
        match found {
            Some(b) => lower(&b.node, phase),
            None => bail!("graph {} has no block for {name} layers", self.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Plan;

    fn spec(config: &str) -> Spec {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(config);
        Spec::describe_hf_config(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn the_dense_graph_lowers_to_the_dense_plan() {
        let graph = Graph::parse(DENSE).unwrap();
        for config in ["qwen3-0.6b.config.json", "llama-3.2-1b.config.json", "qwen2.5-0.5b.config.json"] {
            let spec = spec(config);
            for phase in [Phase::Prefill, Phase::Decode] {
                let plan = Plan::build(&spec, phase, true);
                assert_eq!(graph.head(phase).unwrap(), plan.head, "{config}");
                assert_eq!(graph.tail(phase).unwrap(), plan.tail, "{config}");
                assert_eq!(graph.block_ops(Block::Dense, phase).unwrap(), plan.attention, "{config}");
                assert_eq!(graph.block_ops(Block::Full, phase).unwrap(), plan.attention, "{config}");
            }
        }
    }

    #[test]
    fn an_unknown_op_is_an_error() {
        let text = DENSE.replace("op = \"flash\"", "op = \"nope\"");
        let graph = Graph::parse(&text).unwrap();
        assert!(graph.block_ops(Block::Dense, Phase::Prefill).is_err());
    }
}
