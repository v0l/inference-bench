use crate::spec::Spec;
use crate::vision::VisionConfig;
use crate::vision_encoder::VisionOp;
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
    #[serde(default)]
    pub vision_head: Vec<Node>,
    #[serde(default)]
    pub vision_block: Vec<Node>,
    #[serde(default)]
    pub vision_tail: Vec<Node>,
    #[serde(default)]
    pub mtp: Vec<Node>,
    #[serde(default)]
    pub mtp_head: Vec<Node>,
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
        "fc_planes" => Buf::FcPlanes,
        "mtp_logits" => Buf::MtpLogits,
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
        "mtp_fc" => Weight::MtpFc,
        "mtp_qkv" => Weight::MtpQkv,
        "mtp_o" => Weight::MtpO,
        "mtp_gate_up" => Weight::MtpGateUp,
        "mtp_down" => Weight::MtpDown,
        "mtp_head" => Weight::MtpHead,
        other => bail!("unknown weight {other}"),
    })
}

fn quant_source(name: &str) -> Result<QuantSource> {
    Ok(match name {
        "attn" => QuantSource::Attn,
        "mlp" => QuantSource::Mlp,
        "bf16" => QuantSource::Bf16,
        "head" => QuantSource::Head,
        "draft_head" => QuantSource::DraftHead,
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
            Some("verify") => phase == Phase::Verify,
            Some("rows") => phase != Phase::Prefill,
            Some(_) => false,
        }
    }

    fn lower(&self) -> Result<Op> {
        let field = |v: &Option<String>, what: &str| v.clone().with_context(|| format!("{} needs {what}", self.op));
        Ok(match self.op.as_str() {
            "embed" => Op::Embed,
            "splice" => Op::Splice,
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
            "gdn_verify" => Op::GdnVerify,
            "mtp_in" => Op::MtpIn,
            "mtp_gather" => Op::MtpGather,
            "mtp_norm" => Op::MtpNorm,
            "mtp_draft" => Op::MtpDraft,
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
            "speculative" => spec.speculative().is_some(),
            "vision" => spec.vision.is_some(),
            _ => false,
        })
    })
}

fn lower(nodes: &[Node], phase: Phase, tensor_core: bool, spec: &Spec) -> Result<Vec<Op>> {
    nodes.iter().filter(|n| n.runs_in(phase, tensor_core, spec)).map(Node::lower).collect()
}

fn vision_op(node: &Node, block: usize, deepstack: Option<usize>) -> Result<VisionOp> {
    let which = || -> Result<u8> {
        match node.weight.as_deref() {
            Some("1") => Ok(1),
            Some("2") => Ok(2),
            other => bail!("vnorm needs weight 1 or 2, got {other:?}"),
        }
    };
    Ok(match node.op.as_str() {
        "patch_embed" => VisionOp::PatchEmbed,
        "vnorm" => VisionOp::Norm { block, which: which()? },
        "vqkv" => VisionOp::Qkv { block },
        "vrope" => VisionOp::Rope,
        "vattention" => VisionOp::Attention,
        "vproj" => VisionOp::Proj { block },
        "vmlp_up" => VisionOp::MlpUp { block },
        "vmlp_down" => VisionOp::MlpDown { block },
        "deepstack_merger" => VisionOp::DeepstackMerger { index: deepstack.context("deepstack merger outside a deepstack block")?, after_block: block },
        "vmerger" => VisionOp::Merger,
        other => bail!("unknown vision op {other}"),
    })
}

fn vision_holds(when: Option<&str>, deepstack: Option<usize>) -> bool {
    when.is_none_or(|w| w.split('+').all(|c| c == "deepstack" && deepstack.is_some()))
}

impl Graph {
    pub fn vision_plan(&self, cfg: &VisionConfig) -> Result<Vec<VisionOp>> {
        let mut ops = Vec::new();
        for n in &self.vision_head {
            ops.push(vision_op(n, 0, None)?);
        }
        for block in 0..cfg.depth {
            let deepstack = cfg.deepstack_visual_indexes.iter().position(|&b| b == block);
            for n in self.vision_block.iter().filter(|n| vision_holds(n.when.as_deref(), deepstack)) {
                ops.push(vision_op(n, block, deepstack)?);
            }
        }
        for n in &self.vision_tail {
            ops.push(vision_op(n, 0, None)?);
        }
        Ok(ops)
    }

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
        Ok(Plan { phase, tensor_core, head: lower(&self.head, phase, tensor_core, spec)?, blocks, tail: lower(&self.tail, phase, tensor_core, spec)?, mtp: lower(&self.mtp, phase, tensor_core, spec)?, mtp_head: lower(&self.mtp_head, phase, tensor_core, spec)? })
    }
}

