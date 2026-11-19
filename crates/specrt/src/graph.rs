use crate::spec::Spec;
use crate::vision::VisionConfig;
use crate::vision_encoder::VisionOp;
use crate::plan::{Buf, QuantSource, NormKind, Op, Phase, Plan, PlanBlock, Weight};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Bind {
    Number(f64),
    Name(String),
    Weight { weight: String, len: String },
    Buffer { buffer: String },
    State { state: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub op: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grid: Vec<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub bind: std::collections::BTreeMap<String, Bind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    #[serde(default, rename = "in", skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockGraph {
    pub name: String,
    pub layers: Vec<String>,
    #[serde(default)]
    pub mlp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

    fn lower_described(&self, index: usize) -> Result<Op> {
        ensure!(index < 256, "an attention spec has at most 256 nodes");
        Ok(match self.op.as_str() {
            "qk_rope" => Op::QkRope,
            "decode_rope" => Op::DecodeRope,
            "kernel" => Op::Kernel { node: index as u8 },
            other => bail!("attention node op {other:?} is not qk_rope, decode_rope or kernel"),
        })
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
            "logit_scale" => Op::LogitScale,
            "split" => Op::Split { from: buffer(&field(&self.input, "in")?)?, into: buffer(&field(&self.out, "out")?)? },
            "qk_norm" => Op::QkNorm,
            "add_scaled" => Op::ScaledAdd { from: buffer(&field(&self.input, "in")?)? },
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
            "scaled" => !spec.norm.sandwich && spec.norm.residual_scale.is_some(),
            "pre" => spec.norm.pre,
            "nopre" => !spec.norm.pre,
            "qk_norm_full" => spec.attention.qk_norm_full,
            "branch" => spec.uses_branch(),
            "plain" => !spec.uses_branch(),
            "logit_softcap" => spec.dims.logit_softcap.is_some(),
            "logit_scale" => spec.dims.logit_scale.is_some(),
            "speculative" => spec.speculative().is_some(),
            "vision" => spec.vision.is_some(),
            "moe" => spec.moe.is_some(),
            "described" => !spec.attention.node.is_empty(),
            "stock" => spec.attention.node.is_empty(),
            "dense" => spec.moe.is_none(),
            _ => false,
        })
    })
}

fn lower(nodes: &[Node], phase: Phase, tensor_core: bool, spec: &Spec, block: Option<&str>) -> Result<Vec<Op>> {
    let mut ops = Vec::new();
    for n in nodes.iter().filter(|n| n.runs_in(phase, tensor_core, spec)) {
        if n.op == "attention_nodes" {
            let here = |d: &Node| d.layers.is_empty() || block.is_none_or(|b| d.layers.iter().any(|l| l == b));
            for (i, d) in spec.attention.node.iter().enumerate().filter(|(_, d)| here(d) && d.runs_in(phase, tensor_core, spec)) {
                ops.push(d.lower_described(i)?);
            }
        } else {
            ops.push(n.lower()?);
        }
    }
    Ok(ops)
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
        "vpool" => VisionOp::Pool,
        other => bail!("unknown vision op {other}"),
    })
}

fn vision_holds(when: Option<&str>, deepstack: Option<usize>, cfg: &VisionConfig) -> bool {
    when.is_none_or(|w| {
        w.split('+').all(|c| match c {
            "deepstack" => deepstack.is_some(),
            "pooled" => cfg.pooled(),
            "merged" => !cfg.pooled(),
            _ => false,
        })
    })
}

impl Graph {
    pub fn vision_plan(&self, cfg: &VisionConfig) -> Result<Vec<VisionOp>> {
        let mut ops = Vec::new();
        for n in &self.vision_head {
            ops.push(vision_op(n, 0, None)?);
        }
        for block in 0..cfg.depth {
            let deepstack = cfg.deepstack_visual_indexes.iter().position(|&b| b == block);
            for n in self.vision_block.iter().filter(|n| vision_holds(n.when.as_deref(), deepstack, cfg)) {
                ops.push(vision_op(n, block, deepstack)?);
            }
        }
        for n in self.vision_tail.iter().filter(|n| vision_holds(n.when.as_deref(), None, cfg)) {
            ops.push(vision_op(n, 0, None)?);
        }
        Ok(ops)
    }

    pub fn shards(&self) -> Result<std::collections::BTreeMap<String, crate::tp::Shard>> {
        let mut out = std::collections::BTreeMap::new();
        let nodes = self.head.iter().chain(self.block.iter().flat_map(|b| b.node.iter())).chain(&self.tail).chain(&self.mtp).chain(&self.mtp_head);
        for n in nodes.filter(|n| n.op == "linear") {
            let (Some(w), Some(s)) = (&n.weight, &n.shard) else { bail!("linear node {:?} declares no shard", n.weight) };
            let shard = crate::tp::Shard::parse(s)?;
            if let Some(old) = out.insert(w.clone(), shard) {
                ensure!(old == shard, "weight {w} is sharded two ways");
            }
        }
        Ok(out)
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
            let filtered = spec.attention.node.iter().any(|n| !n.layers.is_empty()) && b.node.iter().any(|n| n.op == "attention_nodes");
            if filtered {
                for name in &b.layers {
                    blocks.push(PlanBlock { layers: vec![name.clone()], sparse, ops: lower(&b.node, phase, tensor_core, spec, Some(name))? });
                }
            } else {
                blocks.push(PlanBlock { layers: b.layers.clone(), sparse, ops: lower(&b.node, phase, tensor_core, spec, None)? });
            }
        }
        Ok(Plan {
            phase,
            tensor_core,
            head: lower(&self.head, phase, tensor_core, spec, None)?,
            blocks,
            tail: lower(&self.tail, phase, tensor_core, spec, None)?,
            mtp: lower(&self.mtp, phase, tensor_core, spec, None)?,
            mtp_head: lower(&self.mtp_head, phase, tensor_core, spec, None)?,
        })
    }
}

