use crate::spec::{Block, GatedStage, Spec};
use crate::vision::VisionConfig;
use crate::vision_encoder::VisionOp;
use crate::plan::{Buf, GATED_MODULE, QuantSource, NormKind, Op, Phase, Plan, PlanBlock, ResStage, Weight};
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
    #[serde(default)]
    pub window: Vec<Node>,
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
        "mix" => Buf::Mix,
        "mix_planes" => Buf::MixPlanes,
        "branch" => Buf::Branch,
        "merged" => Buf::Merged,
        "fc_planes" => Buf::FcPlanes,
        "draft_planes" => Buf::DraftPlanes,
        "streams" => Buf::Streams,
        "mtp_logits" => Buf::MtpLogits,
        "latent" => Buf::Latent,
        "latent_planes" => Buf::LatentPlanes,
        other => bail!("unknown buffer {other}"),
    })
}

fn weight(name: &str) -> Result<Weight> {
    Ok(match name {
        "qkv" => Weight::Qkv,
        "o" => Weight::O,
        "gate_up" => Weight::GateUp,
        "gate_up_wide" => Weight::GateUpWide,
        "down" => Weight::Down,
        "down_wide" => Weight::DownWide,
        "lin_in" => Weight::LinIn,
        "lin_ba" => Weight::LinBa,
        "lin_out" => Weight::LinOut,
        "mix_in" => Weight::MixIn,
        "mix_out" => Weight::MixOut,
        "lm_head" => Weight::LmHead,
        "mtp_fc" => Weight::MtpFc,
        "mtp_fc_hidden" => Weight::MtpFcHidden,
        "mtp_qkv" => Weight::MtpQkv,
        "mtp_o" => Weight::MtpO,
        "mtp_gate_up" => Weight::MtpGateUp,
        "mtp_down" => Weight::MtpDown,
        "mtp_head" => Weight::MtpHead,
        "latent_down" => Weight::LatentDown,
        "latent_up" => Weight::LatentUp,
        "shared_gate_up" => Weight::SharedGateUp,
        "shared_down" => Weight::SharedDown,
        "cross_q" => Weight::CrossQ,
        "cross_o" => Weight::CrossO,
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
        "post_feedforward_layernorm_1.weight" => NormKind::PostDense,
        "post_feedforward_layernorm_2.weight" => NormKind::PostExperts,
        "cross_attn_layernorm.weight" => NormKind::Cross,
        "norm.weight" => NormKind::Final,
        other => bail!("unknown norm weight {other}"),
    })
}

impl Node {
    pub fn owner_only(&self) -> bool {
        self.when.as_deref().is_some_and(|w| w.split('+').any(|c| c == "kv_owner"))
    }

    pub fn present_in(&self, block: Block) -> bool {
        self.layers.is_empty() || self.layers.iter().any(|l| l == block.name())
    }

    fn runs_in(&self, phase: Phase, tensor_core: bool, spec: &Spec, kind: Option<Block>, tail: bool) -> bool {
        kind.is_none_or(|b| self.present_in(b)) && holds(self.when.as_deref(), tensor_core, spec, tail) && match self.phase.as_deref() {
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
            "norm_add" => Op::NormAdd { which: norm(&field(&self.weight, "weight")?)?, from: buffer(&field(&self.input, "in")?)?, into: buffer(self.out.as_deref().unwrap_or("x"))? },
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
            "moe_combine" => match self.out.as_deref() {
                None | Some("x") => Op::MoeCombine,
                Some(other) => Op::MoeCombineInto { into: buffer(other)? },
            },
            "hyper_expand" => Op::HyperExpand,
            "hyper_pre" => Op::HyperPre { site: field(&self.weight, "weight")?.parse().with_context(|| format!("hyper_pre weight is a site index, not {:?}", self.weight))? },
            "hyper_post" => Op::HyperPost,
            "hyper_mean" => Op::HyperMean,
            "block_res" => Op::BlockRes {
                stage: match field(&self.weight, "weight")?.as_str() {
                    "enter" => ResStage::Enter,
                    "mixed" => ResStage::Mixed,
                    "done" => ResStage::Done,
                    "tail" => ResStage::Tail,
                    other => bail!("block_res weight {other:?} is not enter, mixed, done or tail"),
                },
            },
            "latent_norm" => Op::LatentNorm,
            "position_add" => Op::PositionAdd,
            "cross_kv" => Op::CrossKv,
            "cross_attention" => Op::CrossAttention { decode: false },
            "cross_attention_rows" => Op::CrossAttention { decode: true },
            "flash" => Op::Flash,
            "decode_attention" => Op::DecodeAttention,
            other => bail!("unknown op {other}"),
        })
    }
}

fn holds(when: Option<&str>, tensor_core: bool, spec: &Spec, tail: bool) -> bool {
    when.is_none_or(|w| {
        w.split('+').all(|c| match c {
            "kv_owner" => !tail,
            "kv_reader" => tail,
            "tensor_core" => tensor_core,
            "legacy" => !tensor_core,
            "sandwich" => spec.norm.sandwich,
            "scaled" => !spec.norm.sandwich && spec.norm.residual_scale.is_some(),
            "pre" => spec.norm.pre,
            "nopre" => !spec.norm.pre,
            "qk_norm_full" => spec.attention.qk_norm_full,
            "branch" => spec.uses_branch() && !spec.layers.split && spec.hyper.is_none() && spec.gated_residual().is_none(),
            "plain" => !spec.uses_branch() && !spec.layers.split && spec.hyper.is_none() && spec.gated_residual().is_none() && spec.block_residual.is_none() && spec.cross.is_none(),
            "blockres" => spec.block_residual.is_some(),
            "latent" => spec.moe.as_ref().is_some_and(|m| m.latent.is_some()),
            "flatmoe" => spec.moe.as_ref().is_some_and(|m| m.latent.is_none()),
            "hyper" => spec.hyper.is_some(),
            "gated" => spec.gated_residual().is_some(),
            "untagged" => spec.gated_residual().is_some_and(|h| h.tags().is_empty()),
            "tagged" => spec.gated_residual().is_some_and(|h| !h.tags().is_empty()),
            "gated_norm" => spec.gated_residual().is_some_and(|h| h.final_norm),
            "bare" => spec.gated_residual().is_some_and(|h| !h.final_norm),
            "stored" => spec.mlp.stored_gate_up,
            "unstored" => !spec.mlp.stored_gate_up,
            "flat" => spec.gated_residual().is_none(),
            "ple" => spec.per_layer().is_some(),
            "inputs" => spec.inputs.is_some(),
            "split" => spec.layers.split,
            "logit_softcap" => spec.dims.logit_softcap.is_some(),
            "logit_scale" => spec.dims.logit_scale.is_some(),
            "speculative" => spec.speculative().is_some(),
            "vision" => spec.vision.is_some(),
            "moe" => spec.moe.is_some(),
            "parallel" => spec.parallel_moe(),
            "described" => !spec.attention.node.is_empty(),
            "mixer" => spec.mixer.is_some(),
            "gdn" => spec.mixer.is_none(),
            "stock" => spec.attention.node.is_empty(),
            "dense" => spec.moe.is_none(),
            "cross" => spec.cross.is_some(),
            "positions" => spec.dims.positions.is_some(),
            "rms" => spec.norm.kind == crate::spec::NormKind::Rmsnorm,
            _ => false,
        })
    })
}

fn per_layer_stage(op: &str) -> Option<u8> {
    Some(match op {
        "per_layer_head" => crate::plan::PLE_HEAD,
        "per_layer_layer" => crate::plan::PLE_LAYER,
        _ => return None,
    })
}

fn gated_stage(op: &str) -> Option<GatedStage> {
    Some(match op {
        "gated_expand" => GatedStage::Expand,
        "gated_pre" => GatedStage::Pre,
        "gated_post" => GatedStage::Post,
        "gated_collapse" => GatedStage::Collapse,
        _ => return None,
    })
}

fn widen(op: Op) -> Op {
    match op {
        Op::Linear { weight: Weight::GateUp, quant, input, output } => Op::Linear { weight: Weight::GateUpWide, quant, input, output },
        Op::Linear { weight: Weight::Down, quant, input, output } => Op::Linear { weight: Weight::DownWide, quant, input, output },
        other => other,
    }
}

fn lower(nodes: &[Node], phase: Phase, tensor_core: bool, spec: &Spec, kind: Option<Block>, tail: bool) -> Result<Vec<Op>> {
    let mut ops = Vec::new();
    for n in nodes.iter().filter(|n| n.runs_in(phase, tensor_core, spec, None, tail)) {
        if n.op == "mixer_nodes" {
            let count = spec.mixer.as_ref().map_or(0, |m| m.node.len());
            ensure!(count <= 256, "a mixer has at most 256 nodes");
            for (i, d) in spec.mixer.iter().flat_map(|m| m.node.iter()).enumerate() {
                if d.runs_in(phase, tensor_core, spec, kind, tail) {
                    ops.push(Op::MixKernel { node: i as u8 });
                }
            }
        } else if let Some(stage) = per_layer_stage(&n.op) {
            let p = spec.per_layer().with_context(|| format!("{} needs a [per_layer]", n.op))?;
            for (i, _) in p.nodes(if stage == crate::plan::PLE_HEAD { "head" } else { "layer" }) {
                ops.push(Op::PerLayer { node: i as u8, stage });
            }
        } else if n.op == "gated_at" {
            let h = spec.gated_residual().context("gated_at needs a [gated_residual]")?;
            let tag = n.weight.as_deref().context("a gated_at node names its tag in weight")?;
            for (i, _) in h.tagged(tag) {
                ops.push(Op::Gated { node: i as u8, site: 0 });
            }
        } else if let Some(stage) = gated_stage(&n.op) {
            let h = spec.gated_residual().with_context(|| format!("{} needs a [gated_residual]", n.op))?;
            let site = match stage {
                GatedStage::Pre | GatedStage::Post => n.weight.as_deref().context("a gated pre or post node names its site in weight")?.parse::<u8>().context("gated site")?,
                GatedStage::Collapse if n.weight.as_deref() == Some("draft") => crate::plan::GATED_DRAFT,
                GatedStage::Expand | GatedStage::Collapse => GATED_MODULE,
            };
            ensure!(site >= crate::plan::GATED_DRAFT || (site as usize) < h.sites.len(), "gated site {site} is not one of the {} [gated_residual] sites", h.sites.len());
            for (i, _) in h.nodes(stage) {
                ops.push(Op::Gated { node: i as u8, site });
            }
        } else if n.op == "input_nodes" {
            let p = spec.inputs.as_ref().context("input_nodes needs [inputs]")?;
            for i in 0..p.node.len() {
                ops.push(Op::Input { node: i as u8 });
            }
        } else if n.op == "prelude_nodes" {
            let count = spec.prelude.as_ref().map_or(0, |p| p.node.len());
            ensure!(count <= 64, "a prelude has at most 64 nodes");
            for i in 0..count {
                ops.push(Op::Prelude { node: i as u8 });
            }
        } else if n.op == "attention_nodes" {
            for (i, d) in spec.attention.node.iter().enumerate().filter(|(_, d)| d.runs_in(phase, tensor_core, spec, kind, tail)) {
                ops.push(d.lower_described(i)?);
            }
        } else {
            let op = n.lower()?;
            ops.push(if tail && spec.mlp.wide_shared_tail { widen(op) } else { op });
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
        "vshuffle" => VisionOp::Shuffle,
        "vadapter" => VisionOp::Adapter,
        other => bail!("unknown vision op {other}"),
    })
}

fn vision_holds(when: Option<&str>, deepstack: Option<usize>, cfg: &VisionConfig) -> bool {
    when.is_none_or(|w| {
        w.split('+').all(|c| match c {
            "deepstack" => deepstack.is_some(),
            "pooled" => cfg.pooled(),
            "tiled" => cfg.tiled(),
            "merged" => cfg.tower == crate::vision::Tower::Merged,
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
        for b in self.block.iter().filter(|b| holds(b.when.as_deref(), tensor_core, spec, false)) {
            let sparse = match b.mlp.as_deref() {
                None | Some("dense") => false,
                Some("sparse") => true,
                Some(other) => bail!("block {} has unknown mlp {other}", b.name),
            };
            let tails: &[bool] = if spec.layers.shared_tail > 0 { &[false, true] } else { &[false] };
            for &tail in tails {
                if spec.attention.node.iter().any(|n| !n.layers.is_empty()) {
                    for name in &b.layers {
                        let kind = Block::parse(name)?;
                        blocks.push(PlanBlock { layers: vec![name.clone()], sparse, tail, ops: lower(&b.node, phase, tensor_core, spec, Some(kind), tail)? });
                    }
                } else {
                    blocks.push(PlanBlock { layers: b.layers.clone(), sparse, tail, ops: lower(&b.node, phase, tensor_core, spec, None, tail)? });
                }
            }
        }
        let per_layer: Vec<Op> = match spec.per_layer() {
            Some(p) => p.nodes("layer").map(|(i, _)| Op::PerLayer { node: i as u8, stage: crate::plan::PLE_LAYER }).collect(),
            None => Vec::new(),
        };
        Ok(Plan { phase, tensor_core, head: lower(&self.head, phase, tensor_core, spec, None, false)?, blocks, per_layer, tail: lower(&self.tail, phase, tensor_core, spec, None, false)?, mtp: lower(&self.mtp, phase, tensor_core, spec, None, false)?, mtp_head: lower(&self.mtp_head, phase, tensor_core, spec, None, false)?, window: lower(&self.window, phase, tensor_core, spec, None, false)? })
    }
}

