use crate::codegen::{WeightKind, select};
use crate::spec::{Block, Quant, Spec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
    Verify,
    Forward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stream {
    Image,
    Text,
    Joint,
    Seqs,
    Latent,
}

impl Stream {
    pub fn label(self) -> &'static str {
        match self {
            Stream::Image => "image",
            Stream::Text => "text",
            Stream::Joint => "joint",
            Stream::Seqs => "seqs",
            Stream::Latent => "latent",
        }
    }
}

fn short(weight: &str) -> String {
    weight.split('+').map(|w| w.rsplit_once("{i}.").map_or(w, |(_, t)| t)).collect::<Vec<_>>().join("+")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitOp {
    Patchify { from: &'static str, into: &'static str },
    Unpatchify { from: &'static str, into: &'static str, channels_last: bool },
    Split { from: &'static str, into: &'static str, rows: Stream, act: bool },
    Copy { from: &'static str, into: &'static str, rows: Stream },
    TimestepEmbed { from: &'static str, into: &'static str },
    Linear { weight: &'static str, input: &'static str, output: &'static str, rows: Stream, add: bool, bias: bool, act: u8 },
    Modulate { from: &'static str, into: &'static str, rows: Stream, set: &'static str, shift: u8, scale: u8 },
    GatedAdd { from: &'static str, into: &'static str, rows: Stream, set: &'static str, gate: u8 },
    Glu { from: &'static str, into: &'static str, rows: Stream, offset: &'static str },
    AxisRope { from: &'static str, text: &'static str, image: &'static str },
    JointAttention { into: &'static str, rows: Stream },
    TableAdd { table: &'static str, from: &'static str, into: &'static str },
    LayerNorm { from: &'static str, into: &'static str, rows: Stream, weight: &'static str },
    Pack { from: &'static str, into: &'static str, rows: Stream, norms: &'static str, wide: bool, rope: bool },
    CrossAttention { into: &'static str, rows: Stream, kv: &'static str },
    Step { from: &'static str, into: &'static str, rule: &'static str, guided: bool },
    MultiStep { from: &'static str, into: &'static str, rule: &'static str, state: &'static str, scalars: &'static str, guided: bool },
}

impl DitOp {
    pub fn label(&self) -> String {
        match self {
            DitOp::Patchify { .. } => "patchify".into(),
            DitOp::Unpatchify { .. } => "unpatchify".into(),
            DitOp::Split { act: true, from, .. } => format!("silu {from}"),
            DitOp::Split { from, .. } => format!("split {from}"),
            DitOp::Copy { from, .. } => format!("copy {from}"),
            DitOp::TimestepEmbed { .. } => "timestep embed".into(),
            DitOp::Linear { weight, rows, .. } => format!("{} ({})", short(weight), rows.label()),
            DitOp::Modulate { rows, set, shift, scale, .. } => format!("{} modulate {set}[{shift},{scale}]", rows.label()),
            DitOp::GatedAdd { rows, from, set, gate, .. } => format!("{} gated add {from} {set}[{gate}]", rows.label()),
            DitOp::Glu { from, rows, .. } => format!("{} swiglu {from}", rows.label()),
            DitOp::AxisRope { .. } => "qk norm and axis rope".into(),
            DitOp::JointAttention { rows: Stream::Joint, .. } => "joint attention".into(),
            DitOp::JointAttention { rows, .. } => format!("{} self attention", rows.label()),
            DitOp::TableAdd { table, from, .. } => format!("{} plus {from}", short(table)),
            DitOp::LayerNorm { weight, rows, .. } => format!("{} ({})", short(weight), rows.label()),
            DitOp::Pack { rope: true, rows, .. } => format!("{} qk norm, rope and pack", rows.label()),
            DitOp::Pack { into, rows, .. } => format!("{} norm and pack into {into}", rows.label()),
            DitOp::CrossAttention { rows, kv, .. } => format!("{} cross attention to {kv}", rows.label()),
            DitOp::Step { guided: true, .. } | DitOp::MultiStep { guided: true, .. } => "guided scheduler step".into(),
            DitOp::Step { .. } | DitOp::MultiStep { .. } => "scheduler step".into(),
        }
    }

    pub fn flow(&self) -> String {
        match self {
            DitOp::Patchify { from, into } | DitOp::Unpatchify { from, into, .. } | DitOp::TimestepEmbed { from, into } | DitOp::TableAdd { from, into, .. } => format!("{from} to {into}"),
            DitOp::Glu { from, into, rows, .. } => format!("{from} to {into} ({})", rows.label()),
            DitOp::Split { from, into, rows, .. } | DitOp::Copy { from, into, rows } | DitOp::Modulate { from, into, rows, .. } | DitOp::LayerNorm { from, into, rows, .. } => format!("{from} to {into} ({})", rows.label()),
            DitOp::Linear { input, output, rows, add, .. } => format!("{input} to {output} ({}{})", rows.label(), if *add { ", added" } else { "" }),
            DitOp::GatedAdd { from, into, rows, .. } => format!("{from} gated into {into} ({})", rows.label()),
            DitOp::AxisRope { from, .. } => format!("{from} to attention planes"),
            DitOp::Pack { from, into, .. } => format!("{from} to {into}"),
            DitOp::JointAttention { into, .. } => format!("attention planes to {into}"),
            DitOp::CrossAttention { into, kv, .. } => format!("attention planes and {kv} to {into}"),
            DitOp::Step { from, into, .. } | DitOp::MultiStep { from, into, .. } => format!("{into} and {from} to {into}"),
        }
    }

    pub fn kernel(&self) -> &'static str {
        match self {
            DitOp::Patchify { .. } => "dit_patchify",
            DitOp::Unpatchify { channels_last: true, .. } => "dit_unpatchify_last",
            DitOp::Unpatchify { .. } => "dit_unpatchify",
            DitOp::Split { act: true, .. } => "dit_silu_split",
            DitOp::Split { .. } => "dit_split",
            DitOp::Copy { .. } => "dit_copy",
            DitOp::TimestepEmbed { .. } => "dit_timestep",
            DitOp::Linear { .. } => "vis_mma",
            DitOp::Modulate { .. } => "dit_modulate",
            DitOp::GatedAdd { .. } => "dit_gated_add",
            DitOp::Glu { .. } => "dit_glu",
            DitOp::AxisRope { .. } => "vis_axis_rope",
            DitOp::JointAttention { .. } => "vis_flash",
            DitOp::TableAdd { .. } => "dit_table_add",
            DitOp::LayerNorm { .. } => "dit_layernorm",
            DitOp::Pack { .. } => "vis_qk_pack",
            DitOp::CrossAttention { .. } => "vis_flash_kv",
            DitOp::Step { .. } | DitOp::MultiStep { .. } => "dit_step",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
    Mix,
    MixPlanes,
    Branch,
    Logits,
    FcPlanes,
    DraftPlanes,
    MtpLogits,
    Partial,
    Gdn,
    Streams,
    Merged,
    Latent,
    LatentPlanes,
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
            Buf::Mix => "mix",
            Buf::MixPlanes => "mix planes",
            Buf::Branch => "branch",
            Buf::Logits => "logits",
            Buf::FcPlanes => "fc planes",
            Buf::DraftPlanes => "draft stream planes",
            Buf::MtpLogits => "draft logits",
            Buf::Partial => "rank partial",
            Buf::Gdn => "gdn",
            Buf::Streams => "streams",
            Buf::Merged => "merged",
            Buf::Latent => "latent",
            Buf::LatentPlanes => "latent planes",
        }
    }

    pub fn is_planes(self) -> bool {
        matches!(self, Buf::XnPlanes | Buf::AttnPlanes | Buf::ActPlanes | Buf::GdnPlanes | Buf::MixPlanes | Buf::FcPlanes | Buf::LatentPlanes | Buf::DraftPlanes)
    }

    pub fn holds_activation(self) -> bool {
        self.is_planes() || self == Buf::Act
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Qkv,
    O,
    GateUp,
    GateUpWide,
    Down,
    DownWide,
    LinIn,
    LinBa,
    LinOut,
    MixIn,
    MixOut,
    LmHead,
    MtpFc,
    MtpFcHidden,
    MtpQkv,
    MtpO,
    MtpGateUp,
    MtpDown,
    MtpHead,
    LatentDown,
    LatentUp,
    SharedGateUp,
    SharedDown,
}

impl Weight {
    pub fn name(self) -> &'static str {
        match self {
            Weight::Qkv => "qkv",
            Weight::O => "o",
            Weight::GateUp => "gate_up",
            Weight::GateUpWide => "gate_up_wide",
            Weight::Down => "down",
            Weight::DownWide => "down_wide",
            Weight::LinIn => "lin_in",
            Weight::LinBa => "lin_ba",
            Weight::LinOut => "lin_out",
            Weight::MixIn => "mix_in",
            Weight::MixOut => "mix_out",
            Weight::LmHead => "lm_head",
            Weight::MtpFc => "mtp_fc",
            Weight::MtpFcHidden => "mtp_fc_hidden",
            Weight::MtpQkv => "mtp_qkv",
            Weight::MtpO => "mtp_o",
            Weight::MtpGateUp => "mtp_gate_up",
            Weight::MtpDown => "mtp_down",
            Weight::MtpHead => "mtp_head",
            Weight::LatentDown => "latent_down",
            Weight::LatentUp => "latent_up",
            Weight::SharedGateUp => "shared_gate_up",
            Weight::SharedDown => "shared_down",
        }
    }

    pub fn rows(self, spec: &Spec) -> usize {
        let la = spec.linear_attention.as_ref();
        match self {
            Weight::Qkv | Weight::MtpQkv => spec.qkv_rows(),
            Weight::GateUp => spec.gate_up_rows(),
            Weight::GateUpWide => spec.gate_up_rows_wide(),
            Weight::MtpGateUp => 2 * spec.dims.intermediate,
            Weight::O | Weight::Down | Weight::DownWide | Weight::LinOut | Weight::MtpFc | Weight::MtpFcHidden | Weight::MtpO | Weight::MtpDown => spec.dims.hidden,
            Weight::LinIn => la.map_or(0, |l| l.in_proj_rows()),
            Weight::LinBa => la.map_or(0, |l| l.ba_rows()),
            Weight::MixIn => spec.mix_in_rows(),
            Weight::MixOut => spec.dims.hidden,
            Weight::LmHead | Weight::MtpHead => spec.head_rows().len(),
            Weight::LatentDown => spec.expert_width(),
            Weight::LatentUp | Weight::SharedDown => spec.dims.hidden,
            Weight::SharedGateUp => 2 * spec.shared_expert_rows(),
        }
    }

    pub fn tensors(self, spec: &Spec) -> Vec<String> {
        let w = |m: &str| format!("{m}.weight");
        match self {
            Weight::Qkv if spec.attention.projection.iter().any(|p| !p.append) => spec.attention.projection.iter().map(|p| format!("self_attn.{}.weight", p.tensor)).collect(),
            Weight::Qkv => {
                let names: &[&str] = if spec.attention.fused_qkv { &["self_attn.qkv_proj"] } else { &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] };
                let mut t: Vec<String> = names.iter().map(|m| w(m)).collect();
                if spec.attention.qkv_bias {
                    t.extend(["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"].map(String::from));
                }
                t.extend(spec.attention.projection.iter().map(|p| format!("self_attn.{}.weight", p.tensor)));
                t
            }
            Weight::O => {
                let mut t = vec![w(&format!("self_attn.{}", spec.attention.o_tensor.as_deref().unwrap_or("o_proj")))];
                if spec.attention.o_bias {
                    t.push("self_attn.o_proj.bias".into());
                }
                t
            }
            Weight::GateUp | Weight::GateUpWide if spec.mlp.mapped.is_some() => vec![w("mlp.up_proj")],
            Weight::GateUp | Weight::GateUpWide if spec.mlp.fused_gate_up => vec![w("mlp.gate_up_proj")],
            Weight::GateUp | Weight::GateUpWide => vec![w("mlp.gate_proj"), w("mlp.up_proj")],
            Weight::Down | Weight::DownWide => vec![w("mlp.down_proj")],
            Weight::LinIn if spec.linear_attention.as_ref().is_some_and(|l| l.fused_interleaved) => vec![w("linear_attn.in_proj_qkvz")],
            Weight::LinBa if spec.linear_attention.as_ref().is_some_and(|l| l.fused_interleaved) => vec![w("linear_attn.in_proj_ba")],
            Weight::LinIn => vec![w("linear_attn.in_proj_qkv"), w("linear_attn.in_proj_z")],
            Weight::LinBa => vec![w("linear_attn.in_proj_b"), w("linear_attn.in_proj_a")],
            Weight::LinOut => vec![w("linear_attn.out_proj")],
            Weight::MixIn => spec.mixer.iter().flat_map(|m| m.projection.iter().map(move |p| format!("{}.{}.weight", m.module, p.tensor))).collect(),
            Weight::MixOut => spec.mixer.iter().map(|m| format!("{}.{}.weight", m.module, m.out_tensor)).collect(),
            Weight::LmHead if spec.dims.tie_embeddings => vec![w("embed_tokens")],
            Weight::LmHead => vec![w("lm_head")],
            Weight::MtpFc | Weight::MtpFcHidden | Weight::MtpQkv | Weight::MtpO | Weight::MtpGateUp | Weight::MtpDown | Weight::MtpHead => Vec::new(),
            Weight::LatentDown => spec.latent_tensors(false),
            Weight::LatentUp => spec.latent_tensors(true),
            Weight::SharedGateUp => spec.shared_tensors(false),
            Weight::SharedDown => spec.shared_tensors(true),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantSource {
    Attn,
    Mlp,
    Bf16,
    Head,
    DraftHead,
}

impl QuantSource {
    pub fn resolve_for(self, spec: &Spec, layer: usize, weight: Weight) -> Quant {
        let stored = self.resolve(spec, layer);
        match self {
            QuantSource::Attn => spec.linear_quant(weight.name(), layer),
            QuantSource::Mlp => spec.served_mlp_quant(layer),
            _ => stored,
        }
    }

    pub fn resolve(self, spec: &Spec, layer: usize) -> Quant {
        match self {
            QuantSource::Attn => spec.quant,
            QuantSource::Mlp => spec.mlp_quant(layer),
            QuantSource::Bf16 => Quant::Bf16,
            QuantSource::Head => spec.lm_head_quant(),
            QuantSource::DraftHead => spec.draft_head_quant().unwrap_or_else(|| spec.lm_head_quant()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Input,
    PostAttention,
    PreMlp,
    PostMlp,
    PostDense,
    PostExperts,
    Final,
}

impl NormKind {
    pub fn tensor(self) -> &'static str {
        match self {
            NormKind::Input => "input_layernorm.weight",
            NormKind::PostAttention => "post_attention_layernorm.weight",
            NormKind::PreMlp => "pre_feedforward_layernorm.weight",
            NormKind::PostMlp => "post_feedforward_layernorm.weight",
            NormKind::PostDense => "post_feedforward_layernorm_1.weight",
            NormKind::PostExperts => "post_feedforward_layernorm_2.weight",
            NormKind::Final => "norm.weight",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Embed,
    Splice,
    Input { node: u8 },
    Norm { which: NormKind, into: Buf },
    NormAdd { which: NormKind, from: Buf, into: Buf },
    Clear { buf: Buf },
    Split { from: Buf, into: Buf },
    QkNorm,
    ScaledAdd { from: Buf },
    LogitScale,
    LogitSoftcap,
    Linear { weight: Weight, quant: QuantSource, input: Buf, output: Buf },
    KvRope,
    QkRope,
    Flash,
    FlashLegacy,
    DecodeAttention,
    DecodeRope,
    Kernel { node: u8 },
    MixKernel { node: u8 },
    Gated { node: u8, site: u8 },
    Prelude { node: u8 },
    Swiglu,
    GdnPrefill,
    GdnDecode,
    GdnVerify,
    MtpIn,
    MtpGather,
    MtpNorm,
    MtpDraft,
    MoeRoute,
    MoeExperts,
    MoeCombine,
    MoeCombineInto { into: Buf },
    AllReduce,
    AllReduceNorm { which: NormKind },
    Gemv { norm: Option<NormKind>, weight: Weight, quant: QuantSource, input: Buf, output: Buf },
    GemvStore { weight: Weight, quant: QuantSource, input: Buf, output: Buf },
    StepAttention,
    GdnStep,
    HyperExpand,
    HyperPre { site: u8 },
    HyperPost,
    HyperMean,
    PerLayer { node: u8, stage: u8 },
    Dit(DitOp),
    BlockRes { stage: ResStage },
    LatentNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResStage {
    Enter,
    Mixed,
    Done,
    Tail,
}

impl ResStage {
    pub fn name(self) -> &'static str {
        match self {
            ResStage::Enter => "enter",
            ResStage::Mixed => "mixed",
            ResStage::Done => "done",
            ResStage::Tail => "tail",
        }
    }
}

pub const PLE_HEAD: u8 = 0;
pub const PLE_LAYER: u8 = 1;

pub const GATED_MODULE: u8 = u8::MAX;
pub const GATED_DRAFT: u8 = u8::MAX - 1;

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
            Op::Splice => "media splice".into(),
            Op::Input { .. } => "input stream".into(),
            Op::Norm { which, .. } => match which {
                NormKind::Input => "input norm".into(),
                NormKind::PostAttention => "post attention norm".into(),
                NormKind::PreMlp => "pre mlp norm".into(),
                NormKind::PostMlp => "post mlp norm".into(),
                NormKind::PostDense => "post dense mlp norm".into(),
                NormKind::PostExperts => "post experts norm".into(),
                NormKind::Final => "final norm".into(),
            },
            Op::NormAdd { which, .. } => format!("{which:?} norm and add"),
            Op::LogitSoftcap => "logit softcap".into(),
            Op::LogitScale => "logit scale".into(),
            Op::Clear { .. } => "clear".into(),
            Op::Split { .. } => "split".into(),
            Op::QkNorm => "qk norm".into(),
            Op::ScaledAdd { .. } => "scaled add".into(),
            Op::Linear { weight, .. } => weight.name().into(),
            Op::KvRope | Op::QkRope | Op::DecodeRope => "norm and rope".into(),
            Op::Flash | Op::FlashLegacy => "attention".into(),
            Op::DecodeAttention => "decode attention".into(),
            Op::Kernel { .. } => "described kernel".into(),
            Op::MixKernel { .. } => "described mixer kernel".into(),
            Op::Gated { site: GATED_MODULE, .. } => "gated residual mixer".into(),
            Op::Gated { site: GATED_DRAFT, .. } => "gated residual draft mixer".into(),
            Op::Gated { site, .. } => format!("gated residual site {site}"),
            Op::Prelude { .. } => "described prelude kernel".into(),
            Op::Swiglu => "swiglu".into(),
            Op::GdnPrefill | Op::GdnDecode | Op::GdnVerify => "delta rule".into(),
            Op::MtpIn => "draft input".into(),
            Op::MtpGather => "draft rows".into(),
            Op::MtpNorm => "draft norm".into(),
            Op::MtpDraft => "draft tokens".into(),
            Op::MoeRoute => "router".into(),
            Op::MoeExperts => "experts".into(),
            Op::MoeCombine => "combine".into(),
            Op::MoeCombineInto { .. } => "combine".into(),
            Op::AllReduce => "allreduce".into(),
            Op::AllReduceNorm { .. } => "allreduce, add and norm".into(),
            Op::Gemv { norm: Some(_), weight, .. } => format!("norm and {}", weight.name()),
            Op::Gemv { weight, .. } => weight.name().into(),
            Op::GemvStore { weight, .. } => format!("{} store", weight.name()),
            Op::StepAttention => "step attention".into(),
            Op::GdnStep => "delta rule step".into(),
            Op::HyperExpand => "hyper expand".into(),
            Op::HyperPre { site } => format!("hyper pre {site}"),
            Op::HyperPost => "hyper post".into(),
            Op::HyperMean => "hyper mean".into(),
            Op::PerLayer { stage: PLE_HEAD, .. } => "per layer inputs".into(),
            Op::PerLayer { .. } => "per layer residual".into(),
            Op::Dit(d) => d.label(),
            Op::BlockRes { stage } => format!("block residual {}", stage.name()),
            Op::LatentNorm => "latent expert norm".into(),
        }
    }

    pub fn flow(&self) -> String {
        match self {
            Op::Embed => "tokens to x".into(),
            Op::Splice => "media embeddings into x".into(),
            Op::Input { .. } => "input streams and masks into x".into(),
            Op::Norm { into, .. } => format!("x to {}", into.label()),
            Op::NormAdd { from, into, .. } => format!("{} added to {}", from.label(), into.label()),
            Op::LogitSoftcap | Op::LogitScale => "logits to logits".into(),
            Op::Clear { buf } => format!("zero {}", buf.label()),
            Op::Split { from, into } => format!("{} to {}", from.label(), into.label()),
            Op::QkNorm => "qkv to qkv".into(),
            Op::ScaledAdd { from } => format!("{} scaled and added to x", from.label()),
            Op::Linear { input, output, .. } => format!("{} to {}", input.label(), output.label()),
            Op::KvRope | Op::QkRope | Op::DecodeRope => "qkv to kv cache".into(),
            Op::Flash | Op::FlashLegacy | Op::DecodeAttention | Op::Kernel { .. } => "qkv and kv cache to attn".into(),
            Op::MixKernel { .. } => "mix and sequence state to mix planes".into(),
            Op::Gated { .. } => "streams and x".into(),
            Op::Prelude { .. } => "tokens, streams and sequence state to streams".into(),
            Op::Swiglu => "gate_up to act".into(),
            Op::GdnPrefill | Op::GdnDecode | Op::GdnVerify => "lin, ba and state to gdn".into(),
            Op::MtpIn => "tokens and hidden to fc planes".into(),
            Op::MtpGather => "hidden rows to draft rows".into(),
            Op::MtpNorm => "draft rows to xn planes".into(),
            Op::MtpDraft => "draft logits to draft tokens".into(),
            Op::MoeRoute => "x to top-k experts".into(),
            Op::MoeExperts => "x and top-k to expert outputs".into(),
            Op::MoeCombine => "expert outputs to x".into(),
            Op::MoeCombineInto { into } => format!("expert outputs to {}", into.label()),
            Op::AllReduce => "rank partials summed into x".into(),
            Op::AllReduceNorm { .. } => "rank partials summed into x, x to xn planes".into(),
            Op::Gemv { norm: Some(_), output, .. } => format!("x normed to {}", output.label()),
            Op::Gemv { input, output, .. } => format!("{} to {}", input.label(), output.label()),
            Op::GemvStore { input, output, .. } => format!("{} stored to {}", input.label(), output.label()),
            Op::StepAttention => "qkv and kv cache to attn".into(),
            Op::GdnStep => "lin, ba and state to gdn".into(),
            Op::HyperExpand => "x to streams".into(),
            Op::HyperPre { .. } => "streams to x and hyper weights".into(),
            Op::HyperPost => "x and hyper weights into streams".into(),
            Op::HyperMean => "streams to x".into(),
            Op::PerLayer { stage: PLE_HEAD, .. } => "tokens and hidden to per layer inputs".into(),
            Op::PerLayer { .. } => "x and per layer inputs into x".into(),
            Op::Dit(d) => d.flow(),
            Op::BlockRes { stage: ResStage::Done } => "prefix added to x".into(),
            Op::BlockRes { .. } => "x, prefix and block stack to x".into(),
            Op::LatentNorm => "latent to latent planes".into(),
        }
    }

    pub fn tensors(&self, spec: &Spec) -> Vec<String> {
        match self {
            Op::Embed => vec!["embed_tokens.weight".into()],
            Op::Norm { which, .. } | Op::NormAdd { which, .. } => vec![which.tensor().into()],
            Op::Linear { weight, .. } => weight.tensors(spec),
            Op::Gemv { norm, weight, .. } => norm.iter().map(|n| n.tensor().to_string()).chain(weight.tensors(spec)).collect(),
            Op::GemvStore { weight, .. } => weight.tensors(spec),
            Op::KvRope | Op::QkRope | Op::DecodeRope | Op::Flash | Op::DecodeAttention | Op::StepAttention | Op::QkNorm if spec.attention.qk_norm && !spec.attention.qk_norm_unit => {
                vec!["self_attn.q_norm.weight".into(), "self_attn.k_norm.weight".into()]
            }
            Op::Kernel { node } => spec
                .attention
                .node
                .get(*node as usize)
                .into_iter()
                .flat_map(|n| n.bind.values())
                .filter_map(|b| match b {
                    crate::graph::Bind::Weight { weight, .. } => Some(format!("self_attn.{weight}")),
                    crate::graph::Bind::Number(_) | crate::graph::Bind::Name(_) | crate::graph::Bind::Buffer { .. } | crate::graph::Bind::State { .. } => None,
                })
                .collect(),
            Op::MixKernel { node } => spec
                .mixer
                .iter()
                .filter_map(|m| m.node.get(*node as usize).map(|n| (m, n)))
                .flat_map(|(m, n)| {
                    n.bind.values().filter_map(move |b| match b {
                        crate::graph::Bind::Weight { weight, .. } => Some(format!("{}.{weight}", m.module)),
                        _ => None,
                    })
                })
                .collect(),
            Op::Gated { node, site } => spec
                .gated_residual
                .iter()
                .filter_map(|h| h.node.get(*node as usize).map(|n| (h, n)))
                .flat_map(|(h, n)| {
                    let module = h.sites.get(*site as usize).cloned().unwrap_or_else(|| h.module.clone());
                    n.bind.values().filter_map(move |b| match b {
                        crate::graph::Bind::Weight { weight, .. } if module == crate::spec::GATED_ROOT => Some(weight.clone()),
                        crate::graph::Bind::Weight { weight, .. } => Some(format!("{module}.{weight}")),
                        _ => None,
                    })
                })
                .collect(),
            Op::Prelude { node } => spec
                .prelude
                .iter()
                .filter_map(|p| p.node.get(*node as usize).map(|n| (p, n)))
                .flat_map(|(p, n)| {
                    n.bind.values().filter_map(move |b| match b {
                        crate::graph::Bind::Weight { weight, .. } => Some(format!("{}.{weight}", p.module)),
                        _ => None,
                    })
                })
                .collect(),
            Op::GdnPrefill | Op::GdnDecode | Op::GdnStep => ["conv1d.weight", "A_log", "dt_bias", "norm.weight"].iter().map(|t| format!("linear_attn.{t}")).collect(),
            Op::MoeRoute => vec!["mlp.gate.weight".into(), "mlp.shared_expert_gate.weight".into()],
            Op::MoeExperts => spec.expert_tensors(),
            Op::HyperPre { site } => spec.hyper.as_ref().and_then(|h| h.sites.get(*site as usize)).map(|s| ["fn", "base", "scale"].iter().map(|t| format!("{s}_{t}")).collect()).unwrap_or_default(),
            Op::PerLayer { node, .. } => spec.per_layer.as_ref().and_then(|p| p.node.get(*node as usize)).into_iter().flat_map(|n| n.bind.values()).filter_map(|b| match b {
                crate::graph::Bind::Weight { weight, .. } => Some(weight.clone()),
                _ => None,
            }).collect(),
            Op::Input { node } => spec.inputs.as_ref().and_then(|p| p.node.get(*node as usize)).into_iter().flat_map(|n| n.bind.values()).filter_map(|b| match b {
                crate::graph::Bind::Weight { weight, .. } => Some(weight.clone()),
                _ => None,
            }).collect(),
            Op::BlockRes { stage: ResStage::Enter } => spec.block_residual.iter().flat_map(|r| ["norm", "proj"].map(|t| format!("{}_{t}.weight", r.sites[0]))).collect(),
            Op::BlockRes { stage: ResStage::Mixed } => spec.block_residual.iter().flat_map(|r| ["norm", "proj"].map(|t| format!("{}_{t}.weight", r.sites[1]))).collect(),
            Op::BlockRes { stage: ResStage::Tail } => spec.block_residual.iter().flat_map(|r| ["norm", "proj"].map(|t| format!("{}_{t}.weight", r.output))).collect(),
            Op::LatentNorm => spec.latent_norm_tensors(),
            _ => Vec::new(),
        }
    }

    pub fn checkpoint_tensors(&self, spec: &Spec, root: &str, layer: Option<usize>) -> Vec<String> {
        let tied_or_root = |t: &str| format!("{root}.{t}");
        if matches!(self, Op::Input { .. }) {
            return self.tensors(spec);
        }
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
            Op::Linear { quant, weight, .. } | Op::Gemv { quant, weight, .. } | Op::GemvStore { quant, weight, .. } => Some(quant.resolve_for(spec, layer, *weight)),
            _ => None,
        }
    }

    pub fn decoder(&self, spec: &Spec, layer: usize, tokens: usize, target: &kir::Target) -> Option<&'static str> {
        match self {
            Op::Linear { weight, quant, .. } => {
                let kind = weight_kind(quant.resolve_for(spec, layer, *weight));
                select(spec, target, weight.name(), tokens, kind).ok().map(|c| c.decoder)
            }
            _ => None,
        }
    }

    pub fn kernels(&self, spec: &Spec, layer: usize, tokens: usize, target: &kir::Target) -> Vec<String> {
        let names = |k: &[&str]| k.iter().map(|s| s.to_string()).collect();
        match self {
            Op::Embed => names(&["embed"]),
            Op::Splice => names(&["splice_rows", "add_rows"]),
            Op::Input { node } => spec.inputs.as_ref().and_then(|p| p.node.get(*node as usize)).and_then(|n| n.kernel.clone()).into_iter().collect(),
            Op::Norm { into: Buf::XnPlanes, .. } => names(&["rms_split"]),
            Op::Norm { .. } => names(&["rmsnorm"]),
            Op::NormAdd { .. } => names(&["norm_add"]),
            Op::LogitSoftcap => names(&["logit_softcap"]),
            Op::LogitScale => names(&["logit_scale"]),
            Op::Clear { .. } => names(&["clear_branch"]),
            Op::Split { .. } => names(&["split_rows"]),
            Op::QkNorm => names(&["q_norm_rows", "k_norm_rows"]),
            Op::ScaledAdd { .. } => names(&["residual_add"]),
            Op::Linear { weight, quant, .. } => {
                let w = *weight;
                let kind = weight_kind(quant.resolve_for(spec, layer, w));
                let choice = match select(spec, target, w.name(), tokens, kind) {
                    Ok(c) => c,
                    Err(e) => return vec![format!("unsupported: {e}")],
                };
                let mut out: Vec<String> = choice.quant.into_iter().map(String::from).collect();
                out.push(choice.name);
                out.extend(choice.reduce);
                out
            }
            Op::KvRope => names(&["kv_norm_rope"]),
            Op::QkRope | Op::DecodeRope => names(&["qk_norm_rope"]),
            Op::Flash => names(&["flash_mma"]),
            Op::FlashLegacy => names(&["flash_attention"]),
            Op::DecodeAttention => names(&["qk_norm_rope", "batch_decode_attention", "batch_decode_combine"]),
            Op::Kernel { node } => spec.attention.node.get(*node as usize).and_then(|n| n.kernel.clone()).into_iter().collect(),
            Op::MixKernel { node } => spec.mixer.as_ref().and_then(|m| m.node.get(*node as usize)).and_then(|n| n.kernel.clone()).into_iter().collect(),
            Op::Gated { node, .. } => spec.gated_residual().and_then(|h| h.node.get(*node as usize)).and_then(|n| n.kernel.clone()).into_iter().collect(),
            Op::Prelude { node } => spec.prelude.as_ref().and_then(|p| p.node.get(*node as usize)).and_then(|n| n.kernel.clone()).into_iter().collect(),
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
            Op::GdnVerify => names(&["gdn_conv_verify", "gdn_verify", "split_rows"]),
            Op::MtpIn if spec.streamed_mtp() => names(&["mtp_streams_in"]),
            Op::MtpIn => names(&["mtp_fc_in"]),
            Op::MtpGather if spec.streamed_mtp() => names(&["mtp_copy_w", "mtp_copy_w"]),
            Op::MtpGather => names(&["mtp_copy"]),
            Op::MtpNorm => names(&["rms_split"]),
            Op::MtpDraft => names(&["mtp_draft"]),
            Op::MoeRoute => names(&["moe_route_logits", "moe_route"]),
            Op::MoeExperts => names(&["moe_up_bf16", "moe_down_bf16"]),
            Op::MoeCombine | Op::MoeCombineInto { .. } => names(&["moe_combine"]),
            Op::AllReduce => vec![crate::allreduce::kernel_name(crate::allreduce::pick_mode(tokens * spec.dims.hidden * 4), kir::allreduce::Dtype::F32)],
            Op::AllReduceNorm { .. } => names(&["ar_add_rmsnorm_f32"]),
            Op::GemvStore { weight, quant, .. } => match crate::gemm::gemv_into(spec, weight.name(), weight_kind(quant.resolve_for(spec, layer, *weight)), false, true) {
                Some(g) => vec![g.name.to_string()],
                None => vec![format!("unsupported: no store gemv for {}", weight.name())],
            },
            Op::Gemv { norm, weight, quant, .. } => match crate::gemm::gemv(spec, weight.name(), weight_kind(quant.resolve_for(spec, layer, *weight)), norm.is_some()) {
                Some(g) => vec![g.name.to_string()],
                None => vec![format!("unsupported: no gemv for {}", weight.name())],
            },
            Op::StepAttention => names(&["qk_norm_rope", "attention_split", "attention_reduce"]),
            Op::GdnStep => names(&["gdn_conv_step", "gdn_step"]),
            Op::HyperExpand => names(&["hyper_expand"]),
            Op::HyperPre { .. } => names(&["hyper_pre"]),
            Op::HyperPost => names(&["hyper_post"]),
            Op::HyperMean => names(&["hyper_mean"]),
            Op::PerLayer { node, .. } => spec.per_layer.as_ref().and_then(|p| p.node.get(*node as usize)).and_then(|n| n.kernel.clone()).into_iter().collect(),
            Op::Dit(d) => names(&[d.kernel()]),
            Op::BlockRes { stage: ResStage::Enter } => names(&["block_res_set", "block_res_mix", "block_res_push"]),
            Op::BlockRes { stage: ResStage::Mixed } => names(&["block_res_add", "block_res_mix"]),
            Op::BlockRes { stage: ResStage::Done } => names(&["block_res_out"]),
            Op::BlockRes { stage: ResStage::Tail } => names(&["block_res_mix"]),
            Op::LatentNorm => names(&["latent_norm"]),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanBlock {
    pub layers: Vec<String>,
    pub sparse: bool,
    pub tail: bool,
    pub ops: Vec<Op>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub phase: Phase,
    pub tensor_core: bool,
    pub head: Vec<Op>,
    pub blocks: Vec<PlanBlock>,
    pub per_layer: Vec<Op>,
    pub tail: Vec<Op>,
    pub mtp: Vec<Op>,
    pub mtp_head: Vec<Op>,
}

impl Plan {
    pub fn build(spec: &Spec, phase: Phase, tensor_core: bool) -> Plan {
        crate::graph::Graph::decoder().plan(spec, phase, tensor_core).expect("the embedded decoder graph lowers")
    }

    pub fn layer_tail(&self) -> &[Op] {
        &self.per_layer
    }

    pub fn layer(&self, block: Block) -> &[Op] {
        self.layer_of(block, false)
    }

    pub fn layer_of(&self, block: Block, sparse: bool) -> &[Op] {
        self.layer_for(block, sparse, false)
    }

    pub fn layer_for(&self, block: Block, sparse: bool, tail: bool) -> &[Op] {
        let name = block.name();
        self.blocks
            .iter()
            .find(|b| b.sparse == sparse && b.tail == tail && b.layers.iter().any(|l| l == name))
            .map_or(&[], |b| &b.ops)
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

    fn tp_plan(fuse: bool) -> Plan {
        let json = std::fs::read_to_string(fixtures().join("qwen3.8-27b.config.json")).unwrap();
        let spec = Spec::from_hf_config_serving(&json).unwrap();
        let shards = crate::graph::Graph::decoder().shards().unwrap();
        crate::compile::compile(&spec, Plan::build(&spec, Phase::Decode, true), &kir::Target::sm_120(), &crate::compile::Options::ranked(shards, fuse, 4096)).unwrap()
    }

    #[test]
    fn every_row_parallel_linear_is_followed_by_an_allreduce() {
        let plan = tp_plan(false);
        for block in [Block::Linear, Block::Full] {
            let ops = plan.layer(block);
            let partials = ops.iter().filter(|o| matches!(o, Op::Linear { output: Buf::Partial, .. })).count();
            assert_eq!(partials, 2, "{block:?}: mixer out and down");
            assert_eq!(ops.iter().filter(|o| **o == Op::AllReduce).count(), 2);
            for (i, o) in ops.iter().enumerate() {
                if matches!(o, Op::Linear { output: Buf::Partial, .. }) {
                    assert_eq!(ops[i + 1], Op::AllReduce);
                }
            }
            assert!(!ops.iter().any(|o| matches!(o, Op::Linear { output: Buf::X, .. })), "no row linear writes x directly");
        }
        assert_eq!(plan.mtp.iter().filter(|o| **o == Op::AllReduce).count(), 2, "the draft layer is sharded too");
        assert!(plan.tail.iter().all(|o| !matches!(o, Op::AllReduce | Op::AllReduceNorm { .. })), "the head is replicated");
    }

    #[test]
    fn the_allreduce_fuses_with_the_next_norm_when_asked() {
        let plan = tp_plan(true);
        let ops = plan.layer(Block::Full);
        assert!(ops.contains(&Op::AllReduceNorm { which: NormKind::PostAttention }));
        assert!(!ops.iter().any(|o| matches!(o, Op::Norm { which: NormKind::PostAttention, .. })));
        assert_eq!(ops.last(), Some(&Op::AllReduce), "down has no norm after it inside the layer");
    }

    #[test]
    fn tp_one_plans_are_untouched() {
        let json = std::fs::read_to_string(fixtures().join("qwen3.8-27b.config.json")).unwrap();
        let spec = Spec::from_hf_config_serving(&json).unwrap();
        let plain = Plan::build(&spec, Phase::Decode, true);
        assert_eq!(crate::compile::compile(&spec, plain.clone(), &kir::Target::sm_120(), &crate::compile::Options::ranked(Default::default(), true, 4096)).unwrap(), plain);
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
    fn linear_kernels_come_from_select() {
        let (spec, _) = described("qwen3-0.6b.config.json", "qwen3-0.6b.json");
        let op = Op::Linear { weight: Weight::Qkv, quant: QuantSource::Attn, input: Buf::XnPlanes, output: Buf::Qkv };
        let t = kir::Target { sm_count: 84, ..kir::Target::sm_120() };
        let direct = select(&spec, &t, "qkv", 512, WeightKind::Bf16).unwrap();
        assert_eq!(op.kernels(&spec, 0, 512, &t), vec![direct.name]);
        assert_eq!(op.decoder(&spec, 0, 512, &t), Some("DecBf16"));
        assert_eq!(Op::Embed.decoder(&spec, 0, 512, &t), None);
    }
}
