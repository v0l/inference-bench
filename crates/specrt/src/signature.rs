use crate::graph::Bind;
use crate::plan::{Buf, NormKind, Op, Phase, Plan, Weight, weight_kind};
use crate::spec::{KvDtype, Quant, Spec};
use anyhow::{Result, bail};
use kir::gemm::Epilogue;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Elementwise,
    RowReduce,
    Gemm,
    Attention,
    Scan,
    Gather,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    F32,
    Bf16,
    Planes,
    I32,
    Weight(Quant),
    Paged(KvDtype),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    Planes,
    Paged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    Tokens,
    Out,
    Seqs,
    Context,
    Pairs(usize),
    Experts { total: usize, top_k: usize },
    Fixed(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Place {
    Buf(Buf),
    Local(String),
    Weight(String),
    Cache(String),
    State(String),
    Input(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub place: Place,
    pub rows: Rows,
    pub cols: usize,
    pub form: Form,
    pub layout: Layout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub class: Class,
    pub reads: Vec<Tensor>,
    pub writes: Vec<Tensor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count {
    pub tokens: usize,
    pub out: usize,
    pub seqs: usize,
    pub context: usize,
}

impl Count {
    pub fn new(phase: Phase, tokens: usize, context: usize) -> Count {
        match phase {
            Phase::Prefill => Count { tokens, out: 1, seqs: 1, context },
            Phase::Decode | Phase::Verify => Count { tokens, out: tokens, seqs: tokens, context },
        }
    }

    pub(crate) fn rows(&self, r: Rows) -> usize {
        match r {
            Rows::Tokens => self.tokens,
            Rows::Out => self.out,
            Rows::Seqs => self.seqs,
            Rows::Context => self.context * self.seqs,
            Rows::Pairs(p) => self.tokens * p,
            Rows::Experts { total, top_k } => total.min(self.tokens * top_k),
            Rows::Fixed(n) => n,
        }
    }
}

impl Form {
    fn layout(self) -> Layout {
        match self {
            Form::Planes => Layout::Planes,
            Form::Paged(_) => Layout::Paged,
            Form::F32 | Form::Bf16 | Form::I32 | Form::Weight(_) => Layout::RowMajor,
        }
    }

    pub fn label(self) -> String {
        match self {
            Form::F32 => "f32".into(),
            Form::Bf16 => "bf16".into(),
            Form::Planes => "bf16 hi/lo planes".into(),
            Form::I32 => "i32".into(),
            Form::Weight(q) => crate::anatomy::quant_tag(q).into(),
            Form::Paged(KvDtype::F16) => "paged bf16".into(),
            Form::Paged(KvDtype::Fp8) => "paged fp8 with scales".into(),
        }
    }
}

impl Tensor {
    pub(crate) fn new(place: Place, rows: Rows, cols: usize, form: Form) -> Tensor {
        Tensor { place, rows, cols, form, layout: form.layout() }
    }

    fn buf(b: Buf, rows: Rows, cols: usize) -> Tensor {
        Tensor::new(Place::Buf(b), rows, cols, if b.is_planes() { Form::Planes } else { Form::F32 })
    }

    fn weight(name: impl Into<String>, rows: usize, cols: usize, form: Form) -> Tensor {
        Tensor::new(Place::Weight(name.into()), Rows::Fixed(rows), cols, form)
    }

    pub fn bytes(&self, c: &Count) -> u64 {
        let elems = (c.rows(self.rows) * self.cols) as u64;
        match self.form {
            Form::F32 | Form::I32 | Form::Planes => 4 * elems,
            Form::Bf16 => 2 * elems,
            Form::Weight(q) => crate::anatomy::quant_bytes(elems, q),
            Form::Paged(_) => elems,
        }
    }

    pub fn describe(&self) -> String {
        let place = match &self.place {
            Place::Buf(b) => b.label().to_string(),
            Place::Local(n) | Place::Weight(n) | Place::Cache(n) | Place::State(n) => n.clone(),
            Place::Input(n) => n.to_string(),
        };
        format!("{place} [{:?} x {}] {}", self.rows, self.cols, self.form.label())
    }
}

fn class_of(body: &kir::Body) -> Class {
    use kir::rows::RowBody;
    match body {
        kir::Body::RmsNorm { .. } | kir::Body::RmsNormSegment { .. } | kir::Body::AddRmsNorm { .. } | kir::Body::NormAdd { .. } => Class::RowReduce,
        kir::Body::Pointwise { .. } | kir::Body::Add { .. } | kir::Body::GatedActivation { .. } => Class::Elementwise,
        kir::Body::Compose(c) if c.sum.is_some() => Class::RowReduce,
        kir::Body::Compose(_) => Class::Elementwise,
        kir::Body::Scan(_) => Class::Attention,
        kir::Body::Rows(r) => match r {
            RowBody::HeadRms { .. } | RowBody::LayerNorm { .. } => Class::RowReduce,
            RowBody::SpanRope { .. } => Class::Elementwise,
            RowBody::StateWrite { .. } | RowBody::BlockPool { .. } | RowBody::IndexTopk { .. } => Class::Gather,
            RowBody::HeadMatvec { .. } => Class::Gemm,
        },
        kir::Body::Recurrence(_) => Class::Scan,
        kir::Body::Seq(kir::recur::SeqBody::Conv { .. }) => Class::Scan,
        kir::Body::Seq(kir::recur::SeqBody::GroupNorm { .. }) => Class::RowReduce,
    }
}

pub struct Signer<'a> {
    spec: &'a Spec,
    described: Vec<Option<kir::Kernel>>,
    mixer: Vec<kir::Kernel>,
    ranked: bool,
}

impl<'a> Signer<'a> {
    pub fn new(spec: &'a Spec) -> Result<Signer<'a>> {
        let target = crate::codegen::default_target();
        let ctx = crate::engine::DEFAULT_CTX;
        Ok(Signer { spec, described: crate::codegen::described_kernels(spec, 1, ctx, &target)?, mixer: crate::codegen::mixer_kernels(spec, 1, ctx, &target)?, ranked: false })
    }

    pub fn ranked(self, ranked: bool) -> Signer<'a> {
        Signer { ranked, ..self }
    }

    fn hidden(&self) -> usize {
        self.spec.dims.hidden
    }

    fn x(&self) -> Tensor {
        Tensor::buf(Buf::X, Rows::Tokens, self.hidden())
    }

    fn norm_weight(&self, which: NormKind) -> Tensor {
        Tensor::weight(which.tensor(), 1, self.hidden(), Form::F32)
    }

    pub(crate) fn caches(&self, layer: usize, rows: Rows) -> Vec<Tensor> {
        let at = self.spec.attention_before(layer);
        crate::state::positional(self.spec)
            .filter(|s| s.layers == crate::state::Layers::Attention)
            .map(|s| Tensor::new(Place::Cache(s.name.clone()), rows, s.token_layer_bytes(self.spec, at, 1), Form::Paged(self.spec.runtime.kv_dtype)))
            .collect()
    }

    pub(crate) fn qk_weights(&self) -> Vec<Tensor> {
        let a = &self.spec.attention;
        if a.qk_norm && !a.qk_norm_unit && !a.qk_norm_full {
            vec![Tensor::weight("self_attn.q_norm.weight", 1, a.head_dim, Form::F32), Tensor::weight("self_attn.k_norm.weight", 1, a.head_dim, Form::F32)]
        } else {
            Vec::new()
        }
    }

    pub(crate) fn qkv(&self) -> Tensor {
        Tensor::buf(Buf::Qkv, Rows::Tokens, self.spec.qkv_rows())
    }

    fn rope(&self, layer: usize) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut reads = vec![self.qkv()];
        reads.extend(self.qk_weights());
        let mut writes = vec![self.qkv()];
        writes.extend(self.caches(layer, Rows::Tokens));
        (reads, writes)
    }

    fn attention(&self, layer: usize, rope: bool, out: Tensor) -> Signature {
        let (mut reads, mut writes) = if rope { self.rope(layer) } else { (vec![self.qkv()], Vec::new()) };
        reads.extend(self.caches(layer, Rows::Context));
        writes.push(out);
        Signature { class: Class::Attention, reads, writes }
    }

    fn linear(&self, weight: Weight, quant: crate::plan::QuantSource, input: Buf, output: Buf, layer: usize) -> Signature {
        let s = crate::gemm::linear_shape(self.spec, weight.name());
        let q = quant.resolve_for(self.spec, layer, weight);
        let head = matches!(weight, Weight::LmHead | Weight::MtpHead);
        let rows = if weight == Weight::LmHead { Rows::Out } else { Rows::Tokens };
        let out_cols = match s.epi {
            _ if head => self.spec.dims.vocab,
            Epilogue::Act if output.holds_activation() => self.spec.gate_up_out(s.n),
            _ => s.n,
        };
        let mut reads = vec![Tensor::buf(input, rows, s.k), Tensor::weight(weight.name(), weight.rows(self.spec), s.k, Form::Weight(q))];
        if s.bias {
            reads.push(Tensor::weight(format!("{}.bias", weight.name()), 1, s.n, Form::F32));
        }
        let out = Tensor::buf(output, rows, out_cols);
        if s.epi == Epilogue::Residual {
            reads.push(out.clone());
        }
        Signature { class: Class::Gemm, reads, writes: vec![out] }
    }

    fn bound(&self, kernel: &kir::Kernel, bind: &BTreeMap<String, Bind>, scratch: &BTreeMap<String, String>, ints: &BTreeMap<String, String>, defaults: &dyn Fn(&str) -> Option<Tensor>) -> Result<Signature> {
        let (mut reads, mut writes): (Vec<Tensor>, Vec<Tensor>) = (Vec::new(), Vec::new());
        let dim = |s: &str| crate::state::dim(self.spec, s);
        let mut planes_out = false;
        for p in kernel.params.iter().filter(|p| p.role != kir::Role::Value) {
            let t = match bind.get(p.name) {
                Some(Bind::Weight { weight, len }) => Tensor::weight(weight.clone(), 1, self.spec.weight_len(len)?, Form::F32),
                Some(Bind::Buffer { buffer }) => match buffer.as_str() {
                    "qkv" => self.qkv(),
                    "attn" => Tensor::buf(Buf::Attn, Rows::Tokens, self.spec.q_dim().max(self.spec.o_dim())),
                    "mix" => Tensor::buf(Buf::Mix, Rows::Tokens, self.spec.mix_in_rows()),
                    name if ints.contains_key(name) => Tensor::new(Place::Local(name.into()), Rows::Tokens, dim(&ints[name])?, Form::I32),
                    name => match scratch.get(name) {
                        Some(w) => Tensor::new(Place::Local(name.into()), Rows::Tokens, dim(w)?, Form::F32),
                        None => bail!("{} binds {name:?}, which is no buffer", kernel.name),
                    },
                },
                Some(Bind::State { state }) => match crate::state::named(self.spec, state) {
                    Some(s) if s.positional() => Tensor::new(Place::Cache(state.clone()), if p.access == kir::Access::Write { Rows::Tokens } else { Rows::Context }, s.token_layer_bytes(self.spec, 0, 1), Form::Paged(self.spec.runtime.kv_dtype)),
                    Some(s) => Tensor::new(Place::State(state.clone()), Rows::Seqs, s.elements(self.spec)?, Form::F32),
                    None => bail!("{} binds state {state:?}, which the spec does not declare", kernel.name),
                },
                Some(Bind::Number(_) | Bind::Name(_)) => continue,
                None if matches!(p.name, "out_hi" | "out_lo") => {
                    if !planes_out {
                        planes_out = true;
                        if let Some(t) = defaults("out_planes") {
                            writes.push(t);
                        }
                    }
                    continue;
                }
                None => match defaults(p.name) {
                    Some(t) => t,
                    None => continue,
                },
            };
            match (&t.place, p.access) {
                (Place::State(_), _) => {
                    reads.push(t.clone());
                    writes.push(t);
                }
                (_, kir::Access::Read) => reads.push(t),
                (_, kir::Access::Write) => writes.push(t),
            }
        }
        Ok(Signature { class: class_of(&kernel.body), reads, writes })
    }

    fn described(&self, node: u8, layer: usize) -> Result<Signature> {
        let a = &self.spec.attention;
        let Some(Some(kernel)) = self.described.get(node as usize) else { bail!("attention node {node} has no described kernel") };
        let caches = self.caches(layer, Rows::Context);
        let defaults = |name: &str| -> Option<Tensor> {
            match name {
                "qkv" => Some(self.qkv()),
                "out_planes" => Some(Tensor::buf(Buf::AttnPlanes, Rows::Tokens, self.spec.o_dim())),
                "k_cache" => caches.first().cloned(),
                "v_cache" => caches.get(1).or(caches.first()).cloned(),
                "block_table" | "pos" | "slots" => Some(Tensor::new(Place::Input("positions"), Rows::Tokens, 1, Form::I32)),
                _ => None,
            }
        };
        self.bound(kernel, &a.node[node as usize].bind, &a.scratch, &a.int_scratch, &defaults)
    }

    fn mixed(&self, node: u8) -> Result<Signature> {
        let Some(m) = &self.spec.mixer else { bail!("a mixer node needs [mixer]") };
        let Some(kernel) = self.mixer.get(node as usize) else { bail!("mixer node {node} has no kernel") };
        let defaults = |name: &str| -> Option<Tensor> {
            match name {
                "out_planes" => Some(Tensor::buf(Buf::MixPlanes, Rows::Tokens, self.spec.mix_out_cols())),
                "slots" => Some(Tensor::new(Place::Input("positions"), Rows::Tokens, 1, Form::I32)),
                _ => None,
            }
        };
        self.bound(kernel, &m.node[node as usize].bind, &m.scratch, &BTreeMap::new(), &defaults)
    }

    fn gdn(&self) -> Signature {
        let Some(la) = &self.spec.linear_attention else {
            return Signature { class: Class::Scan, reads: Vec::new(), writes: Vec::new() };
        };
        let states: Vec<Tensor> = crate::state::sequence(self.spec)
            .filter(|s| s.layers == crate::state::Layers::Linear)
            .map(|s| Tensor::new(Place::State(s.name.clone()), Rows::Seqs, s.elements(self.spec).unwrap_or(0), Form::F32))
            .collect();
        let mut reads = vec![
            Tensor::buf(Buf::PreLin, Rows::Tokens, la.in_proj_rows()),
            Tensor::buf(Buf::PreBa, Rows::Tokens, la.ba_rows()),
            Tensor::weight("linear_attn.conv1d.weight", la.conv_dim(), la.conv_kernel, Form::F32),
            Tensor::weight("linear_attn.A_log", 1, la.value_heads, Form::F32),
            Tensor::weight("linear_attn.dt_bias", 1, la.value_heads, Form::F32),
            Tensor::weight("linear_attn.norm.weight", 1, la.value_head_dim, Form::F32),
        ];
        reads.extend(states.iter().cloned());
        let mut writes = states;
        writes.push(Tensor::buf(Buf::GdnPlanes, Rows::Tokens, la.value_dim()));
        Signature { class: Class::Scan, reads, writes }
    }

    fn moe(&self, op: &Op, layer: usize) -> Signature {
        let h = self.hidden();
        let Some(m) = &self.spec.moe else {
            return Signature { class: Class::Gemm, reads: Vec::new(), writes: Vec::new() };
        };
        let rows = Tensor::new(Place::Local("moe rows".into()), Rows::Tokens, h, Form::F32);
        let routes = Tensor::new(Place::Local("moe routes".into()), Rows::Pairs(m.pairs()), 2, Form::I32);
        let outs = Tensor::new(Place::Local("moe outputs".into()), Rows::Pairs(m.pairs()), h, Form::F32);
        let q = self.spec.mlp_quant(layer);
        match op {
            Op::MoeRoute => Signature {
                class: Class::Gemm,
                reads: vec![self.x(), self.norm_weight(NormKind::PostAttention), Tensor::weight("mlp.gate.weight", m.experts, h, Form::Bf16)],
                writes: vec![rows, routes],
            },
            Op::MoeExperts => {
                let matrices = if self.spec.mlp.mapped.is_some() { 2 } else { 3 };
                let mut reads = vec![rows, routes, Tensor::new(Place::Weight("mlp.experts".into()), Rows::Experts { total: m.local_experts(), top_k: m.top_k() }, matrices * m.expert_intermediate * h, Form::Weight(q))];
                if m.shared_intermediate > 0 {
                    reads.push(Tensor::weight("mlp.shared_expert", 1, matrices * m.shared_intermediate * h, Form::Weight(q)));
                }
                Signature { class: Class::Gemm, reads, writes: vec![outs] }
            }
            _ => {
                let target = if self.ranked { Tensor::buf(Buf::Partial, Rows::Tokens, h) } else { self.x() };
                Signature { class: Class::Gather, reads: vec![outs, routes, target.clone()], writes: vec![target] }
            }
        }
    }

    pub fn signature(&self, op: &Op, layer: usize) -> Result<Signature> {
        let spec = self.spec;
        let h = self.hidden();
        let tokens_in = Tensor::new(Place::Input("tokens"), Rows::Tokens, 1, Form::I32);
        let sig = |class, reads, writes| Signature { class, reads, writes };
        Ok(match *op {
            Op::Embed => sig(Class::Gather, vec![tokens_in, Tensor::new(Place::Weight("embed_tokens.weight".into()), Rows::Tokens, h, Form::Bf16)], vec![self.x()]),
            Op::Splice => sig(Class::Gather, vec![Tensor::new(Place::Input("media"), Rows::Tokens, h, Form::F32)], vec![self.x()]),
            Op::Norm { which, into } => {
                let rows = if which == NormKind::Final && into.is_planes() { Rows::Out } else { Rows::Tokens };
                sig(Class::RowReduce, vec![Tensor::buf(Buf::X, rows, h), self.norm_weight(which)], vec![Tensor::buf(into, rows, h)])
            }
            Op::NormAdd { which, from } => sig(Class::RowReduce, vec![Tensor::buf(from, Rows::Tokens, h), self.norm_weight(which), self.x()], vec![self.x()]),
            Op::Clear { buf } => sig(Class::Elementwise, Vec::new(), vec![Tensor::buf(buf, Rows::Tokens, h)]),
            Op::Split { from, into } => sig(Class::Elementwise, vec![Tensor::buf(from, Rows::Tokens, h)], vec![Tensor::buf(into, Rows::Tokens, h)]),
            Op::QkNorm => sig(
                Class::RowReduce,
                vec![self.qkv(), Tensor::weight("self_attn.q_norm.weight", 1, spec.q_dim(), Form::F32), Tensor::weight("self_attn.k_norm.weight", 1, spec.kv_dim(), Form::F32)],
                vec![self.qkv()],
            ),
            Op::ScaledAdd { from } => sig(Class::Elementwise, vec![self.x(), Tensor::buf(from, Rows::Tokens, h)], vec![self.x()]),
            Op::LogitScale | Op::LogitSoftcap => {
                let logits = Tensor::buf(Buf::Logits, Rows::Out, spec.dims.vocab);
                sig(Class::Elementwise, vec![logits.clone()], vec![logits])
            }
            Op::Linear { weight, quant, input, output } => self.linear(weight, quant, input, output, layer),
            Op::Gemv { norm, weight, quant, input, output } => {
                let mut s = self.linear(weight, quant, input, output, layer);
                if let Some(which) = norm {
                    let rows = s.reads[0].rows;
                    s.reads[0] = Tensor::buf(Buf::X, rows, h);
                    s.reads.push(self.norm_weight(which));
                }
                s
            }
            Op::GemvStore { weight, quant, input, output } => {
                let mut s = self.linear(weight, quant, input, output, layer);
                s.reads.retain(|t| t.place != Place::Buf(output));
                s
            }
            Op::StepAttention => self.attention(layer, true, Tensor::buf(Buf::Attn, Rows::Tokens, spec.q_dim())),
            Op::GdnStep => {
                let mut s = self.gdn();
                if let Some(planes) = s.writes.iter_mut().find(|t| t.place == Place::Buf(Buf::GdnPlanes)) {
                    *planes = Tensor::buf(Buf::Gdn, Rows::Tokens, planes.cols);
                }
                s
            }
            Op::KvRope | Op::QkRope | Op::DecodeRope => {
                let (reads, writes) = self.rope(layer);
                sig(Class::Elementwise, reads, writes)
            }
            Op::Flash => self.attention(layer, false, Tensor::buf(Buf::AttnPlanes, Rows::Tokens, spec.q_dim())),
            Op::FlashLegacy => self.attention(layer, false, Tensor::buf(Buf::Attn, Rows::Tokens, spec.q_dim())),
            Op::DecodeAttention => self.attention(layer, true, Tensor::buf(Buf::AttnPlanes, Rows::Tokens, spec.q_dim())),
            Op::Kernel { node } => self.described(node, layer)?,
            Op::MixKernel { node } => self.mixed(node)?,
            Op::Swiglu => sig(Class::Elementwise, vec![Tensor::buf(Buf::GateUp, Rows::Tokens, 2 * spec.dims.intermediate)], vec![Tensor::buf(Buf::Act, Rows::Tokens, spec.dims.intermediate)]),
            Op::GdnPrefill | Op::GdnDecode | Op::GdnVerify => self.gdn(),
            Op::MtpIn => sig(
                Class::Gather,
                vec![tokens_in, Tensor::new(Place::Input("hidden"), Rows::Tokens, h, Form::F32), Tensor::new(Place::Weight("embed_tokens.weight".into()), Rows::Tokens, h, Form::Bf16), Tensor::weight("mtp norms", 2, h, Form::F32)],
                vec![Tensor::buf(Buf::FcPlanes, Rows::Tokens, 2 * h)],
            ),
            Op::MtpGather => sig(Class::Gather, vec![self.x()], vec![Tensor::new(Place::Local("draft rows".into()), Rows::Tokens, h, Form::F32)]),
            Op::MtpNorm => sig(
                Class::RowReduce,
                vec![Tensor::new(Place::Local("draft rows".into()), Rows::Tokens, h, Form::F32), Tensor::weight("mtp norm", 1, h, Form::F32)],
                vec![Tensor::buf(Buf::XnPlanes, Rows::Tokens, h)],
            ),
            Op::MtpDraft => sig(Class::RowReduce, vec![Tensor::buf(Buf::MtpLogits, Rows::Tokens, spec.dims.vocab)], vec![Tensor::new(Place::Local("drafts".into()), Rows::Tokens, 1, Form::I32)]),
            Op::MoeRoute | Op::MoeExperts | Op::MoeCombine => self.moe(op, layer),
            Op::AllReduce => {
                let partial = Tensor::buf(Buf::Partial, Rows::Tokens, h);
                sig(Class::Elementwise, vec![partial.clone(), self.x()], vec![self.x(), partial])
            }
            Op::AllReduceNorm { which } => {
                let partial = Tensor::buf(Buf::Partial, Rows::Tokens, h);
                sig(Class::RowReduce, vec![partial.clone(), self.x(), self.norm_weight(which)], vec![self.x(), Tensor::buf(Buf::XnPlanes, Rows::Tokens, h), partial])
            }
        })
    }
}

pub struct Section<'p> {
    pub name: String,
    pub layer: usize,
    pub repeats: usize,
    pub ops: &'p [Op],
}

pub(crate) fn layers_using(spec: &Spec, layers: &[String], sparse: bool) -> Vec<usize> {
    (0..spec.layers.count)
        .filter(|&i| layers.iter().any(|l| l == spec.block(i).name()) && spec.moe.as_ref().is_some_and(|m| m.is_sparse(i) && spec.has_mlp(i)) == sparse)
        .collect()
}

pub fn sections<'p>(spec: &Spec, plan: &'p Plan) -> Vec<Section<'p>> {
    let mut out = vec![Section { name: "head".into(), layer: 0, repeats: 1, ops: &plan.head }];
    for b in &plan.blocks {
        let used = layers_using(spec, &b.layers, b.sparse);
        let Some(&first) = used.first() else { continue };
        let mut kinds: Vec<&str> = used.iter().map(|&i| spec.block(i).name()).collect();
        kinds.sort_unstable();
        kinds.dedup();
        let name = format!("layer {}{}", kinds.join("|"), if b.sparse { " moe" } else { "" });
        out.push(Section { name, layer: first, repeats: used.len(), ops: &b.ops });
    }
    out.push(Section { name: "tail".into(), layer: spec.layers.count.saturating_sub(1), repeats: 1, ops: &plan.tail });
    out.push(Section { name: "mtp".into(), layer: spec.layers.count, repeats: 1, ops: &plan.mtp });
    out.push(Section { name: "mtp head".into(), layer: spec.layers.count, repeats: 1, ops: &plan.mtp_head });
    out
}

pub(crate) fn ranked(plan: &Plan) -> bool {
    let all = plan.head.iter().chain(plan.blocks.iter().flat_map(|b| &b.ops)).chain(&plan.tail).chain(&plan.mtp).chain(&plan.mtp_head);
    all.into_iter().any(|o| matches!(o, Op::AllReduce | Op::AllReduceNorm { .. }))
}

fn rows_fit(read: Rows, written: Rows) -> bool {
    read == written || (read == Rows::Out && written == Rows::Tokens)
}

pub(crate) fn key(place: &Place) -> Place {
    match place {
        Place::Buf(Buf::XnPlanes) => Place::Buf(Buf::Xn),
        Place::Buf(Buf::AttnPlanes) => Place::Buf(Buf::Attn),
        Place::Buf(Buf::ActPlanes) => Place::Buf(Buf::Act),
        p => p.clone(),
    }
}

fn check(defined: &mut BTreeMap<Place, Tensor>, section: &str, index: usize, op: &Op, sig: &Signature) -> Result<()> {
    let at = || format!("{section} node {index} ({} {op:?})", op.label());
    for r in sig.reads.iter().filter(|r| matches!(r.place, Place::Buf(_) | Place::Local(_))) {
        let Some(w) = defined.get(&key(&r.place)) else { bail!("{}: reads {} before any node writes it", at(), r.describe()) };
        if w.form != r.form || w.layout != r.layout {
            bail!("{}: form mismatch, reads {} but it was written as {}", at(), r.describe(), w.describe());
        }
        if w.cols != r.cols || !rows_fit(r.rows, w.rows) {
            bail!("{}: shape mismatch, reads {} but it was written as {}", at(), r.describe(), w.describe());
        }
    }
    for w in &sig.writes {
        defined.insert(key(&w.place), w.clone());
    }
    Ok(())
}

pub fn validate_plan(plan: &Plan, spec: &Spec) -> Result<()> {
    let signer = Signer::new(spec)?.ranked(ranked(plan));
    validate_with(&signer, plan, spec)
}

pub fn validate_with(signer: &Signer, plan: &Plan, spec: &Spec) -> Result<()> {
    let mut entry: BTreeMap<Place, Tensor> = BTreeMap::new();
    entry.insert(Place::Buf(Buf::Partial), Tensor::buf(Buf::Partial, Rows::Tokens, spec.dims.hidden));
    if plan.head.is_empty() {
        entry.insert(Place::Buf(Buf::X), Tensor::buf(Buf::X, Rows::Tokens, spec.dims.hidden));
    }
    let walk = |defined: &mut BTreeMap<Place, Tensor>, name: &str, layer: usize, ops: &[Op]| -> Result<()> {
        for (i, op) in ops.iter().enumerate() {
            let sig = signer.signature(op, layer).map_err(|e| anyhow::anyhow!("{name} node {i} ({op:?}): {e:#}"))?;
            check(defined, name, i, op, &sig)?;
        }
        Ok(())
    };
    walk(&mut entry, "head", 0, &plan.head)?;
    let mut carried: BTreeMap<Place, Tensor> = BTreeMap::new();
    let mut last = entry.clone();
    for i in 0..spec.layers.count {
        let sparse = spec.moe.as_ref().is_some_and(|m| m.is_sparse(i) && spec.has_mlp(i));
        let block = spec.block(i);
        let mut defined = entry.clone();
        defined.extend(carried.clone());
        walk(&mut defined, &format!("layer {i} ({}{})", block.name(), if sparse { ", moe" } else { "" }), i, plan.layer_of(block, sparse))?;
        carried.extend(defined.iter().filter(|(p, _)| matches!(p, Place::Local(_))).map(|(p, t)| (p.clone(), t.clone())));
        last = defined;
    }
    walk(&mut last, "tail", spec.layers.count.saturating_sub(1), &plan.tail)?;
    let mut draft = entry;
    draft.remove(&Place::Buf(Buf::X));
    walk(&mut draft, "mtp", spec.layers.count, &plan.mtp)?;
    walk(&mut draft, "mtp head", spec.layers.count, &plan.mtp_head)
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpCost {
    pub label: String,
    pub class: Class,
    pub read: u64,
    pub written: u64,
    pub launches: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SectionCost {
    pub name: String,
    pub repeats: usize,
    pub ops: Vec<OpCost>,
    pub read: u64,
    pub written: u64,
    pub launches: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanReport {
    pub phase: Phase,
    pub count: Count,
    pub sections: Vec<SectionCost>,
}

fn linear_launches(spec: &Spec, weight: Weight, quant: crate::plan::QuantSource, layer: usize, tokens: usize, target: &kir::Target, listed: usize) -> usize {
    use crate::codegen::{BatchShape, GemmPath};
    let kind = weight_kind(quant.resolve_for(spec, layer, weight));
    let Ok(choice) = crate::codegen::select(spec, target, weight.name(), tokens, kind) else { return listed };
    let s = crate::gemm::linear_shape(spec, weight.name());
    let unsplit = match choice.path {
        GemmPath::F8b | GemmPath::F4b => BatchShape::for_tokens(tokens).splits(target, s.n, s.k) == 1,
        GemmPath::Ggr => BatchShape::for_ggml(tokens).ggml_splits(target, s.n, s.k) == 1,
        _ => false,
    };
    if choice.reduce.is_some() && unsplit { listed - 1 } else { listed }
}

impl Signer<'_> {
    pub(crate) fn launches(&self, op: &Op, phase: Phase, layer: usize, count: &Count, target: &kir::Target) -> usize {
        let spec = self.spec;
        let tokens = if matches!(op, Op::Linear { weight: Weight::LmHead, .. }) { count.out } else { count.tokens };
        let listed = op.kernels(spec, layer, tokens, target).len();
        match *op {
            Op::Linear { weight, quant, .. } => linear_launches(spec, weight, quant, layer, tokens, target, listed),
            Op::Kernel { node } => {
                let tiled = self.described.get(node as usize).and_then(Option::as_ref).is_some_and(|k| k.tile_scan()) && crate::codegen::described_tiles(spec, &spec.attention.node[node as usize]);
                if tiled && phase != Phase::Prefill { 2 } else { listed }
            }
            Op::MixKernel { node } => {
                let rows_per_seq = count.tokens / count.seqs.max(1);
                let chunked = spec.runtime.chunked_scan && self.mixer.get(node as usize).and_then(|k| k.chunk_form()).is_some_and(|f| rows_per_seq >= f.min_rows as usize);
                if chunked { 3 } else { listed }
            }
            Op::MoeRoute if spec.moe.as_ref().is_some_and(|m| m.input_scaled) => listed + 1,
            _ => listed,
        }
    }
}

pub fn plan_report(spec: &Spec, phase: Phase, count: Count, target: &kir::Target) -> Result<PlanReport> {
    report_of(spec, &Plan::build(spec, phase, true), count, target)
}

pub fn compiled_report(spec: &Spec, plan: &Plan, count: Count, target: &kir::Target) -> Result<PlanReport> {
    report_of(spec, plan, count, target)
}

fn report_of(spec: &Spec, plan: &Plan, count: Count, target: &kir::Target) -> Result<PlanReport> {
    let phase = plan.phase;
    let signer = Signer::new(spec)?.ranked(ranked(plan));
    let mut sections = Vec::new();
    for s in sections_of(spec, plan) {
        let mut ops = Vec::new();
        for op in s.ops {
            let sig = signer.signature(op, s.layer)?;
            let read = sig.reads.iter().map(|t| t.bytes(&count)).sum();
            let written = sig.writes.iter().map(|t| t.bytes(&count)).sum();
            ops.push(OpCost { label: op.label(), class: sig.class, read, written, launches: signer.launches(op, phase, s.layer, &count, target) });
        }
        let (read, written, launches) = ops.iter().fold((0, 0, 0), |a, o| (a.0 + o.read, a.1 + o.written, a.2 + o.launches));
        sections.push(SectionCost { name: s.name, repeats: s.repeats, ops, read, written, launches });
    }
    Ok(PlanReport { phase, count, sections })
}

fn sections_of<'p>(spec: &Spec, plan: &'p Plan) -> Vec<Section<'p>> {
    sections(spec, plan).into_iter().filter(|s| !s.ops.is_empty()).collect()
}

impl PlanReport {
    pub fn section(&self, name: &str) -> Option<&SectionCost> {
        self.sections.iter().find(|s| s.name == name)
    }

    pub fn total(&self) -> (u64, u64, usize) {
        self.sections.iter().fold((0, 0, 0), |a, s| (a.0 + s.read * s.repeats as u64, a.1 + s.written * s.repeats as u64, a.2 + s.launches * s.repeats))
    }

    pub fn render(&self) -> String {
        let c = &self.count;
        let mut out = format!("plan {:?} tokens {} context {}\n", self.phase, c.tokens, c.context);
        for s in &self.sections {
            out.push_str(&format!("\n== {} (x{})\n", s.name, s.repeats));
            for (i, o) in s.ops.iter().enumerate() {
                out.push_str(&format!("{i:>3} {:<28} {:<11} read {:>12} write {:>12} launches {}\n", o.label, format!("{:?}", o.class), o.read, o.written, o.launches));
            }
            out.push_str(&format!("    {:<28} {:<11} read {:>12} write {:>12} launches {}\n", "total", "", s.read, s.written, s.launches));
        }
        let (read, written, launches) = self.total();
        out.push_str(&format!("\nstep total read {read} write {written} launches {launches}\n"));
        out
    }
}
