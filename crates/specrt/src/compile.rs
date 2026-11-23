use crate::fusion::{self, Costs, Part, Section, Unit};
use crate::gemm::{self, GemvLaunch};
use crate::plan::{Buf, Op, Phase, Plan, QuantSource, Weight, weight_kind};
use crate::signature::{Count, Place, Signer, layers_using, ranked};
use crate::spec::Spec;
use crate::tp::Shard;
use anyhow::{Result, bail};
use kir::gemm::Epilogue;
use std::collections::{BTreeMap, BTreeSet};

pub const PARTITION: &str = "partition";
pub const GEMV_LAYOUT: &str = "gemv_layout";
pub const FUSE: &str = "fuse";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Planes,
    Gemv,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub shards: BTreeMap<String, Shard>,
    pub fuse: bool,
    pub family: Family,
    pub context: usize,
}

impl Options {
    pub fn planes(context: usize) -> Options {
        Options { shards: BTreeMap::new(), fuse: true, family: Family::Planes, context }
    }

    pub fn ranked(shards: BTreeMap<String, Shard>, fuse: bool, context: usize) -> Options {
        Options { shards, fuse, family: Family::Planes, context }
    }

    pub fn gemv(context: usize) -> Options {
        Options { shards: BTreeMap::new(), fuse: true, family: Family::Gemv, context }
    }
}

pub fn partition(ops: &[Op], shards: &BTreeMap<String, Shard>) -> Vec<Op> {
    let mut out = Vec::with_capacity(ops.len() + 4);
    for op in ops {
        match *op {
            Op::Linear { weight, quant, input, output: Buf::X } if shards.get(weight.name()).is_some_and(|s| s.needs_allreduce()) => {
                out.push(Op::Linear { weight, quant, input, output: Buf::Partial });
                out.push(Op::AllReduce);
            }
            Op::MoeCombine if !shards.is_empty() => {
                out.push(Op::MoeCombine);
                out.push(Op::AllReduce);
            }
            op => out.push(op),
        }
    }
    out
}

fn writes_x(signer: &Signer, op: &Op, layer: usize) -> Result<bool> {
    Ok(signer.signature(op, layer)?.writes.iter().any(|t| t.place == Place::Buf(Buf::X)))
}

fn row_form(b: Buf) -> Buf {
    match b {
        Buf::XnPlanes => Buf::Xn,
        Buf::AttnPlanes => Buf::Attn,
        Buf::ActPlanes => Buf::Act,
        Buf::GdnPlanes => Buf::Gdn,
        b => b,
    }
}

pub fn gemv_layout(signer: &Signer, ops: &[Op], layer: usize) -> Result<Vec<Op>> {
    #[derive(PartialEq)]
    enum Alias {
        None,
        X,
        Stale,
    }
    let mut alias = Alias::None;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        let read = |b: Buf, alias: &Alias| -> Result<Buf> {
            let b = row_form(b);
            Ok(match (b, alias) {
                (Buf::Xn, Alias::X) => Buf::X,
                (Buf::Xn, Alias::Stale) => bail!("pass {GEMV_LAYOUT}: {} reads a copy of x that later ops changed", op.label()),
                (b, _) => b,
            })
        };
        let lowered = match *op {
            Op::Split { from: Buf::X, into } if row_form(into) == Buf::Xn => {
                alias = Alias::X;
                continue;
            }
            Op::Splice => continue,
            Op::Norm { which, into } => {
                alias = Alias::None;
                Op::Norm { which, into: row_form(into) }
            }
            Op::Linear { weight, quant, input, output } => Op::Linear { weight, quant, input: read(input, &alias)?, output: row_form(output) },
            Op::DecodeAttention => Op::StepAttention,
            Op::GdnDecode => Op::GdnStep,
            Op::Embed | Op::Clear { .. } | Op::NormAdd { .. } | Op::ScaledAdd { .. } | Op::QkNorm | Op::LogitScale | Op::LogitSoftcap | Op::MoeRoute | Op::MoeExperts | Op::MoeCombine => *op,
            ref other => bail!("pass {GEMV_LAYOUT}: {} has no single row form", other.label()),
        };
        if alias == Alias::X && writes_x(signer, &lowered, layer)? {
            alias = Alias::Stale;
        }
        out.push(lowered);
    }
    Ok(out)
}

struct Ctx<'a> {
    spec: &'a Spec,
    opts: &'a Options,
    ops: &'a [Op],
    layers: &'a [usize],
    writes_x: Vec<bool>,
}

fn launch(spec: &Spec, weight: Weight, quant: QuantSource, layer: usize, normed: bool) -> Option<GemvLaunch> {
    gemm::gemv(spec, weight.name(), weight_kind(quant.resolve_for(spec, layer, weight)), normed)
}

impl Ctx<'_> {
    fn every_layer(&self, weight: Weight, quant: QuantSource, normed: bool, want: impl Fn(&GemvLaunch) -> bool) -> bool {
        self.layers.iter().all(|&l| launch(self.spec, weight, quant, l, normed).is_some_and(|g| want(&g)))
    }

    fn realise(&self, units: &[&Unit]) -> Option<(usize, Op)> {
        let ops: BTreeSet<usize> = units.iter().map(|u| u.op).collect();
        let ops: Vec<usize> = ops.into_iter().collect();
        match ops[..] {
            [i] => Some((i, self.ops[i])),
            [a, b] => match (self.ops[a], self.ops[b]) {
                (Op::AllReduce, Op::Norm { which, into: Buf::XnPlanes }) if self.opts.fuse && b == a + 1 => Some((b, Op::AllReduceNorm { which })),
                (Op::Norm { which, into: Buf::Xn }, Op::Linear { weight, quant, input: Buf::Xn, output }) if self.opts.family == Family::Gemv && !self.writes_x[a + 1..b].contains(&true) => {
                    let parts: Vec<Part> = units.iter().filter(|u| u.op == b).map(|u| u.part).collect();
                    let act = parts.contains(&Part::Act);
                    let whole = parts.contains(&Part::Main) && parts.iter().all(|p| matches!(p, Part::Main | Part::Act));
                    (whole && self.every_layer(weight, quant, true, |g| g.act == act)).then_some((b, Op::Gemv { norm: Some(which), weight, quant, input: Buf::X, output }))
                }
                (Op::Clear { buf }, Op::Linear { weight, quant, input, output }) if self.opts.family == Family::Gemv && output == buf => {
                    let parts: Vec<Part> = units.iter().map(|u| u.part).collect();
                    let whole = parts.contains(&Part::Main) && parts.contains(&Part::Residual) && parts.iter().all(|p| matches!(p, Part::Whole | Part::Main | Part::Residual));
                    (whole && self.layers.iter().all(|&l| gemm::gemv_into(self.spec, weight.name(), weight_kind(quant.resolve_for(self.spec, l, weight)), false, true).is_some())).then_some((b, Op::GemvStore { weight, quant, input, output }))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn unfused(&self, op: Op, out: &mut Vec<Op>) -> Result<()> {
        let Op::Linear { weight, quant, input, output } = op else {
            out.push(op);
            return Ok(());
        };
        if self.opts.family == Family::Planes {
            out.push(op);
            return Ok(());
        }
        if !self.every_layer(weight, quant, false, |_| true) {
            bail!("pass {GEMV_LAYOUT}: no gemv kernel runs {} on its own", weight.name());
        }
        let shape = gemm::linear_shape(self.spec, weight.name());
        if shape.epi == Epilogue::Act && output == Buf::Act {
            out.push(Op::Gemv { norm: None, weight, quant, input, output: Buf::GateUp });
            out.push(Op::Swiglu);
        } else {
            out.push(Op::Gemv { norm: None, weight, quant, input, output });
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn fuse(spec: &Spec, signer: &Signer, target: &kir::Target, costs: &Costs, opts: &Options, phase: Phase, layers: &[usize], ops: &[Op]) -> Result<Vec<Op>> {
    let layer = layers.first().copied().unwrap_or(0);
    if ops.is_empty() {
        return Ok(Vec::new());
    }
    let writes_x = ops.iter().map(|o| writes_x(signer, o, layer)).collect::<Result<Vec<bool>>>()?;
    let ctx = Ctx { spec, opts, ops, layers, writes_x };
    let library = |units: &[&Unit]| ctx.realise(units).is_some();
    let count = Count::new(phase, 1, opts.context);
    let section = Section { name: String::new(), repeats: layers.len(), phase, layer, ops };
    let fused = fusion::fuse_ops(spec, signer, target, count, costs, section, Some(&library))?;
    let mut at: BTreeMap<usize, Op> = BTreeMap::new();
    let mut dropped: BTreeSet<usize> = BTreeSet::new();
    for g in fused.groups.iter().filter(|g| g.ops.len() > 1) {
        let units: Vec<&Unit> = g.members.iter().map(|&u| &fused.units[u]).collect();
        let complete = fused.units.iter().filter(|u| g.ops.contains(&u.op)).count() == units.len();
        let Some((pos, op)) = ctx.realise(&units).filter(|_| complete) else {
            bail!("pass {FUSE}: group [{}] has no kernel", g.units.join(" + "));
        };
        dropped.extend(g.ops.iter().copied().filter(|&o| o != pos));
        at.insert(pos, op);
    }
    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        match at.get(&i) {
            Some(f) => out.push(*f),
            None if dropped.contains(&i) => {}
            None => ctx.unfused(*op, &mut out)?,
        }
    }
    Ok(out)
}

fn sections_mut<'p>(spec: &Spec, plan: &'p mut Plan) -> Vec<(Vec<usize>, &'p mut Vec<Op>)> {
    let count = spec.layers.count;
    let Plan { head, blocks, tail, mtp, mtp_head, .. } = plan;
    let mut out = vec![(vec![0], head)];
    for b in blocks.iter_mut() {
        out.push((layers_using(spec, &b.layers, b.sparse), &mut b.ops));
    }
    out.push((vec![count.saturating_sub(1)], tail));
    out.push((vec![count], mtp));
    out.push((vec![count], mtp_head));
    out
}

pub fn partition_plan(spec: &Spec, mut plan: Plan, shards: &BTreeMap<String, Shard>) -> Plan {
    for (_, ops) in sections_mut(spec, &mut plan) {
        *ops = partition(ops, shards);
    }
    plan
}

pub fn compile(spec: &Spec, plan: Plan, target: &kir::Target, opts: &Options) -> Result<Plan> {
    let mut plan = partition_plan(spec, plan, &opts.shards);
    let signer = Signer::new(spec)?.ranked(ranked(&plan));
    if opts.family == Family::Gemv {
        for (layers, ops) in sections_mut(spec, &mut plan) {
            *ops = gemv_layout(&signer, ops, layers.first().copied().unwrap_or(0))?;
        }
    }
    let costs = Costs::for_target(target);
    let phase = plan.phase;
    if opts.family == Family::Planes && !(opts.fuse && ranked(&plan)) {
        return Ok(plan);
    }
    for (layers, ops) in sections_mut(spec, &mut plan) {
        *ops = fuse(spec, &signer, target, &costs, opts, phase, &layers, ops)?;
    }
    Ok(plan)
}

pub fn single(spec: &Spec, target: &kir::Target, context: usize) -> Result<Plan> {
    let mut plan = Plan::build(spec, Phase::Decode, true);
    plan.mtp.clear();
    plan.mtp_head.clear();
    compile(spec, plan, target, &Options::gemv(context))
}
