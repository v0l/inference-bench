use super::graph::{Slot, VaeOp, VaePlan};
use crate::fusion::{Costs, Fit, FusionReport, Geometry, Owner, Part, SectionFusion, Unit, fuse_units, unfused_units};
use crate::plan::Phase;
use crate::signature::{Class, Count, Form, Place, Rows, Tensor};
use anyhow::{Result, bail, ensure};
use kir::conv::{ConvKernel, ConvShape, Epilogue as ConvEpilogue, Family as ConvFamily, Prologue as ConvPrologue};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Dims {
    pub fn pixels(&self) -> usize {
        self.t * self.h * self.w
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shapes {
    pub levels: Vec<Dims>,
    pub op_in: Vec<usize>,
    pub op_out: Vec<usize>,
    pub channels_out: Vec<usize>,
}

impl Shapes {
    pub fn count(&self) -> Count {
        let pixels: Vec<usize> = self.levels.iter().map(Dims::pixels).collect();
        Count::grids(&pixels)
    }
}

pub fn shapes(plan: &VaePlan, latent: Dims, first: bool) -> Result<Shapes> {
    let mut levels = vec![latent];
    let mut at: BTreeMap<Slot, (usize, usize)> = BTreeMap::new();
    at.insert(Slot::X, (0, plan.latent_channels));
    let (mut op_in, mut op_out, mut channels_out) = (Vec::new(), Vec::new(), Vec::new());
    for (i, op) in plan.ops.iter().enumerate() {
        let here = |s: Slot| at.get(&s).copied().ok_or_else(|| anyhow::anyhow!("op {i} ({}) reads slot {} before any op writes it", op.label(), s.name()));
        let (level, c) = here(op.reads()[0])?;
        let need = |want: usize| -> Result<()> {
            ensure!(c == want, "op {i} ({}) reads {c} channels, it needs {want}", op.label());
            Ok(())
        };
        let (out_level, out_c) = match op {
            VaeOp::Conv { cin, cout, .. } => {
                need(*cin)?;
                (level, *cout)
            }
            VaeOp::GroupNorm { channels, .. } | VaeOp::RmsNorm { channels, .. } => {
                need(*channels)?;
                (level, c)
            }
            VaeOp::Attention { channels, .. } => {
                need(3 * channels)?;
                (level, *channels)
            }
            VaeOp::Upsample { .. } => {
                let d = levels[level];
                levels.push(Dims { t: d.t, h: 2 * d.h, w: 2 * d.w });
                (levels.len() - 1, c)
            }
            VaeOp::TimeUp { channels, .. } => {
                need(*channels)?;
                let d = levels[level];
                levels.push(Dims { t: if first { d.t } else { 2 * d.t }, ..d });
                (levels.len() - 1, c)
            }
            VaeOp::Add { with, .. } => {
                let (l2, c2) = here(*with)?;
                ensure!((l2, c2) == (level, c), "op {i} ({}) adds tensors of different shapes", op.label());
                (level, c)
            }
            VaeOp::Clamp { .. } => (level, c),
        };
        ensure!(levels.len() <= crate::signature::GRID_LEVELS, "the vae plan has more than {} grid levels", crate::signature::GRID_LEVELS);
        at.insert(op.writes(), (out_level, out_c));
        op_in.push(level);
        op_out.push(out_level);
        channels_out.push(out_c);
    }
    let (_, c) = at[&Slot::X];
    ensure!(c == plan.out_channels, "the plan ends with {c} channels in x, not {}", plan.out_channels);
    Ok(Shapes { levels, op_in, op_out, channels_out })
}

fn short(name: &str) -> String {
    name.strip_prefix("decoder.").unwrap_or(name).to_string()
}

pub fn conv_shape(op: &VaeOp, upsample: bool) -> Option<ConvShape> {
    match op {
        VaeOp::Conv { cin, cout, k, kt, .. } => Some(ConvShape { cin: *cin, cout: *cout, k: *k, kt: *kt, upsample, interleave: false }),
        VaeOp::TimeUp { channels, .. } => Some(ConvShape { cin: *channels, cout: 2 * channels, k: 1, kt: 3, upsample: false, interleave: true }),
        _ => None,
    }
}

fn geometry(t: &kir::Target, shape: ConvShape) -> Option<Geometry> {
    let k = ConvKernel::new(t, shape, ConvPrologue::None, ConvEpilogue { residual: false, stats: None }).ok()?;
    Some(match k.family {
        ConvFamily::Tile => Geometry { col_blocks: shape.cout.div_ceil(k.tile.bn), rows_block: k.tile.bm, owner: Owner::Tile { cols: k.tile.bn }, smem: k.smem_bytes() as u64, regs: k.regs(), k_tile: Some(k.tile.bk) },
        ConvFamily::Direct => Geometry { col_blocks: shape.cout, rows_block: kir::conv::THREADS as usize, owner: Owner::Tile { cols: 1 }, smem: k.smem_bytes() as u64, regs: k.regs(), k_tile: Some(1) },
    })
}

pub struct Lowered {
    pub units: Vec<Unit>,
    pub internal: BTreeSet<Place>,
    pub count: Count,
    pub flops: Vec<u64>,
}

pub fn units(plan: &VaePlan, sh: &Shapes, target: &kir::Target) -> Result<Lowered> {
    let mut units = Vec::new();
    let mut flops = Vec::new();
    let mut channels: BTreeMap<Slot, usize> = BTreeMap::from([(Slot::X, plan.latent_channels)]);
    let grid = |slot: Slot, level: usize, c: usize| Tensor::new(Place::Local(slot.name().into()), Rows::Grid(level), c, Form::F32);
    let weight = |name: &str, rows: usize, cols: usize| Tensor::new(Place::Weight(name.into()), Rows::Fixed(rows), cols, Form::F32);
    for (i, op) in plan.ops.iter().enumerate() {
        let (li, lo, cout) = (sh.op_in[i], sh.op_out[i], sh.channels_out[i]);
        let cin = channels[&op.reads()[0]];
        let input = grid(op.reads()[0], li, cin);
        let output = grid(op.writes(), lo, cout);
        let pixels = sh.levels[lo].pixels() as u64;
        let mut push = |u: Unit, f: u64| {
            units.push(u);
            flops.push(f);
        };
        match op {
            VaeOp::Conv { weights, cin, cout, k, kt, .. } => {
                let shape = conv_shape(op, false).unwrap();
                let mut reads = vec![input, weight(&short(&weights[0]), *cout, cin * k * k * kt), weight(&format!("{} bias", short(&weights[0])), 1, *cout)];
                if *kt > 1 {
                    reads.push(Tensor::new(Place::State(format!("{} frames", short(&weights[0]))), Rows::Fixed((kt - 1) * sh.levels[li].h * sh.levels[li].w), *cin, Form::F32));
                }
                let u = Unit::new(op.label().replace("decoder.", ""), i, Class::Gemm, reads, vec![output]).with_part(Part::Main).with_geometry(geometry(target, shape));
                push(u, pixels * shape.flops_per_pixel());
            }
            VaeOp::TimeUp { weight: w, channels: c, .. } => {
                let shape = conv_shape(op, false).unwrap();
                let state = Tensor::new(Place::State(format!("{} frames", short(w))), Rows::Fixed(2 * sh.levels[li].h * sh.levels[li].w), *c, Form::F32);
                let reads = vec![input, weight(&short(w), 2 * c, 3 * c), weight(&format!("{} bias", short(w)), 1, 2 * c), state.clone()];
                let u = Unit::new(op.label().replace("decoder.", ""), i, Class::Gemm, reads, vec![output, state]).with_part(Part::Main).with_geometry(geometry(target, shape));
                push(u, sh.levels[li].pixels() as u64 * shape.flops_per_pixel());
            }
            VaeOp::GroupNorm { weight: w, groups, channels: c, .. } => {
                let stats = Tensor::new(Place::Local("group stats".into()), Rows::Fixed(*groups), 2, Form::F32);
                let label = short(w);
                push(Unit::new(format!("{label} stats"), i, Class::GridReduce, vec![input.clone()], vec![stats.clone()]).with_part(Part::Stats).with_row(c / groups), 2 * pixels * *c as u64);
                let reads = vec![input, stats, weight(&label, 2, *c)];
                push(Unit::new(format!("{label} apply"), i, Class::Elementwise, reads, vec![output]).with_part(Part::Apply), 6 * pixels * *c as u64);
            }
            VaeOp::RmsNorm { weight: w, channels: c, .. } => {
                let u = Unit::new(short(w), i, Class::RowReduce, vec![input, weight(&short(w), 1, *c)], vec![output]).with_row(*c);
                push(u, 5 * pixels * *c as u64);
            }
            VaeOp::Upsample { .. } => push(Unit::new(format!("upsample {i}"), i, Class::Gather, vec![input], vec![output]).with_part(Part::Resample), 0),
            VaeOp::Attention { channels: c, .. } => {
                let per = sh.levels[li].h * sh.levels[li].w;
                push(Unit::new(format!("attention {i}"), i, Class::Attention, vec![input], vec![output]).with_part(Part::Main), 4 * (per * per * c * sh.levels[li].t) as u64);
            }
            VaeOp::Add { with, .. } => {
                let other = grid(*with, li, cin);
                push(Unit::new(format!("add {i}"), i, Class::Elementwise, vec![input, other], vec![output]).with_part(Part::Residual), pixels * cin as u64);
            }
            VaeOp::Clamp { .. } => push(Unit::new(format!("clamp {i}"), i, Class::Elementwise, vec![input], vec![output]).with_part(Part::Act), pixels * cin as u64),
        }
        channels.insert(op.writes(), cout);
    }
    let internal: BTreeSet<Place> = [Slot::T, Slot::H].iter().map(|s| Place::Local(s.name().into())).chain([Place::Local("group stats".into())]).collect();
    Ok(Lowered { units, internal, count: sh.count(), flops })
}

pub fn library(plan: &VaePlan, target: &kir::Target, units: &[&Unit]) -> Fit {
    if units.len() == 1 {
        return Fit::Kernel;
    }
    let mains: Vec<&&Unit> = units.iter().filter(|u| u.part == Part::Main && u.class == Class::Gemm).collect();
    let [main] = mains.as_slice() else { return Fit::None };
    let op = &plan.ops[main.op];
    if matches!(op, VaeOp::TimeUp { .. }) {
        return Fit::None;
    }
    let (mut apply, mut up, mut residual, mut stats, mut clamp) = (None, false, 0, None, 0);
    for u in units.iter().filter(|u| u.op != main.op) {
        match (u.part, &plan.ops[u.op]) {
            (Part::Apply, VaeOp::GroupNorm { groups, act, .. }) if u.op < main.op && apply.is_none() => apply = Some((*groups, *act == super::graph::Act::Silu)),
            (Part::Resample, VaeOp::Upsample { .. }) if u.op < main.op && !up => up = true,
            (Part::Residual, _) if u.op > main.op => residual += 1,
            (Part::Stats, VaeOp::GroupNorm { groups, .. }) if u.op > main.op && stats.is_none() => stats = Some(*groups),
            (Part::Act, VaeOp::Clamp { .. }) if u.op > main.op => clamp += 1,
            _ => return Fit::None,
        }
    }
    if residual > 1 || clamp > 1 || (apply.is_some() && up) {
        return Fit::None;
    }
    let shape = conv_shape(op, up).unwrap();
    let prologue = apply.map_or(ConvPrologue::None, |(groups, silu)| ConvPrologue::GroupApply { groups, silu });
    match ConvKernel::new(target, shape, prologue, ConvEpilogue { residual: residual == 1, stats }) {
        Ok(_) => Fit::Kernel,
        Err(_) => Fit::None,
    }
}

pub fn fuse(plan: &VaePlan, sh: &Shapes, target: &kir::Target) -> Result<(SectionFusion, Lowered)> {
    let low = units(plan, sh, target)?;
    let lib = |u: &[&Unit]| library(plan, target, u);
    let s = fuse_units(low.units.clone(), &low.internal, &Costs::for_target(target), low.count, format!("vae {}", plan.graph), Some(&lib));
    Ok((s, low))
}

pub fn unfused(plan: &VaePlan, sh: &Shapes, target: &kir::Target) -> Result<(SectionFusion, Lowered)> {
    let low = units(plan, sh, target)?;
    let s = unfused_units(low.units.clone(), low.count, format!("vae {} unfused", plan.graph));
    Ok((s, low))
}

pub fn fusion_report(s: &SectionFusion, count: Count, target: &kir::Target) -> FusionReport {
    FusionReport { phase: Phase::Prefill, count, costs: Costs::for_target(target), sections: vec![s.clone()], capture: Vec::new() }
}

pub fn plan_report(low: &Lowered, sh: &Shapes) -> String {
    let mut out = format!("vae plan, grid levels {}\n", sh.levels.iter().map(|d| format!("{}x{}x{}", d.t, d.h, d.w)).collect::<Vec<_>>().join(" "));
    let (mut read, mut written, mut flops) = (0u64, 0u64, 0u64);
    for (i, u) in low.units.iter().enumerate() {
        let r: u64 = u.reads.iter().map(|t| t.bytes(&low.count)).sum();
        let w: u64 = u.writes.iter().map(|t| t.bytes(&low.count)).sum();
        out.push_str(&format!("{i:>3} {:<44} {:<11} read {:>12} write {:>12} flops {:>14} launches {}\n", u.label, format!("{:?}", u.class), r, w, low.flops[i], u.launches));
        read += r;
        written += w;
        flops += low.flops[i];
    }
    out.push_str(&format!("\ntotal read {read} write {written} flops {flops} launches {}\n", low.units.iter().map(|u| u.launches).sum::<usize>()));
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueInfo {
    pub writer: usize,
    pub place: Place,
    pub rows: Rows,
    pub cols: usize,
    pub readers: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pro {
    None,
    Apply { op: usize, stats: usize },
    Upsample,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Conv { op: usize, input: usize, prologue: Pro, residual: Option<usize>, stats: Option<(usize, usize)>, clamp: Option<usize>, output: usize },
    TimeUp { op: usize, input: usize, output: usize },
    Stats { op: usize, input: usize, output: usize },
    Apply { op: usize, input: usize, stats: usize, output: usize },
    Rms { op: usize, input: usize, output: usize },
    Upsample { op: usize, input: usize, output: usize },
    Attention { op: usize, input: usize, output: usize },
    Add { op: usize, a: usize, b: usize, output: usize },
    Clamp { op: usize, input: usize, output: usize },
}

impl Step {
    pub fn inputs(&self) -> Vec<usize> {
        match self {
            Step::Conv { input, prologue, residual, .. } => {
                let mut v = vec![*input];
                if let Pro::Apply { stats, .. } = prologue {
                    v.push(*stats);
                }
                v.extend(residual);
                v
            }
            Step::Apply { input, stats, .. } => vec![*input, *stats],
            Step::Add { a, b, .. } => vec![*a, *b],
            Step::TimeUp { input, .. } | Step::Stats { input, .. } | Step::Rms { input, .. } | Step::Upsample { input, .. } | Step::Attention { input, .. } | Step::Clamp { input, .. } => vec![*input],
        }
    }

    pub fn outputs(&self) -> Vec<usize> {
        match self {
            Step::Conv { output, stats, .. } => {
                let mut v = vec![*output];
                v.extend(stats.map(|(_, s)| s));
                v
            }
            Step::TimeUp { output, .. } | Step::Stats { output, .. } | Step::Apply { output, .. } | Step::Rms { output, .. } | Step::Upsample { output, .. } | Step::Attention { output, .. } | Step::Add { output, .. } | Step::Clamp { output, .. } => vec![*output],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub steps: Vec<Step>,
    pub values: Vec<ValueInfo>,
    pub input: usize,
    pub output: usize,
}

fn versions(units: &[Unit]) -> (Vec<ValueInfo>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut values: Vec<ValueInfo> = vec![ValueInfo { writer: usize::MAX, place: Place::Local("x".into()), rows: Rows::Grid(0), cols: 0, readers: Vec::new() }];
    let mut current: BTreeMap<Place, usize> = BTreeMap::from([(Place::Local("x".into()), 0)]);
    let (mut reads, mut writes) = (Vec::new(), Vec::new());
    for (u, unit) in units.iter().enumerate() {
        let mut r = Vec::new();
        for t in unit.reads.iter().filter(|t| matches!(t.place, Place::Local(_))) {
            let v = current[&t.place];
            values[v].readers.push(u);
            if values[v].cols == 0 {
                values[v].cols = t.cols;
            }
            r.push(v);
        }
        let mut w = Vec::new();
        for t in unit.writes.iter().filter(|t| matches!(t.place, Place::Local(_))) {
            current.insert(t.place.clone(), values.len());
            w.push(values.len());
            values.push(ValueInfo { writer: u, place: t.place.clone(), rows: t.rows, cols: t.cols, readers: Vec::new() });
        }
        reads.push(r);
        writes.push(w);
    }
    (values, reads, writes)
}

pub fn schedule(plan: &VaePlan, low: &Lowered, fused: &SectionFusion) -> Result<Schedule> {
    let units = &low.units;
    let (values, reads, writes) = versions(units);
    let output = *writes.iter().flatten().rfind(|&&v| values[v].place == Place::Local("x".into())).unwrap_or(&0);
    let group_of: Vec<usize> = {
        let mut g = vec![usize::MAX; units.len()];
        for (gi, grp) in fused.groups.iter().enumerate() {
            for &m in &grp.members {
                g[m] = gi;
            }
        }
        g
    };
    ensure!(group_of.iter().all(|&g| g != usize::MAX), "a unit is in no group");
    let n = fused.groups.len();
    let mut deps: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (u, rs) in reads.iter().enumerate() {
        for &v in rs {
            let w = values[v].writer;
            if w != usize::MAX && group_of[w] != group_of[u] {
                deps[group_of[u]].insert(group_of[w]);
            }
        }
    }
    let mut done = vec![false; n];
    let mut order = Vec::new();
    while order.len() < n {
        let next = (0..n).filter(|&g| !done[g] && deps[g].iter().all(|&d| done[d])).min_by_key(|&g| fused.groups[g].members.iter().min().copied());
        let Some(g) = next else { bail!("the fused vae groups form a cycle") };
        done[g] = true;
        order.push(g);
    }
    let mut steps = Vec::new();
    for g in order {
        let members = &fused.groups[g].members;
        let main = members.iter().copied().find(|&m| units[m].part == Part::Main);
        let step = match main {
            Some(m) if units[m].class == Class::Gemm => {
                let op = units[m].op;
                if matches!(plan.ops[op], VaeOp::TimeUp { .. }) {
                    Step::TimeUp { op, input: reads[m][0], output: writes[m][0] }
                } else {
                    let mut input = reads[m][0];
                    let mut prologue = Pro::None;
                    let (mut residual, mut stats, mut clamp) = (None, None, None);
                    let mut out = writes[m][0];
                    let mut sorted = members.clone();
                    sorted.sort();
                    for &u in &sorted {
                        match units[u].part {
                            Part::Apply => {
                                input = reads[u][0];
                                prologue = Pro::Apply { op: units[u].op, stats: reads[u][1] };
                            }
                            Part::Resample => {
                                input = reads[u][0];
                                prologue = Pro::Upsample;
                            }
                            Part::Residual => {
                                let other = reads[u].iter().copied().find(|&v| v != out).ok_or_else(|| anyhow::anyhow!("a residual add reads the conv output twice"))?;
                                residual = Some(other);
                                out = writes[u][0];
                            }
                            Part::Stats => {
                                ensure!(reads[u][0] == out, "group stats in a conv epilogue read the stored output");
                                stats = Some((units[u].op, writes[u][0]));
                            }
                            Part::Act => {
                                clamp = Some(units[u].op);
                                out = writes[u][0];
                            }
                            _ => {}
                        }
                    }
                    Step::Conv { op, input, prologue, residual, stats, clamp, output: out }
                }
            }
            _ => {
                ensure!(members.len() == 1, "a light vae group without a conv has {} units", members.len());
                let u = members[0];
                let op = units[u].op;
                let (input, output) = (reads[u][0], writes[u][0]);
                match (&plan.ops[op], units[u].part) {
                    (VaeOp::GroupNorm { .. }, Part::Stats) => Step::Stats { op, input, output },
                    (VaeOp::GroupNorm { .. }, _) => Step::Apply { op, input, stats: reads[u][1], output },
                    (VaeOp::RmsNorm { .. }, _) => Step::Rms { op, input, output },
                    (VaeOp::Upsample { .. }, _) => Step::Upsample { op, input, output },
                    (VaeOp::Attention { .. }, _) => Step::Attention { op, input, output },
                    (VaeOp::Add { .. }, _) => Step::Add { op, a: reads[u][0], b: reads[u][1], output },
                    (VaeOp::Clamp { .. }, _) => Step::Clamp { op, input, output },
                    (other, _) => bail!("no step runs {} alone", other.label()),
                }
            }
        };
        steps.push(step);
    }
    Ok(Schedule { steps, values, input: 0, output })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arena {
    pub peak: u64,
    pub at: usize,
    pub size: u64,
    pub offsets: BTreeMap<usize, (u64, u64)>,
}

pub fn value_bytes(v: &ValueInfo, count: &Count) -> u64 {
    let elem = if v.place == Place::Local("group stats".into()) { 8 } else { 4 };
    elem * (count.rows(v.rows) * v.cols) as u64
}

pub fn lifetimes(s: &Schedule) -> BTreeMap<usize, (usize, usize)> {
    let mut life: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (i, st) in s.steps.iter().enumerate() {
        for v in st.outputs() {
            life.entry(v).or_insert((i, i));
        }
        for v in st.inputs() {
            let e = life.entry(v).or_insert((0, i));
            e.1 = e.1.max(i);
        }
    }
    if let Some(e) = life.get_mut(&s.output) {
        e.1 = s.steps.len();
    }
    life
}

pub fn arena(s: &Schedule, count: &Count) -> Arena {
    let life = lifetimes(s);
    let bytes = |v: usize| value_bytes(&s.values[v], count).div_ceil(256) * 256;
    let (mut peak, mut at) = (0, 0);
    for i in 0..=s.steps.len() {
        let live: u64 = life.iter().filter(|(_, (a, b))| *a <= i && i <= *b).map(|(v, _)| bytes(*v)).sum();
        if live > peak {
            (peak, at) = (live, i);
        }
    }
    let mut order: Vec<usize> = life.keys().copied().collect();
    order.sort_by_key(|&v| (std::cmp::Reverse(bytes(v)), v));
    let mut offsets: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    for v in order {
        let (a, b) = life[&v];
        let size = bytes(v);
        let mut busy: Vec<(u64, u64)> = offsets.iter().filter(|(o, _)| {
            let (oa, ob) = life[o];
            oa <= b && a <= ob
        }).map(|(_, r)| *r).collect();
        busy.sort();
        let mut off = 0;
        for (o, sz) in busy {
            if off + size <= o {
                break;
            }
            off = off.max(o + sz);
        }
        offsets.insert(v, (off, size));
    }
    let size = offsets.values().map(|(o, s)| o + s).max().unwrap_or(0);
    Arena { peak, at, size, offsets }
}

pub struct Compiled {
    pub plan: VaePlan,
    pub shapes: Shapes,
    pub fusion: SectionFusion,
    pub lowered: Lowered,
    pub schedule: Schedule,
    pub arena: Arena,
}

pub fn compile(plan: &VaePlan, latent: Dims, target: &kir::Target, fuse_on: bool, first: bool) -> Result<Compiled> {
    let sh = shapes(plan, latent, first)?;
    let (fusion, lowered) = if fuse_on { fuse(plan, &sh, target)? } else { unfused(plan, &sh, target)? };
    let schedule = schedule(plan, &lowered, &fusion)?;
    let arena = arena(&schedule, &lowered.count);
    Ok(Compiled { plan: plan.clone(), shapes: sh, fusion, lowered, schedule, arena })
}
