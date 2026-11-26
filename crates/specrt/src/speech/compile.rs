use super::graph::{SpeechOp, SpeechPlan};
use crate::fusion::{Costs, Fit, Geometry, Owner, Part, SectionFusion, Unit, fuse_units, unfused_units};
use crate::signature::{Count, Place, speech_signature, validate_speech};
use anyhow::{Result, bail, ensure};
use kir::conv::{ConvKernel, ConvShape, Epilogue, Family, Prologue};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Frames,
    Gemm { op: usize, shape: ConvShape, gelu: bool, t_in: usize, t_out: usize },
    Power,
    Log,
    Max,
    Scale,
    Gelu { op: usize },
}

impl Step {
    pub fn label(&self, plan: &SpeechPlan) -> String {
        match self {
            Step::Gemm { op, gelu: true, .. } => format!("{} + gelu", plan.ops[*op].label()),
            Step::Gemm { op, .. } | Step::Gelu { op } => plan.ops[*op].label(),
            other => format!("{other:?}").to_lowercase(),
        }
    }
}

pub fn gemm_shape(plan: &SpeechPlan, op: &SpeechOp) -> Option<ConvShape> {
    match op {
        SpeechOp::Dft => Some(plan.mel.dft_shape()),
        SpeechOp::Bank => Some(plan.mel.bank_shape()),
        op => op.conv_shape(),
    }
}

pub fn conv_kernel(target: &kir::Target, shape: ConvShape, gelu: bool) -> Result<ConvKernel> {
    ConvKernel::new(target, shape, Prologue::None, Epilogue { residual: false, stats: None, gelu }).map_err(anyhow::Error::msg)
}

fn geometry(target: &kir::Target, shape: ConvShape) -> Option<Geometry> {
    let k = conv_kernel(target, shape, false).ok()?;
    Some(match k.family {
        Family::Tile => Geometry { col_blocks: shape.cout.div_ceil(k.tile.bn), rows_block: k.tile.bm, owner: Owner::Tile { cols: k.tile.bn }, smem: k.smem_bytes() as u64, regs: k.regs(), k_tile: Some(k.tile.bk) },
        Family::Direct | Family::Rows => Geometry { col_blocks: shape.cout, rows_block: kir::conv::THREADS as usize, owner: Owner::Tile { cols: 1 }, smem: k.smem_bytes() as u64, regs: k.regs(), k_tile: Some(1) },
    })
}

pub fn units(plan: &SpeechPlan, target: &kir::Target) -> Result<Vec<Unit>> {
    validate_speech(plan)?;
    let mut out = Vec::new();
    for (i, op) in plan.ops.iter().enumerate() {
        let s = speech_signature(plan, op);
        let mut u = Unit::new(op.label(), i, s.class, s.reads, s.writes);
        if let Some(shape) = gemm_shape(plan, op) {
            u = u.with_part(Part::Main).with_geometry(geometry(target, shape));
            let rows = match op {
                SpeechOp::Conv { t_out, .. } => *t_out,
                _ => plan.mel.frames,
            };
            u.flops = rows as u64 * shape.flops_per_pixel();
        } else if matches!(op, SpeechOp::Gelu { .. }) {
            u = u.with_part(Part::Act);
        }
        out.push(u);
    }
    Ok(out)
}

pub fn library(plan: &SpeechPlan, target: &kir::Target, units: &[&Unit]) -> Fit {
    if units.len() == 1 {
        return Fit::Kernel;
    }
    match units {
        [a, b] if a.part == Part::Main && b.part == Part::Act => match plan.ops[a.op].conv_shape() {
            Some(shape) if conv_kernel(target, shape, true).is_ok() => Fit::Kernel,
            _ => Fit::None,
        },
        _ => Fit::None,
    }
}

fn count(plan: &SpeechPlan) -> Count {
    Count { tokens: plan.mel.frames, out: plan.frames, seqs: 1, ..Count::default() }
}

pub fn fuse(plan: &SpeechPlan, target: &kir::Target, on: bool) -> Result<SectionFusion> {
    let units = units(plan, target)?;
    if !on {
        return Ok(unfused_units(units, count(plan), format!("speech {}", plan.graph)));
    }
    let internal: BTreeSet<Place> = units.iter().flat_map(|u| u.reads.iter().chain(&u.writes)).filter(|t| matches!(&t.place, Place::Local(n) if n != "stem")).map(|t| t.place.clone()).collect();
    let lib = |u: &[&Unit]| library(plan, target, u);
    Ok(fuse_units(units, &internal, &Costs::for_target(target), count(plan), format!("speech {}", plan.graph), Some(&lib)))
}

pub fn steps(plan: &SpeechPlan, fused: &SectionFusion) -> Result<Vec<Step>> {
    let mut groups: Vec<Vec<usize>> = fused.groups.iter().map(|g| { let mut m = g.members.clone(); m.sort(); m }).collect();
    groups.sort();
    let mut out = Vec::new();
    for g in groups {
        let ops: Vec<usize> = g.iter().map(|&u| fused.units[u].op).collect();
        let first = &plan.ops[ops[0]];
        let gelu = match ops.as_slice() {
            [_] => false,
            [_, b] if matches!(plan.ops[*b], SpeechOp::Gelu { .. }) => true,
            _ => bail!("the speech fusion made a group no step runs: {ops:?}"),
        };
        out.push(match first {
            SpeechOp::Frames => Step::Frames,
            SpeechOp::Power => Step::Power,
            SpeechOp::Log => Step::Log,
            SpeechOp::Max => Step::Max,
            SpeechOp::Scale => Step::Scale,
            SpeechOp::Gelu { .. } => Step::Gelu { op: ops[0] },
            SpeechOp::Dft | SpeechOp::Bank | SpeechOp::Conv { .. } => {
                let shape = gemm_shape(plan, first).unwrap();
                let (t_in, t_out) = match first {
                    SpeechOp::Conv { t_in, t_out, .. } => (*t_in, *t_out),
                    _ => (plan.mel.frames, plan.mel.frames),
                };
                Step::Gemm { op: ops[0], shape, gelu, t_in, t_out }
            }
        });
    }
    ensure!(!out.is_empty(), "the speech plan has no steps");
    Ok(out)
}

pub fn kernels(steps: &[Step], target: &kir::Target) -> Result<Vec<ConvKernel>> {
    let mut out = Vec::new();
    for s in steps {
        if let Step::Gemm { shape, gelu, .. } = s {
            out.push(conv_kernel(target, *shape, *gelu)?);
        }
        if let Step::Gelu { .. } = s {
            bail!("an unfused gelu has no speech kernel; the stem gelu runs in the conv epilogue");
        }
    }
    out.sort_by_key(ConvKernel::name);
    out.dedup_by_key(|k| k.name());
    Ok(out)
}

pub fn report(plan: &SpeechPlan, fused: &SectionFusion, steps: &[Step]) -> String {
    let c = count(plan);
    let mut out = format!("speech plan {}, {} samples, {} frames of {} bins, stem {} x {}\n", plan.graph, plan.mel.samples, plan.mel.frames, plan.mel.bins, plan.frames, plan.width);
    let (mut read, mut written) = (0u64, 0u64);
    for (i, u) in fused.units.iter().enumerate() {
        let r: u64 = u.reads.iter().map(|t| t.bytes(&c)).sum();
        let w: u64 = u.writes.iter().map(|t| t.bytes(&c)).sum();
        out.push_str(&format!("{i:>3} {:<28} {:<11} read {r:>10} write {w:>10} flops {:>12}\n", u.label, format!("{:?}", u.class), u.flops));
        read += r;
        written += w;
    }
    out.push_str(&format!("launches {} unfused, {} fused: {}\n", fused.launches_before, steps.len(), steps.iter().map(|s| s.label(plan)).collect::<Vec<_>>().join(", ")));
    out.push_str(&format!("total read {read} write {written}\n"));
    out
}
