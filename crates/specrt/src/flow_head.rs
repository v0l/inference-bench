use crate::denoiser::{DenoiserGraph, DenoiserSpec, NameKind, Signer, intern_name, name_kind, weight_names};
use crate::plan::{DitOp, Op, Phase, Plan, PlanBlock};
use crate::signature::{Class, Count, Place};
use crate::weights::Checkpoint;
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const MEANFLOW: &str = include_str!("../graphs/meanflow.toml");
pub const LSD: &str = include_str!("../graphs/lsd.toml");
pub const KIND: &str = "flow-head-graph";

pub fn graphs() -> &'static [DenoiserGraph] {
    static GRAPHS: std::sync::OnceLock<Vec<DenoiserGraph>> = std::sync::OnceLock::new();
    GRAPHS.get_or_init(|| [MEANFLOW, LSD].iter().map(|t| DenoiserGraph::parse_kind(t, KIND).expect("an embedded flow head graph parses")).collect())
}

pub fn graph(name: &str) -> Result<&'static DenoiserGraph> {
    graphs().iter().find(|g| g.name == name).with_context(|| format!("no flow head graph is named {name:?}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Session,
    Frame,
    Step,
}

impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Level::Session => "session",
            Level::Frame => "frame",
            Level::Step => "step",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlowHead {
    pub graph: &'static DenoiserGraph,
    pub spec: DenoiserSpec,
    pub root: String,
    pub start: Vec<Op>,
    pub plan: Plan,
    pub finish: Vec<Op>,
    pub steps: usize,
    pub tensors: BTreeMap<String, String>,
}

fn dit(op: &Op) -> Result<&DitOp> {
    match op {
        Op::Dit(d) => Ok(d),
        other => bail!("{} is not a flow head op", other.label()),
    }
}

impl FlowHead {
    pub fn from_config(config: &serde_json::Value, steps: Option<usize>) -> Result<FlowHead> {
        let model_type = config["model_type"].as_str().context("the config has no model_type")?;
        let family = crate::family::Registry::embedded().for_model_type(model_type).with_context(|| format!("no family runs {model_type}"))?;
        ensure!(family.head.is_set(), "the {} family declares no [head]", family.name);
        let mut values = family.head.read(config)?;
        if let Some(n) = steps {
            values.insert("num_steps".into(), n as f64);
        }
        let graph = graph(&family.head.graph)?;
        let spec = DenoiserSpec::from_values(graph, values, Vec::new())?;
        let mut head = FlowHead::new(graph, spec, &family.head.root)?;
        head.tensors = family.head.tensors.clone();
        Ok(head)
    }

    pub fn from_dir(dir: &Path, steps: Option<usize>) -> Result<FlowHead> {
        let p = dir.join("config.json");
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| p.display().to_string())?)?;
        FlowHead::from_config(&config, steps)
    }

    pub fn new(graph: &'static DenoiserGraph, spec: DenoiserSpec, root: &str) -> Result<FlowHead> {
        let lower = |nodes: &[crate::denoiser::DitNode], prefix: &str| nodes.iter().map(|n| n.lower(prefix).map(Op::Dit)).collect::<Result<Vec<Op>>>();
        let mut blocks = Vec::new();
        for b in &graph.block {
            blocks.push(PlanBlock { layers: vec![b.name.clone()], sparse: false, tail: false, ops: lower(&b.node, &format!("{root}.{}.{{i}}", b.prefix))? });
        }
        let plan = Plan { phase: Phase::Decode, tensor_core: false, head: lower(&graph.head, root)?, blocks, per_layer: Vec::new(), tail: lower(&graph.tail, root)?, mtp: Vec::new(), mtp_head: Vec::new(), window: Vec::new() };
        let steps = spec.size("num_steps")?;
        ensure!(steps >= 1, "num_steps is {steps}");
        let head = FlowHead { graph, spec, root: root.to_string(), start: lower(&graph.start, root)?, plan, finish: lower(&graph.finish, root)?, steps, tensors: BTreeMap::new() };
        head.validate()?;
        Ok(head)
    }

    pub fn validate(&self) -> Result<()> {
        let mut whole = self.plan.clone();
        whole.head = self.start.iter().chain(&self.plan.head).cloned().collect();
        whole.tail = self.plan.tail.iter().chain(&self.finish).cloned().collect();
        crate::denoiser::validate(self.graph, &self.spec, &whole)
    }

    pub fn signer(&self) -> Signer<'_> {
        Signer { graph: self.graph, spec: &self.spec }
    }

    pub fn layers(&self) -> Result<Vec<usize>> {
        self.graph.block.iter().map(|b| self.spec.size(&b.count)).collect()
    }

    pub fn step_scalars(&self) -> Vec<&str> {
        self.graph.scalars.iter().filter(|(_, e)| self.steps > 1 && e.split(|c: char| !c.is_alphanumeric() && c != '_').any(|w| w == "step")).map(|(n, _)| n.as_str()).collect()
    }

    pub fn checkpoint(&self, dir: &Path) -> Result<Checkpoint> {
        Ok(Checkpoint::open(dir)?.renamed(&self.tensors))
    }

    pub fn input_width(&self, name: &str) -> Result<usize> {
        self.spec.size(&self.graph.decl(name).with_context(|| format!("the {} head has no buffer {name}", self.graph.name))?.cols)
    }

    pub fn scalar_values(&self, step: usize, guidance: &[f32]) -> Result<BTreeMap<&str, Vec<f32>>> {
        let mut out = BTreeMap::new();
        if let (Ok(lo), Ok(hi)) = (self.spec.get("guidance_min"), self.spec.get("guidance_max")) {
            for g in guidance {
                ensure!((lo..=hi).contains(&f64::from(*g)), "guidance {g} is outside the model's range [{lo}, {hi}]");
            }
        }
        for (name, text) in &self.graph.scalars {
            let e = kir::expr::Expr::parse(text).map_err(anyhow::Error::msg)?;
            let mut v = Vec::with_capacity(guidance.len());
            for g in guidance {
                let env = |n: &str| match n {
                    "step" => Some(step as f64),
                    "guidance" => Some(f64::from(*g)),
                    other => self.spec.values.get(other).copied(),
                };
                v.push(e.eval(&env).with_context(|| format!("scalar {name} = {text:?} names a value the head does not have"))? as f32);
            }
            out.insert(name.as_str(), v);
        }
        Ok(out)
    }

    pub fn levels(&self) -> Result<Vec<Level>> {
        let signer = self.signer();
        let mut level: BTreeMap<Place, Level> = BTreeMap::new();
        for n in &self.graph.frame {
            level.insert(Place::Input(intern_name(n)), Level::Frame);
        }
        for n in self.step_scalars() {
            level.insert(Place::Input(intern_name(n)), Level::Step);
        }
        for op in &self.start {
            for w in signer.sign(dit(op)?)?.writes {
                level.insert(w.place, Level::Frame);
            }
        }
        for op in self.plan.blocks.iter().flat_map(|b| &b.ops).chain(&self.plan.tail) {
            for w in signer.sign(dit(op)?)?.writes {
                level.insert(w.place, Level::Step);
            }
        }
        let mut out = Vec::with_capacity(self.plan.head.len());
        for op in &self.plan.head {
            let sig = signer.sign(dit(op)?)?;
            let l = sig.reads.iter().chain(&sig.writes).filter_map(|t| level.get(&t.place).copied()).max().unwrap_or(Level::Session);
            for w in &sig.writes {
                let e = level.entry(w.place.clone()).or_insert(l);
                *e = (*e).max(l);
            }
            out.push(l);
        }
        Ok(out)
    }

    pub fn ops_at(&self, level: Level) -> Result<Vec<usize>> {
        Ok(self.levels()?.iter().enumerate().filter(|(_, l)| **l == level).map(|(i, _)| i).collect())
    }

    pub fn count(sessions: usize) -> Count {
        Count { tokens: sessions, out: sessions, seqs: sessions, ..Count::default() }
    }

    pub fn units(&self, ops: &[Op], sessions: usize) -> Result<(Vec<crate::fusion::Unit>, BTreeSet<Place>)> {
        use crate::fusion::{Geometry, Owner, Part, Unit};
        let signer = self.signer();
        let count = FlowHead::count(sessions);
        let mut units = Vec::new();
        let mut internal = BTreeSet::new();
        for (i, op) in ops.iter().enumerate() {
            let d = dit(op)?;
            let sig = signer.sign(d)?;
            let flops = signer.flops(d, &count)?;
            match *d {
                DitOp::Linear { .. } => {
                    let out = sig.writes[0].clone();
                    let geo = Geometry { col_blocks: out.cols, rows_block: sessions, owner: Owner::Tile { cols: 1 }, smem: 0, regs: kir::heavy::BASE_REGS + sessions as u32, k_tile: None };
                    let mut reads = sig.reads.clone();
                    reads.truncate(2);
                    let mut u = Unit::new(d.label(), i, Class::Gemm, reads, vec![out]).with_geometry(Some(geo)).with_part(Part::Main);
                    u.flops = flops;
                    units.push(u);
                }
                _ => {
                    let row = match d {
                        DitOp::Modulate { .. } => self.spec.hidden(),
                        _ => sig.writes.first().map_or(1, |t| t.cols),
                    };
                    let mut u = Unit::new(d.label(), i, sig.class, sig.reads, sig.writes).with_row(row);
                    u.flops = flops;
                    units.push(u);
                }
            }
            for t in units.last().unwrap().reads.iter().chain(&units.last().unwrap().writes) {
                if let Place::Local(n) = &t.place
                    && n != "x"
                {
                    internal.insert(t.place.clone());
                }
            }
        }
        Ok((units, internal))
    }

    pub fn block_fusion(&self, sessions: usize, target: &kir::Target) -> Result<Vec<crate::fusion::SectionFusion>> {
        let costs = crate::fusion::Costs::for_target(target);
        let mut out = Vec::new();
        for (b, g) in self.plan.blocks.iter().zip(&self.graph.block) {
            let (units, internal) = self.units(&b.ops, sessions)?;
            out.push(crate::fusion::fuse_units(units, &internal, &costs, FlowHead::count(sessions), format!("layer {}", g.name), None));
        }
        Ok(out)
    }

    pub fn gated_epilogues(&self, sessions: usize, target: &kir::Target) -> Result<Vec<BTreeSet<usize>>> {
        let mut out = Vec::new();
        for (b, f) in self.plan.blocks.iter().zip(self.block_fusion(sessions, target)?) {
            let mut fused = BTreeSet::new();
            for g in &f.groups {
                let linear = g.ops.iter().copied().find(|&o| matches!(b.ops[o], Op::Dit(DitOp::Linear { .. })));
                let gated = g.ops.iter().copied().find(|&o| matches!(b.ops[o], Op::Dit(DitOp::GatedAdd { .. })));
                if let (Some(l), Some(a)) = (linear, gated)
                    && a == l + 1
                {
                    fused.insert(l);
                }
            }
            out.push(fused);
        }
        Ok(out)
    }

    pub fn report(&self, sessions: usize, target: &kir::Target) -> Result<HeadReport> {
        let signer = self.signer();
        let count = FlowHead::count(sessions);
        let levels = self.levels()?;
        let cost = |d: &DitOp| -> Result<OpCost> {
            let sig = signer.sign(d)?;
            Ok(OpCost { label: d.label(), class: sig.class, read: sig.reads.iter().map(|t| t.bytes(&count)).sum(), written: sig.writes.iter().map(|t| t.bytes(&count)).sum(), flops: signer.flops(d, &count)?, launches: 1 })
        };
        let mut sections = Vec::new();
        for level in [Level::Session, Level::Frame, Level::Step] {
            let ops: Vec<OpCost> = self.plan.head.iter().zip(&levels).filter(|(_, l)| **l == level).map(|(o, _)| cost(dit(o)?)).collect::<Result<_>>()?;
            let name = match level {
                Level::Session => "head, once per session and guidance change".to_string(),
                Level::Frame => "head, once per frame".to_string(),
                Level::Step => format!("head, every step (x{})", self.steps),
            };
            let start = if level == Level::Frame { self.start.iter().map(|o| cost(dit(o)?)).collect::<Result<Vec<_>>>()? } else { Vec::new() };
            sections.push(HeadSection { name, level, repeats: if level == Level::Step { self.steps } else { 1 }, ops: start.into_iter().chain(ops).collect(), fused_launches: None });
        }
        let fusion = self.block_fusion(sessions, target)?;
        for ((b, g), f) in self.plan.blocks.iter().zip(&self.graph.block).zip(&fusion) {
            let n = self.spec.size(&g.count)?;
            sections.push(HeadSection { name: format!("layer {} (x{n}, every step)", g.name), level: Level::Step, repeats: n * self.steps, ops: b.ops.iter().map(|o| cost(dit(o)?)).collect::<Result<_>>()?, fused_launches: Some(f.launches_after) });
        }
        sections.push(HeadSection { name: format!("tail, every step (x{})", self.steps), level: Level::Step, repeats: self.steps, ops: self.plan.tail.iter().map(|o| cost(dit(o)?)).collect::<Result<_>>()?, fused_launches: None });
        sections.push(HeadSection { name: "finish, once per frame".into(), level: Level::Frame, repeats: 1, ops: self.finish.iter().map(|o| cost(dit(o)?)).collect::<Result<_>>()?, fused_launches: None });
        Ok(HeadReport { sessions, steps: self.steps, sections, groups: fusion.iter().flat_map(|f| f.groups.iter().filter(|g| g.units.len() > 1).map(|g| g.units.join(" + "))).collect() })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpCost {
    pub label: String,
    pub class: Class,
    pub read: u64,
    pub written: u64,
    pub flops: u64,
    pub launches: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeadSection {
    pub name: String,
    pub level: Level,
    pub repeats: usize,
    pub ops: Vec<OpCost>,
    pub fused_launches: Option<usize>,
}

impl HeadSection {
    pub fn totals(&self) -> (u64, u64, u64, usize) {
        self.ops.iter().fold((0, 0, 0, 0), |a, o| (a.0 + o.read, a.1 + o.written, a.2 + o.flops, a.3 + o.launches))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeadReport {
    pub sessions: usize,
    pub steps: usize,
    pub sections: Vec<HeadSection>,
    pub groups: Vec<String>,
}

impl HeadReport {
    pub fn per(&self, level: Level, fused: bool) -> (u64, u64, u64, usize) {
        self.sections.iter().filter(|s| (s.level == Level::Session) == (level == Level::Session)).fold((0, 0, 0, 0), |a, s| {
            let (r, w, f, l) = s.totals();
            let l = if fused { s.fused_launches.unwrap_or(l) } else { l };
            let n = s.repeats as u64;
            (a.0 + r * n, a.1 + w * n, a.2 + f * n, a.3 + l * s.repeats)
        })
    }

    pub fn render(&self) -> String {
        let mut out = format!("plan Decode sessions {} steps {} (flow head)\n", self.sessions, self.steps);
        for s in &self.sections {
            out.push_str(&format!("\n== {}\n", s.name));
            for (i, o) in s.ops.iter().enumerate() {
                out.push_str(&format!("{i:>3} {:<44} {:<11} read {:>10} write {:>9} flops {:>11} launches {}\n", o.label, format!("{:?}", o.class), o.read, o.written, o.flops, o.launches));
            }
            let (r, w, f, l) = s.totals();
            let fused = s.fused_launches.map_or(String::new(), |n| format!(" ({n} after fusion)"));
            out.push_str(&format!("    {:<44} {:<11} read {r:>10} write {w:>9} flops {f:>11} launches {l}{fused}\n", "total", ""));
        }
        let (sr, sw, sf, sl) = self.per(Level::Session, false);
        let (fr, fw, ff, fl) = self.per(Level::Frame, false);
        let (_, _, _, fused) = self.per(Level::Frame, true);
        out.push_str(&format!("\nper session and guidance change: read {sr} write {sw} flops {sf} launches {sl}\nper frame: read {fr} write {fw} flops {ff} launches {fl} ({fused} after fusion)\n"));
        for g in &self.groups {
            out.push_str(&format!("fused [{g}]\n"));
        }
        out
    }
}

pub fn frame_noise(seed: u64, frame: u64, width: usize) -> Vec<f32> {
    crate::sampler::noise(seed ^ frame.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15), width)
}

pub struct HostHead {
    pub head: FlowHead,
    linears: BTreeMap<String, (Vec<f32>, usize, usize)>,
    vectors: BTreeMap<String, Vec<f32>>,
}

pub struct HostSession {
    rows: usize,
    bufs: BTreeMap<&'static str, Vec<f32>>,
    pub ran: Vec<String>,
    guidance: Vec<f32>,
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

pub fn load_weights(head: &FlowHead, ck: &Checkpoint, mut linear: impl FnMut(String, Vec<f32>, usize, usize) -> Result<()>, mut vector: impl FnMut(String, Vec<f32>) -> Result<()>) -> Result<()> {
    let signer = head.signer();
    let layers = head.layers()?;
    let mut seen = BTreeSet::new();
    let mut each = |ops: &[Op], blocks: Option<usize>| -> Result<()> {
        for block in blocks.map_or(vec![None], |n| (0..n).map(Some).collect()) {
            for op in ops {
                match *dit(op)? {
                    DitOp::Linear { weight, input, output, bias, .. } => {
                        let key = block.map_or(weight.to_string(), |i| weight.replace("{i}", &i.to_string()));
                        if !seen.insert(key.clone()) {
                            continue;
                        }
                        let (n, k) = signer.linear_shape(weight, input, output)?;
                        let mut w = Vec::with_capacity(n * k);
                        let mut b = Vec::with_capacity(n);
                        for ((name, _), biased) in weight_names(weight, block).into_iter().zip(crate::denoiser::biased_parts(weight)) {
                            let part = ck.f32_values(&name, &ck.shape_of(&name)?)?;
                            let rows = part.len() / k.max(1);
                            w.extend(part);
                            let bname = name.replace(".weight", ".bias");
                            match bias && biased {
                                true => b.extend(ck.f32_values(&bname, &ck.shape_of(&bname)?)?),
                                false => b.extend(std::iter::repeat_n(0.0, rows)),
                            }
                        }
                        ensure!(w.len() == n * k, "{key}: {} values, expected {n} x {k}", w.len());
                        if bias {
                            ensure!(b.len() == n, "{key}: {} bias values, expected {n}", b.len());
                            vector(format!("{key}.bias"), b)?;
                        }
                        linear(key, w, n, k)?;
                    }
                    DitOp::Modulate { weight: Some(w), rms, .. } => {
                        let base = block.map_or(w.to_string(), |i| w.replace("{i}", &i.to_string()));
                        for part in if rms { &["weight"][..] } else { &["weight", "bias"][..] } {
                            let name = format!("{base}.{part}");
                            if seen.insert(name.clone()) {
                                vector(name.clone(), ck.f32_values(&name, &ck.shape_of(&name)?)?)?;
                            }
                        }
                    }
                    DitOp::RmsNorm { weight: w, .. } | DitOp::TimestepEmbed { table: Some(w), .. } => {
                        let name = block.map_or(w.to_string(), |i| w.replace("{i}", &i.to_string()));
                        if seen.insert(name.clone()) {
                            vector(name.clone(), ck.f32_values(&name, &ck.shape_of(&name)?)?)?;
                        }
                    }
                    DitOp::Map { expr, .. } => {
                        let e = kir::expr::Expr::parse(expr).map_err(anyhow::Error::msg)?;
                        let mut names = Vec::new();
                        e.names(&mut names);
                        for n in names.into_iter().filter(|n| name_kind(head.graph, &head.spec, n) == NameKind::Constant) {
                            if seen.insert(n.clone()) {
                                vector(n.clone(), ck.f32_values(&n, &ck.shape_of(&n)?)?)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    };
    each(&head.start, None)?;
    each(&head.plan.head, None)?;
    for (b, n) in head.plan.blocks.iter().zip(&layers) {
        each(&b.ops, Some(*n))?;
    }
    each(&head.plan.tail, None)?;
    each(&head.finish, None)
}

impl HostHead {
    pub fn load(head: FlowHead, dir: &Path) -> Result<HostHead> {
        let ck = head.checkpoint(dir)?;
        let mut linears = BTreeMap::new();
        let mut vectors = BTreeMap::new();
        load_weights(&head, &ck, |k, w, n, kk| {
            linears.insert(k, (w, n, kk));
            Ok(())
        }, |k, v| {
            vectors.insert(k, v);
            Ok(())
        })?;
        Ok(HostHead { head, linears, vectors })
    }

    pub fn session(&self, guidance: &[f32]) -> Result<HostSession> {
        let rows = guidance.len();
        ensure!(rows > 0, "a head batch has at least one session");
        let mut bufs = BTreeMap::new();
        for d in self.head.graph.buffer.iter().chain(&self.head.graph.input).chain(&self.head.graph.output) {
            bufs.insert(intern_name(&d.name), vec![0f32; rows * self.head.spec.size(&d.cols)?]);
        }
        let mut s = HostSession { rows, bufs, ran: Vec::new(), guidance: Vec::new() };
        self.set_guidance(&mut s, guidance)?;
        Ok(s)
    }

    pub fn set_guidance(&self, s: &mut HostSession, guidance: &[f32]) -> Result<()> {
        ensure!(guidance.len() == s.rows, "{} guidance values for {} sessions", guidance.len(), s.rows);
        s.guidance = guidance.to_vec();
        for (name, v) in self.head.scalar_values(0, guidance)? {
            s.bufs.insert(intern_name(name), v);
        }
        s.ran.clear();
        let levels = self.head.levels()?;
        for (op, l) in self.head.plan.head.iter().zip(&levels) {
            if *l == Level::Session {
                self.op(dit(op)?, None, s)?;
            }
        }
        Ok(())
    }

    pub fn frame(&self, s: &mut HostSession, hidden: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let spec = &self.head.spec;
        ensure!(hidden.len() == s.rows * self.head.input_width("hidden")? && noise.len() == s.rows * spec.size("latent_dim")?, "a frame takes one hidden row and one noise row per session");
        s.ran.clear();
        s.bufs.insert("hidden", hidden.to_vec());
        s.bufs.insert("noise", noise.to_vec());
        let levels = self.head.levels()?;
        let layers = self.head.layers()?;
        for op in &self.head.start {
            self.op(dit(op)?, None, s)?;
        }
        for (op, l) in self.head.plan.head.iter().zip(&levels) {
            if *l == Level::Frame {
                self.op(dit(op)?, None, s)?;
            }
        }
        let moving = self.head.step_scalars();
        for step in 0..self.head.steps {
            let guidance = s.guidance.clone();
            for (name, v) in self.head.scalar_values(step, &guidance)? {
                if moving.contains(&name) {
                    s.bufs.insert(intern_name(name), v);
                }
            }
            for (op, l) in self.head.plan.head.iter().zip(&levels) {
                if *l == Level::Step {
                    self.op(dit(op)?, None, s)?;
                }
            }
            for (b, n) in self.head.plan.blocks.iter().zip(&layers) {
                for i in 0..*n {
                    for op in &b.ops {
                        self.op(dit(op)?, Some(i), s)?;
                    }
                }
            }
            for op in &self.head.plan.tail {
                self.op(dit(op)?, None, s)?;
            }
        }
        for op in &self.head.finish {
            self.op(dit(op)?, None, s)?;
        }
        Ok(s.bufs[self.head.graph.output[0].name.as_str()].clone())
    }

    pub fn output<'a>(&self, s: &'a HostSession, name: &str) -> Result<&'a [f32]> {
        ensure!(self.head.graph.output.iter().any(|o| o.name == name), "the {} head has no output {name}", self.head.graph.name);
        Ok(&s.bufs[name])
    }

    fn width(&self, name: &str) -> Result<usize> {
        self.head.spec.size(&self.head.graph.decl(name).with_context(|| format!("no buffer {name}"))?.cols)
    }

    fn op(&self, d: &DitOp, block: Option<usize>, s: &mut HostSession) -> Result<()> {
        s.ran.push(d.label());
        let rows = s.rows;
        let h = self.head.spec.hidden();
        let eps = self.head.spec.get("norm_eps")? as f32;
        let at = |w: &str| block.map_or(w.to_string(), |i| w.replace("{i}", &i.to_string()));
        match *d {
            DitOp::TimestepEmbed { from, into, table } => {
                let freq = self.width(into)?;
                let out: Vec<f32> = match table {
                    Some(t) => {
                        let f = self.vectors.get(&at(t)).with_context(|| format!("timestep table {t} is not loaded"))?;
                        ensure!(f.len() * 2 == freq, "timestep table {t} has {} values for {freq} columns", f.len());
                        s.bufs[from].iter().flat_map(|t| f.iter().map(|x| (t * x).cos()).chain(f.iter().map(|x| (t * x).sin())).collect::<Vec<_>>()).collect()
                    }
                    None => {
                        let scale = self.head.spec.get("time_scale")? as f32;
                        s.bufs[from].iter().flat_map(|t| crate::denoiser_host::timestep_embedding(t * scale, freq)).collect()
                    }
                };
                s.bufs.insert(into, out);
            }
            DitOp::RmsNorm { from, into, weight, eps, .. } => {
                let g = self.vectors.get(&at(weight)).with_context(|| format!("rms norm weight {weight} is not loaded"))?.clone();
                let e = self.head.spec.get(eps)? as f32;
                let w = self.width(into)?;
                let x = s.bufs[from].clone();
                let dst = s.bufs.get_mut(into).unwrap();
                for r in 0..rows {
                    let row = &x[r * w..(r + 1) * w];
                    let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / w as f32 + e).sqrt();
                    for c in 0..w {
                        dst[r * w + c] = g[c] * row[c] * inv;
                    }
                }
            }
            DitOp::Linear { weight, input, output, add, bias, .. } => {
                let (w, n, k) = self.linears.get(&at(weight)).with_context(|| format!("linear {} is not loaded", at(weight)))?;
                let b = if bias { Some(self.vectors.get(&format!("{}.bias", at(weight))).with_context(|| format!("bias of {} is not loaded", at(weight)))?) } else { None };
                let x = s.bufs[input].clone();
                let dst = s.bufs.get_mut(output).unwrap();
                for r in 0..rows {
                    for o in 0..*n {
                        let v: f32 = w[o * k..(o + 1) * k].iter().zip(&x[r * k..(r + 1) * k]).map(|(a, b)| a * b).sum::<f32>() + b.map_or(0.0, |b| b[o]);
                        dst[r * n + o] = if add { dst[r * n + o] + v } else { v };
                    }
                }
            }
            DitOp::Split { from, into, act, .. } => {
                let v: Vec<f32> = s.bufs[from].iter().map(|v| if act { silu(*v) } else { *v }).collect();
                s.bufs.insert(into, v);
            }
            DitOp::Copy { from, into, .. } => {
                let v = s.bufs[from].clone();
                s.bufs.insert(into, v);
            }
            DitOp::Modulate { from, into, set, shift, scale, rms, weight, .. } => {
                let gamma = match weight {
                    Some(w) => Some(self.vectors.get(&format!("{}.weight", at(w))).context("modulate weight is not loaded")?.clone()),
                    None => None,
                };
                let beta = match weight {
                    Some(w) if !rms => Some(self.vectors.get(&format!("{}.bias", at(w))).context("modulate bias is not loaded")?.clone()),
                    _ => None,
                };
                let m = s.bufs[set].clone();
                let ms = self.width(set)?;
                let x = s.bufs[from].clone();
                let dst = s.bufs.get_mut(into).unwrap();
                for r in 0..rows {
                    let row = &x[r * h..(r + 1) * h];
                    let mean = if rms { 0.0 } else { row.iter().sum::<f32>() / h as f32 };
                    let inv = 1.0 / (row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32 + eps).sqrt();
                    for c in 0..h {
                        let g = gamma.as_ref().map_or(1.0, |g| g[c]);
                        let b = beta.as_ref().map_or(0.0, |b| b[c]);
                        dst[r * h + c] = ((row[c] - mean) * inv * g + b) * (1.0 + m[r * ms + scale as usize * h + c]) + m[r * ms + shift as usize * h + c];
                    }
                }
            }
            DitOp::GatedAdd { from, into, set, gate, .. } => {
                let m = s.bufs[set].clone();
                let ms = self.width(set)?;
                let b = s.bufs[from].clone();
                let dst = s.bufs.get_mut(into).unwrap();
                for r in 0..rows {
                    for c in 0..h {
                        dst[r * h + c] += m[r * ms + gate as usize * h + c] * b[r * h + c];
                    }
                }
            }
            DitOp::Glu { from, into, offset, .. } => {
                let (fw, mw, off) = (self.width(from)?, self.width(into)?, self.head.spec.size(offset)?);
                let src = s.bufs[from].clone();
                let dst = s.bufs.get_mut(into).unwrap();
                for r in 0..rows {
                    for c in 0..mw {
                        dst[r * mw + c] = silu(src[r * fw + off + c]) * src[r * fw + off + mw + c];
                    }
                }
            }
            DitOp::Map { into, expr, .. } => {
                let e = kir::expr::Expr::parse(expr).map_err(anyhow::Error::msg)?;
                let wide = self.width(into)?;
                let mut names = Vec::new();
                e.names(&mut names);
                let widths: BTreeMap<String, usize> = names.iter().filter(|n| self.head.graph.decl(n).is_some()).map(|n| Ok((n.clone(), self.width(n)?))).collect::<Result<_>>()?;
                let mut out = vec![0f32; rows * wide];
                for r in 0..rows {
                    for c in 0..wide {
                        let env = |n: &str| -> Option<f64> {
                            match name_kind(self.head.graph, &self.head.spec, n) {
                                NameKind::Buffer => {
                                    let w = widths[n];
                                    Some(f64::from(s.bufs[n][if w == wide { r * w + c } else { r }]))
                                }
                                NameKind::Constant if self.head.graph.vectors.iter().any(|v| v == n) => self.vectors.get(n).and_then(|v| v.get(c)).map(|v| f64::from(*v)),
                                NameKind::Constant => self.vectors.get(n).map(|v| f64::from(v[0])),
                                NameKind::Value => self.head.spec.values.get(n).copied(),
                                NameKind::Unknown => None,
                            }
                        };
                        out[r * wide + c] = e.eval(&env).with_context(|| format!("map {expr:?}"))? as f32;
                    }
                }
                s.bufs.insert(into, out);
            }
            ref other => bail!("{} is not a flow head op", other.label()),
        }
        Ok(())
    }
}
