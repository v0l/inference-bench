use crate::plan::{DitOp, Op, Phase, Plan, PlanBlock, Stream};
use crate::signature::{Class, Count, Form, Place, Rows, Tensor};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const DENOISER: &str = include_str!("../graphs/denoiser.toml");

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferDecl {
    pub name: String,
    pub rows: String,
    pub cols: String,
    #[serde(default)]
    pub planes: bool,
    #[serde(default)]
    pub int: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DitNode {
    pub op: String,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default, rename = "in")]
    pub input: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub rows: Option<String>,
    #[serde(default)]
    pub set: Option<String>,
    #[serde(default)]
    pub shift: Option<u8>,
    #[serde(default)]
    pub scale: Option<u8>,
    #[serde(default)]
    pub gate: Option<u8>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub offset: Option<String>,
    #[serde(default)]
    pub add: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DitBlock {
    pub name: String,
    pub prefix: String,
    pub count: String,
    #[serde(default)]
    pub node: Vec<DitNode>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenoiserGraph {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub fields: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub symbols: BTreeMap<String, String>,
    #[serde(default)]
    pub buffer: Vec<BufferDecl>,
    #[serde(default)]
    pub input: Vec<BufferDecl>,
    #[serde(default)]
    pub output: Vec<BufferDecl>,
    #[serde(default)]
    pub head: Vec<DitNode>,
    #[serde(default)]
    pub block: Vec<DitBlock>,
    #[serde(default)]
    pub tail: Vec<DitNode>,
    #[serde(default)]
    pub step: Vec<DitNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenoiserSpec {
    pub values: BTreeMap<String, f64>,
    pub axes: Vec<usize>,
}

pub fn intern_name(s: &str) -> &'static str {
    intern(s)
}

fn intern(s: &str) -> &'static str {
    static NAMES: std::sync::Mutex<BTreeSet<&'static str>> = std::sync::Mutex::new(BTreeSet::new());
    let mut set = NAMES.lock().unwrap();
    if let Some(n) = set.get(s) {
        return n;
    }
    let n: &'static str = Box::leak(s.to_string().into_boxed_str());
    set.insert(n);
    n
}

fn lookup<'a>(configs: &'a BTreeMap<String, serde_json::Value>, path: &str) -> Option<&'a serde_json::Value> {
    let mut parts = path.split('.');
    let mut v = configs.get(parts.next()?)?;
    for p in parts {
        v = match p.parse::<usize>() {
            Ok(i) => v.get(i)?,
            Err(_) => v.get(p)?,
        };
    }
    (!v.is_null()).then_some(v)
}

impl DenoiserSpec {
    pub fn get(&self, name: &str) -> Result<f64> {
        self.values.get(name).copied().with_context(|| format!("the denoiser spec has no {name:?}"))
    }

    pub fn size(&self, name: &str) -> Result<usize> {
        if let Ok(n) = name.parse::<usize>() {
            return Ok(n);
        }
        let v = self.get(name)?;
        ensure!(v >= 0.0 && v.fract() == 0.0, "{name} = {v} is not a size");
        Ok(v as usize)
    }

    pub fn hidden(&self) -> usize {
        self.size("hidden").unwrap_or(0)
    }

    pub fn kir_shape(&self) -> kir::desc::Shape {
        let mut s = kir::desc::Shape::new();
        for (k, v) in &self.values {
            s = s.num(k, *v);
        }
        s
    }

    pub fn from_configs(graph: &DenoiserGraph, configs: &BTreeMap<String, serde_json::Value>) -> Result<DenoiserSpec> {
        let mut values = BTreeMap::new();
        let mut axes = Vec::new();
        for (name, paths) in &graph.fields {
            let Some(v) = paths.iter().find_map(|p| lookup(configs, p)) else { continue };
            match v {
                serde_json::Value::Array(items) => {
                    axes = items.iter().map(|x| x.as_u64().map(|n| n as usize).context("rope axes are whole numbers")).collect::<Result<_>>()?;
                    values.insert(name.clone(), axes.iter().sum::<usize>() as f64);
                    values.insert(format!("{name}_count"), axes.len() as f64);
                }
                serde_json::Value::Bool(b) => {
                    values.insert(name.clone(), if *b { 1.0 } else { 0.0 });
                }
                v => {
                    values.insert(name.clone(), v.as_f64().with_context(|| format!("{name} is {v}, not a number"))?);
                }
            }
        }
        values.entry("time_scale".into()).or_insert(1000.0);
        values.entry("latent_patch".into()).or_insert(1.0);
        let mut pending: Vec<(&String, kir::expr::Expr, &String)> = Vec::new();
        for (name, text) in &graph.symbols {
            pending.push((name, kir::expr::Expr::parse(text).map_err(anyhow::Error::msg)?, text));
        }
        while !pending.is_empty() {
            let before = pending.len();
            pending.retain(|(name, e, _)| match e.eval(&|n| values.get(n).copied()) {
                Some(v) => {
                    values.insert((*name).clone(), v);
                    false
                }
                None => true,
            });
            if pending.len() == before {
                let (name, _, text) = &pending[0];
                bail!("symbol {name} = {text:?} names a field the configs do not give");
            }
        }
        let spec = DenoiserSpec { values, axes };
        ensure!(spec.size("head_dim")? == spec.axes.iter().sum::<usize>(), "rope axes {:?} do not add up to head_dim {}", spec.axes, spec.size("head_dim")?);
        ensure!(spec.axes.iter().all(|a| a % 2 == 0), "rope axes {:?} must be even", spec.axes);
        Ok(spec)
    }

    pub fn from_transformer_config(json: &str, latent_patch: usize) -> Result<DenoiserSpec> {
        let mut configs = BTreeMap::new();
        configs.insert("transformer".to_string(), serde_json::from_str(json)?);
        configs.insert("vae".to_string(), serde_json::json!({ "patch_size": [latent_patch, latent_patch] }));
        DenoiserSpec::from_configs(DenoiserGraph::embedded(), &configs)
    }

    pub fn from_pipeline(dir: &Path) -> Result<DenoiserSpec> {
        let mut configs = BTreeMap::new();
        let read = |p: &Path| -> Result<serde_json::Value> { Ok(serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?)?) };
        configs.insert("pipeline".to_string(), read(&dir.join("model_index.json"))?);
        for part in ["transformer", "vae", "scheduler"] {
            let file = if part == "scheduler" { "scheduler_config.json" } else { "config.json" };
            let p = dir.join(part).join(file);
            if p.exists() {
                configs.insert(part.to_string(), read(&p)?);
            }
        }
        DenoiserSpec::from_configs(DenoiserGraph::embedded(), &configs)
    }
}

fn stream(name: &str) -> Result<Stream> {
    Ok(match name {
        "image" => Stream::Image,
        "text" => Stream::Text,
        "joint" => Stream::Joint,
        "seqs" => Stream::Seqs,
        "latent" => Stream::Latent,
        other => bail!("rows {other:?} is not image, text, joint, seqs or latent"),
    })
}

pub fn rows_of(s: Stream) -> Rows {
    match s {
        Stream::Image => Rows::Image,
        Stream::Text => Rows::Text,
        Stream::Joint => Rows::Joint,
        Stream::Seqs => Rows::Seqs,
        Stream::Latent => Rows::Latent,
    }
}

impl DitNode {
    fn need<'a>(&'a self, v: &'a Option<String>, what: &str) -> Result<&'static str> {
        v.as_deref().map(intern).with_context(|| format!("denoiser node {} needs {what}", self.op))
    }

    fn rows(&self) -> Result<Stream> {
        stream(self.rows.as_deref().unwrap_or("joint"))
    }

    pub fn lower(&self, prefix: &str) -> Result<DitOp> {
        let full = |w: &str| if prefix.is_empty() { w.to_string() } else { format!("{prefix}.{w}") };
        let from = || self.need(&self.input, "in");
        let into = || self.need(&self.out, "out");
        let set = || self.need(&self.set, "set");
        Ok(match self.op.as_str() {
            "patchify" => DitOp::Patchify { from: from()?, into: into()? },
            "unpatchify" => DitOp::Unpatchify { from: from()?, into: into()? },
            "split" => DitOp::Split { from: from()?, into: into()?, rows: self.rows()?, act: false },
            "silu_split" => DitOp::Split { from: from()?, into: into()?, rows: self.rows()?, act: true },
            "copy" => DitOp::Copy { from: from()?, into: into()?, rows: self.rows()? },
            "timestep" => DitOp::TimestepEmbed { from: from()?, into: into()? },
            "linear" => {
                let w = self.need(&self.weight, "weight")?;
                let weight = intern(&w.split('+').map(|part| full(part)).collect::<Vec<_>>().join("+"));
                DitOp::Linear { weight, input: from()?, output: into()?, rows: self.rows()?, add: self.add }
            }
            "modulate" => DitOp::Modulate { from: from()?, into: into()?, rows: self.rows()?, set: set()?, shift: self.shift.context("modulate needs shift")?, scale: self.scale.context("modulate needs scale")? },
            "gated_add" => DitOp::GatedAdd { from: from()?, into: intern("x"), rows: self.rows()?, set: set()?, gate: self.gate.context("gated_add needs gate")? },
            "glu" => DitOp::Glu { from: from()?, into: into()?, rows: self.rows()?, offset: intern(self.offset.as_deref().unwrap_or("0")) },
            "axis_rope" => DitOp::AxisRope { from: from()?, text: intern(&full(self.need(&self.text, "text")?)), image: intern(&full(self.need(&self.image, "image")?)) },
            "joint_attention" => DitOp::JointAttention { into: into()? },
            "scheduler_step" => bail!("the scheduler step is lowered with the pipeline rule (DenoiserGraph::sampling_plan)"),
            other => bail!("unknown denoiser op {other:?}"),
        })
    }
}

impl DenoiserGraph {
    pub fn parse(text: &str) -> Result<DenoiserGraph> {
        let g: DenoiserGraph = toml::from_str(text)?;
        ensure!(g.kind == "denoiser-graph", "{} is not a denoiser graph", g.name);
        Ok(g)
    }

    pub fn embedded() -> &'static DenoiserGraph {
        static GRAPH: std::sync::OnceLock<DenoiserGraph> = std::sync::OnceLock::new();
        GRAPH.get_or_init(|| DenoiserGraph::parse(DENOISER).expect("the embedded denoiser graph parses"))
    }

    pub fn plan(&self) -> Result<Plan> {
        let lower = |nodes: &[DitNode], prefix: &str| nodes.iter().map(|n| n.lower(prefix).map(Op::Dit)).collect::<Result<Vec<Op>>>();
        let mut blocks = Vec::new();
        for b in &self.block {
            blocks.push(PlanBlock { layers: vec![b.name.clone()], sparse: false, tail: false, ops: lower(&b.node, &format!("{}.{{i}}", b.prefix))? });
        }
        Ok(Plan { phase: Phase::Forward, tensor_core: true, head: lower(&self.head, "")?, blocks, per_layer: Vec::new(), tail: lower(&self.tail, "")?, mtp: Vec::new(), mtp_head: Vec::new() })
    }

    pub fn sampling_plan(&self, rule: &kir::expr::Expr, guided: bool) -> Result<Plan> {
        let mut plan = self.plan()?;
        ensure!(!self.step.is_empty(), "the denoiser graph has no [[step]] node");
        let text = rule.to_string();
        let mut names = Vec::new();
        rule.names(&mut names);
        let inputs: &[&str] = if guided { &["x", "c", "u", "g", "s0", "s1"] } else { &["x", "v", "s0", "s1"] };
        if let Some(n) = names.iter().find(|n| !inputs.contains(&n.as_str())) {
            bail!("the step rule names {n:?}; a {} step reads {}", if guided { "guided" } else { "plain" }, inputs.join(", "));
        }
        for n in &self.step {
            ensure!(n.op == "scheduler_step", "[[step]] holds scheduler_step nodes, not {}", n.op);
            let from = n.need(&n.input, "in")?;
            let into = n.need(&n.out, "out")?;
            plan.tail.push(Op::Dit(DitOp::Step { from, into, rule: intern(&text), guided }));
        }
        Ok(plan)
    }

    pub fn latent(&self) -> Result<&str> {
        self.step.first().and_then(|n| n.out.as_deref()).context("the denoiser graph has no [[step]] node")
    }

    pub fn decl(&self, name: &str) -> Option<&BufferDecl> {
        self.buffer.iter().chain(&self.input).chain(&self.output).find(|b| b.name == name)
    }

    pub fn is_input(&self, name: &str) -> bool {
        self.input.iter().any(|b| b.name == name)
    }

    pub fn is_output(&self, name: &str) -> bool {
        self.output.iter().any(|b| b.name == name)
    }

    pub fn tensor(&self, spec: &DenoiserSpec, name: &str, rows: Option<Stream>) -> Result<Tensor> {
        let d = self.decl(name).with_context(|| format!("the denoiser graph declares no buffer {name:?}"))?;
        let r = rows.map_or_else(|| stream(&d.rows), Ok)?;
        let place = if self.is_input(name) || self.is_output(name) { Place::Input(intern(name)) } else { Place::Local(name.to_string()) };
        Ok(Tensor::new(place, rows_of(r), spec.size(&d.cols)?, if d.int { Form::I32 } else if d.planes { Form::Planes } else { Form::F32 }))
    }

    pub fn sections(&self, spec: &DenoiserSpec) -> Result<Vec<(String, usize, Vec<usize>)>> {
        let mut out = vec![("head".to_string(), 1, vec![0])];
        for b in &self.block {
            let n = spec.size(&b.count)?;
            out.push((format!("layer {}", b.name), n, (0..n).collect()));
        }
        out.push(("tail".to_string(), 1, vec![0]));
        Ok(out)
    }
}

pub fn weight_names(weight: &str, block: Option<usize>) -> Vec<(String, Option<(String, String)>)> {
    weight
        .split('+')
        .map(|w| {
            let w = match block {
                Some(i) => w.replace("{i}", &i.to_string()),
                None => w.to_string(),
            };
            match w.split_once('[') {
                Some((base, range)) => {
                    let range = range.trim_end_matches(']');
                    let (a, b) = range.split_once(':').unwrap_or(("0", range));
                    (format!("{base}.weight"), Some((a.to_string(), b.to_string())))
                }
                None => (format!("{w}.weight"), None),
            }
        })
        .collect()
}

pub struct Signer<'a> {
    pub graph: &'a DenoiserGraph,
    pub spec: &'a DenoiserSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DitSignature {
    pub class: Class,
    pub reads: Vec<Tensor>,
    pub writes: Vec<Tensor>,
    pub flops: u64,
}

impl Signer<'_> {
    fn t(&self, name: &str, rows: Option<Stream>) -> Result<Tensor> {
        self.graph.tensor(self.spec, name, rows)
    }

    pub fn linear_shape(&self, weight: &str, input: &str, output: &str) -> Result<(usize, usize)> {
        let n = self.spec.size(&self.graph.decl(output).with_context(|| format!("no buffer {output}"))?.cols)?;
        let k = match weight_names(weight, Some(0)).last().and_then(|(_, r)| r.clone()) {
            Some((a, b)) => self.spec.size(&b)? - self.spec.size(&a)?,
            None => self.spec.size(&self.graph.decl(input).with_context(|| format!("no buffer {input}"))?.cols)?,
        };
        Ok((n, k))
    }

    pub fn sign(&self, op: &DitOp) -> Result<DitSignature> {
        let h = self.spec.hidden();
        let weight = |name: &str, rows: usize, cols: usize| Tensor::new(Place::Weight(name.to_string()), Rows::Fixed(rows), cols, Form::Weight(crate::spec::Quant::Bf16));
        let sig = |class, reads, writes, flops| DitSignature { class, reads, writes, flops };
        Ok(match *op {
            DitOp::Patchify { from, into } | DitOp::Unpatchify { from, into } => sig(Class::Gather, vec![self.t(from, None)?], vec![self.t(into, None)?], 0),
            DitOp::Split { from, into, rows, .. } | DitOp::Copy { from, into, rows } => sig(Class::Elementwise, vec![self.t(from, Some(rows))?], vec![self.t(into, Some(rows))?], 0),
            DitOp::TimestepEmbed { from, into } => sig(Class::Elementwise, vec![self.t(from, None)?], vec![self.t(into, None)?], 0),
            DitOp::Linear { weight: w, input, output, rows, add } => {
                let (n, k) = self.linear_shape(w, input, output)?;
                let input_t = Tensor { cols: k, ..self.t(input, Some(rows))? };
                let out = self.t(output, Some(rows))?;
                let mut reads = vec![input_t, weight(w, n, k)];
                if add {
                    reads.push(out.clone());
                }
                sig(Class::Gemm, reads, vec![out], (2 * n * k) as u64)
            }
            DitOp::Modulate { from, into, rows, set, .. } => sig(Class::RowReduce, vec![self.t(from, Some(rows))?, self.t(set, None)?], vec![self.t(into, Some(rows))?], 0),
            DitOp::GatedAdd { from, into, rows, set, .. } => {
                let x = self.t(into, Some(rows))?;
                sig(Class::Elementwise, vec![self.t(from, Some(rows))?, self.t(set, None)?, x.clone()], vec![x], 0)
            }
            DitOp::Glu { from, into, rows, .. } => sig(Class::Elementwise, vec![self.t(from, Some(rows))?], vec![self.t(into, Some(rows))?], 0),
            DitOp::AxisRope { from, text, image } => {
                let hd = self.spec.size("head_dim")?;
                let mut reads = vec![self.t(from, None)?, self.t("ids", None)?];
                for w in [text, image] {
                    for qk in ["q", "k"] {
                        reads.push(weight(&format!("{w}_{qk}.weight"), 1, hd));
                    }
                }
                sig(Class::RowReduce, reads, vec![Tensor::new(Place::Local("attention planes".into()), Rows::Joint, 3 * h, Form::Planes)], 0)
            }
            DitOp::JointAttention { into } => sig(Class::Attention, vec![Tensor::new(Place::Local("attention planes".into()), Rows::Joint, 3 * h, Form::Planes)], vec![self.t(into, None)?], 0),
            DitOp::Step { from, into, guided, .. } => {
                let latent = self.t(into, None)?;
                let velocity = self.t(from, guided.then_some(Stream::Latent))?;
                ensure!(velocity.cols == latent.cols, "the step reads {from} ({} wide) into {into} ({} wide)", velocity.cols, latent.cols);
                let mut reads = vec![velocity.clone()];
                if guided {
                    reads.push(velocity);
                }
                reads.extend([latent.clone(), self.t("step", None)?]);
                sig(Class::Elementwise, reads, vec![latent], 0)
            }
        })
    }

    pub fn attention_flops(&self, count: &Count) -> u64 {
        let s = (count.image + count.text) as u64;
        4 * s * s * self.spec.hidden() as u64 * count.seqs.max(1) as u64
    }

    pub fn flops(&self, op: &DitOp, count: &Count) -> Result<u64> {
        let s = self.sign(op)?;
        Ok(match op {
            DitOp::Linear { .. } => s.flops * count.rows(s.writes[0].rows) as u64,
            DitOp::JointAttention { .. } => self.attention_flops(count),
            _ => 0,
        })
    }
}

fn covers(written: &BTreeSet<Rows>, read: Rows) -> bool {
    written.contains(&read) || matches!(read, Rows::Latent | Rows::Image) && (written.contains(&Rows::Image) || written.contains(&Rows::Latent)) || written.contains(&Rows::Joint) && matches!(read, Rows::Image | Rows::Text) || read == Rows::Joint && written.contains(&Rows::Image) && written.contains(&Rows::Text)
}

type Defined = BTreeMap<Place, (Tensor, BTreeSet<Rows>)>;

fn check_reads(defined: &Defined, at: &str, sig: &DitSignature) -> Result<()> {
    for r in sig.reads.iter().filter(|r| matches!(r.place, Place::Local(_) | Place::Input(_))) {
        let Some((w, rows)) = defined.get(&r.place) else { bail!("{at}: reads {} before any node writes it", r.describe()) };
        if w.form != r.form {
            bail!("{at}: form mismatch, reads {} but it was written as {}", r.describe(), w.describe());
        }
        if r.cols > w.cols || !covers(rows, r.rows) {
            bail!("{at}: shape mismatch, reads {} but only {rows:?} rows of {} were written", r.describe(), w.describe());
        }
    }
    Ok(())
}

pub fn validate(graph: &DenoiserGraph, spec: &DenoiserSpec, plan: &Plan) -> Result<()> {
    let signer = Signer { graph, spec };
    let mut defined: Defined = BTreeMap::new();
    for i in &graph.input {
        let t = graph.tensor(spec, &i.name, None)?;
        defined.insert(t.place.clone(), (t.clone(), BTreeSet::from([t.rows])));
    }
    let walk = |name: &str, ops: &[Op], defined: &mut Defined| -> Result<()> {
        for (i, op) in ops.iter().enumerate() {
            let Op::Dit(d) = op else { bail!("{name} node {i}: {} is not a denoiser op", op.label()) };
            let at = format!("{name} node {i} ({})", d.label());
            let sig = signer.sign(d).with_context(|| at.clone())?;
            check_reads(defined, &at, &sig)?;
            for w in sig.writes {
                let entry = defined.entry(w.place.clone()).or_insert_with(|| (w.clone(), BTreeSet::new()));
                entry.1.insert(w.rows);
            }
        }
        Ok(())
    };
    walk("head", &plan.head, &mut defined)?;
    for b in &plan.blocks {
        walk(&format!("layer {}", b.layers.join("|")), &b.ops, &mut defined)?;
    }
    walk("tail", &plan.tail, &mut defined)?;
    for o in &graph.output {
        ensure!(defined.contains_key(&Place::Input(intern(&o.name))), "the plan never writes the output {}", o.name);
    }
    Ok(())
}

pub fn invariant_ops(graph: &DenoiserGraph, spec: &DenoiserSpec, ops: &[Op], varying: &[&str]) -> Result<Vec<bool>> {
    invariant_in(graph, spec, ops, &[], varying)
}

pub fn invariant_in(graph: &DenoiserGraph, spec: &DenoiserSpec, ops: &[Op], later: &[&[Op]], varying: &[&str]) -> Result<Vec<bool>> {
    let signer = Signer { graph, spec };
    let sign = |op: &Op| -> Result<DitSignature> {
        let Op::Dit(d) = op else { bail!("{} is not a denoiser op", op.label()) };
        signer.sign(d)
    };
    let mut rewritten: BTreeSet<Place> = BTreeSet::new();
    for op in later.iter().flat_map(|o| o.iter()) {
        rewritten.extend(sign(op)?.writes.into_iter().map(|w| w.place));
    }
    let mut moving: BTreeSet<Place> = varying.iter().map(|n| Place::Input(intern(n))).collect();
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        let sig = sign(op)?;
        let moves = sig.reads.iter().any(|r| moving.contains(&r.place)) || sig.writes.iter().any(|w| rewritten.contains(&w.place));
        if moves {
            moving.extend(sig.writes.iter().map(|w| w.place.clone()));
        }
        out.push(!moves);
    }
    Ok(out)
}

pub fn hoisted(graph: &DenoiserGraph, spec: &DenoiserSpec, plan: &Plan) -> Result<Vec<bool>> {
    let later: Vec<&[Op]> = plan.blocks.iter().map(|b| b.ops.as_slice()).chain([plan.tail.as_slice()]).collect();
    invariant_in(graph, spec, &plan.head, &later, &["latent", "timestep"])
}

#[derive(Clone, Debug, PartialEq)]
pub struct DitOpCost {
    pub label: String,
    pub class: Class,
    pub read: u64,
    pub written: u64,
    pub flops: u64,
    pub launches: usize,
    pub invariant: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DitSection {
    pub name: String,
    pub repeats: usize,
    pub ops: Vec<DitOpCost>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenoiserReport {
    pub count: Count,
    pub sections: Vec<DitSection>,
}

impl DitSection {
    pub fn totals(&self) -> (u64, u64, u64, usize) {
        self.ops.iter().fold((0, 0, 0, 0), |a, o| (a.0 + o.read, a.1 + o.written, a.2 + o.flops, a.3 + o.launches))
    }
}

impl DenoiserReport {
    pub fn total(&self) -> (u64, u64, u64, usize) {
        self.sections.iter().fold((0, 0, 0, 0), |a, s| {
            let (r, w, f, l) = s.totals();
            let n = s.repeats as u64;
            (a.0 + r * n, a.1 + w * n, a.2 + f * n, a.3 + l * s.repeats)
        })
    }

    pub fn hoisted(&self) -> (usize, usize) {
        let all: usize = self.sections.iter().map(|s| s.ops.iter().map(|o| o.launches).sum::<usize>() * s.repeats).sum();
        let invariant: usize = self.sections.iter().map(|s| s.ops.iter().filter(|o| o.invariant).map(|o| o.launches).sum::<usize>() * s.repeats).sum();
        (all, invariant)
    }

    pub fn render(&self) -> String {
        let c = &self.count;
        let mut out = format!("plan Forward image {} text {} (denoiser)\n", c.image, c.text);
        for s in &self.sections {
            out.push_str(&format!("\n== {} (x{})\n", s.name, s.repeats));
            for (i, o) in s.ops.iter().enumerate() {
                out.push_str(&format!("{i:>3} {:<44} {:<11} read {:>10} write {:>10} flops {:>12} launches {}{}\n", o.label, format!("{:?}", o.class), o.read, o.written, o.flops, o.launches, if o.invariant { " step invariant" } else { "" }));
            }
            let (r, w, f, l) = s.totals();
            out.push_str(&format!("    {:<44} {:<11} read {r:>10} write {w:>10} flops {f:>12} launches {l}\n", "total", ""));
        }
        let (r, w, f, l) = self.total();
        let (all, inv) = self.hoisted();
        out.push_str(&format!("\nstep total read {r} write {w} flops {f} launches {l}\nstep invariant launches {inv} of {all} (hoisted out of the step loop: {} per step)\n", all - inv));
        out
    }
}

pub fn report(graph: &DenoiserGraph, spec: &DenoiserSpec, plan: &Plan, count: Count) -> Result<DenoiserReport> {
    let signer = Signer { graph, spec };
    let mut sections = Vec::new();
    let mut parts: Vec<(String, usize, &[Op])> = vec![("head".into(), 1, &plan.head)];
    for (b, g) in plan.blocks.iter().zip(&graph.block) {
        parts.push((format!("layer {}", g.name), spec.size(&g.count)?, &b.ops));
    }
    parts.push(("tail".into(), 1, &plan.tail));
    for (name, repeats, ops) in parts {
        let invariant = if name == "head" { hoisted(graph, spec, plan)? } else { vec![false; ops.len()] };
        let mut costs = Vec::new();
        for (op, inv) in ops.iter().zip(invariant) {
            let Op::Dit(d) = op else { bail!("{} is not a denoiser op", op.label()) };
            let sig = signer.sign(d)?;
            let read = sig.reads.iter().map(|t| t.bytes(&count)).sum();
            let written = sig.writes.iter().map(|t| t.bytes(&count)).sum();
            costs.push(DitOpCost { label: d.label(), class: sig.class, read, written, flops: signer.flops(d, &count)?, launches: 1, invariant: inv });
        }
        sections.push(DitSection { name, repeats, ops: costs });
    }
    Ok(DenoiserReport { count, sections })
}

pub fn count(image: usize, text: usize) -> Count {
    Count { tokens: image + text, out: image, seqs: 1, context: image + text, image, text, ..Count::default() }
}

pub fn guided_count(image: usize, text: usize) -> Count {
    Count { seqs: 2, ..count(image, text) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shapes {
    pub image: usize,
    pub text: usize,
    pub patch_h: usize,
    pub patch_w: usize,
}

impl Shapes {
    pub fn joint(&self) -> usize {
        self.image + self.text
    }
}

pub fn kernel_key(op: &DitOp, graph: &DenoiserGraph, spec: &DenoiserSpec) -> Result<Option<(String, String, Vec<(String, f64)>)>> {
    let width = |name: &str| -> Result<f64> { Ok(spec.size(&graph.decl(name).with_context(|| format!("no buffer {name}"))?.cols)? as f64) };
    Ok(Some(match *op {
        DitOp::Split { from, act, .. } => {
            let w = width(from)?;
            let base = if act { "dit_silu_split" } else { "dit_split" };
            (format!("{base}_{w}"), base.into(), vec![("split_width".into(), w)])
        }
        DitOp::Copy { from, .. } => {
            let w = width(from)?;
            (format!("dit_copy_{w}"), "dit_copy".into(), vec![("split_width".into(), w)])
        }
        DitOp::Glu { from, offset, .. } => {
            let off = spec.size(offset)? as f64;
            let w = width(from)?;
            (format!("dit_glu_{w}_{off}"), "dit_glu".into(), vec![("glu_stride".into(), w), ("glu_offset".into(), off)])
        }
        DitOp::Modulate { .. } => ("dit_modulate".into(), "dit_modulate".into(), Vec::new()),
        DitOp::GatedAdd { .. } => ("dit_gated_add".into(), "dit_gated_add".into(), Vec::new()),
        DitOp::TimestepEmbed { .. } => ("dit_timestep".into(), "dit_timestep".into(), Vec::new()),
        DitOp::Patchify { .. } => ("dit_patchify".into(), "dit_patchify".into(), Vec::new()),
        DitOp::Unpatchify { .. } => ("dit_unpatchify".into(), "dit_unpatchify".into(), Vec::new()),
        DitOp::Step { guided, .. } => (if guided { "dit_step_cfg" } else { "dit_step" }.into(), "dit_step".into(), Vec::new()),
        DitOp::Linear { .. } | DitOp::AxisRope { .. } | DitOp::JointAttention { .. } => return Ok(None),
    }))
}

pub fn step_description(op: &DitOp, graph: &DenoiserGraph) -> Result<String> {
    let DitOp::Step { from, into, rule, guided } = *op else { bail!("{} is not a scheduler step", op.label()) };
    let width = &graph.decl(into).with_context(|| format!("no buffer {into}"))?.cols;
    graph.decl(from).with_context(|| format!("no buffer {from}"))?;
    let rule = kir::expr::Expr::parse(rule).map_err(anyhow::Error::msg)?;
    let rows: &[&str] = if guided { &["c", "u"] } else { &["v"] };
    let scalars: &[&str] = if guided { &["s0", "s1", "g"] } else { &["s0", "s1"] };
    let mut names = Vec::new();
    rule.names(&mut names);
    let mut t = format!("kind = \"kir-kernel\"\nfamily = \"compose\"\nname = \"dit_step\"\nthreads = 256\n\n[args]\nwidth = \"{width}\"\n\n[rows]\niter = \"grid\"\naddress = \"advance\"\nper_row = [\"p_x\", {}]\n", rows.iter().map(|r| format!("\"p_{r}\"")).collect::<Vec<_>>().join(", "));
    t.push_str("\n[[param]]\nname = \"p_x\"\nrole = \"activation\"\naccess = \"write\"\n");
    for p in rows.iter().chain(scalars) {
        t.push_str(&format!("\n[[param]]\nname = \"p_{p}\"\nrole = \"activation\"\naccess = \"read\"\n"));
    }
    t.push_str("\n[[stage]]\nop = \"tile_load\"\nname = \"x\"\nsrc = \"p_x\"\n");
    for r in rows.iter().filter(|r| names.iter().any(|n| n == *r)) {
        t.push_str(&format!("\n[[stage]]\nop = \"tile_load\"\nname = \"{r}\"\nsrc = \"p_{r}\"\n"));
    }
    for s in scalars.iter().filter(|s| names.iter().any(|n| n == *s)) {
        t.push_str(&format!("\n[[stage]]\nop = \"tile_load\"\nname = \"{s}\"\nsrc = \"p_{s}\"\nlayout = \"(width):(0)\"\nat = [\"i\"]\n"));
    }
    t.push_str(&format!("\n[[stage]]\nop = \"epilogue\"\nname = \"next\"\nexpr = \"{rule}\"\n\n[[stage]]\nop = \"tile_store\"\ndst = \"p_x\"\nvalue = \"next\"\n"));
    Ok(t)
}

pub const STEP_SCALARS: [&str; 3] = ["s0", "s1", "g"];

pub fn described_kernels(graph: &DenoiserGraph, spec: &DenoiserSpec, plan: &Plan, shapes: &Shapes, target: &kir::Target) -> Result<Vec<kir::Kernel>> {
    let registry = crate::kir_kernel::Registry::active();
    let base = spec.kir_shape().num("patch_h", shapes.patch_h as f64).num("patch_w", shapes.patch_w as f64).num("grid_elems", (shapes.image * spec.size("in_channels")?) as f64);
    let mut out: BTreeMap<String, kir::Kernel> = BTreeMap::new();
    for op in plan.head.iter().chain(plan.blocks.iter().flat_map(|b| &b.ops)).chain(&plan.tail) {
        let Op::Dit(d) = op else { continue };
        let Some((name, desc, extra)) = kernel_key(d, graph, spec)? else { continue };
        if out.contains_key(&name) {
            continue;
        }
        let mut shape = base.clone();
        for (k, v) in &extra {
            shape = shape.num(k, *v);
        }
        if desc == "dit_step" {
            let text = step_description(d, graph)?;
            let k = kir::desc::Description::parse(&text).map_err(anyhow::Error::msg)?.build(target, &shape).map_err(anyhow::Error::msg)?.renamed(&name);
            out.insert(name, k);
            continue;
        }
        let d = registry.get(&desc).with_context(|| format!("no kernel description {desc}"))?;
        let k = d.build(target, &shape).map_err(anyhow::Error::msg)?.renamed(&name);
        out.insert(name, k);
    }
    Ok(out.into_values().collect())
}
