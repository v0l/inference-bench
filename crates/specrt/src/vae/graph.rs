use super::expr::{self, Env, Value};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

pub const GRAPH: &str = include_str!("../../graphs/vae.toml");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphFile {
    name: String,
    kind: String,
    decoder: Vec<DecoderGraph>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecoderGraph {
    name: String,
    classes: Vec<String>,
    entry: String,
    #[serde(default)]
    defaults: Vec<(String, String)>,
    #[serde(default, rename = "let")]
    lets: Vec<(String, String)>,
    #[serde(rename = "macro")]
    macros: Vec<Macro>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Macro {
    name: String,
    node: Vec<Node>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    op: Option<String>,
    call: Option<String>,
    prefix: Option<String>,
    #[serde(default)]
    args: BTreeMap<String, String>,
    repeat: Option<String>,
    var: Option<String>,
    when: Option<String>,
    weight: Option<String>,
    #[serde(default)]
    weights: Vec<String>,
    cin: Option<String>,
    cout: Option<String>,
    k: Option<String>,
    kt: Option<String>,
    channels: Option<String>,
    groups: Option<String>,
    eps: Option<String>,
    act: Option<String>,
    #[serde(rename = "in")]
    input: Option<String>,
    out: Option<String>,
    with: Option<String>,
    lo: Option<String>,
    hi: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Slot {
    X,
    T,
    H,
}

impl Slot {
    fn parse(name: Option<&str>) -> Result<Slot> {
        Ok(match name.unwrap_or("x") {
            "x" => Slot::X,
            "t" => Slot::T,
            "h" => Slot::H,
            other => bail!("slot {other:?} is not x, t or h"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Slot::X => "x",
            Slot::T => "t",
            Slot::H => "h",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    Silu,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VaeOp {
    Conv { weights: Vec<String>, cin: usize, cout: usize, k: usize, kt: usize, input: Slot, output: Slot },
    GroupNorm { weight: String, channels: usize, groups: usize, eps: f32, act: Act, input: Slot, output: Slot },
    RmsNorm { weight: String, channels: usize, act: Act, input: Slot, output: Slot },
    Upsample { slot: Slot },
    TimeUp { weight: String, channels: usize, slot: Slot },
    Attention { channels: usize, input: Slot, output: Slot },
    Add { input: Slot, with: Slot, output: Slot },
    Clamp { lo: f32, hi: f32, slot: Slot },
}

impl VaeOp {
    pub fn label(&self) -> String {
        match self {
            VaeOp::Conv { weights, .. } => format!("conv {}", weights.join("+")),
            VaeOp::GroupNorm { weight, .. } => format!("group_norm {weight}"),
            VaeOp::RmsNorm { weight, .. } => format!("rms_norm {weight}"),
            VaeOp::Upsample { .. } => "upsample".into(),
            VaeOp::TimeUp { weight, .. } => format!("time_up {weight}"),
            VaeOp::Attention { .. } => "attention".into(),
            VaeOp::Add { input, with, output } => format!("add {} + {} -> {}", input.name(), with.name(), output.name()),
            VaeOp::Clamp { .. } => "clamp".into(),
        }
    }

    pub fn reads(&self) -> Vec<Slot> {
        match self {
            VaeOp::Conv { input, .. } | VaeOp::GroupNorm { input, .. } | VaeOp::RmsNorm { input, .. } | VaeOp::Attention { input, .. } => vec![*input],
            VaeOp::Upsample { slot } | VaeOp::TimeUp { slot, .. } | VaeOp::Clamp { slot, .. } => vec![*slot],
            VaeOp::Add { input, with, .. } => vec![*input, *with],
        }
    }

    pub fn writes(&self) -> Slot {
        match self {
            VaeOp::Conv { output, .. } | VaeOp::GroupNorm { output, .. } | VaeOp::RmsNorm { output, .. } | VaeOp::Attention { output, .. } | VaeOp::Add { output, .. } => *output,
            VaeOp::Upsample { slot } | VaeOp::TimeUp { slot, .. } | VaeOp::Clamp { slot, .. } => *slot,
        }
    }

    pub fn causal(&self) -> bool {
        matches!(self, VaeOp::Conv { kt, .. } if *kt > 1) || matches!(self, VaeOp::TimeUp { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VaePlan {
    pub graph: String,
    pub class: String,
    pub latent_channels: usize,
    pub out_channels: usize,
    pub temporal: bool,
    pub ops: Vec<VaeOp>,
}

impl VaePlan {
    pub fn spatial_scale(&self) -> usize {
        1 << self.ops.iter().filter(|o| matches!(o, VaeOp::Upsample { .. })).count()
    }

    pub fn time_ups(&self) -> usize {
        self.ops.iter().filter(|o| matches!(o, VaeOp::TimeUp { .. })).count()
    }

    pub fn output_frames(&self, latent_frames: usize) -> usize {
        let mut t = latent_frames;
        for _ in 0..self.time_ups() {
            t = 2 * t - 1;
        }
        t
    }

    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        for op in &self.ops {
            match op {
                VaeOp::Conv { weights, cin, cout, k, kt, .. } => {
                    let each = cout / weights.len();
                    for w in weights {
                        out.push((format!("{w}.weight"), vec![each, *cin, *kt, *k, *k]));
                        out.push((format!("{w}.bias"), vec![each]));
                    }
                }
                VaeOp::GroupNorm { weight, channels, .. } => {
                    out.push((format!("{weight}.weight"), vec![*channels]));
                    out.push((format!("{weight}.bias"), vec![*channels]));
                }
                VaeOp::RmsNorm { weight, channels, .. } => out.push((format!("{weight}.gamma"), vec![*channels])),
                VaeOp::TimeUp { weight, channels, .. } => {
                    out.push((format!("{weight}.weight"), vec![2 * channels, *channels, 3, 1, 1]));
                    out.push((format!("{weight}.bias"), vec![2 * channels]));
                }
                _ => {}
            }
        }
        out
    }

    pub fn check_tensors(&self, shape_of: &dyn Fn(&str) -> Result<Vec<usize>>) -> Result<()> {
        for (name, want) in self.tensors() {
            let have = shape_of(&name).with_context(|| format!("the {} plan needs {name}", self.class))?;
            let product = |s: &[usize]| s.iter().product::<usize>();
            ensure!(product(&have) == product(&want) && have[0] == want[0] && (want.len() == 1 || have[1] == want[1]), "{name} is {have:?}, the plan needs {want:?}");
        }
        Ok(())
    }
}

fn graph() -> Result<GraphFile> {
    let g: GraphFile = toml::from_str(GRAPH).context("graphs/vae.toml")?;
    ensure!(g.kind == "graph" && g.name == "vae", "graphs/vae.toml is not the vae graph");
    Ok(g)
}

pub fn decoder_classes() -> Vec<String> {
    graph().map(|g| g.decoder.iter().flat_map(|d| d.classes.clone()).collect()).unwrap_or_default()
}

fn config_env(config: &serde_json::Value) -> Env {
    let mut env = Env::new();
    if let Some(o) = config.as_object() {
        for (k, v) in o {
            if let Some(v) = Value::from_json(v) {
                env.insert(k.clone(), v);
            }
        }
    }
    env
}

struct Expander<'a> {
    graph: &'a DecoderGraph,
    ops: Vec<VaeOp>,
    depth: usize,
}

impl Expander<'_> {
    fn count(&self, text: &Option<String>, env: &Env, what: &str, default: Option<usize>) -> Result<usize> {
        match (text, default) {
            (Some(t), _) => expr::count(t, env),
            (None, Some(d)) => Ok(d),
            (None, None) => bail!("a node needs {what}"),
        }
    }

    fn act(name: &Option<String>) -> Result<Act> {
        Ok(match name.as_deref() {
            None | Some("none") => Act::None,
            Some("silu") => Act::Silu,
            Some(other) => bail!("activation {other:?} is not silu or none"),
        })
    }

    fn op(&mut self, n: &Node, env: &Env) -> Result<()> {
        let op = n.op.as_deref().unwrap_or_default();
        let input = Slot::parse(n.input.as_deref())?;
        let output = Slot::parse(n.out.as_deref())?;
        let name = |w: &Option<String>| -> Result<String> { expr::interpolate(w.as_deref().context("a node needs weight")?, env) };
        let made = match op {
            "conv" => {
                let weights = if n.weights.is_empty() { vec![name(&n.weight)?] } else { n.weights.iter().map(|w| expr::interpolate(w, env)).collect::<Result<_>>()? };
                let cout = self.count(&n.cout, env, "cout", None)?;
                ensure!(cout.is_multiple_of(weights.len()), "cout {cout} does not split over {} weights", weights.len());
                let k = self.count(&n.k, env, "k", None)?;
                ensure!(k % 2 == 1, "conv k {k} must be odd for same padding");
                VaeOp::Conv { weights, cin: self.count(&n.cin, env, "cin", None)?, cout, k, kt: self.count(&n.kt, env, "kt", Some(1))?, input, output }
            }
            "group_norm" => {
                let channels = self.count(&n.channels, env, "channels", None)?;
                let groups = self.count(&n.groups, env, "groups", None)?;
                ensure!(groups > 0 && channels.is_multiple_of(groups), "group_norm over {channels} channels in {groups} groups");
                let eps = expr::number(n.eps.as_deref().unwrap_or("0.000001"), env)? as f32;
                VaeOp::GroupNorm { weight: name(&n.weight)?, channels, groups, eps, act: Self::act(&n.act)?, input, output }
            }
            "rms_norm" => VaeOp::RmsNorm { weight: name(&n.weight)?, channels: self.count(&n.channels, env, "channels", None)?, act: Self::act(&n.act)?, input, output },
            "upsample" => VaeOp::Upsample { slot: input },
            "time_up" => VaeOp::TimeUp { weight: name(&n.weight)?, channels: self.count(&n.channels, env, "channels", None)?, slot: input },
            "attention" => VaeOp::Attention { channels: self.count(&n.channels, env, "channels", None)?, input, output },
            "add" => VaeOp::Add { input, with: Slot::parse(n.with.as_deref())?, output },
            "clamp" => VaeOp::Clamp { lo: expr::number(n.lo.as_deref().unwrap_or("-1"), env)? as f32, hi: expr::number(n.hi.as_deref().unwrap_or("1"), env)? as f32, slot: input },
            other => bail!("vae graph op {other:?} is not known"),
        };
        self.ops.push(made);
        Ok(())
    }

    fn call(&mut self, n: &Node, env: &Env) -> Result<()> {
        let name = n.call.as_deref().unwrap_or_default();
        let m = self.graph.macros.iter().find(|m| m.name == name).with_context(|| format!("macro {name} is not defined"))?.clone();
        let mut inner = env.clone();
        if let Some(p) = &n.prefix {
            inner.insert("prefix".into(), Value::Text(expr::interpolate(p, env)?));
        }
        for (k, v) in &n.args {
            inner.insert(k.clone(), expr::eval(v, env)?);
        }
        self.depth += 1;
        ensure!(self.depth < 16, "macro {name} nests too deep");
        for node in &m.node {
            self.node(node, &inner)?;
        }
        self.depth -= 1;
        Ok(())
    }

    fn node(&mut self, n: &Node, env: &Env) -> Result<()> {
        let times = match &n.repeat {
            Some(r) => expr::count(r, env)?,
            None => 1,
        };
        for i in 0..times {
            let mut env = env.clone();
            if let Some(v) = &n.var {
                env.insert(v.clone(), Value::Num(i as f64));
            }
            if let Some(w) = &n.when
                && expr::number(w, &env)? == 0.0
            {
                continue;
            }
            match (&n.op, &n.call) {
                (Some(_), None) => self.op(n, &env)?,
                (None, Some(_)) => self.call(n, &env)?,
                _ => bail!("a vae graph node has exactly one of op and call"),
            }
        }
        Ok(())
    }
}

pub fn plan(config: &serde_json::Value) -> Result<VaePlan> {
    let class = config["_class_name"].as_str().context("the vae config names no _class_name")?.to_string();
    let g = graph()?;
    let d = g.decoder.iter().find(|d| d.classes.contains(&class)).with_context(|| format!("no vae graph decodes {class}"))?;
    let mut env = config_env(config);
    for (k, v) in &d.defaults {
        if !env.contains_key(k) {
            let value = expr::eval(v, &env)?;
            env.insert(k.clone(), value);
        }
    }
    for (k, v) in &d.lets {
        let value = expr::eval(v, &env)?;
        env.insert(k.clone(), value);
    }
    let mut x = Expander { graph: d, ops: Vec::new(), depth: 0 };
    x.call(&Node { call: Some(d.entry.clone()), ..Node::default() }, &env)?;
    let first_cin = match x.ops.first() {
        Some(VaeOp::Conv { cin, .. }) => *cin,
        _ => bail!("a vae plan starts with a conv"),
    };
    let last_cout = x.ops.iter().rev().find_map(|o| if let VaeOp::Conv { cout, .. } = o { Some(*cout) } else { None }).context("a vae plan has a conv")?;
    let temporal = x.ops.iter().any(VaeOp::causal);
    Ok(VaePlan { graph: d.name.clone(), class, latent_channels: first_cin, out_channels: last_cout, temporal, ops: x.ops })
}

pub fn load_plan(vae_dir: &Path) -> Result<VaePlan> {
    let text = std::fs::read_to_string(vae_dir.join("config.json")).with_context(|| vae_dir.join("config.json").display().to_string())?;
    plan(&serde_json::from_str(&text)?)
}
