use crate::vae::expr::{self, Env, Value};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;

pub const GRAPH: &str = include_str!("../../graphs/codec.toml");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphFile {
    name: String,
    kind: String,
    codec: Vec<CodecGraph>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CodecGraph {
    name: String,
    model_types: Vec<String>,
    section: String,
    encoder: String,
    decoder: String,
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
    cin: Option<String>,
    cout: Option<String>,
    k: Option<String>,
    stride: Option<String>,
    groups: Option<String>,
    channels: Option<String>,
    eps: Option<String>,
    act: Option<String>,
    #[serde(rename = "in")]
    input: Option<String>,
    out: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Slot {
    X,
    T,
}

impl Slot {
    fn parse(name: Option<&str>) -> Result<Slot> {
        Ok(match name.unwrap_or("x") {
            "x" => Slot::X,
            "t" => Slot::T,
            other => bail!("codec slot {other:?} is not x or t"),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CodecOp {
    Conv { weight: String, cin: usize, cout: usize, k: usize, stride: usize, depthwise: bool, input: Slot, output: Slot },
    ConvUp { weight: String, cin: usize, cout: usize, stride: usize },
    RmsNorm { weight: String, channels: usize, eps: f32, input: Slot, output: Slot },
    Linear { weight: String, cin: usize, cout: usize, gelu: bool, input: Slot, output: Slot },
    ScaleAdd { weight: String, channels: usize, input: Slot },
}

impl CodecOp {
    pub fn label(&self) -> String {
        match self {
            CodecOp::Conv { weight, k, stride, depthwise, .. } => format!("{} {weight} k{k} s{stride}", if *depthwise { "dwconv" } else { "conv" }),
            CodecOp::ConvUp { weight, stride, .. } => format!("conv_up {weight} s{stride}"),
            CodecOp::RmsNorm { weight, .. } => format!("rms_norm {weight}"),
            CodecOp::Linear { weight, gelu, .. } => format!("linear {weight}{}", if *gelu { " gelu" } else { "" }),
            CodecOp::ScaleAdd { weight, .. } => format!("scale_add {weight}"),
        }
    }

    pub fn state_floats(&self) -> usize {
        match self {
            CodecOp::Conv { cin, k, stride, .. } => cin * (k - stride),
            CodecOp::ConvUp { cin, .. } => *cin,
            _ => 0,
        }
    }

    pub fn reference_state_floats(&self) -> usize {
        match self {
            CodecOp::ConvUp { cout, stride, .. } => cout * stride,
            op => op.state_floats(),
        }
    }

    pub fn state_cols(&self) -> usize {
        match self {
            CodecOp::Conv { k, stride, .. } => k - stride,
            CodecOp::ConvUp { .. } => 1,
            _ => 0,
        }
    }

    pub fn input(&self) -> Slot {
        match self {
            CodecOp::Conv { input, .. } | CodecOp::RmsNorm { input, .. } | CodecOp::Linear { input, .. } | CodecOp::ScaleAdd { input, .. } => *input,
            CodecOp::ConvUp { .. } => Slot::X,
        }
    }

    pub fn output(&self) -> Slot {
        match self {
            CodecOp::Conv { output, .. } | CodecOp::RmsNorm { output, .. } | CodecOp::Linear { output, .. } => *output,
            CodecOp::ConvUp { .. } | CodecOp::ScaleAdd { .. } => Slot::X,
        }
    }

    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        match self {
            CodecOp::Conv { weight, cin, cout, k, depthwise, .. } => vec![(format!("{weight}.weight"), vec![*cout, if *depthwise { 1 } else { *cin }, *k]), (format!("{weight}.bias"), vec![*cout])],
            CodecOp::ConvUp { weight, cin, cout, stride } => vec![(format!("{weight}.weight"), vec![*cin, *cout, 2 * stride]), (format!("{weight}.bias"), vec![*cout])],
            CodecOp::RmsNorm { weight, channels, .. } => vec![(format!("{weight}.weight"), vec![*channels])],
            CodecOp::Linear { weight, cin, cout, .. } => vec![(format!("{weight}.weight"), vec![*cout, *cin]), (format!("{weight}.bias"), vec![*cout])],
            CodecOp::ScaleAdd { weight, channels, .. } => vec![(weight.clone(), vec![*channels])],
        }
    }

    pub fn time_scale(&self) -> (usize, usize) {
        match self {
            CodecOp::Conv { stride, .. } => (1, *stride),
            CodecOp::ConvUp { stride, .. } => (*stride, 1),
            _ => (1, 1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Encode,
    Decode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub direction: Direction,
    pub ops: Vec<CodecOp>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub in_per_frame: usize,
    pub out_per_frame: usize,
}

impl Stack {
    pub fn state_floats(&self) -> usize {
        self.ops.iter().map(CodecOp::state_floats).sum()
    }

    pub fn reference_state_floats(&self) -> usize {
        self.ops.iter().map(CodecOp::reference_state_floats).sum()
    }

    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        self.ops.iter().flat_map(CodecOp::tensors).collect()
    }

    pub fn params(&self) -> usize {
        self.tensors().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }

    pub fn steps(&self, frames: usize) -> Vec<usize> {
        let mut t = frames * self.in_per_frame;
        let mut out = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let (up, down) = op.time_scale();
            t = t.div_ceil(down) * up;
            out.push(t);
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CodecPlan {
    pub graph: String,
    pub encoder: Stack,
    pub decoder: Stack,
    pub frame_samples: usize,
    pub latent_dim: usize,
}

impl CodecPlan {
    pub fn check_tensors(&self, shape_of: &dyn Fn(&str) -> Result<Vec<usize>>) -> Result<()> {
        for (name, want) in self.encoder.tensors().into_iter().chain(self.decoder.tensors()) {
            let have = shape_of(&name).with_context(|| format!("the {} codec plan needs {name}", self.graph))?;
            ensure!(have == want, "{name} is {have:?}, the codec plan needs {want:?}");
        }
        Ok(())
    }
}

fn graph() -> Result<GraphFile> {
    let g: GraphFile = toml::from_str(GRAPH).context("graphs/codec.toml")?;
    ensure!(g.kind == "graph" && g.name == "codec", "graphs/codec.toml is not the codec graph");
    Ok(g)
}

pub fn model_types() -> Vec<String> {
    graph().map(|g| g.codec.iter().flat_map(|d| d.model_types.clone()).collect()).unwrap_or_default()
}

struct Expander<'a> {
    graph: &'a CodecGraph,
    ops: Vec<CodecOp>,
    depth: usize,
}

impl Expander<'_> {
    fn count(text: &Option<String>, env: &Env, what: &str, default: Option<usize>) -> Result<usize> {
        match (text, default) {
            (Some(t), _) => expr::count(t, env),
            (None, Some(d)) => Ok(d),
            (None, None) => bail!("a codec node needs {what}"),
        }
    }

    fn op(&mut self, n: &Node, env: &Env) -> Result<()> {
        let input = Slot::parse(n.input.as_deref())?;
        let output = Slot::parse(n.out.as_deref())?;
        let weight = expr::interpolate(n.weight.as_deref().context("a codec node needs weight")?, env)?;
        let made = match n.op.as_deref().unwrap_or_default() {
            "conv" => {
                let (cin, cout) = (Self::count(&n.cin, env, "cin", None)?, Self::count(&n.cout, env, "cout", None)?);
                let k = Self::count(&n.k, env, "k", None)?;
                let stride = Self::count(&n.stride, env, "stride", Some(1))?;
                let groups = Self::count(&n.groups, env, "groups", Some(1))?;
                ensure!(stride >= 1 && k >= stride, "a causal conv with k {k} and stride {stride}");
                ensure!(groups == 1 || (groups == cin && cin == cout), "a codec conv is dense or depthwise, not {groups} groups over {cin} to {cout}");
                CodecOp::Conv { weight, cin, cout, k, stride, depthwise: groups > 1, input, output }
            }
            "conv_up" => {
                ensure!(input == Slot::X && output == Slot::X, "conv_up runs on x");
                CodecOp::ConvUp { weight, cin: Self::count(&n.cin, env, "cin", None)?, cout: Self::count(&n.cout, env, "cout", None)?, stride: Self::count(&n.stride, env, "stride", None)? }
            }
            "rms_norm" => CodecOp::RmsNorm { weight, channels: Self::count(&n.channels, env, "channels", None)?, eps: expr::number(n.eps.as_deref().unwrap_or("0.00001"), env)? as f32, input, output },
            "linear" => {
                let gelu = match n.act.as_deref() {
                    None | Some("none") => false,
                    Some("gelu") => true,
                    Some(other) => bail!("codec activation {other:?} is not gelu or none"),
                };
                CodecOp::Linear { weight, cin: Self::count(&n.cin, env, "cin", None)?, cout: Self::count(&n.cout, env, "cout", None)?, gelu, input, output }
            }
            "scale_add" => CodecOp::ScaleAdd { weight, channels: Self::count(&n.channels, env, "channels", None)?, input },
            other => bail!("codec graph op {other:?} is not known"),
        };
        self.ops.push(made);
        Ok(())
    }

    fn call(&mut self, n: &Node, env: &Env) -> Result<()> {
        let name = n.call.as_deref().unwrap_or_default();
        let m = self.graph.macros.iter().find(|m| m.name == name).with_context(|| format!("codec macro {name} is not defined"))?.clone();
        let mut inner = env.clone();
        if let Some(p) = &n.prefix {
            inner.insert("prefix".into(), Value::Text(expr::interpolate(p, env)?));
        }
        for (k, v) in &n.args {
            inner.insert(k.clone(), expr::eval(v, env)?);
        }
        self.depth += 1;
        ensure!(self.depth < 16, "codec macro {name} nests too deep");
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
                _ => bail!("a codec graph node has exactly one of op and call"),
            }
        }
        Ok(())
    }
}

fn stack(g: &CodecGraph, entry: &str, env: &Env, direction: Direction) -> Result<Stack> {
    let mut x = Expander { graph: g, ops: Vec::new(), depth: 0 };
    x.call(&Node { call: Some(entry.into()), ..Node::default() }, env)?;
    let mut channels: BTreeMap<Slot, usize> = BTreeMap::new();
    let in_channels = match x.ops.first() {
        Some(CodecOp::Conv { cin, .. }) => *cin,
        _ => bail!("a codec stack starts with a conv"),
    };
    channels.insert(Slot::X, in_channels);
    let (mut up, mut down) = (1usize, 1usize);
    for (i, op) in x.ops.iter().enumerate() {
        let have = *channels.get(&op.input()).with_context(|| format!("op {i} ({}) reads a slot nothing wrote", op.label()))?;
        let (want, out) = match op {
            CodecOp::Conv { cin, cout, .. } | CodecOp::ConvUp { cin, cout, .. } | CodecOp::Linear { cin, cout, .. } => (*cin, *cout),
            CodecOp::RmsNorm { channels, .. } => (*channels, *channels),
            CodecOp::ScaleAdd { channels: c, .. } => {
                ensure!(channels.get(&Slot::X) == Some(c), "op {i} ({}) adds into x of another width", op.label());
                (*c, *c)
            }
        };
        ensure!(have == want, "op {i} ({}) reads {have} channels, it needs {want}", op.label());
        channels.insert(op.output(), out);
        let (u, d) = op.time_scale();
        up *= u;
        down *= d;
    }
    let out_channels = channels[&Slot::X];
    let (in_per_frame, out_per_frame) = match direction {
        Direction::Encode => (down / up, 1),
        Direction::Decode => (1, up / down),
    };
    ensure!(in_per_frame * up == down * out_per_frame, "the {direction:?} stack does not map whole frames");
    Ok(Stack { direction, ops: x.ops, in_channels, out_channels, in_per_frame, out_per_frame })
}

pub fn find(config: &serde_json::Value) -> Option<String> {
    let mt = config["model_type"].as_str()?;
    graph().ok()?.codec.into_iter().find(|g| g.model_types.iter().any(|m| m == mt)).map(|g| g.section)
}

pub fn plan(config: &serde_json::Value) -> Result<CodecPlan> {
    let model_type = config["model_type"].as_str().context("the config names no model_type")?;
    let g = graph()?;
    let d = g.codec.iter().find(|d| d.model_types.iter().any(|m| m == model_type)).with_context(|| format!("no codec graph for model_type {model_type}"))?;
    let section = config.get(&d.section).with_context(|| format!("the config has no {} section", d.section))?;
    let mut env = Env::new();
    for src in [config, section] {
        for (k, v) in src.as_object().into_iter().flatten() {
            if let Some(v) = Value::from_json(v) {
                env.insert(k.clone(), v);
            }
        }
    }
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
    let encoder = stack(d, &d.encoder, &env, Direction::Encode)?;
    let decoder = stack(d, &d.decoder, &env, Direction::Decode)?;
    ensure!(encoder.in_per_frame == decoder.out_per_frame && encoder.out_channels == decoder.in_channels, "the codec encoder and decoder do not mirror");
    if let Some(f) = config["frame_samples"].as_u64() {
        ensure!(f as usize == encoder.in_per_frame, "frame_samples {f} but the codec graph maps {} samples to a latent", encoder.in_per_frame);
    }
    Ok(CodecPlan { graph: d.name.clone(), frame_samples: encoder.in_per_frame, latent_dim: encoder.out_channels, encoder, decoder })
}
