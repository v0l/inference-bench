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
    encoder: Option<String>,
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
    bias: Option<String>,
    cin: Option<String>,
    cout: Option<String>,
    k: Option<String>,
    stride: Option<String>,
    groups: Option<String>,
    channels: Option<String>,
    eps: Option<String>,
    act: Option<String>,
    heads: Option<String>,
    context: Option<String>,
    theta: Option<String>,
    layout: Option<String>,
    rope: Option<String>,
    dilation: Option<String>,
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
    Conv { weight: String, cin: usize, cout: usize, k: usize, stride: usize, depthwise: bool, bias: bool, act: Option<&'static str>, input: Slot, output: Slot },
    ConvUp { weight: String, cin: usize, cout: usize, stride: usize, depthwise: bool, bias: bool, act: Option<&'static str> },
    RmsNorm { weight: String, channels: usize, eps: f32, input: Slot, output: Slot },
    LayerNorm { weight: String, channels: usize, eps: f32, input: Slot, output: Slot },
    Linear { weight: String, cin: usize, cout: usize, gelu: bool, bias: bool, input: Slot, output: Slot },
    Attention { weight: String, out_weight: String, channels: usize, heads: usize, context: usize, theta: f32, input: Slot, output: Slot },
    ScaleAdd { weight: String, channels: usize, input: Slot },
    Add { channels: usize, input: Slot },
}

fn act_name(text: Option<&str>) -> Result<Option<&'static str>> {
    match text {
        None | Some("none") => Ok(None),
        Some(a) => kir::conv::LINE_ACTS.iter().find(|n| **n == a).map(|n| Some(*n)).with_context(|| format!("codec input activation {a:?} is not one of {:?}", kir::conv::LINE_ACTS)),
    }
}

impl CodecOp {
    pub fn label(&self) -> String {
        let act = |a: &Option<&str>| a.map_or(String::new(), |a| format!("{a} "));
        match self {
            CodecOp::Conv { weight, k, stride, depthwise, act: a, .. } => format!("{}{} {weight} k{k} s{stride}", act(a), if *depthwise { "dwconv" } else { "conv" }),
            CodecOp::ConvUp { weight, stride, depthwise, act: a, .. } => format!("{}{} {weight} s{stride}", act(a), if *depthwise { "dwconv_up" } else { "conv_up" }),
            CodecOp::RmsNorm { weight, .. } => format!("rms_norm {weight}"),
            CodecOp::LayerNorm { weight, .. } => format!("layer_norm {weight}"),
            CodecOp::Linear { weight, gelu, .. } => format!("linear {weight}{}", if *gelu { " gelu" } else { "" }),
            CodecOp::Attention { weight, context, .. } => format!("attention {weight} ring {context}"),
            CodecOp::ScaleAdd { weight, .. } => format!("scale_add {weight}"),
            CodecOp::Add { .. } => "add".into(),
        }
    }

    pub fn state_floats(&self) -> usize {
        match self {
            CodecOp::Conv { cin, k, stride, .. } => cin * (k - stride),
            CodecOp::ConvUp { cin, .. } => *cin,
            CodecOp::Attention { channels, context, .. } => 2 * context * channels + 1,
            _ => 0,
        }
    }

    pub fn reference_state_floats(&self) -> usize {
        match self {
            CodecOp::ConvUp { cout, stride, .. } => cout * stride,
            op => op.state_floats(),
        }
    }

    pub fn input(&self) -> Slot {
        match self {
            CodecOp::Conv { input, .. }
            | CodecOp::RmsNorm { input, .. }
            | CodecOp::LayerNorm { input, .. }
            | CodecOp::Linear { input, .. }
            | CodecOp::Attention { input, .. }
            | CodecOp::ScaleAdd { input, .. }
            | CodecOp::Add { input, .. } => *input,
            CodecOp::ConvUp { .. } => Slot::X,
        }
    }

    pub fn output(&self) -> Slot {
        match self {
            CodecOp::Conv { output, .. } | CodecOp::RmsNorm { output, .. } | CodecOp::LayerNorm { output, .. } | CodecOp::Linear { output, .. } | CodecOp::Attention { output, .. } => *output,
            CodecOp::ConvUp { .. } | CodecOp::ScaleAdd { .. } | CodecOp::Add { .. } => Slot::X,
        }
    }

    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let bias = |on: bool, w: &str, n: usize| if on { vec![(format!("{w}.bias"), vec![n])] } else { vec![] };
        match self {
            CodecOp::Conv { weight, cin, cout, k, depthwise, bias: b, .. } => [vec![(format!("{weight}.weight"), vec![*cout, if *depthwise { 1 } else { *cin }, *k])], bias(*b, weight, *cout)].concat(),
            CodecOp::ConvUp { weight, cin, cout, stride, depthwise, bias: b, .. } => [vec![(format!("{weight}.weight"), vec![*cin, if *depthwise { 1 } else { *cout }, 2 * stride])], bias(*b, weight, *cout)].concat(),
            CodecOp::RmsNorm { weight, channels, .. } => vec![(format!("{weight}.weight"), vec![*channels])],
            CodecOp::LayerNorm { weight, channels, .. } => vec![(format!("{weight}.weight"), vec![*channels]), (format!("{weight}.bias"), vec![*channels])],
            CodecOp::Linear { weight, cin, cout, bias: b, .. } => [vec![(format!("{weight}.weight"), vec![*cout, *cin])], bias(*b, weight, *cout)].concat(),
            CodecOp::Attention { weight, out_weight, channels, .. } => vec![(format!("{weight}.weight"), vec![3 * channels, *channels]), (format!("{out_weight}.weight"), vec![*channels, *channels])],
            CodecOp::ScaleAdd { weight, channels, .. } => vec![(weight.clone(), vec![*channels])],
            CodecOp::Add { .. } => vec![],
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

    pub fn empty(&self) -> bool {
        self.ops.is_empty()
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
    pub fn has_encoder(&self) -> bool {
        !self.encoder.empty()
    }

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

    fn flag(text: &Option<String>, env: &Env, default: bool) -> Result<bool> {
        Ok(match text {
            Some(t) => expr::number(t, env)? != 0.0,
            None => default,
        })
    }

    fn op(&mut self, n: &Node, env: &Env) -> Result<()> {
        let env = &bind_args(n, env)?;
        let input = Slot::parse(n.input.as_deref())?;
        let output = Slot::parse(n.out.as_deref())?;
        let name = n.op.as_deref().unwrap_or_default();
        let weight = match &n.weight {
            Some(w) => expr::interpolate(w, env)?,
            None if name == "add" => String::new(),
            None => bail!("a codec {name} node needs weight"),
        };
        let bias = Self::flag(&n.bias, env, true)?;
        let made = match name {
            "conv" => {
                let (cin, cout) = (Self::count(&n.cin, env, "cin", None)?, Self::count(&n.cout, env, "cout", None)?);
                let k = Self::count(&n.k, env, "k", None)?;
                let stride = Self::count(&n.stride, env, "stride", Some(1))?;
                let groups = Self::count(&n.groups, env, "groups", Some(1))?;
                let dilation = Self::count(&n.dilation, env, "dilation", Some(1))?;
                ensure!(dilation == 1, "a codec conv with dilation {dilation}: dilated causal convs are not lowered yet");
                ensure!(stride >= 1 && k >= stride, "a causal conv with k {k} and stride {stride}");
                ensure!(groups == 1 || (groups == cin && cin == cout), "a codec conv is dense or depthwise, not {groups} groups over {cin} to {cout}");
                CodecOp::Conv { weight, cin, cout, k, stride, depthwise: groups > 1, bias, act: act_name(n.act.as_deref())?, input, output }
            }
            "conv_up" => {
                ensure!(input == Slot::X && output == Slot::X, "conv_up runs on x");
                let (cin, cout) = (Self::count(&n.cin, env, "cin", None)?, Self::count(&n.cout, env, "cout", None)?);
                let groups = Self::count(&n.groups, env, "groups", Some(1))?;
                ensure!(groups == 1 || (groups == cin && cin == cout), "a codec conv_up is dense or depthwise, not {groups} groups over {cin} to {cout}");
                let k = Self::count(&n.k, env, "k", None).ok();
                let stride = Self::count(&n.stride, env, "stride", None)?;
                ensure!(k.is_none_or(|k| k == 2 * stride), "conv_up lowers a kernel of twice its stride, not k {k:?} with stride {stride}");
                CodecOp::ConvUp { weight, cin, cout, stride, depthwise: groups > 1, bias, act: act_name(n.act.as_deref())? }
            }
            "rms_norm" => CodecOp::RmsNorm { weight, channels: Self::count(&n.channels, env, "channels", None)?, eps: expr::number(n.eps.as_deref().unwrap_or("0.00001"), env)? as f32, input, output },
            "layer_norm" => CodecOp::LayerNorm { weight, channels: Self::count(&n.channels, env, "channels", None)?, eps: expr::number(n.eps.as_deref().unwrap_or("0.00001"), env)? as f32, input, output },
            "linear" => {
                let gelu = match n.act.as_deref() {
                    None | Some("none") => false,
                    Some("gelu") => true,
                    Some(other) => bail!("codec activation {other:?} is not gelu or none"),
                };
                CodecOp::Linear { weight, cin: Self::count(&n.cin, env, "cin", None)?, cout: Self::count(&n.cout, env, "cout", None)?, gelu, bias, input, output }
            }
            "attention" => {
                let channels = Self::count(&n.channels, env, "channels", None)?;
                let heads = Self::count(&n.heads, env, "heads", None)?;
                let context = Self::count(&n.context, env, "context", None)?;
                ensure!(heads > 0 && channels.is_multiple_of(heads) && (channels / heads).is_multiple_of(2) && context > 0, "attention over {channels} channels in {heads} heads with a ring of {context}");
                let out_weight = match env.get("out") {
                    Some(Value::Text(t)) => t.clone(),
                    _ => bail!("an attention node names its output projection in args.out"),
                };
                ensure!(n.layout.as_deref().unwrap_or("blocks") == "blocks", "qkv layout {:?}: only q, k and v blocks per position are lowered", n.layout);
                ensure!(n.rope.as_deref().unwrap_or("pairs") == "pairs", "rope {:?}: only rope over pairs 2i, 2i+1 is lowered", n.rope);
                CodecOp::Attention { weight, out_weight, channels, heads, context, theta: expr::number(n.theta.as_deref().unwrap_or("10000"), env)? as f32, input, output }
            }
            "scale_add" => CodecOp::ScaleAdd { weight, channels: Self::count(&n.channels, env, "channels", None)?, input },
            "add" => CodecOp::Add { channels: Self::count(&n.channels, env, "channels", None)?, input },
            other => bail!("codec graph op {other:?} is not known"),
        };
        self.ops.push(made);
        Ok(())
    }

    fn call(&mut self, n: &Node, env: &Env) -> Result<()> {
        let name = n.call.as_deref().unwrap_or_default();
        let m = self.graph.macros.iter().find(|m| m.name == name).with_context(|| format!("codec macro {name} is not defined"))?.clone();
        let mut inner = bind_args(n, env)?;
        if let Some(p) = &n.prefix {
            let prefix = Value::Text(expr::interpolate(p, &inner)?);
            inner.insert("prefix".into(), prefix);
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

fn bind_args(n: &Node, env: &Env) -> Result<Env> {
    let mut inner = env.clone();
    for (k, v) in &n.args {
        let value = match v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
            Some(text) => Value::Text(expr::interpolate(text, env)?),
            None => expr::eval(v, env)?,
        };
        inner.insert(k.clone(), value);
    }
    Ok(inner)
}

fn op_io(op: &CodecOp) -> (usize, usize) {
    match op {
        CodecOp::Conv { cin, cout, .. } | CodecOp::ConvUp { cin, cout, .. } | CodecOp::Linear { cin, cout, .. } => (*cin, *cout),
        CodecOp::RmsNorm { channels, .. } | CodecOp::LayerNorm { channels, .. } | CodecOp::Attention { channels, .. } | CodecOp::ScaleAdd { channels, .. } | CodecOp::Add { channels, .. } => (*channels, *channels),
    }
}

fn stack(g: &CodecGraph, entry: &str, env: &Env, direction: Direction) -> Result<Stack> {
    let mut x = Expander { graph: g, ops: Vec::new(), depth: 0 };
    x.call(&Node { call: Some(entry.into()), ..Node::default() }, env)?;
    let mut channels: BTreeMap<Slot, usize> = BTreeMap::new();
    let in_channels = match x.ops.first() {
        Some(op @ (CodecOp::Conv { .. } | CodecOp::ConvUp { .. } | CodecOp::Linear { .. })) => op_io(op).0,
        _ => bail!("a codec stack starts with a conv or a linear"),
    };
    channels.insert(Slot::X, in_channels);
    let (mut up, mut down) = (1usize, 1usize);
    for (i, op) in x.ops.iter().enumerate() {
        let have = *channels.get(&op.input()).with_context(|| format!("op {i} ({}) reads a slot nothing wrote", op.label()))?;
        let (want, out) = op_io(op);
        if matches!(op, CodecOp::ScaleAdd { .. } | CodecOp::Add { .. }) {
            ensure!(channels.get(&Slot::X) == Some(&want), "op {i} ({}) adds into x of another width", op.label());
        }
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
    let decoder = stack(d, &d.decoder, &env, Direction::Decode)?;
    let encoder = match &d.encoder {
        Some(e) => {
            let encoder = stack(d, e, &env, Direction::Encode)?;
            ensure!(encoder.in_per_frame == decoder.out_per_frame && encoder.out_channels == decoder.in_channels, "the codec encoder and decoder do not mirror");
            encoder
        }
        None => Stack { direction: Direction::Encode, ops: Vec::new(), in_channels: decoder.out_channels, out_channels: decoder.in_channels, in_per_frame: decoder.out_per_frame, out_per_frame: 1 },
    };
    if let Some(f) = config["frame_samples"].as_u64() {
        ensure!(f as usize == decoder.out_per_frame, "frame_samples {f} but the codec graph maps a latent to {} samples", decoder.out_per_frame);
    }
    Ok(CodecPlan { graph: d.name.clone(), frame_samples: decoder.out_per_frame, latent_dim: decoder.in_channels, encoder, decoder })
}
