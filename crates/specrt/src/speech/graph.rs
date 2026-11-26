use crate::vae::expr::{self, Env, Value};
use anyhow::{Context, Result, bail, ensure};
use kir::expr::Expr;
use kir::mel::{LogMel, MelScale, Norm, Pad, Window};
use serde::Deserialize;
use std::path::Path;

pub const GRAPH: &str = include_str!("../../graphs/speech.toml");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphFile {
    name: String,
    kind: String,
    front: Vec<Front>,
    stem: Vec<StemGraph>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Front {
    extractor: String,
    op: String,
    samples: String,
    n_fft: String,
    hop: String,
    bins: String,
    rate: String,
    frames: String,
    window: String,
    mel_scale: String,
    norm: String,
    fmin: String,
    fmax: String,
    pad: String,
    log: String,
    floor: String,
    out: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StemGraph {
    name: String,
    model_types: Vec<String>,
    root: String,
    features: String,
    positions: String,
    node: Vec<StemNode>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StemNode {
    op: String,
    weight: String,
    cin: String,
    cout: String,
    k: String,
    stride: Option<String>,
    pad: Option<String>,
    act: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpeechOp {
    Frames,
    Dft,
    Power,
    Bank,
    Log,
    Max,
    Scale,
    Conv { weight: String, cin: usize, cout: usize, k: usize, stride: usize, pad: usize, t_in: usize, t_out: usize },
    Gelu { rows: usize, width: usize },
}

impl SpeechOp {
    pub fn label(&self) -> String {
        match self {
            SpeechOp::Frames => "mel frames".into(),
            SpeechOp::Dft => "mel dft".into(),
            SpeechOp::Power => "mel power".into(),
            SpeechOp::Bank => "mel bank".into(),
            SpeechOp::Log => "mel log".into(),
            SpeechOp::Max => "mel max".into(),
            SpeechOp::Scale => "mel scale".into(),
            SpeechOp::Conv { weight, k, stride, pad, .. } => format!("conv {weight} k{k} s{stride} p{pad}"),
            SpeechOp::Gelu { .. } => "gelu".into(),
        }
    }

    pub fn conv_shape(&self) -> Option<kir::conv::ConvShape> {
        match self {
            SpeechOp::Conv { cin, cout, k, stride, pad, .. } => Some(kir::conv::ConvShape::padded(*cin, *cout, *k, *stride, *pad)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechPlan {
    pub graph: String,
    pub mel: LogMel,
    pub root: String,
    pub ops: Vec<SpeechOp>,
    pub frames: usize,
    pub positions: usize,
    pub width: usize,
}

impl SpeechPlan {
    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                SpeechOp::Conv { weight, cin, cout, k, .. } => Some([(format!("{}.{weight}.weight", self.root), vec![*cout, *cin, *k]), (format!("{}.{weight}.bias", self.root), vec![*cout])]),
                _ => None,
            })
            .flatten()
            .collect()
    }

    pub fn check_tensors(&self, shape_of: &dyn Fn(&str) -> Result<Vec<usize>>) -> Result<()> {
        for (name, want) in self.tensors() {
            let have = shape_of(&name).with_context(|| format!("the {} speech plan needs {name}", self.graph))?;
            ensure!(have == want, "{name} is {have:?}, the speech plan needs {want:?}");
        }
        Ok(())
    }
}

fn graph() -> Result<GraphFile> {
    let g: GraphFile = toml::from_str(GRAPH).context("graphs/speech.toml")?;
    ensure!(g.kind == "graph" && g.name == "speech", "graphs/speech.toml is not the speech graph");
    Ok(g)
}

fn env_of(json: &serde_json::Value) -> Env {
    json.as_object().into_iter().flatten().filter_map(|(k, v)| Value::from_json(v).map(|v| (k.clone(), v))).collect()
}

fn device_expr(text: &str) -> Result<Expr> {
    Expr::parse(text).map_err(anyhow::Error::msg)
}

pub fn log_mel(preprocessor: &serde_json::Value) -> Result<LogMel> {
    let kind = preprocessor["feature_extractor_type"].as_str().context("the preprocessor config names no feature_extractor_type")?;
    let g = graph()?;
    let f = g.front.iter().find(|f| f.extractor == kind).with_context(|| format!("no speech front end for feature extractor {kind}"))?;
    ensure!(f.op == "log_mel", "speech front end op {:?} is not log_mel", f.op);
    let env = env_of(preprocessor);
    let n = |t: &str| expr::count(t, &env);
    let x = |t: &str| expr::number(t, &env);
    let window = match f.window.as_str() {
        "hann_periodic" => Window::Hann { periodic: true },
        "hann" => Window::Hann { periodic: false },
        w => bail!("window {w:?} is not hann or hann_periodic"),
    };
    let scale = match f.mel_scale.as_str() {
        "slaney" => MelScale::Slaney,
        "htk" => MelScale::Htk,
        s => bail!("mel scale {s:?} is not slaney or htk"),
    };
    let norm = match f.norm.as_str() {
        "slaney" => Norm::Slaney,
        "none" => Norm::None,
        s => bail!("mel norm {s:?} is not slaney or none"),
    };
    let pad = match f.pad.as_str() {
        "reflect" => Pad::Reflect,
        "zero" => Pad::Zero,
        s => bail!("pad {s:?} is not reflect or zero"),
    };
    let mel = LogMel {
        samples: n(&f.samples)?,
        n_fft: n(&f.n_fft)?,
        hop: n(&f.hop)?,
        frames: n(&f.frames)?,
        bins: n(&f.bins)?,
        rate: x(&f.rate)?,
        fmin: x(&f.fmin)?,
        fmax: x(&f.fmax)?,
        window,
        scale,
        norm,
        pad,
        log: device_expr(&f.log)?,
        floor: device_expr(&f.floor)?,
        out: device_expr(&f.out)?,
    };
    mel.validate().map_err(anyhow::Error::msg)?;
    Ok(mel)
}

pub fn plan(config: &serde_json::Value, preprocessor: &serde_json::Value) -> Result<SpeechPlan> {
    let mel = log_mel(preprocessor)?;
    let model_type = config["model_type"].as_str().context("the config names no model_type")?;
    let g = graph()?;
    let s = g.stem.iter().find(|s| s.model_types.iter().any(|m| m == model_type)).with_context(|| format!("no speech stem for model_type {model_type}"))?;
    let env = env_of(config);
    let features = expr::count(&s.features, &env)?;
    ensure!(features == mel.bins, "the model reads {features} mel bins, the preprocessor makes {}", mel.bins);
    let mut ops = vec![SpeechOp::Frames, SpeechOp::Dft, SpeechOp::Power, SpeechOp::Bank, SpeechOp::Log, SpeechOp::Max, SpeechOp::Scale];
    let (mut t, mut c) = (mel.frames, mel.bins);
    for node in &s.node {
        ensure!(node.op == "conv", "speech stem op {:?} is not conv", node.op);
        let (cin, cout, k) = (expr::count(&node.cin, &env)?, expr::count(&node.cout, &env)?, expr::count(&node.k, &env)?);
        let stride = node.stride.as_deref().map_or(Ok(1), |v| expr::count(v, &env))?;
        let pad = node.pad.as_deref().map_or(Ok(0), |v| expr::count(v, &env))?;
        ensure!(cin == c, "stem conv {} reads {cin} channels, it is given {c}", node.weight);
        let shape = kir::conv::ConvShape::padded(cin, cout, k, stride, pad);
        let t_out = shape.out_len(t);
        ops.push(SpeechOp::Conv { weight: node.weight.clone(), cin, cout, k, stride, pad, t_in: t, t_out });
        match node.act.as_deref() {
            None | Some("none") => {}
            Some("gelu") => ops.push(SpeechOp::Gelu { rows: t_out, width: cout }),
            Some(a) => bail!("stem activation {a:?} is not gelu or none"),
        }
        (t, c) = (t_out, cout);
    }
    let positions = expr::count(&s.positions, &env)?;
    ensure!(t == positions, "the stem gives {t} positions, the model has {positions}");
    Ok(SpeechPlan { graph: s.name.clone(), mel, root: s.root.clone(), ops, frames: t, positions, width: c })
}

pub fn read_json(path: &Path) -> Result<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?).with_context(|| path.display().to_string())
}

pub fn plan_dir(dir: &Path) -> Result<SpeechPlan> {
    plan(&read_json(&dir.join("config.json"))?, &read_json(&dir.join("preprocessor_config.json"))?)
}
