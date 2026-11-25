use anyhow::{Context, Result, bail, ensure};
use kir::expr::Expr;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

pub const GRAPH: &str = include_str!("../graphs/pipeline.toml");

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Num {
    Value(f64),
    Text(String),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleDecl {
    request: Vec<(String, String)>,
    step: Vec<(String, String)>,
    last: String,
    model_timestep: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StepDecl {
    update: String,
    combine: String,
    latent_in: String,
    output: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    #[serde(default = "yes")]
    pub offload_text_encoder: bool,
    #[serde(default)]
    pub offload_vae: bool,
}

fn yes() -> bool {
    true
}

impl Default for Options {
    fn default() -> Options {
        Options { offload_text_encoder: true, offload_vae: false }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineDecl {
    name: String,
    classes: Vec<String>,
    schedulers: Vec<String>,
    prompt: String,
    pad_keys: String,
    text_layers: Vec<Num>,
    max_length: usize,
    steps: Num,
    guidance: Num,
    cfg: String,
    #[serde(default)]
    fields: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    defaults: BTreeMap<String, f64>,
    #[serde(default)]
    tensors: BTreeMap<String, String>,
    schedule: ScheduleDecl,
    step: StepDecl,
    #[serde(default)]
    options: Options,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphFile {
    name: String,
    kind: String,
    pipeline: Vec<PipelineDecl>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub name: String,
    pub values: BTreeMap<String, f64>,
    pub tensors: BTreeMap<String, String>,
    pub chat_prompt: bool,
    pub masked_pad_keys: bool,
    pub text_layers: Vec<usize>,
    pub max_length: usize,
    pub steps: usize,
    pub guidance: f32,
    pub cfg: Expr,
    request: Vec<(String, Expr)>,
    per_step: Vec<(String, Expr)>,
    last: Expr,
    model_timestep: Expr,
    pub update: Expr,
    pub combine: Expr,
    pub latent_in: Expr,
    pub output: Expr,
    pub options: Options,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub sigmas: Vec<f32>,
    pub values: BTreeMap<String, f64>,
    timesteps: Vec<f32>,
}

impl Schedule {
    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn model_timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }
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

fn parse(text: &str) -> Result<Expr> {
    Expr::parse(text).map_err(anyhow::Error::msg)
}

fn eval(e: &Expr, values: &BTreeMap<String, f64>) -> Result<f64> {
    let mut names = Vec::new();
    e.names(&mut names);
    if let Some(n) = names.iter().find(|n| !values.contains_key(n.as_str())) {
        bail!("the pipeline expression names {n:?}, which no config field or symbol gives");
    }
    e.eval(&|n| values.get(n).copied()).context("the pipeline expression does not evaluate")
}

fn number(n: &Num, values: &BTreeMap<String, f64>) -> Result<f64> {
    match n {
        Num::Value(v) => Ok(*v),
        Num::Text(t) => eval(&parse(t)?, values),
    }
}

fn read(p: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?)?)
}

pub fn configs(dir: &Path) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut configs = BTreeMap::new();
    configs.insert("pipeline".to_string(), read(&dir.join("model_index.json"))?);
    for (part, file) in [("scheduler", "scheduler_config.json"), ("transformer", "config.json"), ("vae", "config.json"), ("text_encoder", "config.json")] {
        let p = dir.join(part).join(file);
        if p.exists() {
            configs.insert(part.to_string(), read(&p)?);
        }
    }
    Ok(configs)
}

fn class_of(configs: &BTreeMap<String, serde_json::Value>, part: &str) -> Option<String> {
    let v = configs.get(part)?;
    v["_class_name"].as_str().map(str::to_string).or_else(|| configs["pipeline"][part][1].as_str().map(str::to_string))
}

impl Pipeline {
    pub fn from_dir(dir: &Path) -> Result<Pipeline> {
        Pipeline::from_configs(&configs(dir)?)
    }

    pub fn from_configs(configs: &BTreeMap<String, serde_json::Value>) -> Result<Pipeline> {
        let g: GraphFile = toml::from_str(GRAPH).context("graphs/pipeline.toml")?;
        ensure!(g.kind == "pipeline-graph" && g.name == "pipeline", "graphs/pipeline.toml is not the pipeline graph");
        let class = configs.get("pipeline").and_then(|p| p["_class_name"].as_str()).context("model_index.json has no _class_name")?;
        let scheduler = class_of(configs, "scheduler").context("the pipeline names no scheduler")?;
        let Some(d) = g.pipeline.iter().find(|p| p.classes.iter().any(|c| c == class)) else {
            let known: Vec<&str> = g.pipeline.iter().flat_map(|p| p.classes.iter().map(String::as_str)).collect();
            bail!("no pipeline graph for {class} (graphs/pipeline.toml has {})", known.join(", "));
        };
        ensure!(d.schedulers.contains(&scheduler), "the {} pipeline runs {}, not {scheduler}", d.name, d.schedulers.join(", "));
        let mut values = d.defaults.clone();
        for (name, paths) in &d.fields {
            let Some(v) = paths.iter().find_map(|p| lookup(configs, p)) else { continue };
            let v = match v {
                serde_json::Value::Bool(b) => f64::from(u8::from(*b)),
                v => v.as_f64().with_context(|| format!("{name} is {v}, not a number"))?,
            };
            values.insert(name.clone(), v);
        }
        let text_layers = d.text_layers.iter().map(|n| number(n, &values).map(|v| v as usize)).collect::<Result<Vec<_>>>()?;
        let steps = number(&d.steps, &values)? as usize;
        let guidance = number(&d.guidance, &values)? as f32;
        let pairs = |list: &[(String, String)]| list.iter().map(|(n, t)| Ok((n.clone(), parse(t)?))).collect::<Result<Vec<_>>>();
        Ok(Pipeline {
            name: d.name.clone(),
            values,
            tensors: d.tensors.clone(),
            chat_prompt: match d.prompt.as_str() {
                "chat" => true,
                "plain" => false,
                other => bail!("prompt {other:?} is not chat or plain"),
            },
            masked_pad_keys: match d.pad_keys.as_str() {
                "masked" => true,
                "zeroed" => false,
                other => bail!("pad_keys {other:?} is not masked or zeroed"),
            },
            text_layers,
            max_length: d.max_length,
            steps,
            guidance,
            cfg: parse(&d.cfg)?,
            request: pairs(&d.schedule.request)?,
            per_step: pairs(&d.schedule.step)?,
            last: parse(&d.schedule.last)?,
            model_timestep: parse(&d.schedule.model_timestep)?,
            update: parse(&d.step.update)?,
            combine: parse(&d.step.combine)?,
            latent_in: parse(&d.step.latent_in)?,
            output: parse(&d.step.output)?,
            options: d.options,
        })
    }

    pub fn guided(&self, guidance: f32) -> Result<bool> {
        let mut v = self.values.clone();
        v.insert("guidance".into(), f64::from(guidance));
        Ok(eval(&self.cfg, &v)? != 0.0)
    }

    pub fn schedule(&self, steps: usize, image_tokens: usize) -> Result<Schedule> {
        ensure!(steps > 0, "a run needs at least one step");
        let mut v = self.values.clone();
        v.insert("steps".into(), steps as f64);
        v.insert("image_tokens".into(), image_tokens as f64);
        for (n, e) in &self.request {
            let x = eval(e, &v)?;
            v.insert(n.clone(), x);
        }
        let mut sigmas = Vec::with_capacity(steps + 1);
        for k in 0..steps {
            let mut s = v.clone();
            s.insert("k".into(), k as f64);
            for (n, e) in &self.per_step {
                let x = eval(e, &s)?;
                s.insert(n.clone(), x);
            }
            sigmas.push(s["sigma"] as f32);
        }
        sigmas.push(eval(&self.last, &v)? as f32);
        let timesteps = (0..steps)
            .map(|i| {
                let mut s = v.clone();
                s.insert("s0".into(), f64::from(sigmas[i]));
                s.insert("s1".into(), f64::from(sigmas[i + 1]));
                eval(&self.model_timestep, &s).map(|t| t as f32)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Schedule { sigmas, values: v, timesteps })
    }

    pub fn step_rule(&self, guided: bool) -> Result<Expr> {
        if !guided {
            return Ok(self.update.clone());
        }
        let combine = self.combine.clone();
        self.update.subst(&|n| Some(if n == "v" { combine.clone() } else { Expr::Name(n.to_string()) })).map_err(anyhow::Error::msg)
    }

    pub fn step_host(&self, x: &mut [f32], cond: &[f32], uncond: Option<&[f32]>, sched: &Schedule, i: usize, guidance: f32) -> Result<()> {
        ensure!(i < sched.steps(), "step {i} of {}", sched.steps());
        ensure!(cond.len() == x.len() && uncond.is_none_or(|u| u.len() == x.len()), "the velocity and the latent differ in size");
        let rule = self.step_rule(uncond.is_some())?;
        let (s0, s1, g) = (sched.sigmas[i], sched.sigmas[i + 1], guidance);
        for j in 0..x.len() {
            let (xv, c, u) = (x[j], cond[j], uncond.map_or(0.0, |u| u[j]));
            let env = |n: &str| -> Option<f64> {
                Some(f64::from(match n {
                    "x" => xv,
                    "v" | "c" => c,
                    "u" => u,
                    "g" => g,
                    "s0" => s0,
                    "s1" => s1,
                    _ => return None,
                }))
            };
            x[j] = f32_eval(&rule, &env)?;
        }
        Ok(())
    }

    pub fn denormalise(&self, x: &mut [f32], var: &[f32], mean: &[f32]) -> Result<()> {
        ensure!(var.len() == mean.len() && !var.is_empty() && x.len().is_multiple_of(var.len()), "the latent is not rows of {} channels", var.len());
        let eps = self.values.get("bn_eps").copied().unwrap_or(0.0) as f32;
        let c = var.len();
        for (j, v) in x.iter_mut().enumerate() {
            let (vr, mn, xv) = (var[j % c], mean[j % c], *v);
            *v = f32_eval(&self.latent_in, &|n| {
                Some(f64::from(match n {
                    "x" => xv,
                    "bn_var" => vr,
                    "bn_mean" => mn,
                    "bn_eps" => eps,
                    _ => return None,
                }))
            })?;
        }
        Ok(())
    }

    pub fn postprocess(&self, x: &[f32]) -> Result<Vec<f32>> {
        x.iter().map(|&v| f32_eval(&self.output, &|n| (n == "x").then_some(f64::from(v)))).collect()
    }
}

fn f32_eval(e: &Expr, env: &dyn Fn(&str) -> Option<f64>) -> Result<f32> {
    let mut names = Vec::new();
    e.names(&mut names);
    let values: BTreeMap<String, f32> = names.iter().map(|n| env(n).map(|v| (n.clone(), v as f32)).with_context(|| format!("the step rule names {n:?}, which is not a step input"))).collect::<Result<_>>()?;
    eval32(e, &values).context("the step rule does not evaluate")
}

fn eval32(e: &Expr, v: &BTreeMap<String, f32>) -> Option<f32> {
    use kir::expr::Op;
    Some(match e {
        Expr::Lit(t) => t.parse().ok()?,
        Expr::Const(_, c) => *c,
        Expr::Name(n) => *v.get(n)?,
        Expr::Neg(a) => -eval32(a, v)?,
        Expr::Bin(op, a, b) => {
            let (a, b) = (eval32(a, v)?, eval32(b, v)?);
            match op {
                Op::Add => a + b,
                Op::Sub => a - b,
                Op::Mul => a * b,
                Op::Div => a / b,
                _ => f32::from(u8::from(e.eval(&|n| v.get(n).map(|x| f64::from(*x)))? != 0.0)),
            }
        }
        Expr::Call(f, args) => {
            let a: Vec<f32> = args.iter().map(|x| eval32(x, v)).collect::<Option<_>>()?;
            match (*f, a.as_slice()) {
                ("min", [p, q]) => p.min(*q),
                ("max", [p, q]) => p.max(*q),
                ("sqrt", [p]) => p.sqrt(),
                ("exp", [p]) => p.exp(),
                ("select", [c, p, q]) => if *c != 0.0 { *p } else { *q },
                _ => e.eval(&|n| v.get(n).map(|x| f64::from(*x)))? as f32,
            }
        }
        Expr::Raw(_) | Expr::Width => return None,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub ids: Vec<u32>,
    pub mask: Vec<u8>,
}

impl Prompt {
    pub fn real(&self) -> usize {
        self.mask.iter().filter(|&&m| m == 1).count()
    }
}

pub struct PromptEncoder {
    tokenizer: tokenizers::Tokenizer,
    template: Option<crate::chat::ChatTemplate>,
    pad: u32,
}

impl PromptEncoder {
    pub fn new(tokenizer: tokenizers::Tokenizer, tokenizer_config: &str) -> Result<PromptEncoder> {
        let config: serde_json::Value = serde_json::from_str(tokenizer_config).context("tokenizer_config.json")?;
        let pad_text = match &config["pad_token"] {
            serde_json::Value::String(s) => s.clone(),
            v => v["content"].as_str().context("the tokenizer config names no pad_token")?.to_string(),
        };
        let pad = tokenizer.token_to_id(&pad_text).with_context(|| format!("pad token {pad_text:?} is not in the vocabulary"))?;
        let tokenizer = crate::engine::without_truncation(tokenizer)?;
        Ok(PromptEncoder { tokenizer, template: crate::chat::ChatTemplate::from_tokenizer_config(tokenizer_config)?, pad })
    }

    pub fn open(dir: &Path) -> Result<PromptEncoder> {
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        PromptEncoder::new(tokenizer, &std::fs::read_to_string(dir.join("tokenizer_config.json"))?)
    }

    pub fn text(&self, pipeline: &Pipeline, prompt: &str) -> Result<String> {
        if !pipeline.chat_prompt {
            return Ok(prompt.to_string());
        }
        self.template.as_ref().context("the pipeline renders a chat prompt and the tokenizer has no chat template")?.render_user(prompt)
    }

    pub fn encode(&self, pipeline: &Pipeline, prompt: &str, max_length: usize) -> Result<Prompt> {
        let text = self.text(pipeline, prompt)?;
        let enc = self.tokenizer.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().iter().copied().take(max_length).collect();
        let mut mask = vec![1u8; ids.len()];
        ids.resize(max_length, self.pad);
        mask.resize(max_length, 0);
        Ok(Prompt { ids, mask })
    }
}

fn splitmix(seed: u64, i: u64) -> f64 {
    let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let pairs = n.div_ceil(2);
    let mut out = Vec::with_capacity(2 * pairs);
    for p in 0..pairs as u64 {
        let u1 = splitmix(seed, 2 * p + 1);
        let u2 = splitmix(seed, 2 * p + 2);
        let r = (-2.0 * (-u1).ln_1p()).sqrt();
        let a = 2.0 * std::f64::consts::PI * u2;
        out.push((r * a.cos()) as f32);
        out.push((r * a.sin()) as f32);
    }
    out.truncate(n);
    out
}
