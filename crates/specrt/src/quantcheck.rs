use crate::quantize::{LogitStats, MAX_KL_WEIGHT, logit_stats, prompt_set, prompt_set_hash};
use crate::spec::{LayerQuant, Quant, QuantPlan, QuantizeChoice, Spec};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const RECIPE_VERSION: u32 = 1;
pub const ENGINE_COMMIT: &str = env!("SPECRT_ENGINE_COMMIT");
pub const REFINE_ROUNDS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    pub mean_kl: f64,
    pub max_kl: f64,
    pub top1: f64,
}

impl From<LogitStats> for Measured {
    fn from(s: LogitStats) -> Self {
        Measured { mean_kl: s.mean_kl, max_kl: s.max_kl, top1: s.top1 }
    }
}

impl Measured {
    pub fn score(&self) -> f64 {
        self.mean_kl + MAX_KL_WEIGHT * self.max_kl
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub layer: usize,
    pub quant: LayerQuant,
    pub bytes: u64,
    pub kl: Measured,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerOption {
    pub quant: LayerQuant,
    pub bytes: u64,
    pub kl: Measured,
}

pub fn layer_options(probes: &[Probe], bf16_bytes: &[u64]) -> Vec<Vec<LayerOption>> {
    bf16_bytes
        .iter()
        .enumerate()
        .map(|(layer, &bytes)| {
            let mut out: Vec<LayerOption> = Vec::new();
            for p in probes.iter().filter(|p| p.layer == layer && p.quant.format != Quant::Bf16) {
                let option = LayerOption { quant: p.quant, bytes: p.bytes, kl: p.kl };
                match out.iter_mut().find(|o| o.quant.format == p.quant.format) {
                    Some(o) if option.kl.score() < o.kl.score() => *o = option,
                    Some(_) => {}
                    None => out.push(option),
                }
            }
            out.retain(|o| o.bytes < bytes);
            out.push(LayerOption { quant: LayerQuant::BF16, bytes, kl: Measured { top1: 1.0, ..Measured::default() } });
            out.sort_by_key(|o| o.bytes);
            out
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    pub kl: f64,
    pub bytes: Option<u64>,
}

pub fn estimated_kl(options: &[Vec<LayerOption>], at: &[usize]) -> f64 {
    options.iter().zip(at).map(|(o, &i)| o[i].kl.mean_kl).sum()
}

pub fn total_bytes(options: &[Vec<LayerOption>], at: &[usize], fixed: u64) -> u64 {
    fixed + options.iter().zip(at).map(|(o, &i)| o[i].bytes).sum::<u64>()
}

pub fn choose(options: &[Vec<LayerOption>], fixed: u64, budget: Budget) -> Vec<usize> {
    let mut at = vec![0usize; options.len()];
    while estimated_kl(options, &at) > budget.kl {
        let total = total_bytes(options, &at, fixed);
        let next = (0..options.len())
            .filter(|&l| at[l] + 1 < options[l].len())
            .filter(|&l| budget.bytes.is_none_or(|cap| total - options[l][at[l]].bytes + options[l][at[l] + 1].bytes <= cap))
            .max_by(|&a, &b| options[a][at[a]].kl.score().total_cmp(&options[b][at[b]].kl.score()).then(b.cmp(&a)));
        let Some(l) = next else { break };
        at[l] += 1;
    }
    at
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeLayer {
    pub index: usize,
    pub format: Quant,
    pub search: bool,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub version: u32,
    pub model: String,
    pub engine: String,
    pub format_version: u32,
    pub prompt_set: String,
    pub positions: usize,
    pub kl_budget: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_bytes: Option<u64>,
    pub bytes: u64,
    pub bf16_bytes: u64,
    pub estimated_kl: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<Measured>,
    #[serde(rename = "layer")]
    pub layers: Vec<RecipeLayer>,
}

impl Recipe {
    pub fn plan(&self) -> QuantPlan {
        QuantPlan { layers: self.layers.iter().map(|l| LayerQuant::new(l.format, l.search)).collect(), kernels: Vec::new() }
    }

    pub fn load(path: &Path) -> Result<Recipe> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let recipe: Recipe = toml::from_str(&text).with_context(|| path.display().to_string())?;
        ensure!(recipe.version == RECIPE_VERSION, "{}: recipe version {} is not {RECIPE_VERSION}", path.display(), recipe.version);
        Ok(recipe)
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string(self)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_toml()?).with_context(|| path.display().to_string())
    }

    pub fn counts(&self) -> Vec<(Quant, usize)> {
        let mut out: Vec<(Quant, usize)> = Vec::new();
        for l in &self.layers {
            match out.iter_mut().find(|(q, _)| *q == l.format) {
                Some((_, n)) => *n += 1,
                None => out.push((l.format, 1)),
            }
        }
        out
    }
}

pub fn recipe_from(spec: &Spec, options: &[Vec<LayerOption>], at: &[usize], fixed: u64, budget: Budget, positions: usize) -> Recipe {
    let bf16_bytes = fixed + options.iter().map(|o| o.last().map_or(0, |b| b.bytes)).sum::<u64>();
    Recipe {
        version: RECIPE_VERSION,
        model: spec.repo.clone(),
        engine: ENGINE_COMMIT.to_string(),
        format_version: crate::quantize::FORMAT_VERSION,
        prompt_set: prompt_set_hash(),
        positions,
        kl_budget: budget.kl,
        budget_bytes: budget.bytes,
        bytes: total_bytes(options, at, fixed),
        bf16_bytes,
        estimated_kl: estimated_kl(options, at),
        check: None,
        layers: options
            .iter()
            .zip(at)
            .enumerate()
            .map(|(index, (o, &i))| RecipeLayer { index, format: o[i].quant.format, search: o[i].quant.search, mean_kl: o[i].kl.mean_kl, max_kl: o[i].kl.max_kl, bytes: o[i].bytes })
            .collect(),
    }
}

pub fn prompt_tokens(spec: &Spec) -> Result<Vec<Vec<u32>>> {
    let path = crate::weights::find_in_snapshot(&spec.repo, "tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    prompt_set().iter().map(|p| Ok(tokenizer.encode(p.text.as_str(), false).map_err(|e| anyhow::anyhow!("{}: {e}", p.name))?.get_ids().to_vec())).collect()
}

pub fn probe_spec(spec: &Spec, formats: &[Quant]) -> Result<Spec> {
    let mut s = spec.clone();
    s.runtime.quantize = QuantizeChoice::Off;
    s.quant_plan = QuantPlan { layers: vec![LayerQuant::BF16; s.layers.count], kernels: formats.to_vec() };
    s.apply_quantize()?;
    Ok(s)
}

pub fn blanket_plan(spec: &Spec, format: Quant) -> QuantPlan {
    let n = spec.layers.count;
    QuantPlan { layers: (0..n).map(|l| if spec.quant_keep.keeps_layer(l, n) { LayerQuant::BF16 } else { LayerQuant::new(format, true) }).collect(), kernels: Vec::new() }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub formats: Vec<Quant>,
    pub budgets: Vec<Budget>,
    pub max_ctx: usize,
    pub tokens: Option<Vec<Vec<u32>>>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub recipes: Vec<Recipe>,
    pub probes: Vec<Probe>,
    pub blanket: Vec<(Quant, Measured, u64)>,
    pub bf16_bytes: u64,
    pub secs: f64,
}

pub struct Session {
    rt: crate::runtime::Runtime,
    tokens: Vec<Vec<u32>>,
    reference: Vec<Vec<f32>>,
    vocab: usize,
}

impl Session {
    pub fn open(spec: &Spec, formats: &[Quant], max_ctx: usize, tokens: Option<Vec<Vec<u32>>>) -> Result<Session> {
        let s = probe_spec(spec, formats)?;
        let tokens = match tokens {
            Some(t) => t,
            None => prompt_tokens(&s)?,
        };
        let longest = tokens.iter().map(Vec::len).max().unwrap_or(0);
        ensure!(longest <= max_ctx, "the longest quantcheck prompt has {longest} tokens, more than max_ctx {max_ctx}");
        let mut rt = crate::runtime::Runtime::load(&s, max_ctx)?;
        let reference = tokens.iter().map(|t| rt.prompt_logits(t)).collect::<Result<Vec<_>>>()?;
        Ok(Session { rt, tokens, reference, vocab: s.dims.vocab })
    }

    pub fn runtime(&self) -> &crate::runtime::Runtime {
        &self.rt
    }

    pub fn positions(&self) -> usize {
        self.tokens.iter().map(Vec::len).sum()
    }

    pub fn logits(&mut self) -> Result<Vec<Vec<f32>>> {
        self.tokens.iter().map(|t| self.rt.prompt_logits(t)).collect()
    }

    pub fn measure(&mut self) -> Result<Measured> {
        let got = self.logits()?;
        let a: Vec<&[f32]> = self.reference.iter().map(Vec::as_slice).collect();
        let b: Vec<&[f32]> = got.iter().map(Vec::as_slice).collect();
        Ok(logit_stats(&a, &b, self.vocab).into())
    }

    pub fn set_layer(&mut self, layer: usize, quant: LayerQuant) -> Result<u64> {
        if self.rt.spec().quant_plan.layers[layer] == quant {
            return Ok(self.rt.layer_bytes()[layer]);
        }
        self.rt.requantize_layer(layer, quant)
    }

    pub fn apply(&mut self, plan: &QuantPlan) -> Result<u64> {
        for (l, &q) in plan.layers.iter().enumerate() {
            self.set_layer(l, q)?;
        }
        Ok(self.rt.weight_bytes)
    }
}

pub fn run(spec: &Spec, opts: &Options, log: &mut dyn FnMut(String)) -> Result<Report> {
    let started = std::time::Instant::now();
    ensure!(!opts.formats.is_empty(), "quantcheck needs at least one format");
    let mut formats = opts.formats.clone();
    formats.sort_by_key(|q| *q as u8);
    formats.dedup();
    let mut session = Session::open(spec, &formats, opts.max_ctx, opts.tokens.clone())?;
    let n = spec.layers.count;
    let bf16_bytes = session.runtime().layer_bytes();
    let fixed = session.runtime().weight_bytes - bf16_bytes.iter().sum::<u64>();
    log(format!("loaded {} layers in bf16: {} weight bytes, {} of them outside the layers; {} prompt positions", n, session.runtime().weight_bytes, fixed, session.positions()));
    let mut probes = Vec::new();
    for layer in 0..n {
        for &format in &formats {
            if !session.runtime().spec().layer_takes(layer, format) {
                continue;
            }
            for search in [true, false] {
                let quant = LayerQuant::new(format, search);
                let bytes = session.set_layer(layer, quant)?;
                let kl = session.measure()?;
                log(format!("layer {layer} {format:?} {}: mean KL {:.5}, max KL {:.4}, top-1 {:.1}%, {} bytes", if search { "search" } else { "plain" }, kl.mean_kl, kl.max_kl, 100.0 * kl.top1, bytes));
                probes.push(Probe { layer, quant, bytes, kl });
            }
        }
        session.set_layer(layer, LayerQuant::BF16)?;
    }
    let options = layer_options(&probes, &bf16_bytes);
    let mut recipes = Vec::new();
    for &budget in &opts.budgets {
        recipes.push(settle(spec, &mut session, &options, fixed, budget, log)?);
    }
    let mut blanket = Vec::new();
    for &format in &formats {
        let bytes = session.apply(&blanket_plan(spec, format))?;
        let kl = session.measure()?;
        log(format!("blanket {format:?} with the keep list: mean KL {:.5}, max KL {:.4}, top-1 {:.1}%, {bytes} bytes", kl.mean_kl, kl.max_kl, 100.0 * kl.top1));
        blanket.push((format, kl, bytes));
    }
    Ok(Report { recipes, probes, blanket, bf16_bytes: fixed + bf16_bytes.iter().sum::<u64>(), secs: started.elapsed().as_secs_f64() })
}

fn settle(spec: &Spec, session: &mut Session, options: &[Vec<LayerOption>], fixed: u64, budget: Budget, log: &mut dyn FnMut(String)) -> Result<Recipe> {
    let mut target = budget;
    let mut last: Option<(Vec<usize>, Recipe)> = None;
    for round in 1..=REFINE_ROUNDS {
        let at = choose(options, fixed, target);
        if let Some((prev, r)) = &last
            && *prev == at
        {
            return Ok(r.clone());
        }
        let mut r = recipe_from(spec, options, &at, fixed, budget, session.positions());
        session.apply(&r.plan())?;
        let check = session.measure()?;
        log(format!("budget KL {} bytes {:?}, round {round}: estimated KL {:.5}, whole model mean KL {:.5}, max KL {:.4}, top-1 {:.1}%, {} bytes", budget.kl, budget.bytes, r.estimated_kl, check.mean_kl, check.max_kl, 100.0 * check.top1, r.bytes));
        r.check = Some(check);
        let room = at.iter().zip(options).any(|(&i, o)| i + 1 < o.len());
        let done = check.mean_kl <= budget.kl || r.estimated_kl <= 0.0 || !room;
        target.kl = r.estimated_kl * budget.kl / check.mean_kl.max(f64::MIN_POSITIVE) * 0.95;
        last = Some((at, r));
        if done {
            break;
        }
    }
    last.map(|(_, r)| r).context("no recipe")
}

pub fn default_path(spec: &Spec) -> std::path::PathBuf {
    let name = spec.repo.rsplit('/').find(|s| !s.is_empty()).unwrap_or(&spec.name);
    std::path::PathBuf::from(format!("{name}.quantrecipe.toml"))
}
