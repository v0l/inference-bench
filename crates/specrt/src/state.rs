use crate::spec::Spec;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Per {
    Token,
    Sequence,
    Layer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layers {
    Attention,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rollback {
    Positional,
    Checkpoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub name: String,
    pub per: Per,
    pub layers: Layers,
    pub shape: Vec<String>,
    pub dtype: Dtype,
    #[serde(default = "yes")]
    pub snapshot: bool,
    #[serde(default = "yes")]
    pub prefix_share: bool,
    pub rollback: Rollback,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
}

fn yes() -> bool {
    true
}

fn declared(name: &str, per: Per, layers: Layers, shape: &[&str], dtype: Dtype, rollback: Rollback) -> State {
    State { name: name.into(), per, layers, shape: shape.iter().map(|s| s.to_string()).collect(), dtype, snapshot: true, prefix_share: true, rollback, owner: None }
}

pub fn stock() -> Vec<State> {
    vec![
        declared("k", Per::Token, Layers::Attention, &["kv_heads", "head_dim"], Dtype::Bf16, Rollback::Positional),
        declared("v", Per::Token, Layers::Attention, &["kv_heads", "head_dim"], Dtype::Bf16, Rollback::Positional),
        declared("conv", Per::Sequence, Layers::Linear, &["conv_dim", "conv_taps"], Dtype::F32, Rollback::Checkpoint),
        declared("ssm", Per::Sequence, Layers::Linear, &["value_heads", "key_head_dim", "key_head_dim"], Dtype::F32, Rollback::Checkpoint),
    ]
}

pub fn is_stock(states: &[State]) -> bool {
    states == stock().as_slice()
}

fn symbol(spec: &Spec, name: &str) -> Option<usize> {
    let a = &spec.attention;
    let la = spec.linear_attention.as_ref();
    Some(match name {
        "heads" => a.heads,
        "kv_heads" => a.kv_heads,
        "head_dim" => a.head_dim,
        "q_dim" => spec.q_dim(),
        "kv_dim" => spec.kv_dim(),
        "hidden" => spec.dims.hidden,
        "conv_dim" => la?.conv_dim(),
        "conv_taps" => la?.conv_kernel.checked_sub(1)?,
        "value_heads" => la?.value_heads,
        "key_heads" => la?.key_heads,
        "key_head_dim" => la?.key_head_dim,
        "value_head_dim" => la?.value_head_dim,
        other => match other.parse() {
            Ok(n) => n,
            Err(_) => {
                let v = *a.params.get(other)?;
                (v >= 0.0 && v.fract() == 0.0).then_some(v as usize)?
            }
        },
    })
}

pub fn dim(spec: &Spec, text: &str) -> Result<usize> {
    let mut total = 0usize;
    for term in text.split('+') {
        let mut product = 1usize;
        for factor in term.split('*') {
            let f = factor.trim();
            product *= symbol(spec, f).with_context(|| format!("state shape symbol {f:?} is not a size this model provides"))?;
        }
        total += product;
    }
    Ok(total)
}

impl State {
    pub fn layer_count(&self, spec: &Spec) -> usize {
        match self.layers {
            Layers::Attention => spec.kv_layers(),
            Layers::Linear => spec.linear_layers(),
        }
    }

    pub fn dims(&self, spec: &Spec) -> Result<Vec<usize>> {
        self.shape.iter().map(|s| dim(spec, s)).collect()
    }

    pub fn elements(&self, spec: &Spec) -> Result<usize> {
        Ok(self.dims(spec)?.iter().product())
    }

    pub fn rows_and_width(&self, spec: &Spec) -> Result<(usize, usize)> {
        let d = self.dims(spec)?;
        let (width, rows) = d.split_last().context("a state needs a shape")?;
        Ok((rows.iter().product(), *width))
    }

    pub fn positional(&self) -> bool {
        self.per == Per::Token
    }

    pub fn token_layer_bytes(&self, spec: &Spec, attention_index: usize, planes: usize) -> usize {
        let Ok((rows, width)) = self.rows_and_width(spec) else { return 0 };
        if spec.runtime.kv_dtype.is_fp8() {
            return if attention_index < spec.kv_f16_layers() { rows * width * 2 } else { rows * (width + 4) };
        }
        rows * width * planes * 2
    }

    pub fn block_bytes(&self, spec: &Spec, planes: usize) -> usize {
        (0..self.layer_count(spec)).map(|i| self.token_layer_bytes(spec, i, planes)).sum::<usize>() * crate::codegen::PAGE
    }

    pub fn sequence_bytes(&self, spec: &Spec) -> usize {
        let size = match self.dtype {
            Dtype::Bf16 => 2,
            Dtype::F32 => 4,
        };
        self.layer_count(spec) * self.elements(spec).unwrap_or(0) * size
    }
}

pub fn positional(spec: &Spec) -> impl Iterator<Item = &State> {
    spec.state.iter().filter(|s| s.positional())
}

pub fn sequence(spec: &Spec) -> impl Iterator<Item = &State> {
    spec.state.iter().filter(|s| s.per == Per::Sequence)
}

pub fn named<'a>(spec: &'a Spec, name: &str) -> Option<&'a State> {
    spec.state.iter().find(|s| s.name == name)
}

pub fn kv_block_bytes(spec: &Spec, planes: usize) -> usize {
    positional(spec).map(|s| s.block_bytes(spec, planes)).sum()
}

pub fn kv_bytes_per_token_at(spec: &Spec, attention_index: usize) -> u64 {
    positional(spec).filter(|s| s.layers == Layers::Attention).map(|s| s.token_layer_bytes(spec, attention_index, 1) as u64).sum()
}

pub fn sequence_bytes(spec: &Spec) -> u64 {
    sequence(spec).map(|s| s.sequence_bytes(spec) as u64).sum()
}

pub fn sequence_bytes_per_layer(spec: &Spec, layers: Layers) -> u64 {
    let size = |s: &State| match s.dtype {
        Dtype::Bf16 => 2,
        Dtype::F32 => 4,
    };
    sequence(spec).filter(|s| s.layers == layers).map(|s| (s.elements(spec).unwrap_or(0) * size(s)) as u64).sum()
}

pub fn validate(spec: &Spec) -> Result<()> {
    if spec.attention.node.is_empty() && spec.attention_layers() > 0 {
        for want in ["k", "v"] {
            ensure!(named(spec, want).is_some_and(|s| s.positional() && s.layers == Layers::Attention), "stock attention needs a per token [[state]] named {want:?}");
        }
    }
    if spec.linear_layers() > 0 {
        for want in ["conv", "ssm"] {
            ensure!(named(spec, want).is_some_and(|s| s.per == Per::Sequence && s.layers == Layers::Linear), "delta-net layers need a per sequence [[state]] named {want:?}");
        }
    }
    ensure!(positional(spec).count() <= 2, "at most two per token [[state]] caches are supported");
    for (i, s) in spec.state.iter().enumerate() {
        let at = format!("[[state]] {:?}", s.name);
        ensure!(!s.name.is_empty() && spec.state[..i].iter().all(|o| o.name != s.name), "{at} has an empty or repeated name");
        ensure!(!s.shape.is_empty(), "{at} has no shape");
        match (s.per, s.rollback) {
            (Per::Token, Rollback::Positional) | (Per::Sequence | Per::Layer, Rollback::Checkpoint) => {}
            (Per::Token, _) => bail!("{at} is per token, so its rollback is positional"),
            _ => bail!("{at} is not per token, so its rollback is checkpoint"),
        }
        ensure!(s.per != Per::Token || s.dtype == Dtype::Bf16, "{at} is per token, which the paged cache stores as bf16");
        if let Some(owner) = &s.owner {
            let layer = owner.strip_prefix("layer:").and_then(|n| n.parse::<usize>().ok());
            ensure!(layer.is_some_and(|l| l < spec.layers.count), "{at} owner {owner:?} is not layer:N inside the model");
        }
        if s.layer_count(spec) > 0 {
            let d = s.dims(spec).with_context(|| at.clone())?;
            ensure!(d.iter().all(|&x| x > 0), "{at} has a zero sized axis");
        }
    }
    Ok(())
}
