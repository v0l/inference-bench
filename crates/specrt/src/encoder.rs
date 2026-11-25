use crate::weights::Checkpoint;
use anyhow::{Context, Result, bail, ensure};
use kir::expr::Expr;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

pub const GRAPH: &str = include_str!("../graphs/encoder.toml");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    op: String,
    #[serde(default)]
    weight: Option<String>,
    #[serde(default, rename = "in")]
    input: Option<String>,
    #[serde(default)]
    with: Option<String>,
    out: String,
    #[serde(default)]
    add: bool,
    #[serde(default)]
    act: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionDecl {
    scale: String,
    bias: Vec<String>,
    bias_fallback_block: usize,
    half: String,
    exact: String,
    bucket: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EncoderDecl {
    name: String,
    classes: Vec<String>,
    embed: Vec<String>,
    layers: String,
    prefix: String,
    final_norm: String,
    eps: String,
    pad_rows: String,
    fields: BTreeMap<String, String>,
    attention: AttentionDecl,
    node: Vec<Node>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphFile {
    name: String,
    kind: String,
    encoder: Vec<EncoderDecl>,
}

pub struct HostEncoder {
    decl: EncoderDecl,
    values: BTreeMap<String, f64>,
    weights: BTreeMap<String, Vec<f32>>,
    pub weight_bytes: u64,
}

fn number(config: &serde_json::Value, key: &str) -> Result<f64> {
    config[key].as_f64().with_context(|| format!("the text encoder config has no number {key}"))
}

fn eval(text: &str, v: &BTreeMap<String, f64>) -> Result<f64> {
    Expr::parse(text).map_err(anyhow::Error::msg)?.eval(&|n| v.get(n).copied()).with_context(|| format!("{text:?} names a value the encoder config does not give"))
}

pub fn class_of(dir: &Path) -> Result<String> {
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    config["architectures"][0].as_str().map(String::from).context("the text encoder config names no architecture")
}

pub fn supports(class: &str) -> bool {
    toml::from_str::<GraphFile>(GRAPH).is_ok_and(|g| g.encoder.iter().any(|e| e.classes.iter().any(|c| c == class)))
}

impl HostEncoder {
    pub fn open(dir: &Path) -> Result<HostEncoder> {
        let g: GraphFile = toml::from_str(GRAPH).context("graphs/encoder.toml")?;
        ensure!(g.kind == "encoder-graph" && g.name == "encoder", "graphs/encoder.toml is not the encoder graph");
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let class = class_of(dir)?;
        let decl = g.encoder.into_iter().find(|e| e.classes.contains(&class)).with_context(|| format!("no encoder graph runs {class}"))?;
        let mut values = BTreeMap::new();
        for (name, key) in &decl.fields {
            values.insert(name.clone(), number(&config, key)?);
        }
        values.insert("layers".into(), number(&config, &decl.layers)?);
        values.insert("eps".into(), number(&config, &decl.eps)?);
        let ck = Checkpoint::open(dir)?;
        let mut weights = BTreeMap::new();
        let mut weight_bytes = 0u64;
        for name in ck.names_under("") {
            let v = ck.f32_values(&name, &ck.shape_of(&name)?)?;
            weight_bytes += 4 * v.len() as u64;
            weights.insert(name, v);
        }
        ensure!(decl.embed.iter().any(|e| weights.contains_key(e)), "the {} encoder has none of {:?}", decl.name, decl.embed);
        Ok(HostEncoder { decl, values, weights, weight_bytes })
    }

    fn size(&self, n: &str) -> usize {
        self.values[n] as usize
    }

    fn w(&self, name: &str) -> Result<&[f32]> {
        self.weights.get(name).map(|v| v.as_slice()).with_context(|| format!("weight {name} is not in the encoder checkpoint"))
    }

    fn bias_table(&self, block: usize) -> Result<&[f32]> {
        let a = &self.decl.attention;
        for b in [block, a.bias_fallback_block] {
            for n in &a.bias {
                let name = format!("{}.{n}", self.decl.prefix.replace("{i}", &b.to_string()));
                if let Some(v) = self.weights.get(&name) {
                    return Ok(v);
                }
            }
        }
        bail!("block {block} has no relative attention bias")
    }

    pub fn buckets(&self, rows: usize) -> Result<Vec<usize>> {
        let mut v = self.values.clone();
        let half = eval(&self.decl.attention.half, &v)?;
        v.insert("half".into(), half);
        let exact = eval(&self.decl.attention.exact, &v)?;
        v.insert("exact".into(), exact);
        let e = Expr::parse(&self.decl.attention.bucket).map_err(anyhow::Error::msg)?;
        let mut out = Vec::with_capacity(rows * rows);
        for q in 0..rows {
            for k in 0..rows {
                v.insert("d".into(), k as f64 - q as f64);
                out.push(e.eval(&|n| v.get(n).copied()).context("the bucket rule does not evaluate")? as usize);
            }
        }
        Ok(out)
    }

    pub fn encode(&self, ids: &[u32], mask: &[u8]) -> Result<Vec<f32>> {
        let h = self.size("hidden");
        let n = ids.len();
        let embed = self.decl.embed.iter().find_map(|e| self.weights.get(e)).context("no embedding")?;
        let mut bufs: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut x = Vec::with_capacity(n * h);
        for &id in ids {
            let r = id as usize;
            ensure!((r + 1) * h <= embed.len(), "token id {id} is outside the vocabulary");
            x.extend_from_slice(&embed[r * h..(r + 1) * h]);
        }
        bufs.insert("x".into(), x);
        let buckets = self.buckets(n)?;
        let eps = self.values["eps"] as f32;
        for i in 0..self.size("layers") {
            let prefix = self.decl.prefix.replace("{i}", &i.to_string());
            let table = self.bias_table(i)?.to_vec();
            for node in &self.decl.node {
                let input = node.input.as_deref().map(|s| bufs.get(s).cloned().with_context(|| format!("{s} is read before it is written"))).transpose()?;
                match node.op.as_str() {
                    "rms_norm" => {
                        let g = self.w(&format!("{prefix}.{}.weight", node.weight.as_deref().context("rms_norm needs a weight")?))?;
                        bufs.insert(node.out.clone(), rms(&input.context("rms_norm needs in")?, g, eps));
                    }
                    "linear" => {
                        let w = self.w(&format!("{prefix}.{}.weight", node.weight.as_deref().context("linear needs a weight")?))?;
                        let x = input.context("linear needs in")?;
                        let k = x.len() / n;
                        let o = w.len() / k;
                        let mut y = vec![0f32; n * o];
                        for r in 0..n {
                            for c in 0..o {
                                let acc: f32 = (0..k).map(|j| x[r * k + j] * w[c * k + j]).sum();
                                y[r * o + c] = match node.act.as_deref() {
                                    Some("gelu_tanh") => 0.5 * acc * (1.0 + (0.797_884_6 * (acc + 0.044715 * acc * acc * acc)).tanh()),
                                    Some(other) => bail!("encoder act {other:?} is not gelu_tanh"),
                                    None => acc,
                                };
                            }
                        }
                        if node.add {
                            let base = bufs.get(&node.out).context("an added linear writes onto an existing value")?;
                            y.iter_mut().zip(base).for_each(|(a, b)| *a += b);
                        }
                        bufs.insert(node.out.clone(), y);
                    }
                    "mul" => {
                        let a = input.context("mul needs in")?;
                        let b = bufs.get(node.with.as_deref().context("mul needs with")?).context("mul reads an unwritten value")?;
                        bufs.insert(node.out.clone(), a.iter().zip(b).map(|(p, q)| p * q).collect());
                    }
                    "attention" => {
                        let heads = self.size("heads");
                        let hd = self.size("head_dim");
                        let inner = heads * hd;
                        let (q, k, v) = (&bufs["q"], &bufs["k"], &bufs["v"]);
                        let scale = eval(&self.decl.attention.scale, &self.values)? as f32;
                        let mut out = vec![0f32; n * inner];
                        for hh in 0..heads {
                            for qi in 0..n {
                                let scores: Vec<f32> = (0..n)
                                    .map(|ki| {
                                        let dot: f32 = (0..hd).map(|d| q[qi * inner + hh * hd + d] * k[ki * inner + hh * hd + d]).sum();
                                        if mask[ki] == 0 { f32::NEG_INFINITY } else { dot * scale + table[buckets[qi * n + ki] * heads + hh] }
                                    })
                                    .collect();
                                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                                let z: f32 = e.iter().sum();
                                for d in 0..hd {
                                    out[qi * inner + hh * hd + d] = (0..n).map(|ki| e[ki] * v[ki * inner + hh * hd + d]).sum::<f32>() / z;
                                }
                            }
                        }
                        bufs.insert(node.out.clone(), out);
                    }
                    other => bail!("unknown encoder op {other:?}"),
                }
            }
        }
        let mut y = rms(&bufs["x"], self.w(&format!("{}.weight", self.decl.final_norm))?, eps);
        if self.decl.pad_rows == "zeroed" {
            let real = mask.iter().filter(|m| **m == 1).count();
            y[real * h..].fill(0.0);
        }
        Ok(y)
    }
}

fn rms(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let h = g.len();
    let mut out = vec![0f32; x.len()];
    for (r, row) in x.chunks(h).enumerate() {
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / h as f32 + eps).sqrt();
        for c in 0..h {
            out[r * h + c] = g[c] * (row[c] * inv);
        }
    }
    out
}
