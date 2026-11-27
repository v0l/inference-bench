use crate::plan::{Buf, NormKind, Op, Phase, Plan, Weight};
use crate::spec::Spec;
use crate::weights::Checkpoint;
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;

pub struct HostDecoder {
    pub spec: Spec,
    pub prefill: Plan,
    pub decode: Plan,
    weights: BTreeMap<String, Vec<f32>>,
    inputs: Vec<kir::Kernel>,
    pub bf16: bool,
}

#[derive(Clone, Default)]
pub struct Sequence {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub cross_kv: Vec<Vec<f32>>,
    pub len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Out {
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
}

struct Rows<'a> {
    seqs: Vec<&'a mut Sequence>,
    tokens: Vec<Vec<u32>>,
    streams: BTreeMap<String, Vec<f32>>,
}

fn bf16_round(v: f32) -> f32 {
    crate::weights::bf16_to_f32(crate::weights::f32_to_bf16(v))
}

impl HostDecoder {
    pub fn open(dir: &std::path::Path, bf16: bool) -> Result<HostDecoder> {
        let json = std::fs::read_to_string(dir.join("config.json")).with_context(|| dir.display().to_string())?;
        let spec = Spec::from_hf_config_speech(&json)?;
        let ck = Checkpoint::open(dir)?.renamed(&spec.tensors);
        HostDecoder::load(spec, &ck, bf16)
    }

    pub fn load(spec: Spec, ck: &Checkpoint, bf16: bool) -> Result<HostDecoder> {
        HostDecoder::load_with(spec, ck, bf16, BTreeMap::new())
    }

    pub fn load_with(spec: Spec, ck: &Checkpoint, bf16: bool, extra: BTreeMap<String, Vec<f32>>) -> Result<HostDecoder> {
        ensure!(spec.cross.is_some() || spec.inputs.is_some(), "the {} spec has neither a [cross] section nor [inputs]", spec.name);
        ensure!(spec.attention.kv_heads == spec.attention.heads, "the host decoder runs equal query and key heads");
        let prefill = Plan::build(&spec, Phase::Prefill, true);
        let decode = Plan::build(&spec, Phase::Decode, true);
        for plan in [&prefill, &decode] {
            crate::signature::validate_plan(plan, &spec)?;
        }
        let mut names: Vec<String> = Vec::new();
        for op in prefill.head.iter().chain(&prefill.tail) {
            names.extend(op.checkpoint_tensors(&spec, "model", None));
        }
        for i in 0..spec.layers.count {
            for op in prefill.layer(spec.block(i)).iter().chain(&prefill.window).chain(decode.layer(spec.block(i))) {
                names.extend(op.checkpoint_tensors(&spec, "model", Some(i)));
            }
        }
        names.sort();
        names.dedup();
        let mut weights = BTreeMap::new();
        for n in names {
            if let Some(v) = extra.get(&n) {
                weights.insert(n, v.clone());
                continue;
            }
            let shape = ck.shape_of(&n).with_context(|| format!("the {} plan reads {n}", spec.name))?;
            let mut v = ck.f32_values(&n, &shape)?;
            if bf16 {
                v.iter_mut().for_each(|x| *x = bf16_round(*x));
            }
            weights.insert(n, v);
        }
        let inputs = crate::codegen::input_kernels(&spec, &kir::Target::sm_120())?;
        Ok(HostDecoder { spec, prefill, decode, weights, inputs, bf16 })
    }

    fn w(&self, name: &str) -> Result<&[f32]> {
        self.weights.get(name).map(Vec::as_slice).with_context(|| format!("weight {name} is not loaded"))
    }

    pub fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.w(name)
    }

    pub fn layer_w(&self, layer: Option<usize>, t: &str) -> Result<&[f32]> {
        match layer {
            Some(i) => self.w(&format!("model.layers.{i}.{t}")),
            None if t.starts_with("lm_head") => self.w(t),
            None => self.w(&format!("model.{t}")),
        }
    }

    pub fn new_sequence(&self) -> Sequence {
        let n = self.spec.layers.count;
        Sequence { keys: vec![Vec::new(); n], values: vec![Vec::new(); n], cross_kv: vec![Vec::new(); n], len: 0 }
    }

    pub fn window(&self, seq: &mut Sequence, states: &[f32]) -> Result<()> {
        let c = self.spec.cross.as_ref().context("no [cross] section")?;
        let h = self.spec.dims.hidden;
        ensure!(states.len() == c.rows * h, "{} encoder values, the [cross] section reads {} rows of {h}", states.len(), c.rows);
        for op in &self.prefill.window {
            match op {
                Op::CrossKv => {
                    for i in 0..self.spec.layers.count {
                        seq.cross_kv[i] = self.linear(Weight::CrossKv, Some(i), states, c.rows)?;
                    }
                }
                other => bail!("{} is not a window op", other.label()),
            }
        }
        *seq = Sequence { keys: vec![Vec::new(); self.spec.layers.count], values: vec![Vec::new(); self.spec.layers.count], cross_kv: std::mem::take(&mut seq.cross_kv), len: 0 };
        Ok(())
    }

    pub fn prefill(&self, seq: &mut Sequence, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.prefill_with(seq, tokens, BTreeMap::new())?.logits)
    }

    pub fn prefill_with(&self, seq: &mut Sequence, tokens: &[u32], streams: BTreeMap<String, Vec<f32>>) -> Result<Out> {
        let mut rows = Rows { seqs: vec![seq], tokens: vec![tokens.to_vec()], streams };
        self.run(&self.prefill, &mut rows)
    }

    pub fn step(&self, seqs: &mut [&mut Sequence], tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.step_with(seqs, tokens, BTreeMap::new())?.logits)
    }

    pub fn step_with(&self, seqs: &mut [&mut Sequence], tokens: &[u32], streams: BTreeMap<String, Vec<f32>>) -> Result<Out> {
        ensure!(seqs.len() == tokens.len(), "{} sequences and {} tokens", seqs.len(), tokens.len());
        let mut rows = Rows { seqs: seqs.iter_mut().map(|s| &mut **s).collect(), tokens: tokens.iter().map(|t| vec![*t]).collect(), streams };
        self.run(&self.decode, &mut rows)
    }

    fn run(&self, plan: &Plan, rows: &mut Rows) -> Result<Out> {
        let h = self.spec.dims.hidden;
        let total: usize = rows.tokens.iter().map(Vec::len).sum();
        let mut bufs: BTreeMap<Buf, Vec<f32>> = BTreeMap::new();
        bufs.insert(Buf::X, vec![0.0; total * h]);
        for op in &plan.head {
            self.op(op, None, rows, &mut bufs)?;
        }
        for i in 0..self.spec.layers.count {
            for op in plan.layer(self.spec.block(i)) {
                self.op(op, Some(i), rows, &mut bufs)?;
            }
        }
        for op in &plan.tail {
            self.op(op, None, rows, &mut bufs)?;
        }
        for (s, t) in rows.seqs.iter_mut().zip(&rows.tokens) {
            s.len += t.len();
        }
        let hidden = bufs.remove(&Buf::XnPlanes).unwrap_or_default();
        Ok(Out { logits: bufs.remove(&Buf::Logits).context("the plan wrote no logits")?, hidden })
    }

    fn norm(&self, which: NormKind, layer: Option<usize>, x: &[f32]) -> Result<Vec<f32>> {
        let h = self.spec.dims.hidden;
        let t = which.tensor();
        let g = self.layer_w(layer, t)?;
        let eps = self.spec.norm.eps as f32;
        let layer_norm = self.spec.norm.kind == crate::spec::NormKind::LayerNorm;
        let b = if layer_norm { Some(self.layer_w(layer, &t.replace(".weight", ".bias"))?) } else { None };
        let mut out = vec![0f32; x.len()];
        for (r, row) in x.chunks(h).enumerate() {
            let mean = if layer_norm { row.iter().sum::<f32>() / h as f32 } else { 0.0 };
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..h {
                let off = if self.spec.norm.unit_offset { 1.0 } else { 0.0 };
                out[r * h + c] = (row[c] - mean) * inv * (g[c] + off) + b.map_or(0.0, |b| b[c]);
            }
        }
        Ok(out)
    }

    pub fn linear_weights(&self, weight: Weight, layer: Option<usize>) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let s = crate::gemm::linear_shape(&self.spec, weight.name());
        let names = weight.tensors(&self.spec);
        let mut w = Vec::with_capacity(s.n * s.k);
        let mut bias = vec![0f32; s.n];
        let mut at = 0;
        for name in names.iter().filter(|n| n.ends_with(".weight")) {
            let part = self.layer_w(layer, name)?;
            let n = part.len() / s.k;
            w.extend_from_slice(part);
            let b = name.replace(".weight", ".bias");
            if names.contains(&b) {
                bias[at..at + n].copy_from_slice(self.layer_w(layer, &b)?);
            }
            at += n;
        }
        ensure!(w.len() == s.n * s.k && at == s.n, "{}: {} weights for {} x {}", weight.name(), w.len(), s.n, s.k);
        Ok((w, bias, s.n, s.k))
    }

    pub fn norm_weights(&self, which: NormKind, layer: Option<usize>) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = which.tensor();
        let g = self.layer_w(layer, t)?.to_vec();
        let b = match self.spec.norm.kind {
            crate::spec::NormKind::LayerNorm => self.layer_w(layer, &t.replace(".weight", ".bias"))?.to_vec(),
            _ => vec![0f32; g.len()],
        };
        Ok((g, b))
    }

    fn linear(&self, weight: Weight, layer: Option<usize>, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (w, bias, n, k) = self.linear_weights(weight, layer)?;
        ensure!(x.len() == rows * k, "{}: {} inputs for {rows} rows of {k}", weight.name(), x.len());
        Ok(crate::speech::host::matmul(x, rows, k, &w, n, Some(&bias)))
    }

    fn act(&self, v: &mut [f32]) -> Result<()> {
        let m = self.spec.mlp.mapped.as_ref().context("the host decoder runs a mapped (non gated) mlp")?;
        let e = kir::expr::Expr::parse(&m.expr).map_err(anyhow::Error::msg)?;
        for x in v.iter_mut() {
            let env = |n: &str| if n == "x" { Some(f64::from(*x)) } else { m.params.get(n).copied() };
            *x = e.eval(&env).context("the mapped activation does not evaluate on the host")? as f32;
        }
        Ok(())
    }

    pub fn attend_cross(&self, seq: &Sequence, layer: usize, q: &[f32]) -> Result<Vec<f32>> {
        let c = self.spec.cross.as_ref().context("no [cross] section")?;
        let qd = self.spec.q_dim();
        let kv = &seq.cross_kv[layer];
        ensure!(kv.len() == c.rows * 2 * qd, "layer {layer}: the sequence has no {} state; run the window first", c.state);
        let keys: Vec<f32> = kv.chunks(2 * qd).flat_map(|r| r[..qd].to_vec()).collect();
        let values: Vec<f32> = kv.chunks(2 * qd).flat_map(|r| r[qd..].to_vec()).collect();
        Ok(self.attend(q, &keys, &values, c.rows, None))
    }

    fn attend(&self, q: &[f32], keys: &[f32], values: &[f32], n_keys: usize, causal_from: Option<usize>) -> Vec<f32> {
        let a = &self.spec.attention;
        let (hd, heads) = (a.head_dim, a.heads);
        let width = heads * hd;
        let scale = a.query_scalar.map_or(1.0 / (hd as f32).sqrt(), |s| (1.0 / s.sqrt()) as f32);
        let rows = q.len() / width;
        let mut out = vec![0f32; q.len()];
        for r in 0..rows {
            let visible = causal_from.map_or(n_keys, |start| start + r + 1);
            for hh in 0..heads {
                let qv = &q[r * width + hh * hd..r * width + (hh + 1) * hd];
                let scores: Vec<f32> = (0..visible).map(|k| (0..hd).map(|i| qv[i] * keys[k * width + hh * hd + i]).sum::<f32>() * scale).collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for i in 0..hd {
                    out[r * width + hh * hd + i] = (0..visible).map(|k| e[k] * values[k * width + hh * hd + i]).sum::<f32>() / z;
                }
            }
        }
        out
    }

    fn per_seq<'b>(tokens: &[Vec<u32>], buf: &'b [f32], width: usize) -> Vec<&'b [f32]> {
        let mut at = 0;
        tokens.iter().map(|t| {
            let s = &buf[at * width..(at + t.len()) * width];
            at += t.len();
            s
        }).collect()
    }

    fn op(&self, op: &Op, layer: Option<usize>, rows: &mut Rows, bufs: &mut BTreeMap<Buf, Vec<f32>>) -> Result<()> {
        let h = self.spec.dims.hidden;
        let total: usize = rows.tokens.iter().map(Vec::len).sum();
        let qd = self.spec.q_dim();
        match *op {
            Op::Embed => {
                let table = self.w("model.embed_tokens.weight")?;
                let x = bufs.get_mut(&Buf::X).unwrap();
                for (r, t) in rows.tokens.iter().flatten().enumerate() {
                    x[r * h..(r + 1) * h].copy_from_slice(&table[*t as usize * h..(*t as usize + 1) * h]);
                }
            }
            Op::PositionAdd => {
                let name = self.spec.dims.positions.clone().context("no [embedding] positions")?;
                let table = self.w(&format!("model.{name}"))?;
                let x = bufs.get_mut(&Buf::X).unwrap();
                let mut r = 0;
                for (s, t) in rows.seqs.iter().zip(&rows.tokens) {
                    for p in s.len..s.len + t.len() {
                        ensure!((p + 1) * h <= table.len(), "position {p} is past the learned table");
                        for c in 0..h {
                            x[r * h + c] += table[p * h + c];
                        }
                        r += 1;
                    }
                }
            }
            Op::Norm { which, into } => {
                let out = self.norm(which, layer, &bufs[&Buf::X])?;
                bufs.insert(into, out);
            }
            Op::Linear { weight, input, output, .. } => {
                let mut y = self.linear(weight, layer, &bufs[&input], total)?;
                let s = crate::gemm::linear_shape(&self.spec, weight.name());
                match s.epi {
                    kir::gemm::Epilogue::Residual => {
                        let x = bufs.get_mut(&output).context("a residual linear needs its output")?;
                        x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
                    }
                    kir::gemm::Epilogue::Map => {
                        self.act(&mut y)?;
                        bufs.insert(output, y);
                    }
                    kir::gemm::Epilogue::Store => {
                        bufs.insert(output, y);
                    }
                    other => bail!("the host decoder has no {other:?} epilogue for {}", weight.name()),
                }
            }
            Op::KvRope | Op::DecodeRope => {
                let i = layer.context("the kv write runs inside a layer")?;
                self.rope(rows, bufs, i)?;
                let qkv = bufs[&Buf::Qkv].clone();
                for (s, part) in rows.seqs.iter_mut().zip(Self::per_seq(&rows.tokens, &qkv, 3 * qd)) {
                    for row in part.chunks(3 * qd) {
                        s.keys[i].extend_from_slice(&row[qd..2 * qd]);
                        s.values[i].extend_from_slice(&row[2 * qd..]);
                    }
                }
            }
            Op::Flash | Op::DecodeAttention => {
                let i = layer.context("attention runs inside a layer")?;
                if *op == Op::DecodeAttention {
                    self.op(&Op::DecodeRope, layer, rows, bufs)?;
                }
                let qkv = bufs[&Buf::Qkv].clone();
                let mut out = Vec::with_capacity(total * qd);
                for (s, part) in rows.seqs.iter().zip(Self::per_seq(&rows.tokens, &qkv, 3 * qd)) {
                    let q: Vec<f32> = part.chunks(3 * qd).flat_map(|r| r[..qd].to_vec()).collect();
                    let n_keys = s.keys[i].len() / qd;
                    let start = n_keys - part.len() / (3 * qd);
                    out.extend(self.attend(&q, &s.keys[i], &s.values[i], n_keys, Some(start)));
                }
                bufs.insert(Buf::AttnPlanes, out);
            }
            Op::CrossAttention { .. } => {
                let i = layer.context("cross attention runs inside a layer")?;
                let q = bufs[&Buf::Qkv].clone();
                let mut out = Vec::with_capacity(total * qd);
                for (s, part) in rows.seqs.iter().zip(Self::per_seq(&rows.tokens, &q, qd)) {
                    out.extend(self.attend_cross(s, i, part)?);
                }
                bufs.insert(Buf::AttnPlanes, out);
            }
            Op::Input { node } => self.input(node as usize, rows, bufs)?,
            ref other => bail!("the host decoder does not run {}", other.label()),
        }
        Ok(())
    }

    fn rope(&self, rows: &Rows, bufs: &mut BTreeMap<Buf, Vec<f32>>, layer: usize) -> Result<()> {
        if self.spec.nope(layer) {
            return Ok(());
        }
        let a = &self.spec.attention;
        let (hd, qd) = (a.head_dim, self.spec.q_dim());
        let rot = self.spec.rotary_dim();
        let theta = self.spec.rope.theta.context("the spec has no rope theta")?;
        ensure!(self.spec.rope.scaling.is_none() && self.spec.rope.mrope.is_none(), "the host decoder runs plain rope");
        let half = rot / 2;
        let inv: Vec<f64> = (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f64 / rot as f64)).collect();
        let qkv = bufs.get_mut(&Buf::Qkv).context("rope reads qkv")?;
        let mut r = 0;
        for (s, t) in rows.seqs.iter().zip(&rows.tokens) {
            for p in s.len..s.len + t.len() {
                let row = &mut qkv[r * 3 * qd..(r + 1) * 3 * qd];
                for part in 0..2 {
                    for head in 0..qd / hd {
                        let v = &mut row[part * qd + head * hd..part * qd + (head + 1) * hd];
                        for (i, f) in inv.iter().enumerate() {
                            let (sin, cos) = ((p as f64) * f).sin_cos();
                            let (a, b) = match self.spec.rope.style {
                                crate::spec::RopeStyle::Pairs => (2 * i, 2 * i + 1),
                                crate::spec::RopeStyle::Neox => (i, i + half),
                            };
                            let (x, y) = (v[a], v[b]);
                            v[a] = (f64::from(x) * cos - f64::from(y) * sin) as f32;
                            v[b] = (f64::from(y) * cos + f64::from(x) * sin) as f32;
                        }
                    }
                }
                r += 1;
            }
        }
        Ok(())
    }

    fn input(&self, node: usize, rows: &Rows, bufs: &mut BTreeMap<Buf, Vec<f32>>) -> Result<()> {
        let p = self.spec.inputs.as_ref().context("an input node needs [inputs]")?;
        let n = &p.node[node];
        let kernel = self.inputs.get(node).context("the input node has no kernel")?;
        let kir::Body::Seq(kir::recur::SeqBody::GroupMap(g)) = &kernel.body else { bail!("input node {node} ({}) is not a group_map kernel", kernel.name) };
        let total: usize = rows.tokens.iter().map(Vec::len).sum();
        let x_in = bufs[&Buf::X].clone();
        let mut owned: Vec<Vec<f32>> = Vec::new();
        let mut out_at = None;
        for param in &kernel.params {
            let v = match n.bind.get(param.name) {
                Some(crate::graph::Bind::Weight { weight, .. }) => self.w(weight)?.to_vec(),
                Some(crate::graph::Bind::Buffer { buffer }) if buffer == "x" && param.access == kir::Access::Write => {
                    out_at = Some(owned.len());
                    x_in.clone()
                }
                Some(crate::graph::Bind::Buffer { buffer }) if buffer == "x" => x_in.clone(),
                Some(crate::graph::Bind::Buffer { buffer }) => match rows.streams.get(buffer) {
                    Some(v) => v.clone(),
                    None => vec![0.0; total * p.width_of(buffer).with_context(|| format!("input buffer {buffer} has no width"))?],
                },
                other => bail!("input node {node} binds {} to {other:?}, which the host does not run", param.name),
            };
            owned.push(v);
        }
        let groups = crate::state::dim(&self.spec, &n.grid[0])?;
        let mut views: Vec<&mut [f32]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
        g.eval(&kernel.params, (groups, total), &mut views).map_err(|e| anyhow::anyhow!("input node {node} ({}): {e}", kernel.name))?;
        let at = out_at.context("the input node writes no x")?;
        bufs.insert(Buf::X, std::mem::take(&mut owned[at]));
        Ok(())
    }
}
