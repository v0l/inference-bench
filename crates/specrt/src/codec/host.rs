use super::graph::{CodecOp, CodecPlan, Slot, Stack};
use crate::vae::host::Weights;
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Lowered {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub taps: usize,
    pub stride: usize,
    pub phases: usize,
    pub depthwise: bool,
    pub act: Option<&'static str>,
}

impl Lowered {
    pub fn state_cols(&self) -> usize {
        self.taps - self.stride
    }
}

pub fn lower_up(w: &[f32], b: &[f32], cin: usize, cout: usize, s: usize) -> Lowered {
    let n = s * cout;
    let mut out = vec![0.0; n * 2 * cin];
    for r in 0..s {
        for co in 0..cout {
            let row = r * cout + co;
            for ci in 0..cin {
                let at = |j: usize| w[(ci * cout + co) * 2 * s + j];
                out[(row * 2) * cin + ci] = at(r + s);
                out[(row * 2 + 1) * cin + ci] = at(r);
            }
        }
    }
    let bias = (0..n).map(|i| b[i % cout]).collect();
    Lowered { w: out, b: bias, cin, cout: n, taps: 2, stride: 1, phases: s, depthwise: false, act: None }
}

pub fn lower_dw_up(w: &[f32], b: &[f32], channels: usize, s: usize) -> Lowered {
    let n = s * channels;
    let mut out = vec![0.0; n * 2];
    for r in 0..s {
        for c in 0..channels {
            out[(r * channels + c) * 2] = w[c * 2 * s + r + s];
            out[(r * channels + c) * 2 + 1] = w[c * 2 * s + r];
        }
    }
    let bias = (0..n).map(|i| b[i % channels]).collect();
    Lowered { w: out, b: bias, cin: channels, cout: n, taps: 2, stride: 1, phases: s, depthwise: true, act: None }
}

pub fn taps_major(w: &[f32], cout: usize, cin: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    for o in 0..cout {
        for c in 0..cin {
            for t in 0..k {
                out[(o * k + t) * cin + c] = w[(o * cin + c) * k + t];
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub enum Param {
    Conv(Lowered),
    Vec(Vec<f32>),
    Norm { w: Vec<f32>, b: Vec<f32> },
    Linear { w: Vec<f32>, b: Vec<f32> },
    Attention { qkv: Vec<f32>, out: Vec<f32> },
}

fn bias_or_zero(weights: &dyn Weights, weight: &str, on: bool, n: usize) -> Result<Vec<f32>> {
    if on { weights.floats(&format!("{weight}.bias")) } else { Ok(vec![0.0; n]) }
}

pub fn load_stack(stack: &Stack, weights: &dyn Weights) -> Result<Vec<Param>> {
    let mut out = Vec::with_capacity(stack.ops.len());
    for op in &stack.ops {
        out.push(match op {
            CodecOp::Conv { weight, cin, cout, k, stride, depthwise, bias, act, .. } => {
                let w = weights.floats(&format!("{weight}.weight"))?;
                let b = bias_or_zero(weights, weight, *bias, *cout)?;
                let wc = if *depthwise { 1 } else { *cin };
                ensure!(w.len() == cout * wc * k && b.len() == *cout, "{} has {} weights", op.label(), w.len());
                Param::Conv(Lowered { w: taps_major(&w, *cout, wc, *k), b, cin: *cin, cout: *cout, taps: *k, stride: *stride, phases: 1, depthwise: *depthwise, act: *act })
            }
            CodecOp::ConvUp { weight, cin, cout, stride, depthwise, bias, act } => {
                let w = weights.floats(&format!("{weight}.weight"))?;
                let b = bias_or_zero(weights, weight, *bias, *cout)?;
                let wc = if *depthwise { 1 } else { *cout };
                ensure!(w.len() == cin * wc * 2 * stride && b.len() == *cout, "{} has {} weights", op.label(), w.len());
                let l = if *depthwise { lower_dw_up(&w, &b, *cin, *stride) } else { lower_up(&w, &b, *cin, *cout, *stride) };
                Param::Conv(Lowered { act: *act, ..l })
            }
            CodecOp::RmsNorm { weight, .. } => Param::Vec(weights.floats(&format!("{weight}.weight"))?),
            CodecOp::LayerNorm { weight, .. } => Param::Norm { w: weights.floats(&format!("{weight}.weight"))?, b: weights.floats(&format!("{weight}.bias"))? },
            CodecOp::ScaleAdd { weight, .. } => Param::Vec(weights.floats(weight)?),
            CodecOp::Add { channels, .. } => Param::Vec(vec![1.0; *channels]),
            CodecOp::Linear { weight, cout, bias, .. } => Param::Linear { w: weights.floats(&format!("{weight}.weight"))?, b: bias_or_zero(weights, weight, *bias, *cout)? },
            CodecOp::Attention { weight, out_weight, .. } => Param::Attention { qkv: weights.floats(&format!("{weight}.weight"))?, out: weights.floats(&format!("{out_weight}.weight"))? },
        });
    }
    Ok(out)
}

pub fn gelu(x: f32) -> f32 {
    (0.5 * x as f64 * (1.0 + kir::expr::erf(x as f64 * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

pub fn act(name: Option<&str>, x: f32) -> f32 {
    match name {
        None => x,
        Some(n) => {
            let e = kir::expr::apply_macro(n, kir::expr::Expr::Name("x".into())).expect("a codec activation is a kir macro");
            e.eval(&|v| (v == "x").then_some(x as f64)).expect("a codec activation evaluates") as f32
        }
    }
}

pub fn conv(x: &[f32], t_in: usize, c: &Lowered, state: &mut [f32]) -> (Vec<f32>, usize) {
    let ctx = c.state_cols();
    let padded = t_in.div_ceil(c.stride) * c.stride;
    let mut xx = state.to_vec();
    xx.extend_from_slice(&x[..t_in * c.cin]);
    xx.resize((ctx + padded) * c.cin, 0.0);
    let end = xx.len() / c.cin;
    state.copy_from_slice(&xx[(end - ctx) * c.cin..end * c.cin]);
    if c.act.is_some() {
        for v in &mut xx {
            *v = act(c.act, *v);
        }
    }
    let t_out = padded / c.stride;
    let mut y = vec![0.0f32; t_out * c.cout];
    for o in 0..t_out {
        for n in 0..c.cout {
            let mut acc = c.b[n];
            for tap in 0..c.taps {
                let row = &xx[(o * c.stride + tap) * c.cin..];
                if c.depthwise {
                    acc += c.w[n * c.taps + tap] * row[n % c.cin];
                } else {
                    let w = &c.w[(n * c.taps + tap) * c.cin..(n * c.taps + tap + 1) * c.cin];
                    acc += w.iter().zip(row).map(|(a, b)| a * b).sum::<f32>();
                }
            }
            y[o * c.cout + n] = acc;
        }
    }
    if c.phases == 1 {
        return (y, t_out);
    }
    let per = c.cout / c.phases;
    let mut out = vec![0.0f32; t_out * c.cout];
    for o in 0..t_out {
        for r in 0..c.phases {
            out[(o * c.phases + r) * per..(o * c.phases + r + 1) * per].copy_from_slice(&y[o * c.cout + r * per..o * c.cout + (r + 1) * per]);
        }
    }
    (out, t_out * c.phases)
}

pub fn rms_norm(x: &[f32], c: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for (row, out) in x.chunks(c).zip(y.chunks_mut(c)) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for i in 0..c {
            out[i] = row[i] * r * gamma[i];
        }
    }
    y
}

pub fn layer_norm(x: &[f32], c: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for (row, out) in x.chunks(c).zip(y.chunks_mut(c)) {
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let r = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            out[i] = (row[i] - mean) * r * gamma[i] + beta[i];
        }
    }
    y
}

pub fn linear(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32], act: bool) -> Vec<f32> {
    let mut y = Vec::with_capacity(x.len() / cin * cout);
    for row in x.chunks(cin) {
        for o in 0..cout {
            let v = w[o * cin..(o + 1) * cin].iter().zip(row).map(|(a, b)| a * b).sum::<f32>() + b[o];
            y.push(if act { gelu(v) } else { v });
        }
    }
    y
}

pub fn rope_pairs(v: &mut [f32], heads: usize, pos: usize, theta: f32) {
    let d = v.len() / heads;
    for h in 0..heads {
        for i in 0..d / 2 {
            let inv = 1.0 / theta.powf((2 * i) as f32 / d as f32);
            let f = (pos as f32 * inv) as f64;
            let (c, s) = (f.cos() as f32, f.sin() as f32);
            let at = h * d + 2 * i;
            let (a, e) = (v[at], v[at + 1]);
            v[at] = a * c - e * s;
            v[at + 1] = e * c + a * s;
        }
    }
}

pub struct Ring<'a> {
    pub pos: &'a mut f32,
    pub k: &'a mut [f32],
    pub v: &'a mut [f32],
}

pub fn split_ring(state: &mut [f32], context: usize, channels: usize) -> Ring<'_> {
    let (pos, kv) = state.split_first_mut().expect("a ring state has a position");
    let (k, v) = kv.split_at_mut(context * channels);
    Ring { pos, k, v }
}

#[allow(clippy::too_many_arguments)]
pub fn ring_attention(x: &[f32], channels: usize, heads: usize, context: usize, theta: f32, qkv_w: &[f32], out_w: &[f32], ring: Ring<'_>) -> Vec<f32> {
    let c = channels;
    let d = c / heads;
    let t = x.len() / c;
    let p0 = *ring.pos as usize;
    let zero = vec![0.0; 3 * c];
    let mut qkv = linear(x, c, 3 * c, qkv_w, &zero, false);
    for (i, row) in qkv.chunks_mut(3 * c).enumerate() {
        rope_pairs(&mut row[..c], heads, p0 + i, theta);
        rope_pairs(&mut row[c..2 * c], heads, p0 + i, theta);
    }
    let key = |k: usize| if k >= p0 { &qkv[(k - p0) * 3 * c + c..(k - p0) * 3 * c + 2 * c] } else { &ring.k[(k % context) * c..(k % context + 1) * c] };
    let val = |k: usize| if k >= p0 { &qkv[(k - p0) * 3 * c + 2 * c..(k - p0) * 3 * c + 3 * c] } else { &ring.v[(k % context) * c..(k % context + 1) * c] };
    let scale = 1.0 / (d as f32).sqrt();
    let mut att = vec![0.0f32; t * c];
    for i in 0..t {
        let a = p0 + i;
        let first = (a + 1).saturating_sub(context);
        for h in 0..heads {
            let q = &qkv[i * 3 * c + h * d..i * 3 * c + (h + 1) * d];
            let scores: Vec<f32> = (first..=a).map(|k| key(k)[h * d..(h + 1) * d].iter().zip(q).map(|(x, y)| x * y).sum::<f32>() * scale).collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            for (j, k) in (first..=a).enumerate() {
                let vr = &val(k)[h * d..(h + 1) * d];
                for dd in 0..d {
                    att[i * c + h * d + dd] += e[j] / sum * vr[dd];
                }
            }
        }
    }
    for i in t.saturating_sub(context)..t {
        let slot = (p0 + i) % context;
        ring.k[slot * c..(slot + 1) * c].copy_from_slice(&qkv[i * 3 * c + c..i * 3 * c + 2 * c]);
        ring.v[slot * c..(slot + 1) * c].copy_from_slice(&qkv[i * 3 * c + 2 * c..i * 3 * c + 3 * c]);
    }
    *ring.pos += t as f32;
    linear(&att, c, c, out_w, &vec![0.0; c], false)
}

pub struct HostStack {
    pub stack: Stack,
    pub params: Vec<Param>,
}

#[derive(Clone, Debug)]
pub struct State {
    pub layers: BTreeMap<usize, Vec<f32>>,
}

impl State {
    pub fn floats(&self) -> usize {
        self.layers.values().map(Vec::len).sum()
    }
}

impl HostStack {
    pub fn new_state(&self) -> State {
        let layers = self
            .params
            .iter()
            .zip(&self.stack.ops)
            .enumerate()
            .filter_map(|(i, (p, op))| match (p, op) {
                (Param::Conv(c), _) => Some((i, vec![0.0; c.state_cols() * c.cin])),
                (Param::Attention { .. }, CodecOp::Attention { .. }) => Some((i, vec![0.0; op.state_floats()])),
                _ => None,
            })
            .collect();
        State { layers }
    }

    pub fn run(&self, input: &[f32], state: &mut State) -> Result<Vec<f32>> {
        let mut t = input.len() / self.stack.in_channels;
        ensure!(t * self.stack.in_channels == input.len(), "{} values are not whole steps of {} channels", input.len(), self.stack.in_channels);
        let mut slots: BTreeMap<Slot, Vec<f32>> = BTreeMap::from([(Slot::X, input.to_vec())]);
        for (i, (op, p)) in self.stack.ops.iter().zip(&self.params).enumerate() {
            let x = &slots[&op.input()];
            let y = match (op, p) {
                (CodecOp::Conv { .. } | CodecOp::ConvUp { .. }, Param::Conv(c)) => {
                    let (y, t_out) = conv(x, t, c, state.layers.get_mut(&i).unwrap());
                    t = t_out;
                    y
                }
                (CodecOp::RmsNorm { channels, eps, .. }, Param::Vec(g)) => rms_norm(x, *channels, g, *eps),
                (CodecOp::LayerNorm { channels, eps, .. }, Param::Norm { w, b }) => layer_norm(x, *channels, w, b, *eps),
                (CodecOp::Linear { cin, cout, gelu, .. }, Param::Linear { w, b }) => linear(x, *cin, *cout, w, b, *gelu),
                (CodecOp::Attention { channels, heads, context, theta, .. }, Param::Attention { qkv, out }) => {
                    let ring = split_ring(state.layers.get_mut(&i).unwrap(), *context, *channels);
                    ring_attention(x, *channels, *heads, *context, *theta, qkv, out, ring)
                }
                (CodecOp::ScaleAdd { .. } | CodecOp::Add { .. }, Param::Vec(g)) => {
                    let c = g.len();
                    let mut base = slots[&Slot::X].clone();
                    for (j, v) in base.iter_mut().enumerate() {
                        *v += g[j % c] * x[j];
                    }
                    base
                }
                _ => unreachable!(),
            };
            slots.insert(op.output(), y);
        }
        Ok(slots.remove(&Slot::X).unwrap())
    }
}

pub struct HostCodec {
    pub plan: CodecPlan,
    pub encoder: HostStack,
    pub decoder: HostStack,
}

impl HostCodec {
    pub fn load(plan: CodecPlan, weights: &dyn Weights) -> Result<HostCodec> {
        let encoder = HostStack { params: load_stack(&plan.encoder, weights)?, stack: plan.encoder.clone() };
        let decoder = HostStack { params: load_stack(&plan.decoder, weights)?, stack: plan.decoder.clone() };
        Ok(HostCodec { plan, encoder, decoder })
    }

    pub fn open(dir: &std::path::Path) -> Result<HostCodec> {
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let plan = super::graph::plan(&config)?;
        let ck = crate::weights::Checkpoint::open(dir)?;
        plan.check_tensors(&|n| ck.shape_of(n))?;
        HostCodec::load(plan, &ck)
    }

    pub fn encode(&self, audio: &[f32]) -> Result<Vec<f32>> {
        ensure!(self.plan.has_encoder(), "the {} codec graph has no encoder", self.plan.graph);
        self.encoder.run(audio, &mut self.encoder.new_state())
    }

    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        self.decoder.run(latents, &mut self.decoder.new_state())
    }

    pub fn silence(&self) -> Result<Vec<f32>> {
        let z = self.encode(&vec![0.0; 16 * self.plan.frame_samples])?;
        Ok(z[z.len() - self.plan.latent_dim..].to_vec())
    }
}
