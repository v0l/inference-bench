use super::graph::{Act, Slot, VaeOp, VaePlan};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Grid {
    pub fn zeros(c: usize, t: usize, h: usize, w: usize) -> Grid {
        Grid { c, t, h, w, data: vec![0.0; c * t * h * w] }
    }

    pub fn pixels(&self) -> usize {
        self.t * self.h * self.w
    }

    pub fn from_channels_first(x: &[f32], c: usize, t: usize, h: usize, w: usize) -> Grid {
        let mut g = Grid::zeros(c, t, h, w);
        let plane = t * h * w;
        for ch in 0..c {
            for p in 0..plane {
                g.data[p * c + ch] = x[ch * plane + p];
            }
        }
        g
    }

    pub fn to_channels_first(&self) -> Vec<f32> {
        let plane = self.pixels();
        let mut out = vec![0.0; self.data.len()];
        for p in 0..plane {
            for ch in 0..self.c {
                out[ch * plane + p] = self.data[p * self.c + ch];
            }
        }
        out
    }

    pub fn frames(&self, from: usize, count: usize) -> Grid {
        let per = self.h * self.w * self.c;
        Grid { c: self.c, t: count, h: self.h, w: self.w, data: self.data[from * per..(from + count) * per].to_vec() }
    }

    pub fn append_frames(&mut self, other: &Grid) {
        assert_eq!((self.c, self.h, self.w), (other.c, other.h, other.w));
        self.data.extend_from_slice(&other.data);
        self.t += other.t;
    }

    pub fn crop(&self, y0: usize, x0: usize, h: usize, w: usize) -> Grid {
        let mut g = Grid::zeros(self.c, self.t, h, w);
        for t in 0..self.t {
            for y in 0..h {
                let src = ((t * self.h + y0 + y) * self.w + x0) * self.c;
                let dst = (t * h + y) * w * self.c;
                g.data[dst..dst + w * self.c].copy_from_slice(&self.data[src..src + w * self.c]);
            }
        }
        g
    }
}

pub fn uniform(seed: u64, n: usize, scale: f64) -> Vec<f32> {
    (1..=n as u64)
        .map(|i| {
            let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            ((2.0 * u - 1.0) * scale) as f32
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvWeights {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub kt: usize,
}

impl ConvWeights {
    pub fn tap(&self, o: usize, i: usize, dt: usize, dy: usize, dx: usize) -> f32 {
        self.w[(((o * self.cin + i) * self.kt + dt) * self.k + dy) * self.k + dx]
    }

    pub fn taps_major(&self) -> Vec<f32> {
        let (k, kt, cin) = (self.k, self.kt, self.cin);
        let mut out = vec![0.0; self.w.len()];
        for o in 0..self.cout {
            for dt in 0..kt {
                for dy in 0..k {
                    for dx in 0..k {
                        for i in 0..cin {
                            out[(((o * kt + dt) * k + dy) * k + dx) * cin + i] = self.tap(o, i, dt, dy, dx);
                        }
                    }
                }
            }
        }
        out
    }
}

pub fn conv(x: &Grid, c: &ConvWeights, history: Option<&Grid>) -> Grid {
    assert_eq!(x.c, c.cin, "conv input channels");
    let pad = c.k / 2;
    let lead = c.kt - 1;
    let frame = |t: isize| -> Option<(&Grid, usize)> {
        if t >= 0 {
            return Some((x, t as usize));
        }
        let h = history?;
        let back = (-t) as usize;
        (back <= h.t).then(|| (h, h.t - back))
    };
    let mut out = Grid::zeros(c.cout, x.t, x.h, x.w);
    let taps = c.taps_major();
    let mut acc = vec![0.0f64; c.cout];
    for t in 0..x.t {
        for y in 0..x.h {
            for xx in 0..x.w {
                acc.iter_mut().enumerate().for_each(|(o, a)| *a = c.b[o] as f64);
                for dt in 0..c.kt {
                    let Some((src, st)) = frame(t as isize + dt as isize - lead as isize) else { continue };
                    for dy in 0..c.k {
                        let sy = y as isize + dy as isize - pad as isize;
                        if sy < 0 || sy >= x.h as isize {
                            continue;
                        }
                        for dx in 0..c.k {
                            let sx = xx as isize + dx as isize - pad as isize;
                            if sx < 0 || sx >= x.w as isize {
                                continue;
                            }
                            let at = ((st * src.h + sy as usize) * src.w + sx as usize) * src.c;
                            let input = &src.data[at..at + c.cin];
                            let tap = (dt * c.k + dy) * c.k + dx;
                            for (o, a) in acc.iter_mut().enumerate() {
                                let w = &taps[(o * c.kt * c.k * c.k + tap) * c.cin..][..c.cin];
                                *a += input.iter().zip(w).map(|(p, q)| (*p as f64) * (*q as f64)).sum::<f64>();
                            }
                        }
                    }
                }
                let at = ((t * x.h + y) * x.w + xx) * c.cout;
                for (o, a) in acc.iter().enumerate() {
                    out.data[at + o] = *a as f32;
                }
            }
        }
    }
    out
}

pub fn upsample(x: &Grid) -> Grid {
    let mut out = Grid::zeros(x.c, x.t, 2 * x.h, 2 * x.w);
    for t in 0..x.t {
        for y in 0..out.h {
            for xx in 0..out.w {
                let src = ((t * x.h + y / 2) * x.w + xx / 2) * x.c;
                let dst = ((t * out.h + y) * out.w + xx) * x.c;
                out.data[dst..dst + x.c].copy_from_slice(&x.data[src..src + x.c]);
            }
        }
    }
    out
}

pub fn interleave(y: &Grid) -> Grid {
    let half = y.c / 2;
    let mut out = Grid::zeros(half, 2 * y.t, y.h, y.w);
    let plane = y.h * y.w;
    for t in 0..y.t {
        for p in 0..plane {
            let src = (t * plane + p) * y.c;
            for s in 0..2 {
                let dst = ((2 * t + s) * plane + p) * half;
                out.data[dst..dst + half].copy_from_slice(&y.data[src + s * half..src + (s + 1) * half]);
            }
        }
    }
    out
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn act(a: Act, v: f32) -> f32 {
    match a {
        Act::None => v,
        Act::Silu => silu(v),
    }
}

pub fn group_stats(x: &Grid, groups: usize) -> Vec<(f64, f64)> {
    let per = x.c / groups;
    let mut sums = vec![(0.0f64, 0.0f64); groups];
    for p in 0..x.pixels() {
        for ch in 0..x.c {
            let v = x.data[p * x.c + ch] as f64;
            let s = &mut sums[ch / per];
            s.0 += v;
            s.1 += v * v;
        }
    }
    sums
}

pub fn group_apply(x: &mut Grid, sums: &[(f64, f64)], gamma: &[f32], beta: &[f32], eps: f32, a: Act) {
    let groups = sums.len();
    let per = x.c / groups;
    let n = (x.pixels() * per) as f64;
    let moments: Vec<(f32, f32)> = sums
        .iter()
        .map(|(s, q)| {
            let mean = s / n;
            let var = (q / n - mean * mean).max(0.0);
            (mean as f32, (1.0 / (var + eps as f64).sqrt()) as f32)
        })
        .collect();
    for p in 0..x.pixels() {
        for ch in 0..x.c {
            let (mean, inv) = moments[ch / per];
            let v = &mut x.data[p * x.c + ch];
            *v = act(a, (*v - mean) * inv * gamma[ch] + beta[ch]);
        }
    }
}

pub fn rms_norm(x: &mut Grid, gamma: &[f32], a: Act) {
    let scale = (x.c as f32).sqrt();
    for p in 0..x.pixels() {
        let row = &mut x.data[p * x.c..(p + 1) * x.c];
        let norm = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt().max(1e-12) as f32;
        for (ch, v) in row.iter_mut().enumerate() {
            *v = act(a, *v / norm * scale * gamma[ch]);
        }
    }
}

pub fn attention(qkv: &Grid, c: usize) -> Grid {
    let plane = qkv.h * qkv.w;
    let scale = 1.0 / (c as f64).sqrt();
    let mut out = Grid::zeros(c, qkv.t, qkv.h, qkv.w);
    let mut scores = vec![0.0f64; plane];
    for t in 0..qkv.t {
        let row = |p: usize| &qkv.data[(t * plane + p) * 3 * c..(t * plane + p + 1) * 3 * c];
        for i in 0..plane {
            let q = &row(i)[..c];
            let mut max = f64::MIN;
            for (j, s) in scores.iter_mut().enumerate() {
                let k = &row(j)[c..2 * c];
                *s = q.iter().zip(k).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() * scale;
                max = max.max(*s);
            }
            let mut total = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                total += *s;
            }
            let dst = &mut out.data[(t * plane + i) * c..(t * plane + i + 1) * c];
            for (j, s) in scores.iter().enumerate() {
                let v = &row(j)[2 * c..];
                for (d, vv) in dst.iter_mut().zip(v) {
                    *d += (s / total * *vv as f64) as f32;
                }
            }
        }
    }
    out
}

pub trait Weights {
    fn floats(&self, name: &str) -> Result<Vec<f32>>;
}

impl Weights for crate::weights::Checkpoint {
    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let v = self.view(name)?;
        Ok(match v.dtype {
            safetensors::Dtype::F32 => v.data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            safetensors::Dtype::BF16 => v.data.as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect(),
            safetensors::Dtype::F16 => v.data.as_chunks::<2>().0.iter().map(|b| half_to_f32(u16::from_le_bytes(*b))).collect(),
            d => bail!("{name} is {d:?}, the vae reads f32, bf16 and f16"),
        })
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            let mut e = -14;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | (((exp - 15 + 127) as u32) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[derive(Clone, Debug)]
pub enum Param {
    Conv(ConvWeights),
    Norm { gamma: Vec<f32>, beta: Vec<f32> },
    None,
}

pub struct HostVae {
    pub plan: VaePlan,
    params: Vec<Param>,
}

#[derive(Clone, Debug, Default)]
pub struct Stream {
    history: BTreeMap<usize, Grid>,
    started: BTreeMap<usize, bool>,
}

impl HostVae {
    pub fn load(plan: VaePlan, weights: &dyn Weights) -> Result<HostVae> {
        let mut params = Vec::with_capacity(plan.ops.len());
        for op in &plan.ops {
            params.push(match op {
                VaeOp::Conv { weights: names, cin, cout, k, kt, .. } => {
                    let (mut w, mut b) = (Vec::new(), Vec::new());
                    for n in names {
                        w.extend(weights.floats(&format!("{n}.weight"))?);
                        b.extend(weights.floats(&format!("{n}.bias"))?);
                    }
                    ensure!(w.len() == cout * cin * k * k * kt && b.len() == *cout, "{} has {} weights and {} biases", op.label(), w.len(), b.len());
                    Param::Conv(ConvWeights { w, b, cin: *cin, cout: *cout, k: *k, kt: *kt })
                }
                VaeOp::TimeUp { weight, channels, .. } => Param::Conv(ConvWeights { w: weights.floats(&format!("{weight}.weight"))?, b: weights.floats(&format!("{weight}.bias"))?, cin: *channels, cout: 2 * channels, k: 1, kt: 3 }),
                VaeOp::GroupNorm { weight, .. } => Param::Norm { gamma: weights.floats(&format!("{weight}.weight"))?, beta: weights.floats(&format!("{weight}.bias"))? },
                VaeOp::RmsNorm { weight, .. } => Param::Norm { gamma: weights.floats(&format!("{weight}.gamma"))?, beta: Vec::new() },
                _ => Param::None,
            });
        }
        Ok(HostVae { plan, params })
    }

    pub fn open(vae_dir: &std::path::Path) -> Result<HostVae> {
        let plan = super::graph::load_plan(vae_dir)?;
        let ck = crate::weights::Checkpoint::open(vae_dir)?;
        plan.check_tensors(&|n| ck.shape_of(n))?;
        HostVae::load(plan, &ck)
    }

    pub fn conv_weights(&self, op: usize) -> Option<&ConvWeights> {
        match &self.params[op] {
            Param::Conv(c) => Some(c),
            _ => None,
        }
    }

    pub fn norm_weights(&self, op: usize) -> Option<(&[f32], &[f32])> {
        match &self.params[op] {
            Param::Norm { gamma, beta } => Some((gamma, beta)),
            _ => None,
        }
    }

    pub fn conv_step(&self, op: usize, x: &Grid, c: &ConvWeights, stream: &mut Stream) -> Grid {
        if c.kt == 1 {
            return conv(x, c, None);
        }
        let prev = stream.history.get(&op);
        let y = conv(x, c, prev);
        let tail = causal_tail(prev, x, c.kt - 1);
        stream.history.insert(op, tail);
        y
    }

    pub fn time_up_step(&self, op: usize, x: &Grid, stream: &mut Stream) -> Grid {
        let Param::Conv(c) = &self.params[op] else { unreachable!() };
        time_up(x, c, stream, op)
    }

    pub fn run(&self, latent: &Grid, stream: &mut Stream) -> Result<Grid> {
        ensure!(latent.c == self.plan.latent_channels, "the latent has {} channels, the vae reads {}", latent.c, self.plan.latent_channels);
        let mut slots: BTreeMap<Slot, Grid> = BTreeMap::new();
        slots.insert(Slot::X, latent.clone());
        let take = |slots: &BTreeMap<Slot, Grid>, s: Slot| slots.get(&s).cloned().with_context(|| format!("slot {} is read before it is written", s.name()));
        for (i, op) in self.plan.ops.iter().enumerate() {
            let out = match op {
                VaeOp::Conv { input, .. } => {
                    let x = take(&slots, *input)?;
                    let Param::Conv(c) = &self.params[i] else { unreachable!() };
                    self.conv_step(i, &x, c, stream)
                }
                VaeOp::GroupNorm { groups, eps, act: a, input, .. } => {
                    let mut x = take(&slots, *input)?;
                    let Param::Norm { gamma, beta } = &self.params[i] else { unreachable!() };
                    let sums = group_stats(&x, *groups);
                    group_apply(&mut x, &sums, gamma, beta, *eps, *a);
                    x
                }
                VaeOp::RmsNorm { act: a, input, .. } => {
                    let mut x = take(&slots, *input)?;
                    let Param::Norm { gamma, .. } = &self.params[i] else { unreachable!() };
                    rms_norm(&mut x, gamma, *a);
                    x
                }
                VaeOp::Upsample { slot } => upsample(&take(&slots, *slot)?),
                VaeOp::TimeUp { slot, .. } => {
                    let x = take(&slots, *slot)?;
                    let Param::Conv(c) = &self.params[i] else { unreachable!() };
                    time_up(&x, c, stream, i)
                }
                VaeOp::Attention { channels, input, .. } => attention(&take(&slots, *input)?, *channels),
                VaeOp::Add { input, with, .. } => {
                    let mut a = take(&slots, *input)?;
                    let b = take(&slots, *with)?;
                    ensure!(a.data.len() == b.data.len(), "add of {} and {} values", a.data.len(), b.data.len());
                    a.data.iter_mut().zip(&b.data).for_each(|(x, y)| *x += y);
                    a
                }
                VaeOp::Clamp { lo, hi, slot } => {
                    let mut x = take(&slots, *slot)?;
                    x.data.iter_mut().for_each(|v| *v = v.clamp(*lo, *hi));
                    x
                }
            };
            slots.insert(op.writes(), out);
        }
        take(&slots, Slot::X)
    }

    pub fn decode(&self, latent: &Grid) -> Result<Grid> {
        self.decode_with(latent, &mut |x, s| self.run(x, s))
    }

    pub fn decode_with(&self, latent: &Grid, run: &mut dyn FnMut(&Grid, &mut Stream) -> Result<Grid>) -> Result<Grid> {
        if !self.plan.temporal {
            return run(latent, &mut Stream::default());
        }
        let mut stream = Stream::default();
        let mut out: Option<Grid> = None;
        for f in 0..latent.t {
            let y = run(&latent.frames(f, 1), &mut stream)?;
            match &mut out {
                Some(o) => o.append_frames(&y),
                None => out = Some(y),
            }
        }
        out.context("an empty latent")
    }
}

fn causal_tail(prev: Option<&Grid>, x: &Grid, keep: usize) -> Grid {
    let mut all = prev.cloned().unwrap_or_else(|| Grid::zeros(x.c, 0, x.h, x.w));
    all.append_frames(x);
    let from = all.t.saturating_sub(keep);
    all.frames(from, all.t - from)
}

fn time_up(x: &Grid, c: &ConvWeights, stream: &mut Stream, op: usize) -> Grid {
    if !stream.started.get(&op).copied().unwrap_or(false) {
        stream.started.insert(op, true);
        return x.clone();
    }
    let prev = stream.history.get(&op);
    let y = conv(x, c, prev);
    stream.history.insert(op, causal_tail(prev, x, 2));
    interleave(&y)
}
