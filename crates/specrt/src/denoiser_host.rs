use crate::denoiser::{DenoiserGraph, DenoiserSpec, Signer, weight_names};
use crate::plan::{DitOp, Op, Plan, Stream};
use crate::weights::Checkpoint;
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;

pub struct Inputs<'a> {
    pub latent: &'a [f32],
    pub text: &'a [f32],
    pub timestep: f32,
    pub ids: &'a [i32],
    pub image: usize,
    pub text_rows: usize,
    pub patch_h: usize,
    pub patch_w: usize,
    pub frames: usize,
}

pub struct HostDenoiser<'a> {
    pub graph: &'a DenoiserGraph,
    pub spec: &'a DenoiserSpec,
    pub plan: &'a Plan,
    weights: BTreeMap<String, Vec<f32>>,
    pub bf16: bool,
}

#[derive(Default)]
pub struct Trace {
    pub blocks: Vec<(String, Vec<f32>)>,
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

impl<'a> HostDenoiser<'a> {
    pub fn load(graph: &'a DenoiserGraph, spec: &'a DenoiserSpec, plan: &'a Plan, ck: &Checkpoint, bf16: bool) -> Result<Self> {
        let mut weights = BTreeMap::new();
        let mut names: Vec<String> = Vec::new();
        let mut collect = |ops: &[Op], blocks: Option<usize>| {
            for op in ops {
                let Op::Dit(d) = op else { continue };
                let ws: Vec<String> = match d {
                    DitOp::Linear { weight, bias, .. } => weight_names(weight, None).into_iter().zip(crate::denoiser::biased_parts(weight)).flat_map(|((n, _), biased)| {
                        let b = n.strip_suffix(".weight").map(|s| format!("{s}.bias"));
                        std::iter::once(n).chain(b.filter(|_| *bias && biased))
                    }).collect(),
                    DitOp::AxisRope { text, image, .. } => [text, image].iter().flat_map(|w| ["q", "k"].map(|q| format!("{w}_{q}.weight"))).collect(),
                    DitOp::TableAdd { table, .. } => vec![table.to_string()],
                    DitOp::LayerNorm { weight, .. } => vec![format!("{weight}.weight"), format!("{weight}.bias")],
                    DitOp::Pack { norms, .. } => crate::denoiser::pack_parts(norms).into_iter().filter(|n| !n.is_empty()).map(|n| format!("{n}.weight")).collect(),
                    _ => Vec::new(),
                };
                for w in ws {
                    match blocks {
                        Some(n) => names.extend((0..n).map(|i| w.replace("{i}", &i.to_string()))),
                        None => names.push(w),
                    }
                }
            }
        };
        collect(&plan.head, None);
        for (b, g) in plan.blocks.iter().zip(&graph.block) {
            collect(&b.ops, Some(spec.size(&g.count)?));
        }
        collect(&plan.tail, None);
        for n in names {
            let shape = ck.shape_of(&n)?;
            let mut v: Vec<f32> = ck.bf16(&n, &shape).map(|b| b.iter().map(|&x| crate::weights::bf16_to_f32(x)).collect())?;
            if !bf16 {
                v = ck.f32_values(&n, &shape)?;
            }
            weights.insert(n, v);
        }
        Ok(HostDenoiser { graph, spec, plan, weights, bf16 })
    }

    fn weight(&self, name: &str) -> Result<&[f32]> {
        self.weights.get(name).map(|v| v.as_slice()).with_context(|| format!("weight {name} is not loaded"))
    }

    fn block_weight(&self, name: &str, block: Option<usize>) -> Result<&[f32]> {
        self.weight(&name.replace("{i}", &block.unwrap_or(0).to_string()))
    }

    fn rows_in(&self, rows: Stream, x: &Inputs) -> usize {
        match rows {
            Stream::Text => x.text_rows,
            Stream::Image | Stream::Latent => x.image,
            Stream::Joint => x.text_rows + x.image,
            Stream::Seqs => 1,
        }
    }

    pub fn forward(&self, x: &Inputs, trace: &mut Trace) -> Result<Vec<f32>> {
        let inputs = [("latent", x.latent.to_vec()), ("text", x.text.to_vec()), ("timestep", vec![x.timestep])];
        self.run(x, &inputs, "velocity", trace)
    }

    pub fn run(&self, x: &Inputs, inputs: &[(&str, Vec<f32>)], output: &str, trace: &mut Trace) -> Result<Vec<f32>> {
        let mut bufs: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        let width = |name: &str| -> Result<usize> { self.spec.size(&self.graph.decl(name).with_context(|| format!("no buffer {name}"))?.cols) };
        let total = x.image + x.text_rows;
        for b in self.graph.buffer.iter().chain(&self.graph.output) {
            let rows = match b.rows.as_str() {
                "joint" => total,
                "image" | "latent" => x.image,
                "text" => x.text_rows,
                _ => 1,
            };
            bufs.insert(Box::leak(b.name.clone().into_boxed_str()), vec![0.0; rows * width(&b.name)?]);
        }
        for (name, v) in inputs {
            bufs.insert(crate::denoiser::intern_name(name), v.clone());
        }
        let mut planes = vec![0f32; total * 3 * self.spec.hidden()];        let run = |ops: &[Op], block: Option<usize>, bufs: &mut BTreeMap<&str, Vec<f32>>, planes: &mut Vec<f32>| -> Result<()> {
            for op in ops {
                let Op::Dit(d) = op else { bail!("{} is not a denoiser op", op.label()) };
                self.op(d, block, x, bufs, planes)?;
            }
            Ok(())
        };
        run(&self.plan.head, None, &mut bufs, &mut planes)?;
        for (b, g) in self.plan.blocks.iter().zip(&self.graph.block) {
            for i in 0..self.spec.size(&g.count)? {
                run(&b.ops, Some(i), &mut bufs, &mut planes)?;
                trace.blocks.push((format!("{}{i}", g.name), bufs["x"].clone()));
            }
        }
        run(&self.plan.tail, None, &mut bufs, &mut planes)?;
        bufs.remove(output).with_context(|| format!("the plan wrote no {output}"))
    }

    fn patch_dims(&self, x: &Inputs) -> Result<(usize, usize, usize, usize)> {
        Ok((self.spec.size("grid_channels")?, self.spec.size("latent_patch")?, x.patch_h * x.frames.max(1), x.patch_w))
    }

    fn span(&self, rows: Stream, x: &Inputs) -> (usize, usize) {
        match rows {
            Stream::Text => (0, x.text_rows),
            Stream::Image => (x.text_rows, x.text_rows + x.image),
            Stream::Joint => (0, x.text_rows + x.image),
            Stream::Seqs => (0, 1),
            Stream::Latent => (0, x.image),
        }
    }

    fn local_span(&self, name: &str, rows: Stream, x: &Inputs) -> (usize, usize) {
        let decl = self.graph.decl(name).map(|d| d.rows.as_str()).unwrap_or("joint");
        if decl == "joint" { self.span(rows, x) } else { (0, match rows { Stream::Image | Stream::Latent => x.image, Stream::Text => x.text_rows, Stream::Joint => x.image + x.text_rows, Stream::Seqs => 1 }) }
    }

    fn op(&self, d: &DitOp, block: Option<usize>, x: &Inputs, bufs: &mut BTreeMap<&str, Vec<f32>>, planes: &mut Vec<f32>) -> Result<()> {
        let h = self.spec.hidden();
        let width = |name: &str| -> Result<usize> { self.spec.size(&self.graph.decl(name).with_context(|| format!("no buffer {name}"))?.cols) };
        let eps = self.spec.get("norm_eps")? as f32;
        let act_round = |v: f32| v;
        match *d {
            DitOp::Patchify { from, into } => {
                let (c, p, ph, pw) = self.patch_dims(x)?;
                let v = patchify(&bufs[from], c, p, ph, pw).into_iter().map(act_round).collect();
                bufs.insert(into, v);
            }
            DitOp::Unpatchify { from, into, channels_last } => {
                let (c, p, ph, pw) = self.patch_dims(x)?;
                let v = if channels_last { unpatchify_last(&bufs[from], c, p, ph, pw) } else { unpatchify(&bufs[from], c, p, ph, pw) };
                bufs.insert(into, v);
            }
            DitOp::Split { from, into, rows, act } => {
                let w = width(from)?;
                let (a, b) = self.local_span(from, rows, x);
                let (c, _) = self.local_span(into, rows, x);
                let src = bufs[from][a * w..b * w].to_vec();
                let dst = bufs.get_mut(into).unwrap();
                for (i, v) in src.iter().enumerate() {
                    dst[c * w + i] = act_round(if act { silu(*v) } else { *v });
                }
            }
            DitOp::Copy { from, into, rows } => {
                let w = width(from)?;
                let (a, b) = self.local_span(from, rows, x);
                let (c, _) = self.local_span(into, rows, x);
                let src = bufs[from][a * w..b * w].to_vec();
                bufs.get_mut(into).unwrap()[c * w..c * w + src.len()].copy_from_slice(&src);
            }
            DitOp::TimestepEmbed { into, .. } => {
                let freq = width(into)?;
                let t = x.timestep * self.spec.get("time_scale")? as f32;
                bufs.insert(into, timestep_embedding(t, freq).into_iter().map(act_round).collect());
            }
            DitOp::Linear { weight, input, output, rows, add, bias, act } => {
                let signer = Signer { graph: self.graph, spec: self.spec };
                let (n, k) = signer.linear_shape(weight, input, output)?;
                let parts = weight_names(weight, block);
                let mut w = Vec::with_capacity(n * k);
                let mut bv: Vec<f32> = Vec::new();
                for ((name, range), biased) in parts.iter().zip(crate::denoiser::biased_parts(weight)) {
                    let full = self.weight(name)?;
                    let part_rows = match range {
                        Some(_) => n,
                        None => full.len() / k,
                    };
                    if bias && biased {
                        bv.extend_from_slice(self.weight(&format!("{}.bias", name.strip_suffix(".weight").unwrap_or(name)))?);
                    } else if bias {
                        bv.extend(std::iter::repeat_n(0.0, part_rows));
                    }
                    match range {
                        Some((a, b)) => {
                            let (a, b) = (self.spec.size(a)?, self.spec.size(b)?);
                            let stride = full.len() / n;
                            for r in 0..n {
                                w.extend_from_slice(&full[r * stride + a..r * stride + b]);
                            }
                        }
                        None => w.extend_from_slice(full),
                    }
                }
                ensure!(w.len() == n * k, "{weight}: {} values, expected {n} x {k}", w.len());
                ensure!(!bias || bv.len() == n, "{weight}: {} bias values, expected {n}", bv.len());
                let iw = width(input)?;
                let ow = width(output)?;
                let (ia, ib) = self.local_span(input, rows, x);
                let (oa, _) = self.local_span(output, rows, x);
                let src = bufs[input].clone();
                let dst = bufs.get_mut(output).unwrap();
                for r in 0..ib - ia {
                    for o in 0..n {
                        let mut acc = 0f32;
                        for c in 0..k {
                            acc += src[(ia + r) * iw + c] * w[o * k + c];
                        }
                        if bias {
                            acc += bv[o];
                        }
                        acc = activate(acc, act);
                        let at = (oa + r) * ow + o;
                        dst[at] = if add { dst[at] + acc } else { acc };
                    }
                }
            }
            DitOp::Modulate { rms: true, .. } | DitOp::Map { .. } => bail!("{} runs in the flow head runner (flow_head), not the DiT host", d.label()),
            DitOp::Modulate { from, into, rows, set, shift, scale, .. } => {
                let m = bufs[set].clone();
                let (a, b) = self.local_span(from, rows, x);
                let src = bufs[from].clone();
                let dst = bufs.get_mut(into).unwrap();
                for r in a..b {
                    let row = &src[r * h..(r + 1) * h];
                    let mean = row.iter().sum::<f32>() / h as f32;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
                    let inv = 1.0 / (var + eps).sqrt();
                    for c in 0..h {
                        dst[r * h + c] = act_round((row[c] - mean) * inv * (1.0 + m[scale as usize * h + c]) + m[shift as usize * h + c]);
                    }
                }
            }
            DitOp::LayerNorm { from, into, rows, weight } => {
                let g = self.block_weight(&format!("{weight}.weight"), block)?.to_vec();
                let be = self.block_weight(&format!("{weight}.bias"), block)?.to_vec();
                let (a, b) = self.local_span(from, rows, x);
                let src = bufs[from].clone();
                let dst = bufs.get_mut(into).unwrap();
                for r in a..b {
                    let row = &src[r * h..(r + 1) * h];
                    let mean = row.iter().sum::<f32>() / h as f32;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
                    let inv = 1.0 / (var + eps).sqrt();
                    for c in 0..h {
                        dst[r * h + c] = (row[c] - mean) * inv * g[c] + be[c];
                    }
                }
            }
            DitOp::TableAdd { table, from, into } => {
                let t = self.block_weight(table, block)?.to_vec();
                let src = bufs[from].clone();
                let dst = bufs.get_mut(into).unwrap();
                ensure!(t.len() == dst.len() && dst.len().is_multiple_of(src.len()), "{table}: {} values for {} outputs from {} inputs", t.len(), dst.len(), src.len());
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = t[j] + src[j % src.len()];
                }
            }
            DitOp::Pack { from, into, rows, norms, rope, .. } => {
                let parts = crate::denoiser::pack_parts(norms);
                let n = self.rows_in(rows, x);
                let fw = width(from)?;
                let (a, _) = self.local_span(from, rows, x);
                let hd = self.spec.size("head_dim")?;
                let (cos, sin) = if rope { rope_tables(&self.spec.axes, x.ids, self.spec.get("rope_theta")?) } else { (Vec::new(), Vec::new()) };
                let src = bufs[from].clone();
                let mut out = vec![0f32; parts.len() * n * h];
                for (pi, norm) in parts.iter().enumerate() {
                    let w = if norm.is_empty() { None } else { Some(self.block_weight(&format!("{norm}.weight"), block)?) };
                    for t in 0..n {
                        let s = &src[(a + t) * fw + pi * h..(a + t) * fw + (pi + 1) * h];
                        let o = &mut out[(pi * n + t) * h..(pi * n + t + 1) * h];
                        let inv = 1.0 / (s.iter().map(|v| v * v).sum::<f32>() / h as f32 + eps).sqrt();
                        let normed: Vec<f32> = (0..h).map(|i| w.map_or(s[i], |w| s[i] * inv * w[i])).collect();
                        for i in 0..h {
                            let d = i % hd;
                            o[i] = if rope && pi < 2 {
                                let other = normed[i ^ 1];
                                let (c, sn) = (cos[t * hd + d], sin[t * hd + d]);
                                if i & 1 == 1 { normed[i] * c + other * sn } else { normed[i] * c - other * sn }
                            } else {
                                normed[i]
                            };
                        }
                    }
                }
                if self.graph.decl(into).is_some() {
                    bufs.insert(into, out);
                } else {
                    planes[..out.len()].copy_from_slice(&out);
                }
            }
            DitOp::CrossAttention { into, rows, kv } => {
                let hd = self.spec.size("head_dim")?;
                let n = self.rows_in(rows, x);
                let m = x.text_rows;
                let scale = 1.0 / (hd as f32).sqrt();
                let kvb = bufs[kv].clone();
                let (oa, _) = self.local_span(into, rows, x);
                let dst = bufs.get_mut(into).unwrap();
                for hh in 0..h / hd {
                    for q in 0..n {
                        let qv = &planes[q * h + hh * hd..q * h + (hh + 1) * hd];
                        let scores: Vec<f32> = (0..m).map(|k| (0..hd).map(|i| qv[i] * kvb[k * h + hh * hd + i]).sum::<f32>() * scale).collect();
                        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                        let z: f32 = e.iter().sum();
                        for i in 0..hd {
                            let v: f32 = (0..m).map(|k| e[k] * kvb[(m + k) * h + hh * hd + i]).sum();
                            dst[(oa + q) * h + hh * hd + i] = v / z;
                        }
                    }
                }
            }
            DitOp::GatedAdd { from, into, rows, set, gate } => {
                let m = bufs[set].clone();
                let (a, b) = self.local_span(into, rows, x);
                let src = bufs[from].clone();
                let dst = bufs.get_mut(into).unwrap();
                for r in a..b {
                    for c in 0..h {
                        dst[r * h + c] += m[gate as usize * h + c] * src[r * h + c];
                    }
                }
            }
            DitOp::Glu { from, into, rows, offset } => {
                let fw = width(from)?;
                let mw = width(into)?;
                let off = self.spec.size(offset)?;
                let (a, b) = self.local_span(from, rows, x);
                let src = bufs[from].clone();
                let dst = bufs.get_mut(into).unwrap();
                for r in a..b {
                    for c in 0..mw {
                        dst[r * mw + c] = act_round(silu(src[r * fw + off + c]) * src[r * fw + off + mw + c]);
                    }
                }
            }
            DitOp::AxisRope { from, text, image } => {
                let fw = width(from)?;
                let hd = self.spec.size("head_dim")?;
                let heads = h / hd;
                let theta = self.spec.get("rope_theta")?;
                let (cos, sin) = rope_tables(&self.spec.axes, x.ids, theta);
                let src = &bufs[from];
                let tw = [self.weight(&format!("{}_q.weight", text.replace("{i}", &block.unwrap_or(0).to_string())))?, self.weight(&format!("{}_k.weight", text.replace("{i}", &block.unwrap_or(0).to_string())))?];
                let iw = [self.weight(&format!("{}_q.weight", image.replace("{i}", &block.unwrap_or(0).to_string())))?, self.weight(&format!("{}_k.weight", image.replace("{i}", &block.unwrap_or(0).to_string())))?];
                let total = x.text_rows + x.image;
                for t in 0..total {
                    for part in 0..3 {
                        for hh in 0..heads {
                            let s = &src[t * fw + part * h + hh * hd..t * fw + part * h + (hh + 1) * hd];
                            let out = &mut planes[(part * total + t) * h + hh * hd..(part * total + t) * h + (hh + 1) * hd];
                            if part == 2 {
                                out.copy_from_slice(s);
                                continue;
                            }
                            let w = if t < x.text_rows { tw[part] } else { iw[part] };
                            let inv = 1.0 / (s.iter().map(|v| v * v).sum::<f32>() / hd as f32 + eps).sqrt();
                            let n: Vec<f32> = (0..hd).map(|i| s[i] * inv * w[i]).collect();
                            for i in 0..hd {
                                let other = n[i ^ 1];
                                let (c, sn) = (cos[t * hd + i], sin[t * hd + i]);
                                out[i] = if i & 1 == 1 { n[i] * c + other * sn } else { n[i] * c - other * sn };
                            }
                        }
                    }
                }
            }
            DitOp::Step { .. } | DitOp::MultiStep { .. } => bail!("the host denoiser runs the network; the scheduler step is sampler::Pipeline::step_host"),
            DitOp::JointAttention { into, rows } => {
                let hd = self.spec.size("head_dim")?;
                let total = self.rows_in(rows, x);
                let (oa, _) = self.local_span(into, rows, x);
                let scale = 1.0 / (hd as f32).sqrt();
                let dst = bufs.get_mut(into).unwrap();
                for hh in 0..h / hd {
                    for q in 0..total {
                        let qv = &planes[q * h + hh * hd..q * h + (hh + 1) * hd];
                        let scores: Vec<f32> = (0..total).map(|k| (0..hd).map(|i| qv[i] * planes[(total + k) * h + hh * hd + i]).sum::<f32>() * scale).collect();
                        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                        let z: f32 = e.iter().sum();
                        for i in 0..hd {
                            let v: f32 = (0..total).map(|k| e[k] * planes[(2 * total + k) * h + hh * hd + i]).sum();
                            dst[(oa + q) * h + hh * hd + i] = act_round(v / z);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn patchify(grid: &[f32], channels: usize, p: usize, ph: usize, pw: usize) -> Vec<f32> {
    let width = channels * p * p;
    let mut out = vec![0f32; ph * pw * width];
    for c in 0..channels {
        for y in 0..ph * p {
            for xx in 0..pw * p {
                let token = (y / p) * pw + xx / p;
                let col = c * p * p + (y % p) * p + xx % p;
                out[token * width + col] = grid[(c * ph * p + y) * pw * p + xx];
            }
        }
    }
    out
}

pub fn unpatchify(tokens: &[f32], channels: usize, p: usize, ph: usize, pw: usize) -> Vec<f32> {
    let width = channels * p * p;
    let mut out = vec![0f32; tokens.len()];
    for c in 0..channels {
        for y in 0..ph * p {
            for xx in 0..pw * p {
                let token = (y / p) * pw + xx / p;
                let col = c * p * p + (y % p) * p + xx % p;
                out[(c * ph * p + y) * pw * p + xx] = tokens[token * width + col];
            }
        }
    }
    out
}

pub fn unpatchify_last(tokens: &[f32], channels: usize, p: usize, ph: usize, pw: usize) -> Vec<f32> {
    let width = channels * p * p;
    let mut out = vec![0f32; tokens.len()];
    for c in 0..channels {
        for y in 0..ph * p {
            for xx in 0..pw * p {
                let token = (y / p) * pw + xx / p;
                let col = ((y % p) * p + xx % p) * channels + c;
                out[(c * ph * p + y) * pw * p + xx] = tokens[token * width + col];
            }
        }
    }
    out
}

fn activate(v: f32, act: u8) -> f32 {
    match act {
        1 => 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v * v * v)).tanh()),
        2 => crate::codec::host::gelu(v),
        _ => v,
    }
}

pub fn video_ids(frames: usize, side_h: usize, side_w: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(frames * side_h * side_w * 3);
    for f in 0..frames {
        for y in 0..side_h {
            for xx in 0..side_w {
                ids.extend([f as i32, y as i32, xx as i32]);
            }
        }
    }
    ids
}

pub fn timestep_embedding(t: f32, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let mut out = vec![0f32; channels];
    for i in 0..half {
        let f = (-(10000f32).ln() * i as f32 / half as f32).exp();
        let a = t * f;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

pub fn rope_tables(axes: &[usize], ids: &[i32], theta: f64) -> (Vec<f32>, Vec<f32>) {
    let hd: usize = axes.iter().sum();
    let tokens = ids.len() / axes.len();
    let mut cos = vec![0f32; tokens * hd];
    let mut sin = vec![0f32; tokens * hd];
    for t in 0..tokens {
        let mut at = 0;
        for (a, &dim) in axes.iter().enumerate() {
            let pos = ids[t * axes.len() + a] as f64;
            for j in 0..dim / 2 {
                let freq = 1.0 / theta.powf((2 * j) as f64 / dim as f64);
                let angle = pos * freq;
                for e in 0..2 {
                    cos[t * hd + at + 2 * j + e] = angle.cos() as f32;
                    sin[t * hd + at + 2 * j + e] = angle.sin() as f32;
                }
            }
            at += dim;
        }
    }
    (cos, sin)
}

pub fn grid_ids(text: usize, side_h: usize, side_w: usize, axes: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity((text + side_h * side_w) * axes);
    for l in 0..text {
        let mut row = vec![0i32; axes];
        row[axes - 1] = l as i32;
        ids.extend(row);
    }
    for y in 0..side_h {
        for xx in 0..side_w {
            let mut row = vec![0i32; axes];
            row[1] = y as i32;
            row[2] = xx as i32;
            ids.extend(row);
        }
    }
    ids
}
