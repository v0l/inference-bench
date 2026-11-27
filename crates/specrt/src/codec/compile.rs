use super::graph::{CodecOp, Slot, Stack};
use anyhow::{Result, bail, ensure};
use kir::conv::{ConvKernel, ConvShape, Epilogue, Prologue};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Conv { shape: ConvShape, gelu: bool, residual: bool, rows: bool, act: Option<&'static str> },
    Depthwise { channels: usize, taps: usize },
    DwUp { channels: usize, stride: usize },
    Rms { channels: usize, eps: f32 },
    Layer { channels: usize, eps: f32 },
    Shift { channels: usize, context: usize },
    Rope { heads: usize, head_dim: usize, context: usize, theta: f32 },
    Ring { heads: usize, head_dim: usize, context: usize },
    RingStore { channels: usize, context: usize },
    Advance { context: usize, channels: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    Main,
    Out,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub op: usize,
    pub part: Part,
    pub scale: Option<usize>,
    pub kind: Kind,
    pub input: usize,
    pub output: usize,
    pub layer: Option<usize>,
    pub t_in: usize,
    pub t_out: usize,
    pub t_pad: usize,
    pub cin: usize,
    pub cout: usize,
}

impl Launch {
    pub fn label(&self, stack: &Stack) -> String {
        let op = stack.ops[self.op].label();
        match &self.kind {
            Kind::Shift { .. } => format!("state {op}"),
            Kind::Rope { .. } => format!("rope {op}"),
            Kind::Ring { .. } => format!("ring {op}"),
            Kind::RingStore { .. } => format!("ring store {op}"),
            Kind::Advance { .. } => format!("advance {op}"),
            Kind::Conv { .. } if matches!(stack.ops[self.op], CodecOp::Attention { .. }) => {
                let part = if self.part == Part::Main { "qkv" } else { "out" };
                match self.scale {
                    Some(s) => format!("{part} {op} + {}", stack.ops[s].label()),
                    None => format!("{part} {op}"),
                }
            }
            Kind::Conv { gelu: true, .. } => op,
            _ => match self.scale {
                Some(s) => format!("{op} + {}", stack.ops[s].label()),
                None => op,
            },
        }
    }

    pub fn weight_floats(&self) -> usize {
        match &self.kind {
            Kind::Conv { shape, .. } => shape.cout * shape.cin * shape.kt + shape.cout,
            Kind::Depthwise { channels, taps } => channels * taps + channels,
            Kind::DwUp { channels, stride } => 3 * channels * stride,
            Kind::Rms { channels, .. } => *channels,
            Kind::Layer { channels, .. } => 2 * channels,
            _ => 0,
        }
    }

    pub fn bytes(&self, sessions: usize) -> (u64, u64) {
        let f = |n: usize| 4 * n as u64;
        let b = sessions;
        let t = self.t_in;
        match &self.kind {
            Kind::Conv { shape, residual, .. } => {
                let state = b * shape.context() * shape.cin;
                let out = b * self.t_out * shape.cout;
                (f(self.weight_floats() + b * t * shape.cin + state + if *residual { out } else { 0 }), f(out))
            }
            Kind::Depthwise { channels, taps } => {
                let out = b * self.t_out * channels;
                (f(self.weight_floats() + b * t * channels + b * (taps - 1) * channels + out), f(out))
            }
            Kind::DwUp { channels, stride } => (f(self.weight_floats() + b * (t + 1) * channels), f(b * t * stride * channels)),
            Kind::Rms { channels, .. } | Kind::Layer { channels, .. } => (f(self.weight_floats() + b * t * channels), f(b * t * channels)),
            Kind::Shift { channels, context } => (f(b * context * channels), f(b * context * channels)),
            Kind::Rope { heads, head_dim, .. } => (f(b * t * 2 * heads * head_dim), f(b * t * 2 * heads * head_dim)),
            Kind::Ring { heads, head_dim, context } => {
                let c = heads * head_dim;
                (f(b * t * c + b * (context + t).min(2 * context) * 2 * c), f(b * t * c))
            }
            Kind::RingStore { channels, context } => (f(b * t.min(*context) * 2 * channels), f(b * t.min(*context) * 2 * channels)),
            Kind::Advance { .. } => (f(b), f(b)),
        }
    }

    pub fn flops(&self, sessions: usize) -> u64 {
        let rows = (sessions * self.t_out) as u64;
        match &self.kind {
            Kind::Conv { shape, .. } => rows * 2 * (shape.cout * shape.cin * shape.kt) as u64,
            Kind::Depthwise { channels, taps } => rows * 2 * (channels * taps) as u64,
            Kind::DwUp { channels, stride } => rows * 4 * (channels * stride) as u64,
            Kind::Rms { channels, .. } => rows * 4 * *channels as u64,
            Kind::Layer { channels, .. } => rows * 6 * *channels as u64,
            Kind::Rope { heads, head_dim, .. } => rows * 6 * (heads * head_dim) as u64,
            Kind::Ring { heads, head_dim, context } => rows * 4 * (heads * head_dim * context) as u64,
            Kind::Shift { .. } | Kind::RingStore { .. } | Kind::Advance { .. } => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchPlan {
    pub launches: Vec<Launch>,
    pub buffers: usize,
    pub buffer_floats: Vec<usize>,
    pub input: usize,
    pub output: usize,
    pub t_out: usize,
    pub out_channels: usize,
    pub states: BTreeMap<usize, usize>,
}

impl LaunchPlan {
    pub fn kernels(&self, target: &kir::Target) -> Result<Vec<ConvKernel>> {
        let mut out = Vec::new();
        for l in &self.launches {
            if let Kind::Conv { shape, gelu, residual, act, .. } = &l.kind {
                out.push(conv_kernel(target, *shape, *gelu, *residual, false, *act)?);
                if kir::conv::rows_fit(shape, 1) {
                    out.push(conv_kernel(target, *shape, *gelu, *residual, true, *act)?);
                }
            }
            if let Kind::Ring { head_dim, context, .. } = &l.kind {
                ensure!(kir::conv::ring_fits(target, *context, *head_dim), "a ring of {context} keys of {head_dim} needs {} bytes of shared memory, the target has {}", kir::conv::ring_attention_smem(*context, *head_dim), target.smem_limit);
            }
        }
        out.sort_by_key(ConvKernel::name);
        out.dedup_by_key(|k| k.name());
        Ok(out)
    }

    pub fn state_floats(&self) -> usize {
        self.states.values().sum()
    }
}

pub fn conv_kernel(target: &kir::Target, shape: ConvShape, gelu: bool, residual: bool, rows: bool, act: Option<&'static str>) -> Result<ConvKernel> {
    let epilogue = Epilogue { residual, stats: None, gelu };
    let prologue = act.map_or(Prologue::None, Prologue::Act);
    if rows {
        return ConvKernel::rows_with(target, shape, prologue, epilogue).map_err(anyhow::Error::msg);
    }
    ConvKernel::new(target, shape, prologue, epilogue).map_err(anyhow::Error::msg)
}

struct Buffers {
    slot: BTreeMap<Slot, usize>,
    floats: Vec<usize>,
}

impl Buffers {
    fn fresh(&mut self, keep: &[usize]) -> usize {
        let live: Vec<usize> = self.slot.values().copied().chain(keep.iter().copied()).collect();
        match (0..self.floats.len()).find(|b| !live.contains(b)) {
            Some(b) => b,
            None => {
                self.floats.push(0);
                self.floats.len() - 1
            }
        }
    }

    fn grow(&mut self, b: usize, floats: usize) {
        self.floats[b] = self.floats[b].max(floats);
    }

    fn target(&mut self, op: &CodecOp, input: usize, fused: bool, floats: usize) -> usize {
        let out = if fused {
            self.slot[&Slot::X]
        } else {
            if op.output() != op.input() {
                self.slot.remove(&op.output());
            }
            self.fresh(&[input])
        };
        self.grow(out, floats);
        out
    }
}

fn folds_into(op: &CodecOp, next: Option<&CodecOp>) -> bool {
    matches!(next, Some(CodecOp::ScaleAdd { input, .. } | CodecOp::Add { input, .. }) if *input == op.output() && op.output() != Slot::X)
}

pub fn plan(stack: &Stack, steps: usize, sessions: usize) -> Result<LaunchPlan> {
    ensure!(steps > 0 && sessions > 0, "a codec run needs steps and sessions");
    ensure!(!stack.empty(), "the codec graph has no {:?} stack", stack.direction);
    let mut bufs = Buffers { slot: BTreeMap::from([(Slot::X, 0)]), floats: vec![sessions * steps * stack.in_channels] };
    let mut t = steps;
    let mut launches = Vec::new();
    let mut states = BTreeMap::new();
    let ops = &stack.ops;
    let mut i = 0;
    while i < ops.len() {
        let op = &ops[i];
        let fused_scale = folds_into(op, ops.get(i + 1)).then_some(i + 1);
        let input = bufs.slot[&op.input()];
        let at = |kind: Kind, part: Part, scale: Option<usize>, input: usize, output: usize, t_in: usize, t_out: usize, t_pad: usize, cin: usize, cout: usize| Launch { op: i, part, scale, kind, input, output, layer: Some(i), t_in, t_out, t_pad, cin, cout };
        let mut step = 1;
        match op {
            CodecOp::Conv { cin, cout, depthwise: true, k, stride, act, .. } => {
                ensure!(*stride == 1 && cin == cout && act.is_none(), "a depthwise codec conv has stride 1 and no input activation");
                let Some(s) = fused_scale else { bail!("op {i} ({}): a depthwise conv feeds a scale_add into x", op.label()) };
                let x = bufs.slot[&Slot::X];
                launches.push(at(Kind::Depthwise { channels: *cin, taps: *k }, Part::Main, Some(s), input, x, t, t, t, *cin, *cout));
                launches.push(at(Kind::Shift { channels: *cin, context: k - 1 }, Part::Main, None, input, input, t, t, t, *cin, *cin));
                states.insert(i, (k - 1) * cin);
                step = 2;
            }
            CodecOp::ConvUp { cin, cout, stride, depthwise: true, act, .. } => {
                ensure!(cin == cout && act.is_none(), "a depthwise codec conv_up keeps its width and has no input activation");
                let out = bufs.target(op, input, false, sessions * t * stride * cout);
                launches.push(at(Kind::DwUp { channels: *cin, stride: *stride }, Part::Main, None, input, out, t, t, t, *cin, stride * cout));
                launches.push(at(Kind::Shift { channels: *cin, context: 1 }, Part::Main, None, input, input, t, t, t, *cin, *cin));
                states.insert(i, *cin);
                bufs.slot.insert(Slot::X, out);
                t *= stride;
            }
            CodecOp::Conv { cin, cout, act, .. } | CodecOp::ConvUp { cin, cout, act, .. } => {
                let (up, k, stride) = match op {
                    CodecOp::Conv { k, stride, .. } => (false, *k, *stride),
                    CodecOp::ConvUp { stride, .. } => (true, 2, *stride),
                    _ => unreachable!(),
                };
                let shape = if up { ConvShape::line(*cin, stride * cout, 2, 1, stride) } else { ConvShape::line(*cin, *cout, k, stride, 1) };
                let residual = fused_scale.filter(|_| !up && stride == 1);
                let t_out = t.div_ceil(shape.stride);
                let t_pad = t_out * shape.stride;
                let out = bufs.target(op, input, residual.is_some(), sessions * t_out * shape.cout);
                let rows = kir::conv::rows_fit(&shape, sessions * t_out);
                launches.push(at(Kind::Conv { shape, gelu: false, residual: residual.is_some(), rows, act: *act }, Part::Main, residual, input, out, t, t_out, t_pad, *cin, shape.cout));
                if shape.context() > 0 {
                    launches.push(at(Kind::Shift { channels: *cin, context: shape.context() }, Part::Main, None, input, input, t, t_out, t_pad, *cin, *cin));
                    states.insert(i, shape.context() * cin);
                }
                if residual.is_none() {
                    bufs.slot.insert(op.output(), out);
                } else {
                    step = 2;
                }
                t = t_out * shape.phases;
            }
            CodecOp::RmsNorm { channels, eps, .. } | CodecOp::LayerNorm { channels, eps, .. } => {
                let out = bufs.target(op, input, false, sessions * t * channels);
                let kind = match op {
                    CodecOp::RmsNorm { .. } => Kind::Rms { channels: *channels, eps: *eps },
                    _ => Kind::Layer { channels: *channels, eps: *eps },
                };
                launches.push(Launch { layer: None, ..at(kind, Part::Main, None, input, out, t, t, t, *channels, *channels) });
                bufs.slot.insert(op.output(), out);
            }
            CodecOp::Linear { cin, cout, gelu, .. } => {
                let shape = ConvShape::line(*cin, *cout, 1, 1, 1);
                let out = bufs.target(op, input, fused_scale.is_some(), sessions * t * cout);
                let rows = kir::conv::rows_fit(&shape, sessions * t);
                launches.push(Launch { layer: None, ..at(Kind::Conv { shape, gelu: *gelu, residual: fused_scale.is_some(), rows, act: None }, Part::Main, fused_scale, input, out, t, t, t, *cin, *cout) });
                match fused_scale {
                    Some(_) => step = 2,
                    None => {
                        bufs.slot.insert(op.output(), out);
                    }
                }
            }
            CodecOp::Attention { channels, heads, context, theta, .. } => {
                let c = *channels;
                let d = c / heads;
                let qkv_shape = ConvShape::line(c, 3 * c, 1, 1, 1);
                let qkv = bufs.fresh(&[input]);
                bufs.grow(qkv, sessions * t * 3 * c);
                let att = bufs.fresh(&[input, qkv]);
                bufs.grow(att, sessions * t * c);
                launches.push(Launch { layer: None, ..at(Kind::Conv { shape: qkv_shape, gelu: false, residual: false, rows: kir::conv::rows_fit(&qkv_shape, sessions * t), act: None }, Part::Main, None, input, qkv, t, t, t, c, 3 * c) });
                launches.push(at(Kind::Rope { heads: *heads, head_dim: d, context: *context, theta: *theta }, Part::Main, None, qkv, qkv, t, t, t, c, c));
                launches.push(at(Kind::Ring { heads: *heads, head_dim: d, context: *context }, Part::Main, None, qkv, att, t, t, t, c, c));
                launches.push(at(Kind::RingStore { channels: c, context: *context }, Part::Main, None, qkv, qkv, t, t, t, c, c));
                launches.push(at(Kind::Advance { context: *context, channels: c }, Part::Main, None, qkv, qkv, t, t, t, c, c));
                states.insert(i, op.state_floats());
                let out_shape = ConvShape::line(c, c, 1, 1, 1);
                let out = bufs.target(op, att, fused_scale.is_some(), sessions * t * c);
                launches.push(Launch { layer: None, ..at(Kind::Conv { shape: out_shape, gelu: false, residual: fused_scale.is_some(), rows: kir::conv::rows_fit(&out_shape, sessions * t), act: None }, Part::Out, fused_scale, att, out, t, t, t, c, c) });
                match fused_scale {
                    Some(_) => step = 2,
                    None => {
                        bufs.slot.insert(op.output(), out);
                    }
                }
            }
            CodecOp::ScaleAdd { .. } | CodecOp::Add { .. } => bail!("op {i} ({}) has no producer to fold into", op.label()),
        }
        i += step;
    }
    let output = bufs.slot[&Slot::X];
    let buffers = bufs.floats.len();
    Ok(LaunchPlan { launches, buffers, buffer_floats: bufs.floats, input: 0, output, t_out: t, out_channels: stack.out_channels, states })
}

pub fn report(stack: &Stack, p: &LaunchPlan, sessions: usize) -> String {
    let mut out = String::new();
    let (mut read, mut written, mut flops) = (0u64, 0u64, 0u64);
    for (i, l) in p.launches.iter().enumerate() {
        let (r, w) = l.bytes(sessions);
        let f = l.flops(sessions);
        out.push_str(&format!("{i:>3} {:<52} rows {:>5} read {:>11} write {:>9} flops {:>12}\n", l.label(stack), sessions * l.t_out, r, w, f));
        read += r;
        written += w;
        flops += f;
    }
    let weights: u64 = p.launches.iter().map(|l| 4 * l.weight_floats() as u64).sum();
    out.push_str(&format!(
        "total {} launches, read {read} bytes ({weights} of f32 weights), write {written} bytes, {flops} flops, state {} values per session, activations {} bytes\n",
        p.launches.len(),
        p.state_floats(),
        4 * p.buffer_floats.iter().sum::<usize>()
    ));
    out
}
