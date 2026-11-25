use super::graph::{CodecOp, Slot, Stack};
use anyhow::{Result, bail, ensure};
use kir::conv::{ConvKernel, ConvShape, Epilogue, Prologue};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Conv { shape: ConvShape, gelu: bool, residual: bool },
    Depthwise { channels: usize, taps: usize },
    Rms { channels: usize, eps: f32 },
    Shift { channels: usize, context: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub op: usize,
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
        match self.kind {
            Kind::Shift { .. } => format!("state {op}"),
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
            Kind::Rms { channels, .. } => *channels,
            Kind::Shift { .. } => 0,
        }
    }

    pub fn bytes(&self, sessions: usize) -> (u64, u64) {
        let f = |n: usize| 4 * n as u64;
        let b = sessions;
        match &self.kind {
            Kind::Conv { shape, residual, .. } => {
                let state = b * shape.context() * shape.cin;
                let out = b * self.t_out * shape.cout;
                (f(self.weight_floats() + b * self.t_in * shape.cin + state + if *residual { out } else { 0 }), f(out))
            }
            Kind::Depthwise { channels, taps } => {
                let out = b * self.t_out * channels;
                (f(self.weight_floats() + b * self.t_in * channels + b * (taps - 1) * channels + out), f(out))
            }
            Kind::Rms { channels, .. } => (f(self.weight_floats() + b * self.t_in * channels), f(b * self.t_in * channels)),
            Kind::Shift { channels, context } => (f(b * context * channels), f(b * context * channels)),
        }
    }

    pub fn flops(&self, sessions: usize) -> u64 {
        let rows = (sessions * self.t_out) as u64;
        match &self.kind {
            Kind::Conv { shape, .. } => rows * 2 * (shape.cout * shape.cin * shape.kt) as u64,
            Kind::Depthwise { channels, taps } => rows * 2 * (channels * taps) as u64,
            Kind::Rms { channels, .. } => rows * 4 * *channels as u64,
            Kind::Shift { .. } => 0,
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
            if let Kind::Conv { shape, gelu, residual } = &l.kind {
                out.push(conv_kernel(target, *shape, *gelu, *residual)?);
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

pub fn conv_kernel(target: &kir::Target, shape: ConvShape, gelu: bool, residual: bool) -> Result<ConvKernel> {
    ConvKernel::new(target, shape, Prologue::None, Epilogue { residual, stats: None, gelu }).map_err(anyhow::Error::msg)
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
}

pub fn plan(stack: &Stack, steps: usize, sessions: usize) -> Result<LaunchPlan> {
    ensure!(steps > 0 && sessions > 0, "a codec run needs steps and sessions");
    let mut bufs = Buffers { slot: BTreeMap::from([(Slot::X, 0)]), floats: vec![sessions * steps * stack.in_channels] };
    let mut t = steps;
    let mut launches = Vec::new();
    let mut states = BTreeMap::new();
    let ops = &stack.ops;
    let mut i = 0;
    while i < ops.len() {
        let op = &ops[i];
        let fused_scale = match ops.get(i + 1) {
            Some(CodecOp::ScaleAdd { input, .. }) if *input == op.output() && op.output() != Slot::X => Some(i + 1),
            _ => None,
        };
        let input = bufs.slot[&op.input()];
        match op {
            CodecOp::Conv { cin, cout, k, stride, depthwise: true, .. } => {
                ensure!(*stride == 1 && cin == cout, "a depthwise codec conv has stride 1");
                let Some(s) = fused_scale else { bail!("op {i} ({}): a depthwise conv feeds a scale_add into x", op.label()) };
                let x = bufs.slot[&Slot::X];
                launches.push(Launch { op: i, scale: Some(s), kind: Kind::Depthwise { channels: *cin, taps: *k }, input, output: x, layer: Some(i), t_in: t, t_out: t, t_pad: t, cin: *cin, cout: *cout });
                launches.push(Launch { op: i, scale: None, kind: Kind::Shift { channels: *cin, context: k - 1 }, input, output: input, layer: Some(i), t_in: t, t_out: t, t_pad: t, cin: *cin, cout: *cin });
                states.insert(i, (k - 1) * cin);
                i += 2;
                continue;
            }
            CodecOp::Conv { cin, cout, .. } | CodecOp::ConvUp { cin, cout, .. } => {
                let (up, k, stride) = match op {
                    CodecOp::Conv { k, stride, .. } => (false, *k, *stride),
                    CodecOp::ConvUp { stride, .. } => (true, 2, *stride),
                    _ => unreachable!(),
                };
                let shape = if up { ConvShape::line(*cin, stride * cout, 2, 1, stride) } else { ConvShape::line(*cin, *cout, k, stride, 1) };
                let t_out = t.div_ceil(shape.stride);
                let t_pad = t_out * shape.stride;
                let out = bufs.fresh(&[input]);
                bufs.grow(out, sessions * t_out * shape.cout);
                launches.push(Launch { op: i, scale: None, kind: Kind::Conv { shape, gelu: false, residual: false }, input, output: out, layer: Some(i), t_in: t, t_out, t_pad, cin: *cin, cout: shape.cout });
                if shape.context() > 0 {
                    launches.push(Launch { op: i, scale: None, kind: Kind::Shift { channels: *cin, context: shape.context() }, input, output: input, layer: Some(i), t_in: t, t_out, t_pad, cin: *cin, cout: *cin });
                    states.insert(i, shape.context() * cin);
                }
                bufs.slot.insert(op.output(), out);
                t = t_out * shape.phases;
            }
            CodecOp::RmsNorm { channels, eps, .. } => {
                if op.output() != op.input() {
                    bufs.slot.remove(&op.output());
                }
                let out = bufs.fresh(&[input]);
                bufs.grow(out, sessions * t * channels);
                launches.push(Launch { op: i, scale: None, kind: Kind::Rms { channels: *channels, eps: *eps }, input, output: out, layer: None, t_in: t, t_out: t, t_pad: t, cin: *channels, cout: *channels });
                bufs.slot.insert(op.output(), out);
            }
            CodecOp::Linear { cin, cout, gelu, .. } => {
                let shape = ConvShape::line(*cin, *cout, 1, 1, 1);
                let (out, scale) = match fused_scale {
                    Some(s) => (bufs.slot[&Slot::X], Some(s)),
                    None => {
                        if op.output() != op.input() {
                            bufs.slot.remove(&op.output());
                        }
                        (bufs.fresh(&[input]), None)
                    }
                };
                bufs.grow(out, sessions * t * cout);
                launches.push(Launch { op: i, scale, kind: Kind::Conv { shape, gelu: *gelu, residual: scale.is_some() }, input, output: out, layer: None, t_in: t, t_out: t, t_pad: t, cin: *cin, cout: *cout });
                if scale.is_none() {
                    bufs.slot.insert(op.output(), out);
                } else {
                    i += 2;
                    continue;
                }
            }
            CodecOp::ScaleAdd { .. } => bail!("op {i} ({}) has no producer to fold into", op.label()),
        }
        i += 1;
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
