use super::Backend;
use crate::gguf::DeviceRows;
use crate::runtime::{Generation, Sampling};
use crate::executor::PlanBackend;
use crate::plan::{Buf, NormKind, Op, Phase, Plan, Weight};
use crate::spec::{Block, Spec};
use crate::weights::{Checkpoint, Matrix, Mixer};
use crate::wgsl::{self, LINEAR_TOKENS, ROW_SPAN};
use anyhow::{Context, Result, bail, ensure};
use std::cell::RefCell;
use std::ops::Range;
use std::time::Instant;

const PREFILL_CHUNK: usize = 64;
const DECODE_STEPS_PER_SUBMIT: usize = 4;

pub struct WMat<B: Backend> {
    data: B::Buffer,
    rows: B::Buffer,
    nrows: u32,
    cols: u32,
    bytes: u64,
}

struct LayerBuffers<B: Backend> {
    ln1: B::Buffer,
    ln2: B::Buffer,
    qkv: WMat<B>,
    bias: Option<B::Buffer>,
    q_norm: B::Buffer,
    k_norm: B::Buffer,
    o: WMat<B>,
    gate_up: WMat<B>,
    down: WMat<B>,
    k_cache: B::Buffer,
    v_cache: B::Buffer,
}

struct Scratch<B: Backend> {
    x: B::Buffer,
    xn: B::Buffer,
    qkv: B::Buffer,
    attn: B::Buffer,
    gate_up: B::Buffer,
    act: B::Buffer,
    logits: B::Buffer,
    tokens: B::Buffer,
}

struct Model<B: Backend> {
    spec: Spec,
    max_ctx: usize,
    embed: WMat<B>,
    layers: Vec<LayerBuffers<B>>,
    final_norm: B::Buffer,
    head: WMat<B>,
    scratch: Scratch<B>,
    weight_bytes: u64,
    prefill: Plan,
    decode: Plan,
}

pub struct Dense<B: Backend> {
    be: B,
    m: Model<B>,
}

fn rows_of(m: &Matrix, nrows: usize, cols: usize, what: &str) -> Result<DeviceRows> {
    match m {
        Matrix::Bf16(w) => {
            ensure!(w.len() == nrows * cols, "{what} has {} values, expected {}", w.len(), nrows * cols);
            DeviceRows::from_bf16(w, cols)
        }
        Matrix::Ggml(g) => {
            ensure!(g.rows.len() == nrows && g.cols == cols, "{what} is {}x{}, expected {nrows}x{cols}", g.rows.len(), g.cols);
            Ok(DeviceRows { data: g.data.clone(), rows: g.rows.clone(), cols })
        }
        _ => bail!("{what} uses weights that are not yet on wgpu"),
    }
}

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn upload_rows<B: Backend>(be: &B, rows: DeviceRows, nrows: usize, what: &str) -> Result<WMat<B>> {
    let data = be.alloc(rows.data.len() as u64).with_context(|| format!("{what} weights"))?;
    be.upload(&data, 0, &rows.data);
    let table = be.alloc(rows.rows.len() as u64 * 4)?;
    be.upload(&table, 0, &words(&rows.rows));
    Ok(WMat { bytes: rows.data.len() as u64 + rows.rows.len() as u64 * 4, data, rows: table, nrows: nrows as u32, cols: rows.cols as u32 })
}

fn upload_floats<B: Backend>(be: &B, v: &[f32]) -> Result<B::Buffer> {
    let b = be.alloc(v.len() as u64 * 4)?;
    be.upload(&b, 0, &floats(v));
    Ok(b)
}

struct Step {
    pos0: usize,
    rows: usize,
    head: Range<usize>,
}

struct Exec<'a, B: Backend> {
    be: RefCell<&'a mut B>,
    m: &'a Model<B>,
}

impl<B: Backend> Exec<'_, B> {
    fn buf(&self, b: Buf) -> &B::Buffer {
        let s = &self.m.scratch;
        match b {
            Buf::X => &s.x,
            Buf::Xn => &s.xn,
            Buf::Qkv => &s.qkv,
            Buf::Attn => &s.attn,
            Buf::GateUp => &s.gate_up,
            Buf::Act => &s.act,
            Buf::Logits => &s.logits,
            other => unreachable!("{} is not on wgpu", other.label()),
        }
    }

    fn weight<'a>(&'a self, w: Weight, l: Option<&'a LayerBuffers<B>>) -> &'a WMat<B> {
        let l = || l.expect("layer weights need a layer");
        match w {
            Weight::Qkv => &l().qkv,
            Weight::O => &l().o,
            Weight::GateUp => &l().gate_up,
            Weight::Down => &l().down,
            Weight::LmHead => &self.m.head,
            other => unreachable!("{} is not on wgpu", other.name()),
        }
    }
}

impl<B: Backend> PlanBackend<Step> for Exec<'_, B> {
    fn layers(&self) -> usize {
        self.m.layers.len()
    }

    fn block(&self, layer: usize) -> Block {
        self.m.spec.block(layer)
    }

    fn run_op(&self, op: &Op, layer: Option<usize>, run: &Step) -> Result<()> {
        let m = self.m;
        let spec = &m.spec;
        let l = layer.map(|i| &m.layers[i]);
        let mut be = self.be.borrow_mut();
        let be = &mut **be;
        let head = op_is_head(op);
        let (rows, row0) = if head { (run.head.len(), run.head.start) } else { (run.rows, 0) };
        let kernels = wgsl::kernels_for(op, spec, rows)?;
        let r = rows as u32;
        match *op {
            Op::Embed => be.dispatch(kernels[0], &[&m.scratch.tokens, &m.embed.data, &m.embed.rows, &m.scratch.x], [0, run.pos0 as u32, 0, 0, 0, 0, 0, 0], [r, 1, 1]),
            Op::Norm { which, into } => {
                let w = match which {
                    NormKind::Input => &l.expect("layer norm needs a layer").ln1,
                    NormKind::PostAttention => &l.expect("layer norm needs a layer").ln2,
                    NormKind::Final => &m.final_norm,
                };
                be.dispatch(kernels[0], &[&m.scratch.x, w, self.buf(into)], [row0 as u32, 0, 0, 0, 0, 0, 0, 0], [r, 1, 1])
            }
            Op::Linear { weight, input, output } => {
                let w = self.weight(weight, l);
                let grid = [w.nrows.min(ROW_SPAN), w.nrows.div_ceil(ROW_SPAN), rows.div_ceil(if rows == 1 { 1 } else { LINEAR_TOKENS }) as u32];
                let accumulate = output == Buf::X;
                be.dispatch(kernels[0], &[self.buf(input), self.buf(output), &w.data, &w.rows], [w.nrows, w.cols, r, 0, 0, accumulate as u32, 0, 0], grid)?;
                if let (Weight::Qkv, Some(b)) = (weight, l.and_then(|l| l.bias.as_ref())) {
                    let width = spec.qkv_rows();
                    be.dispatch(kernels[1], &[self.buf(output), b], [r, width as u32, 0, 0, 0, 0, 0, 0], grid_1d(rows * width))?;
                }
                Ok(())
            }
            Op::KvRope | Op::QkRope => {
                let l = l.expect("rope needs a layer");
                let a = &spec.attention;
                be.dispatch(kernels[0], &[&m.scratch.qkv, &l.q_norm, &l.k_norm, &l.k_cache, &l.v_cache], [run.pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [(a.heads + 2 * a.kv_heads) as u32, r, 1])
            }
            Op::Flash | Op::FlashLegacy | Op::DecodeAttention => {
                let l = l.expect("attention needs a layer");
                be.dispatch(kernels[0], &[&m.scratch.qkv, &l.k_cache, &l.v_cache, &m.scratch.attn], [run.pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [spec.attention.heads as u32, r, 1])
            }
            Op::Swiglu => be.dispatch(kernels[0], &[&m.scratch.gate_up, &m.scratch.act], [r, 0, 0, 0, 0, 0, 0, 0], grid_1d(rows * spec.dims.intermediate)),
            Op::GdnPrefill | Op::GdnDecode => bail!("delta-net is not yet on wgpu"),
        }
    }
}

fn op_is_head(op: &Op) -> bool {
    matches!(op, Op::Norm { which: NormKind::Final, .. } | Op::Linear { weight: Weight::LmHead, .. })
}

fn grid_1d(items: usize) -> [u32; 3] {
    let groups = items.div_ceil(256) as u32;
    [groups.min(ROW_SPAN), groups.div_ceil(ROW_SPAN), 1]
}

fn mat<B: Backend>(be: &B, m: &Matrix, nrows: usize, cols: usize, what: &str) -> Result<WMat<B>> {
    upload_rows(be, rows_of(m, nrows, cols, what)?, nrows, what)
}

impl<B: Backend> Dense<B> {
    pub fn load(mut be: B, spec: &Spec, max_ctx: usize) -> Result<Self> {
        if let Some(why) = wgsl::unsupported(spec) {
            bail!("{} is not supported on backend wgpu: {why}", spec.name);
        }
        let limits = be.limits();
        ensure!(limits.max_storage_buffers >= 5, "this adapter allows {} storage buffers per stage, specrt needs 5", limits.max_storage_buffers);
        be.compile(&wgsl::generate(spec, max_ctx)?)?;
        let checkpoint = Checkpoint::for_repo(&spec.repo)?;
        let d = &spec.dims;
        let a = &spec.attention;
        let out = checkpoint.outside(spec)?;
        let embed = upload_rows(&be, DeviceRows::from_bf16(&out.embed, d.hidden)?, d.vocab, "embed")?;
        let head = mat(&be, &out.lm_head, d.vocab, d.hidden, "lm_head")?;
        let final_norm = upload_floats(&be, &out.norm)?;
        let kv_floats = a.kv_heads * max_ctx * a.head_dim;
        let mut layers = Vec::with_capacity(spec.layers.count);
        let mut weight_bytes = embed.bytes + head.bytes;
        for i in 0..spec.layers.count {
            let l = checkpoint.layer(spec, i).with_context(|| format!("layer {i}"))?;
            let Mixer::Attention(at) = &l.mixer else { bail!("layer {i} is not attention, which is the only mixer on wgpu") };
            let norm = |v: &Option<Vec<f32>>| -> Result<B::Buffer> { upload_floats(&be, v.as_deref().unwrap_or(&[0.0; 4])) };
            let lb = LayerBuffers {
                ln1: upload_floats(&be, &l.ln1)?,
                ln2: upload_floats(&be, &l.ln2)?,
                qkv: mat(&be, &at.qkv, spec.qkv_rows(), d.hidden, "qkv")?,
                bias: at.qkv_bias.as_ref().map(|b| upload_floats(&be, b)).transpose()?,
                q_norm: norm(&at.q_norm)?,
                k_norm: norm(&at.k_norm)?,
                o: mat(&be, &at.o, d.hidden, spec.q_dim(), "o")?,
                gate_up: mat(&be, &l.gate_up, 2 * d.intermediate, d.hidden, "gate_up")?,
                down: mat(&be, &l.down, d.hidden, d.intermediate, "down")?,
                k_cache: be.alloc(kv_floats as u64 * 4)?,
                v_cache: be.alloc(kv_floats as u64 * 4)?,
            };
            weight_bytes += lb.qkv.bytes + lb.o.bytes + lb.gate_up.bytes + lb.down.bytes;
            layers.push(lb);
        }
        let rows = PREFILL_CHUNK;
        let scratch = Scratch {
            x: be.alloc((rows * d.hidden * 4) as u64)?,
            xn: be.alloc((rows * d.hidden * 4) as u64)?,
            qkv: be.alloc((rows * spec.qkv_rows() * 4) as u64)?,
            attn: be.alloc((rows * spec.q_dim() * 4) as u64)?,
            gate_up: be.alloc((rows * 2 * d.intermediate * 4) as u64)?,
            act: be.alloc((rows * d.intermediate * 4) as u64)?,
            logits: be.alloc((rows * d.vocab * 4) as u64).context("logits buffer")?,
            tokens: be.alloc(((max_ctx + 2) * 4) as u64)?,
        };
        Ok(Dense { be, m: Model { spec: spec.clone(), max_ctx, embed, layers, final_norm, head, scratch, weight_bytes, prefill: wgsl::plan(spec, Phase::Prefill), decode: wgsl::plan(spec, Phase::Decode) } })
    }

    pub fn description(&self) -> String {
        self.be.description()
    }

    pub fn spec(&self) -> &Spec {
        &self.m.spec
    }

    pub fn max_ctx(&self) -> usize {
        self.m.max_ctx
    }

    pub fn weight_bytes(&self) -> u64 {
        self.m.weight_bytes
    }

    fn argmax(be: &mut B, m: &Model<B>, sampling: Sampling, step: usize, slot: usize) -> Result<()> {
        let inv = sampling.inverse_temperature().to_bits();
        be.dispatch("argmax", &[&m.scratch.logits, &m.scratch.tokens], [0, inv, sampling.seed, step as u32, slot as u32, 0, 0, 0], [1, 1, 1])
    }

    fn run(be: &mut B, m: &Model<B>, plan: &Plan, pos0: usize, rows: usize, head: Range<usize>) -> Result<()> {
        let exec = Exec { be: RefCell::new(be), m };
        crate::executor::run_plan(&exec, plan, &Step { pos0, rows, head }, &mut |_| Ok(()))
    }

    fn prefill(&mut self, prompt: &[u32], all_logits: bool, sampling: Sampling) -> Result<Vec<f32>> {
        let Self { be, m } = self;
        let vocab = m.spec.dims.vocab;
        be.upload(&m.scratch.tokens, 0, &words(prompt));
        let mut logits = Vec::new();
        let n = prompt.len();
        let mut c0 = 0;
        while c0 < n {
            let rows = PREFILL_CHUNK.min(n - c0);
            let last = c0 + rows == n;
            let head = if all_logits { 0..rows } else if last { rows - 1..rows } else { 0..0 };
            Self::run(be, m, &m.prefill, c0, rows, head)?;
            if all_logits {
                be.submit()?;
                let bytes = be.download(&m.scratch.logits, 0, (rows * vocab * 4) as u64)?;
                logits.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))));
            } else {
                if last {
                    Self::argmax(be, m, sampling, 0, n)?;
                }
                be.submit()?;
            }
            c0 += rows;
        }
        Ok(logits)
    }

    fn decode_step(be: &mut B, m: &Model<B>, pos: usize, sampling: Sampling) -> Result<()> {
        Self::run(be, m, &m.decode, pos, 1, 0..1)?;
        Self::argmax(be, m, sampling, pos, pos + 1)
    }

    pub fn prompt_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty() && tokens.len() <= self.m.max_ctx, "prompt is empty or longer than max_ctx");
        self.prefill(tokens, true, Sampling::GREEDY)
    }

    pub fn generate(&mut self, prompt: &[u32], new_tokens: usize, sampling: Sampling) -> Result<Generation> {
        self.generate_with(prompt, new_tokens, sampling, &[], false, &mut |_| true)
    }

    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
        stop: &[u32],
        _chunked: bool,
        on_tokens: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Generation> {
        ensure!(!prompt.is_empty() && new_tokens > 0, "empty prompt or nothing to generate");
        ensure!(prompt.len() + new_tokens <= self.m.max_ctx, "prompt plus generation exceeds max_ctx");
        let n = prompt.len();
        let started = Instant::now();
        self.prefill(prompt, false, sampling)?;
        let mut tokens = self.read_tokens(n, 1)?;
        let prefill_secs = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut shown = 0;
        loop {
            let (kept, stopped) = crate::runtime::cut_at_stop(&tokens, stop);
            let mut wanted = true;
            if kept > shown {
                wanted = on_tokens(&tokens[shown..kept]);
                shown = kept;
            }
            if stopped || !wanted || tokens.len() >= new_tokens {
                tokens.truncate(kept);
                break;
            }
            let steps = DECODE_STEPS_PER_SUBMIT.min(new_tokens - tokens.len());
            let base = n + tokens.len() - 1;
            let Self { be, m } = self;
            for k in 0..steps {
                Self::decode_step(be, m, base + k, sampling)?;
            }
            be.submit()?;
            let fresh = self.read_tokens(n + tokens.len(), steps)?;
            tokens.extend(fresh);
        }
        let decode_secs = started.elapsed().as_secs_f64();
        Ok(Generation { tokens, prefill_secs, decode_secs, peak_device_bytes: self.m.weight_bytes })
    }

    fn read_tokens(&mut self, at: usize, count: usize) -> Result<Vec<u32>> {
        let bytes = self.be.download(&self.m.scratch.tokens, at as u64 * 4, count as u64 * 4)?;
        Ok(bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes"))).collect())
    }
}
