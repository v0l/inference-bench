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

const PREFILL_CHUNK: usize = 512;
const DECODE_STEPS_PER_SUBMIT: usize = 4;

pub struct WMat<B: Backend> {
    data: B::Buffer,
    rows: B::Buffer,
    nrows: u32,
    cols: u32,
    bytes: u64,
    sg_kernel: &'static str,
    coop_kernel: &'static str,
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
    k_cache: Vec<B::Buffer>,
    v_cache: Vec<B::Buffer>,
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
    slots: usize,
    first: bool,
    last: bool,
    embed: Option<WMat<B>>,
    layers: Vec<LayerBuffers<B>>,
    final_norm: Option<B::Buffer>,
    head: Option<WMat<B>>,
    scratch: Scratch<B>,
    weight_bytes: u64,
    prefill: Plan,
    decode: Plan,
}

pub struct Dense<B: Backend> {
    be: B,
    m: Model<B>,
}

fn rows_of<'a>(m: &'a Matrix, nrows: usize, cols: usize, what: &str) -> Result<DeviceRows<'a>> {
    match m {
        Matrix::Bf16(w) => {
            ensure!(w.len() == nrows * cols, "{what} has {} values, expected {}", w.len(), nrows * cols);
            DeviceRows::from_bf16(w, cols)
        }
        Matrix::Ggml(g) => {
            ensure!(g.rows.len() == nrows && g.cols == cols, "{what} is {}x{}, expected {nrows}x{cols}", g.rows.len(), g.cols);
            Ok(DeviceRows { parts: g.parts.iter().map(|p| std::borrow::Cow::Borrowed(p.as_ref())).collect(), rows: g.rows.clone(), cols })
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
    let parts: Vec<&[u8]> = rows.parts.iter().map(|p| p.as_ref()).collect();
    let data = be.alloc_filled(&parts).with_context(|| format!("{what} weights"))?;
    let table = be.alloc(rows.rows.len() as u64 * 4)?;
    be.upload(&table, 0, &words(&rows.rows));
    let sg_kernel = wgsl::sg_kernel_for(&rows.rows);
    let coop_kernel = wgsl::coop_kernel_for(&rows.rows);
    Ok(WMat { bytes: rows.len_bytes() as u64 + rows.rows.len() as u64 * 4, sg_kernel, coop_kernel, data, rows: table, nrows: nrows as u32, cols: rows.cols as u32 })
}

fn upload_floats<B: Backend>(be: &B, v: &[f32]) -> Result<B::Buffer> {
    let b = be.alloc(v.len() as u64 * 4)?;
    be.upload(&b, 0, &floats(v));
    Ok(b)
}

struct Step {
    slot: usize,
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
            Weight::LmHead => self.m.head.as_ref().expect("the last stage owns the head"),
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
            Op::Embed => be.dispatch(kernels[0], &[&m.scratch.tokens, &m.embed.as_ref().expect("the first stage owns the embedding").data, &m.embed.as_ref().expect("the first stage owns the embedding").rows, &m.scratch.x], [0, run.pos0 as u32, 0, 0, 0, 0, 0, 0], [r, 1, 1]),
            Op::Norm { which, into } => {
                let w = match which {
                    NormKind::Input => &l.expect("layer norm needs a layer").ln1,
                    NormKind::PostAttention => &l.expect("layer norm needs a layer").ln2,
                    NormKind::Final => m.final_norm.as_ref().expect("the last stage owns the final norm"),
                    NormKind::PreMlp | NormKind::PostMlp | NormKind::PostDense | NormKind::PostExperts => bail!("sandwich norms are not yet on wgpu"),
                };
                be.dispatch(kernels[0], &[&m.scratch.x, w, self.buf(into)], [row0 as u32, 0, 0, 0, 0, 0, 0, 0], [r, 1, 1])
            }
            Op::Linear { weight, input, output, .. } => {
                let w = self.weight(weight, l);
                let coop = kernels[0] == "linear_tile" && be.limits().coop_matrix;
                let sg = rows == 1 && be.limits().subgroup_size == 32;
                let kernel = if coop { w.coop_kernel } else if sg { w.sg_kernel } else { kernels[0] };
                let grid = if sg {
                    let groups = w.nrows.div_ceil(wgsl::SG_ROWS);
                    [groups.min(ROW_SPAN), groups.div_ceil(ROW_SPAN), 1]
                } else if kernels[0] == "linear_tile" {
                    [w.nrows.div_ceil(wgsl::TILE_ROWS as u32), 1, rows.div_ceil(wgsl::TILE_TOKENS) as u32]
                } else {
                    [w.nrows.min(ROW_SPAN), w.nrows.div_ceil(ROW_SPAN), rows.div_ceil(if rows == 1 { 1 } else { LINEAR_TOKENS }) as u32]
                };
                let accumulate = output == Buf::X;
                let glue = sg && weight == Weight::GateUp;
                let target = if glue { Buf::Act } else { output };
                be.dispatch(kernel, &[self.buf(input), self.buf(target), &w.data, &w.rows], [w.nrows, w.cols, r, 0, 0, accumulate as u32, glue as u32, 0], grid)?;
                if let (Weight::Qkv, Some(b)) = (weight, l.and_then(|l| l.bias.as_ref())) {
                    let width = spec.qkv_rows();
                    be.dispatch(kernels[1], &[self.buf(output), b], [r, width as u32, 0, 0, 0, 0, 0, 0], grid_1d(rows * width))?;
                }
                Ok(())
            }
            Op::KvRope | Op::QkRope => {
                let l = l.expect("rope needs a layer");
                let a = &spec.attention;
                be.dispatch(kernels[0], &[&m.scratch.qkv, &l.q_norm, &l.k_norm, &l.k_cache[run.slot], &l.v_cache[run.slot]], [run.pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [(a.heads + 2 * a.kv_heads) as u32, r, 1])
            }
            Op::Flash | Op::FlashLegacy | Op::DecodeAttention => {
                let l = l.expect("attention needs a layer");
                if rows > 1 && be.limits().coop_matrix && spec.attention.head_dim.is_multiple_of(8) && spec.attention.head_dim <= 256 {
                    return be.dispatch("flash_coop", &[&m.scratch.qkv, &l.k_cache[run.slot], &l.v_cache[run.slot], &m.scratch.attn], [run.pos0 as u32, r, 0, 0, 0, 0, 0, 0], [spec.attention.heads as u32, r.div_ceil(wgsl::FLASH_COOP_QUERIES), 1]);
                }
                if rows > 1 && be.limits().subgroup_size == 32 {
                    return be.dispatch("flash", &[&m.scratch.qkv, &l.k_cache[run.slot], &l.v_cache[run.slot], &m.scratch.attn], [run.pos0 as u32, r, 0, 0, 0, 0, 0, 0], [spec.attention.heads as u32, r.div_ceil(wgsl::FLASH_TOKENS), 1]);
                }
                if rows == 1 && be.limits().subgroup_size == 32 && spec.attention.head_dim.is_multiple_of(16) && spec.attention.head_dim <= 256 {
                    return be.dispatch("flash_decode", &[&m.scratch.qkv, &l.k_cache[run.slot], &l.v_cache[run.slot], &m.scratch.attn], [run.pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [spec.attention.heads as u32, 1, 1]);
                }
                be.dispatch(kernels[0], &[&m.scratch.qkv, &l.k_cache[run.slot], &l.v_cache[run.slot], &m.scratch.attn], [run.pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [spec.attention.heads as u32, r, 1])
            }
            Op::Swiglu if rows == 1 && be.limits().subgroup_size == 32 => Ok(()),
            Op::Swiglu => be.dispatch(kernels[0], &[&m.scratch.gate_up, &m.scratch.act], [r, 0, 0, 0, 0, 0, 0, 0], grid_1d(rows * spec.dims.intermediate)),
            Op::Kernel { .. } | Op::MixKernel { .. } | Op::DecodeRope => bail!("described kernels are not yet on wgpu"),
            Op::NormAdd { .. } | Op::Clear { .. } | Op::Split { .. } | Op::QkNorm | Op::ScaledAdd { .. } | Op::LogitScale | Op::LogitSoftcap => bail!("{} is not yet on wgpu", op.label()),
            Op::Splice => bail!("vision is not yet on wgpu"),
            Op::GdnPrefill | Op::GdnDecode | Op::GdnVerify => bail!("delta-net is not yet on wgpu"),
            Op::MtpIn | Op::MtpGather | Op::MtpNorm | Op::MtpDraft => bail!("MTP is not yet on wgpu"),
            Op::MoeRoute | Op::MoeExperts | Op::MoeCombine | Op::MoeCombineInto { .. } => bail!("mixture of experts is not yet on wgpu"),
            Op::AllReduce | Op::AllReduceNorm { .. } => bail!("tensor parallel is not on wgpu"),
            Op::Gemv { .. } | Op::GemvStore { .. } | Op::StepAttention | Op::GdnStep => bail!("the single row cuda plan is not on wgpu"),
            Op::HyperExpand | Op::HyperPre { .. } | Op::HyperPost | Op::HyperMean => bail!("hyper-connections are not yet on wgpu"),
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

pub enum StageInput<'a> {
    Tokens(&'a [u32]),
    Hidden(&'a [f32]),
}

pub enum StageOutput {
    Token(u32),
    Hidden(Vec<f32>),
}

fn stage_plan(spec: &Spec, phase: Phase, first: bool, last: bool) -> Plan {
    let mut plan = wgsl::plan(spec, phase);
    if !first {
        plan.head.clear();
    }
    if !last {
        plan.tail.clear();
    }
    plan
}

impl<B: Backend> Dense<B> {
    pub fn load(be: B, spec: &Spec, max_ctx: usize) -> Result<Self> {
        Self::load_stage(be, spec, max_ctx, 1, 0, spec.layers.count)
    }

    pub fn load_stage(mut be: B, spec: &Spec, max_ctx: usize, slots: usize, start: usize, end: usize) -> Result<Self> {
        let spec = &crate::engine::apply_rope_extension(spec)?;
        if let Some(why) = wgsl::unsupported(spec) {
            bail!("{} is not supported on backend wgpu: {why}", spec.name);
        }
        ensure!(start < end && end <= spec.layers.count && slots >= 1, "stage layers {start}..{end} with {slots} slots do not fit the {} layers of {}", spec.layers.count, spec.name);
        let (first, last) = (start == 0, end == spec.layers.count);
        let staged = if first && last { spec.clone() } else { Spec { vision: None, ..spec.stage(start, end) } };
        let limits = be.limits();
        ensure!(limits.max_storage_buffers >= 5, "this adapter allows {} storage buffers per stage, specrt needs 5", limits.max_storage_buffers);
        be.compile(&wgsl::generate(&staged, max_ctx, &limits.target())?)?;
        let checkpoint = Checkpoint::for_repo(&spec.repo)?;
        let d = &spec.dims;
        let a = &spec.attention;
        let (mut embed, mut head, mut final_norm, mut weight_bytes) = (None, None, None, 0);
        if first || last {
            if first {
                let e = upload_rows(&be, checkpoint.embedding_rows(spec)?, d.vocab, "embed")?;
                weight_bytes += e.bytes;
                embed = Some(e);
            }
            if last {
                let (norm, lm_head) = checkpoint.head(spec)?;
                let h = mat(&be, &lm_head, d.vocab, d.hidden, "lm_head")?;
                weight_bytes += h.bytes;
                head = Some(h);
                final_norm = Some(upload_floats(&be, &norm)?);
            }
        }
        let kv_floats = a.kv_heads * max_ctx * a.head_dim;
        let mut layers = Vec::with_capacity(end - start);
        for i in 0..end - start {
            let global = start + i;
            let l = checkpoint.layer_at(&staged, i, global).with_context(|| format!("layer {global}"))?;
            let Mixer::Attention(at) = &l.mixer else { bail!("layer {global} is not attention, which is the only mixer on wgpu") };
            let norm = |v: &Option<Vec<f32>>| -> Result<B::Buffer> { upload_floats(&be, v.as_deref().unwrap_or(&[0.0; 4])) };
            let caches = |be: &B| -> Result<Vec<B::Buffer>> { (0..slots).map(|_| be.alloc(kv_floats as u64 * 4)).collect() };
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
                k_cache: caches(&be)?,
                v_cache: caches(&be)?,
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
            logits: be.alloc(if last { (rows * d.vocab * 4) as u64 } else { 16 }).context("logits buffer")?,
            tokens: be.alloc(((max_ctx + 2) * 4) as u64)?,
        };
        let prefill = stage_plan(&staged, Phase::Prefill, first, last);
        let decode = stage_plan(&staged, Phase::Decode, first, last);
        Ok(Dense { be, m: Model { spec: staged, max_ctx, slots, first, last, embed, layers, final_norm, head, scratch, weight_bytes, prefill, decode } })
    }

    pub fn hidden(&self) -> usize {
        self.m.spec.dims.hidden
    }

    pub fn stage_forward(&mut self, slot: usize, pos0: usize, input: StageInput<'_>, decode: bool, sample: bool) -> Result<StageOutput> {
        let Self { be, m } = self;
        ensure!(slot < m.slots, "slot {slot} out of {}", m.slots);
        let hidden = m.spec.dims.hidden;
        let n = match &input {
            StageInput::Tokens(t) => {
                ensure!(m.first, "this stage has no embedding, send hidden rows");
                t.len()
            }
            StageInput::Hidden(h) => {
                ensure!(!m.first, "this stage owns the embedding, send token ids");
                ensure!(!h.is_empty() && h.len() % hidden == 0, "hidden state of {} floats", h.len());
                h.len() / hidden
            }
        };
        ensure!(n >= 1 && pos0 + n <= m.max_ctx, "{n} rows at position {pos0} exceed max_ctx {}", m.max_ctx);
        let plan = if decode { &m.decode } else { &m.prefill };
        let mut carried = Vec::new();
        let mut token = 0;
        let mut c0 = 0;
        while c0 < n {
            let rows = PREFILL_CHUNK.min(n - c0);
            let last_chunk = c0 + rows == n;
            match &input {
                StageInput::Tokens(t) => be.upload(&m.scratch.tokens, ((pos0 + c0) * 4) as u64, &words(&t[c0..c0 + rows])),
                StageInput::Hidden(h) => be.upload(&m.scratch.x, 0, &floats(&h[c0 * hidden..(c0 + rows) * hidden])),
            }
            let sampled = m.last && sample && last_chunk;
            let head = if sampled { rows - 1..rows } else { 0..0 };
            let exec = Exec { be: RefCell::new(&mut *be), m };
            crate::executor::run_plan(&exec, plan, &Step { slot, pos0: pos0 + c0, rows, head }, &mut |_| Ok(()))?;
            if sampled {
                Self::argmax(be, m, Sampling::GREEDY, 0, m.max_ctx + 1)?;
            }
            be.submit()?;
            if sampled {
                let bytes = be.download(&m.scratch.tokens, ((m.max_ctx + 1) * 4) as u64, 4)?;
                token = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes"));
            } else if !m.last {
                let bytes = be.download(&m.scratch.x, 0, (rows * hidden * 4) as u64)?;
                carried.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))));
            }
            c0 += rows;
        }
        Ok(if m.last { StageOutput::Token(token) } else { StageOutput::Hidden(carried) })
    }

    pub fn stage_role(&self) -> (bool, bool) {
        (self.m.first, self.m.last)
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

    fn run(be: &mut B, m: &Model<B>, plan: &Plan, slot: usize, pos0: usize, rows: usize, head: Range<usize>) -> Result<()> {
        let exec = Exec { be: RefCell::new(be), m };
        crate::executor::run_plan(&exec, plan, &Step { slot, pos0, rows, head }, &mut |_| Ok(()))
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
            Self::run(be, m, &m.prefill, 0, c0, rows, head)?;
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
        Self::run(be, m, &m.decode, 0, pos, 1, 0..1)?;
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
