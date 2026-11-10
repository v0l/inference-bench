use super::Backend;
use crate::gguf::DeviceRows;
use crate::runtime::{Generation, Sampling};
use crate::spec::Spec;
use crate::weights::{Checkpoint, Matrix, Mixer};
use crate::wgsl::{self, LINEAR_TOKENS, ROW_SPAN};
use anyhow::{Context, Result, bail, ensure};
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
        Ok(Dense { be, m: Model { spec: spec.clone(), max_ctx, embed, layers, final_norm, head, scratch, weight_bytes } })
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

    #[allow(clippy::too_many_arguments)]
    fn linear(be: &mut B, w: &WMat<B>, x: &B::Buffer, y: &B::Buffer, tokens: usize, x_row0: usize, y_row0: usize, accumulate: bool) -> Result<()> {
        let (name, tt) = if tokens == 1 { ("linear", 1) } else { ("linear_t8", LINEAR_TOKENS) };
        let nrows = w.nrows;
        let grid = [nrows.min(ROW_SPAN), nrows.div_ceil(ROW_SPAN), tokens.div_ceil(tt) as u32];
        be.dispatch(name, &[x, y, &w.data, &w.rows], [nrows, w.cols, tokens as u32, x_row0 as u32, y_row0 as u32, accumulate as u32, 0, 0], grid)
    }

    fn norm(be: &mut B, w: &B::Buffer, x: &B::Buffer, out: &B::Buffer, src_row0: usize, dst_row0: usize, rows: usize) -> Result<()> {
        be.dispatch("rmsnorm", &[x, w, out], [src_row0 as u32, dst_row0 as u32, 0, 0, 0, 0, 0, 0], [rows as u32, 1, 1])
    }

    fn forward_layers(be: &mut B, m: &Model<B>, pos0: usize, rows: usize) -> Result<()> {
        let s = &m.scratch;
        let spec = &m.spec;
        let heads = (spec.attention.heads + 2 * spec.attention.kv_heads) as u32;
        for l in &m.layers {
            Self::norm(be, &l.ln1, &s.x, &s.xn, 0, 0, rows)?;
            Self::linear(be, &l.qkv, &s.xn, &s.qkv, rows, 0, 0, false)?;
            if let Some(b) = &l.bias {
                be.dispatch("bias", &[&s.qkv, b], [rows as u32, spec.qkv_rows() as u32, 0, 0, 0, 0, 0, 0], grid_1d(rows * spec.qkv_rows()))?;
            }
            be.dispatch("rope", &[&s.qkv, &l.q_norm, &l.k_norm, &l.k_cache, &l.v_cache], [pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [heads, rows as u32, 1])?;
            be.dispatch("attention", &[&s.qkv, &l.k_cache, &l.v_cache, &s.attn], [pos0 as u32, 0, 0, 0, 0, 0, 0, 0], [spec.attention.heads as u32, rows as u32, 1])?;
            Self::linear(be, &l.o, &s.attn, &s.x, rows, 0, 0, true)?;
            Self::norm(be, &l.ln2, &s.x, &s.xn, 0, 0, rows)?;
            Self::linear(be, &l.gate_up, &s.xn, &s.gate_up, rows, 0, 0, false)?;
            be.dispatch("swiglu", &[&s.gate_up, &s.act], [rows as u32, 0, 0, 0, 0, 0, 0, 0], grid_1d(rows * spec.dims.intermediate))?;
            Self::linear(be, &l.down, &s.act, &s.x, rows, 0, 0, true)?;
        }
        Ok(())
    }

    fn argmax(be: &mut B, m: &Model<B>, sampling: Sampling, step: usize, slot: usize) -> Result<()> {
        let inv = sampling.inverse_temperature().to_bits();
        be.dispatch("argmax", &[&m.scratch.logits, &m.scratch.tokens], [0, inv, sampling.seed, step as u32, slot as u32, 0, 0, 0], [1, 1, 1])
    }

    fn embed(be: &mut B, m: &Model<B>, token0: usize, rows: usize) -> Result<()> {
        be.dispatch("embed", &[&m.scratch.tokens, &m.embed.data, &m.embed.rows, &m.scratch.x], [0, token0 as u32, 0, 0, 0, 0, 0, 0], [rows as u32, 1, 1])
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
            Self::embed(be, m, c0, rows)?;
            Self::forward_layers(be, m, c0, rows)?;
            let last = c0 + rows == n;
            if all_logits {
                Self::norm(be, &m.final_norm, &m.scratch.x, &m.scratch.xn, 0, 0, rows)?;
                Self::linear(be, &m.head, &m.scratch.xn, &m.scratch.logits, rows, 0, 0, false)?;
                be.submit()?;
                let bytes = be.download(&m.scratch.logits, 0, (rows * vocab * 4) as u64)?;
                logits.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))));
            } else if last {
                Self::norm(be, &m.final_norm, &m.scratch.x, &m.scratch.xn, rows - 1, 0, 1)?;
                Self::linear(be, &m.head, &m.scratch.xn, &m.scratch.logits, 1, 0, 0, false)?;
                Self::argmax(be, m, sampling, 0, n)?;
                be.submit()?;
            } else {
                be.submit()?;
            }
            c0 += rows;
        }
        Ok(logits)
    }

    fn decode_step(be: &mut B, m: &Model<B>, pos: usize, sampling: Sampling) -> Result<()> {
        Self::embed(be, m, pos, 1)?;
        Self::forward_layers(be, m, pos, 1)?;
        Self::norm(be, &m.final_norm, &m.scratch.x, &m.scratch.xn, 0, 0, 1)?;
        Self::linear(be, &m.head, &m.scratch.xn, &m.scratch.logits, 1, 0, 0, false)?;
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
