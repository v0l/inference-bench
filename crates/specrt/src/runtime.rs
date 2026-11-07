use crate::codegen::{Module, PAGE};
use crate::jit::{Jit, default_cache_dir};
use crate::spec::Spec;
use crate::weights::{Matrix, Weights, find_safetensors};
use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, CudaView, LaunchConfig,
    PushKernelArg,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

enum DeviceMatrix {
    Bf16(CudaSlice<u16>),
    Q8 { values: CudaSlice<i8>, scales: CudaSlice<f32> },
}

struct DeviceLayer {
    ln1: CudaSlice<f32>,
    ln2: CudaSlice<f32>,
    qkv: DeviceMatrix,
    qkv_bias: Option<CudaSlice<f32>>,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    o: DeviceMatrix,
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
}

struct Buffers {
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    attn_part: CudaSlice<f32>,
    gate_up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    k_cache: CudaSlice<u16>,
    v_cache: CudaSlice<u16>,
    block_table: CudaSlice<i32>,
    token: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    history: CudaSlice<i32>,
    count: CudaSlice<i32>,
    dummy: CudaSlice<f32>,
    scores: CudaSlice<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub seed: u32,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling { temperature: 0.0, seed: 0 };

    pub fn inverse_temperature(&self) -> f32 {
        if self.temperature > 0.0 { 1.0 / self.temperature } else { 0.0 }
    }
}

struct Planes {
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
}

impl Planes {
    fn new(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        Ok(Planes { hi: stream.alloc_zeros(len)?, lo: stream.alloc_zeros(len)? })
    }

    fn rows(&self, from: usize, to: usize, width: usize) -> (CudaView<'_, u16>, CudaView<'_, u16>) {
        (self.hi.slice(from * width..to * width), self.lo.slice(from * width..to * width))
    }

    fn all(&self) -> (CudaView<'_, u16>, CudaView<'_, u16>) {
        (self.hi.as_view(), self.lo.as_view())
    }
}

enum GemmOut<'a> {
    F32(&'a CudaSlice<f32>),
    Planes(&'a Planes),
}

struct PrefillBuffers {
    tokens: CudaSlice<i32>,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    gate_up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    xn_planes: Planes,
    attn_planes: Planes,
    act_planes: Planes,
    dummy_u16: CudaSlice<u16>,
}

pub struct BatchGeneration {
    pub tokens: Vec<Vec<u32>>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

struct BatchBuffers {
    tokens: CudaSlice<i32>,
    logits: CudaSlice<f32>,
    history: CudaSlice<i32>,
    part_acc: CudaSlice<f32>,
    part_ml: CudaSlice<f32>,
}

pub struct Generation {
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub peak_device_bytes: u64,
}

pub struct Runtime {
    spec: Spec,
    max_ctx: usize,
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    smem: std::collections::BTreeMap<&'static str, u32>,
    embed: CudaSlice<u16>,
    norm: CudaSlice<f32>,
    lm_head: DeviceMatrix,
    layers: Vec<DeviceLayer>,
    buf: Buffers,
    pre: PrefillBuffers,
    chunk: usize,
    use_mma: bool,
    sms: u32,
    layer_base: i32,
    batch: usize,
    pub cache_hit: bool,
    pub jit_key: String,
    pub arch: String,
    pub device_bytes: u64,
    ctx: Arc<CudaContext>,
    baseline_free: usize,
}

fn grid(n: usize, per_block: usize) -> u32 {
    n.div_ceil(per_block) as u32
}

fn cfg(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

const LINEAR_ROWS_PER_BLOCK: usize = 8;
const PREFILL_CHUNK: usize = 2048;
const SHARED_SCORES_MAX_CTX: usize = 4096;
const SCORE_BUDGET_FLOATS: usize = 1 << 26;
const ATTN_SPLITS: usize = 64;
const ATTN_BLOCK: u32 = 128;
const GEMM_TILE: usize = 64;

impl Runtime {
    pub fn load(spec: &Spec, max_ctx: usize) -> Result<Self> {
        Self::load_batched(spec, max_ctx, 1)
    }

    pub fn load_batched(spec: &Spec, max_ctx: usize, batch: usize) -> Result<Self> {
        ensure!(batch >= 1, "batch must be at least 1");
        let load_started = Instant::now();
        let ctx = CudaContext::new(0).context("cuda init")?;
        let free_before = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let jit = Jit::new(ctx.clone(), default_cache_dir())?;
        let module = Module::generate(spec, max_ctx)?;
        let jit_started = Instant::now();
        let loaded = jit.load(&module)?;
        let jit_secs = jit_started.elapsed().as_secs_f64();
        let mut kernels = HashMap::new();
        for &name in &module.kernels {
            let f = loaded.function(name)?;
            if let Some(&bytes) = module.smem.get(name) {
                use cudarc::driver::sys::CUfunction_attribute as A;
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)?;
                f.set_attribute(
                    A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .with_context(|| format!("{name}: {bytes} bytes of shared memory"))?;
            }
            kernels.insert(name, f);
        }
        let smem = module.smem.clone();
        let weights = Weights::load(spec, &find_safetensors(&spec.repo)?)?;
        let up_m = |m: &Matrix| -> Result<DeviceMatrix> {
            Ok(match m {
                Matrix::Bf16(w) => DeviceMatrix::Bf16(stream.clone_htod(w)?),
                Matrix::Q8 { values, scales } => DeviceMatrix::Q8 {
                    values: stream.clone_htod(values)?,
                    scales: stream.clone_htod(scales)?,
                },
            })
        };
        let up_opt = |v: &Option<Vec<f32>>| -> Result<Option<CudaSlice<f32>>> {
            Ok(match v {
                Some(v) => Some(stream.clone_htod(v)?),
                None => None,
            })
        };
        let mut layers = Vec::with_capacity(weights.layers.len());
        for l in &weights.layers {
            layers.push(DeviceLayer {
                ln1: stream.clone_htod(&l.ln1)?,
                ln2: stream.clone_htod(&l.ln2)?,
                qkv: up_m(&l.qkv)?,
                qkv_bias: up_opt(&l.qkv_bias)?,
                q_norm: up_opt(&l.q_norm)?,
                k_norm: up_opt(&l.k_norm)?,
                o: up_m(&l.o)?,
                gate_up: up_m(&l.gate_up)?,
                down: up_m(&l.down)?,
            });
        }
        let d = &spec.dims;
        let pages = max_ctx.div_ceil(PAGE);
        let cache_len = batch
            * spec.layers.count
            * pages
            * spec.attention.kv_heads
            * PAGE
            * spec.attention.head_dim;
        let table: Vec<i32> = (0..pages as i32).rev().collect();
        let score_rows = (SCORE_BUDGET_FLOATS / (spec.attention.heads * max_ctx)).max(1);
        let chunk = max_ctx.min(PREFILL_CHUNK).min(score_rows).max(batch);
        let buf = Buffers {
            x: stream.alloc_zeros(d.hidden)?,
            xn: stream.alloc_zeros(d.hidden)?,
            qkv: stream.alloc_zeros(spec.q_dim() + 2 * spec.kv_dim())?,
            attn: stream.alloc_zeros(spec.q_dim())?,
            attn_part: stream.alloc_zeros(spec.attention.heads * ATTN_SPLITS * (spec.attention.head_dim + 2))?,
            gate_up: stream.alloc_zeros(2 * d.intermediate)?,
            act: stream.alloc_zeros(d.intermediate)?,
            logits: stream.alloc_zeros(d.vocab)?,
            k_cache: stream.alloc_zeros(2 * cache_len)?,
            v_cache: stream.alloc_zeros(2 * cache_len)?,
            block_table: stream.clone_htod(&table)?,
            token: stream.alloc_zeros(1)?,
            pos: stream.alloc_zeros(1)?,
            history: stream.alloc_zeros(max_ctx)?,
            count: stream.alloc_zeros(1)?,
            dummy: stream.alloc_zeros(1)?,
            scores: stream.alloc_zeros(if max_ctx > SHARED_SCORES_MAX_CTX {
                spec.attention.heads * max_ctx * chunk
            } else {
                1
            })?,
        };
        let pre = PrefillBuffers {
            tokens: stream.alloc_zeros(chunk)?,
            x: stream.alloc_zeros(chunk * d.hidden)?,
            xn: stream.alloc_zeros(chunk * d.hidden)?,
            qkv: stream.alloc_zeros(chunk * (spec.q_dim() + 2 * spec.kv_dim()))?,
            attn: stream.alloc_zeros(chunk * spec.q_dim())?,
            gate_up: stream.alloc_zeros(chunk * 2 * d.intermediate)?,
            act: stream.alloc_zeros(chunk * d.intermediate)?,
            xn_planes: Planes::new(&stream, chunk * d.hidden)?,
            attn_planes: Planes::new(&stream, chunk * spec.q_dim())?,
            act_planes: Planes::new(&stream, chunk * d.intermediate)?,
            dummy_u16: stream.alloc_zeros(8)?,
        };
        let rt = Runtime {
            smem,
            pre,
            chunk,
            layer_base: 0,
            batch,
            sms: ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )? as u32,
            use_mma: ctx.compute_capability()?.0 >= 8
                && std::env::var_os("SPECRT_GEMM_FP32").is_none(),
            spec: spec.clone(),
            max_ctx,
            embed: stream.clone_htod(&weights.embed)?,
            norm: stream.clone_htod(&weights.norm)?,
            lm_head: up_m(&weights.lm_head)?,
            layers,
            buf,
            kernels,
            stream,
            cache_hit: loaded.cache_hit,
            jit_key: loaded.key,
            arch: jit.arch().to_string(),
            device_bytes: 0,
            ctx: ctx.clone(),
            baseline_free: free_before,
        };
        rt.stream.synchronize()?;
        let free_after = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        let mut rt = rt;
        rt.device_bytes = free_before.saturating_sub(free_after) as u64;
        tracing::info!(
            spec = %rt.spec.name,
            jit_key = %rt.jit_key,
            jit_cache_hit = rt.cache_hit,
            jit_secs = jit_secs,
            arch = %rt.arch,
            vram_before_bytes = free_before,
            vram_after_bytes = free_after,
            device_bytes = rt.device_bytes,
            load_secs = load_started.elapsed().as_secs_f64(),
            "runtime loaded"
        );
        Ok(rt)
    }

    fn kernel(&self, name: &str) -> &CudaFunction {
        &self.kernels[name]
    }

    fn linear(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel(name));
        match w {
            DeviceMatrix::Bf16(w) => b.arg(w).arg(&self.buf.dummy),
            DeviceMatrix::Q8 { values, scales } => b.arg(values).arg(scales),
        };
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(x).arg(out);
        unsafe { b.launch(cfg(grid(rows, LINEAR_ROWS_PER_BLOCK), 256)) }?;
        Ok(())
    }

    fn linear_norm(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        norm: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel(name));
        match w {
            DeviceMatrix::Bf16(w) => b.arg(w).arg(&self.buf.dummy),
            DeviceMatrix::Q8 { values, scales } => b.arg(values).arg(scales),
        };
        if name != "gate_up_act" {
            b.arg(bias.unwrap_or(&self.buf.dummy));
        }
        b.arg(x).arg(norm).arg(out);
        unsafe { b.launch(cfg(grid(rows, LINEAR_ROWS_PER_BLOCK), 256)) }?;
        Ok(())
    }

    fn rmsnorm(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, out: &CudaSlice<f32>) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel("rmsnorm"));
        b.arg(x).arg(w).arg(out);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn step(&self) -> Result<()> {
        let s = &self.spec;
        let a = &s.attention;
        let buf = &self.buf;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&buf.token).arg(&buf.x);
        unsafe { b.launch(cfg(grid(s.dims.hidden, 256), 256)) }?;
        let qkv_rows = s.q_dim() + 2 * s.kv_dim();
        for (i, l) in self.layers.iter().enumerate() {
            let layer = i as i32;
            self.linear_norm(
                "qkv_proj_norm",
                &l.qkv,
                l.qkv_bias.as_ref(),
                &buf.x,
                &l.ln1,
                &buf.qkv,
                qkv_rows,
            )?;
            let mut b = self.stream.launch_builder(self.kernel("qk_norm_rope"));
            b.arg(&buf.qkv)
                .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer)
                .arg(&0i32);
            unsafe { b.launch(cfg((a.heads + 2 * a.kv_heads) as u32, a.head_dim.max(32) as u32)) }?;
            let mut b = self.stream.launch_builder(self.kernel("attention_split"));
            b.arg(&buf.qkv)
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer)
                .arg(&buf.attn_part);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (a.heads as u32, ATTN_SPLITS as u32, 1),
                    block_dim: (ATTN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let mut b = self.stream.launch_builder(self.kernel("attention_reduce"));
            b.arg(&buf.attn_part).arg(&buf.pos).arg(&buf.attn);
            unsafe { b.launch(cfg(a.heads as u32, a.head_dim.max(32) as u32)) }?;
            self.linear("o_proj", &l.o, None, &buf.attn, &buf.x, s.dims.hidden)?;
            self.linear_norm(
                "gate_up_act",
                &l.gate_up,
                None,
                &buf.x,
                &l.ln2,
                &buf.act,
                s.dims.intermediate,
            )?;
            self.linear("down_proj", &l.down, None, &buf.act, &buf.x, s.dims.hidden)?;
        }
        self.linear_norm(
            "lm_head_norm",
            &self.lm_head,
            None,
            &buf.x,
            &self.norm,
            &buf.logits,
            s.dims.vocab,
        )?;
        Ok(())
    }

    fn gemm(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaView<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel(name));
        match w {
            DeviceMatrix::Bf16(w) => b.arg(w).arg(&self.buf.dummy),
            DeviceMatrix::Q8 { values, scales } => b.arg(values).arg(scales),
        };
        let n = tokens as i32;
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(x).arg(out).arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(rows, GEMM_TILE), grid(tokens, GEMM_TILE), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        use crate::codegen::Tile;
        let big = Tile::big();
        let blocks = |t: Tile| grid(rows, t.bn) * grid(tokens, t.bm);
        let suffix = match std::env::var("SPECRT_TILE").as_deref() {
            Ok("small") => "",
            Ok("big") => "_big",
            Ok("narrow") => "_narrow",
            _ if tokens <= 16 => "_tiny",
            _ if blocks(big) >= 2 * self.sms => "_big",
            _ if blocks(Tile::SMALL) >= 2 * self.sms => "",
            _ => "_narrow",
        };
        let tile = Tile::variant(suffix);
        let kernel = format!("{name}_mma{suffix}");
        let mut b = self.stream.launch_builder(self.kernel(&kernel));
        match w {
            DeviceMatrix::Bf16(w) => b.arg(w).arg(&self.buf.dummy),
            DeviceMatrix::Q8 { values, scales } => b.arg(values).arg(scales),
        };
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let n = tokens as i32;
        b.arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(rows, tile.bn), grid(tokens, tile.bm), 1),
            block_dim: (tile.threads(), 1, 1),
            shared_mem_bytes: self.smem[kernel.as_str()],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn rms_split(&self, x: &CudaView<f32>, w: &CudaSlice<f32>, out: &Planes, tokens: usize) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel("rms_split"));
        b.arg(x).arg(w).arg(&out.hi).arg(&out.lo);
        unsafe { b.launch(cfg(tokens as u32, 256)) }?;
        Ok(())
    }

    fn rmsnorm_rows(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel("rmsnorm"));
        b.arg(x).arg(w).arg(out);
        unsafe { b.launch(cfg(tokens as u32, 1024)) }?;
        Ok(())
    }

    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        start: usize,
        all_logits: Option<&CudaSlice<f32>>,
    ) -> Result<()> {
        let n = tokens.len();
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.stream.memcpy_htod(&ids, &mut self.pre.tokens.slice_mut(..n))?;
        self.stream.memcpy_htod(&[start as i32], &mut self.buf.pos)?;
        let s = &self.spec;
        let a = &s.attention;
        let hidden = s.dims.hidden;
        let qkv_rows = s.q_dim() + 2 * s.kv_dim();
        let pre = &self.pre;
        let buf = &self.buf;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&pre.tokens).arg(&pre.x);
        let config = LaunchConfig {
            grid_dim: (grid(hidden, 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        for (i, l) in self.layers.iter().enumerate() {
            let layer = self.layer_base + i as i32;
            if self.use_mma {
                self.rms_split(&pre.x.as_view(), &l.ln1, &pre.xn_planes, n)?;
                self.tc(
                    "qkv",
                    &l.qkv,
                    l.qkv_bias.as_ref(),
                    pre.xn_planes.all(),
                    GemmOut::F32(&pre.qkv),
                    qkv_rows,
                    n,
                )?;
            } else {
                self.rmsnorm_rows(&pre.x, &l.ln1, &pre.xn, n)?;
                self.gemm(
                    "qkv_gemm",
                    &l.qkv,
                    l.qkv_bias.as_ref(),
                    &pre.xn.as_view(),
                    &pre.qkv,
                    qkv_rows,
                    n,
                )?;
            }
            let rope_kernel = if self.use_mma { "kv_norm_rope" } else { "qk_norm_rope" };
            let rope_heads = if self.use_mma { 2 * a.kv_heads } else { a.heads + 2 * a.kv_heads };
            let mut b = self.stream.launch_builder(self.kernel(rope_kernel));
            b.arg(&pre.qkv)
                .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer);
            if !self.use_mma {
                b.arg(&0i32);
            }
            let config = LaunchConfig {
                grid_dim: (rope_heads as u32, n as u32, 1),
                block_dim: (a.head_dim.max(32) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(config) }?;
            let attn_kernel = if self.use_mma { "flash_mma" } else { "flash_attention" };
            let mut b = self.stream.launch_builder(self.kernel(attn_kernel));
            let ntok = n as i32;
            b.arg(&pre.qkv);
            if self.use_mma {
                b.arg(l.q_norm.as_ref().unwrap_or(&buf.dummy));
            }
            b.arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer);
            if self.use_mma {
                b.arg(&pre.attn_planes.hi).arg(&pre.attn_planes.lo);
            } else {
                b.arg(&pre.attn);
            }
            b.arg(&ntok);
            let rows = if self.use_mma { crate::codegen::FLASH_ROWS } else { 16 };
            let config = LaunchConfig {
                grid_dim: (grid(n, rows), a.heads as u32, 1),
                block_dim: (if self.use_mma { rows as u32 * 2 } else { 128 }, 1, 1),
                shared_mem_bytes: if self.use_mma { self.smem["flash_mma"] } else { 0 },
            };
            unsafe { b.launch(config) }?;
            let inter = s.dims.intermediate;
            if self.use_mma {
                self.tc("o", &l.o, None, pre.attn_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                self.rms_split(&pre.x.as_view(), &l.ln2, &pre.xn_planes, n)?;
                self.tc(
                    "gate_up",
                    &l.gate_up,
                    None,
                    pre.xn_planes.all(),
                    GemmOut::Planes(&pre.act_planes),
                    2 * inter,
                    n,
                )?;
                self.tc("down", &l.down, None, pre.act_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                continue;
            }
            self.gemm("o_gemm", &l.o, None, &pre.attn.as_view(), &pre.x, hidden, n)?;
            self.rmsnorm_rows(&pre.x, &l.ln2, &pre.xn, n)?;
            self.gemm(
                "gate_up_gemm",
                &l.gate_up,
                None,
                &pre.xn.as_view(),
                &pre.gate_up,
                2 * inter,
                n,
            )?;
            let mut b = self.stream.launch_builder(self.kernel("swiglu"));
            b.arg(&pre.gate_up).arg(&pre.act);
            let config = LaunchConfig {
                grid_dim: (grid(inter, 256), n as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(config) }?;
            self.gemm("down_gemm", &l.down, None, &pre.act.as_view(), &pre.x, hidden, n)?;
        }
        if self.use_mma {
            let (out, rows) = match all_logits {
                Some(out) => (out, 0..n),
                None => (&buf.logits, n - 1..n),
            };
            let x = pre.x.slice(rows.start * hidden..rows.end * hidden);
            self.rms_split(&x, &self.norm, &pre.xn_planes, rows.len())?;
            self.tc(
                "lm_head",
                &self.lm_head,
                None,
                pre.xn_planes.rows(0, rows.len(), hidden),
                GemmOut::F32(out),
                s.dims.vocab,
                rows.len(),
            )?;
            return Ok(());
        }
        self.rmsnorm_rows(&pre.x, &self.norm, &pre.xn, n)?;
        match all_logits {
            Some(out) => self.gemm(
                "lm_head_gemm",
                &self.lm_head,
                None,
                &pre.xn.as_view(),
                out,
                s.dims.vocab,
                n,
            )?,
            None => {
                let last = pre.xn.slice((n - 1) * hidden..n * hidden);
                self.gemm("lm_head_gemm", &self.lm_head, None, &last, &buf.logits, s.dims.vocab, 1)?
            }
        }
        Ok(())
    }

    fn prefill(&mut self, tokens: &[u32], all_logits: bool) -> Result<Vec<f32>> {
        let vocab = self.spec.dims.vocab;
        let scratch = if all_logits {
            Some(self.stream.alloc_zeros::<f32>(self.chunk * vocab)?)
        } else {
            None
        };
        let mut out = Vec::new();
        for (c, piece) in tokens.chunks(self.chunk).enumerate() {
            self.prefill_chunk(piece, c * self.chunk, scratch.as_ref())?;
            if let Some(scratch) = &scratch {
                out.extend(self.stream.clone_dtoh(&scratch.slice(..piece.len() * vocab))?);
            }
        }
        Ok(out)
    }

    fn advance(&self, sampling: Sampling) -> Result<()> {
        let buf = &self.buf;
        let mut b = self.stream.launch_builder(self.kernel("argmax_advance"));
        b.arg(&buf.logits).arg(&buf.token).arg(&buf.pos).arg(&buf.history).arg(&buf.count);
        let inv_temp = sampling.inverse_temperature();
        b.arg(&inv_temp).arg(&sampling.seed);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn set(&mut self, token: u32, pos: usize) -> Result<()> {
        self.stream.memcpy_htod(&[token as i32], &mut self.buf.token)?;
        self.stream.memcpy_htod(&[pos as i32], &mut self.buf.pos)?;
        Ok(())
    }

    pub fn prompt_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(tokens.len() <= self.max_ctx, "prompt longer than max_ctx");
        self.prefill(tokens, true)
    }

    fn capture(&self, sampling: Sampling) -> Result<CudaGraph> {
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.step().and_then(|_| self.advance(sampling));
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        graph.context("empty graph")
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
    ) -> Result<Generation> {
        self.generate_with(prompt, new_tokens, sampling, &[], false, &mut |_| true)
    }

    pub fn generate_until(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
        stop: &[u32],
        on_tokens: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Generation> {
        self.generate_with(prompt, new_tokens, sampling, stop, true, on_tokens)
    }

    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
        stop: &[u32],
        chunked: bool,
        on_tokens: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Generation> {
        ensure!(!prompt.is_empty() && new_tokens > 0, "empty prompt or nothing to generate");
        ensure!(
            prompt.len() + new_tokens <= self.max_ctx,
            "prompt plus generation exceeds max_ctx"
        );
        let started = Instant::now();
        self.stream.memcpy_htod(&[0i32], &mut self.buf.count)?;
        self.prefill(prompt, false)?;
        self.stream.memcpy_htod(&[prompt.len() as i32 - 1], &mut self.buf.pos)?;
        self.advance(sampling)?;
        self.stream.synchronize()?;
        let prefill_secs = started.elapsed().as_secs_f64();
        let graph = self.capture(sampling)?;
        graph.upload()?;
        self.stream.synchronize()?;
        let started = Instant::now();
        let mut made = 1;
        let mut tokens: Vec<u32> = Vec::new();
        let mut shown = 0;
        loop {
            let seen = self.stream.clone_dtoh(&self.buf.history.slice(..made))?;
            tokens = seen.iter().map(|&t| t as u32).collect();
            let (kept, stopped) = cut_at_stop(&tokens, stop);
            let mut wanted = true;
            if kept > shown {
                wanted = on_tokens(&tokens[shown..kept]);
                shown = kept;
            }
            if stopped || !wanted || made >= new_tokens {
                tokens.truncate(kept);
                break;
            }
            let step = if chunked { DECODE_CHUNK.min(new_tokens - made) } else { new_tokens - made };
            for _ in 0..step {
                graph.launch()?;
            }
            made += step;
        }
        self.stream.synchronize()?;
        let decode_secs = started.elapsed().as_secs_f64();
        let free_now = self.ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        let peak_device_bytes =
            (self.baseline_free.saturating_sub(free_now) as u64).max(self.device_bytes);
        Ok(Generation { tokens, prefill_secs, decode_secs, peak_device_bytes })
    }

    fn first_token(&self, bb: &BatchBuffers, seq: usize) -> Result<()> {
        let stride = self.max_ctx as i32;
        let mut b = self.stream.launch_builder(self.kernel("argmax_rows"));
        let tokens = bb.tokens.slice(seq..seq + 1);
        let history = bb.history.slice(seq * self.max_ctx..);
        b.arg(&self.buf.logits).arg(&tokens).arg(&history).arg(&self.buf.count).arg(&stride);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn batch_step(&self, bb: &BatchBuffers) -> Result<()> {
        let s = &self.spec;
        let a = &s.attention;
        let n = self.batch;
        let hidden = s.dims.hidden;
        let inter = s.dims.intermediate;
        let pre = &self.pre;
        let buf = &self.buf;
        let seq_layers = s.layers.count as i32;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&bb.tokens).arg(&pre.x);
        let config = LaunchConfig {
            grid_dim: (grid(hidden, 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        for (i, l) in self.layers.iter().enumerate() {
            let layer = i as i32;
            self.rms_split(&pre.x.as_view(), &l.ln1, &pre.xn_planes, n)?;
            let qkv_rows = s.q_dim() + 2 * s.kv_dim();
            self.tc(
                "qkv",
                &l.qkv,
                l.qkv_bias.as_ref(),
                pre.xn_planes.all(),
                GemmOut::F32(&pre.qkv),
                qkv_rows,
                n,
            )?;
            let mut b = self.stream.launch_builder(self.kernel("qk_norm_rope"));
            b.arg(&pre.qkv)
                .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer)
                .arg(&seq_layers);
            let config = LaunchConfig {
                grid_dim: ((a.heads + 2 * a.kv_heads) as u32, n as u32, 1),
                block_dim: (a.head_dim.max(32) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(config) }?;
            let mut b = self.stream.launch_builder(self.kernel("batch_decode_attention"));
            b.arg(&pre.qkv)
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer)
                .arg(&bb.part_acc)
                .arg(&bb.part_ml)
                .arg(&seq_layers);
            let config = LaunchConfig {
                grid_dim: (a.kv_heads as u32, n as u32, crate::codegen::BATCH_DECODE_SPLITS as u32),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(config) }?;
            let mut b = self.stream.launch_builder(self.kernel("batch_decode_combine"));
            b.arg(&bb.part_acc).arg(&bb.part_ml).arg(&pre.attn_planes.hi).arg(&pre.attn_planes.lo);
            let config = LaunchConfig {
                grid_dim: (a.heads as u32, n as u32, 1),
                block_dim: (a.head_dim as u32 / 2, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(config) }?;
            self.tc("o", &l.o, None, pre.attn_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
            self.rms_split(&pre.x.as_view(), &l.ln2, &pre.xn_planes, n)?;
            self.tc(
                "gate_up",
                &l.gate_up,
                None,
                pre.xn_planes.all(),
                GemmOut::Planes(&pre.act_planes),
                2 * inter,
                n,
            )?;
            self.tc("down", &l.down, None, pre.act_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
        }
        self.rms_split(&pre.x.as_view(), &self.norm, &pre.xn_planes, n)?;
        self.tc(
            "lm_head",
            &self.lm_head,
            None,
            pre.xn_planes.all(),
            GemmOut::F32(&bb.logits),
            s.dims.vocab,
            n,
        )?;
        let stride = self.max_ctx as i32;
        let mut b = self.stream.launch_builder(self.kernel("argmax_rows"));
        b.arg(&bb.logits).arg(&bb.tokens).arg(&bb.history).arg(&buf.count).arg(&stride);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        let mut b = self.stream.launch_builder(self.kernel("bump"));
        b.arg(&buf.pos).arg(&buf.count);
        unsafe { b.launch(cfg(1, 1)) }?;
        Ok(())
    }

    pub fn generate_batch(
        &mut self,
        prompts: &[Vec<u32>],
        new_tokens: usize,
    ) -> Result<BatchGeneration> {
        ensure!(self.use_mma, "batch decode needs the tensor core path");
        ensure!(prompts.len() == self.batch, "expected {} prompts", self.batch);
        let len = prompts[0].len();
        ensure!(len > 0 && new_tokens > 0, "empty prompt or nothing to generate");
        ensure!(prompts.iter().all(|p| p.len() == len), "batch prompts must have the same length");
        ensure!(len + new_tokens <= self.max_ctx, "prompt plus generation exceeds max_ctx");
        let parts = self.batch * self.spec.attention.heads * crate::codegen::BATCH_DECODE_SPLITS;
        let bb = BatchBuffers {
            tokens: self.stream.alloc_zeros(self.batch)?,
            logits: self.stream.alloc_zeros(self.batch * self.spec.dims.vocab)?,
            history: self.stream.alloc_zeros(self.batch * self.max_ctx)?,
            part_acc: self.stream.alloc_zeros(parts * self.spec.attention.head_dim)?,
            part_ml: self.stream.alloc_zeros(parts * 2)?,
        };
        let started = Instant::now();
        self.stream.memcpy_htod(&[0i32], &mut self.buf.count)?;
        for (seq, p) in prompts.iter().enumerate() {
            self.layer_base = (seq * self.spec.layers.count) as i32;
            let prefilled = self.prefill(p, false);
            self.layer_base = 0;
            prefilled?;
            self.first_token(&bb, seq)?;
        }
        self.stream.memcpy_htod(&[len as i32], &mut self.buf.pos)?;
        self.stream.memcpy_htod(&[1i32], &mut self.buf.count)?;
        self.stream.synchronize()?;
        let prefill_secs = started.elapsed().as_secs_f64();
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.batch_step(&bb);
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        let graph = graph.context("empty graph")?;
        graph.upload()?;
        self.stream.synchronize()?;
        let started = Instant::now();
        for _ in 1..new_tokens {
            graph.launch()?;
        }
        self.stream.synchronize()?;
        let decode_secs = started.elapsed().as_secs_f64();
        let history = self.stream.clone_dtoh(&bb.history)?;
        let tokens = history
            .chunks(self.max_ctx)
            .map(|h| h[..new_tokens].iter().map(|&t| t as u32).collect())
            .collect();
        Ok(BatchGeneration { tokens, prefill_secs, decode_secs })
    }
}

const DECODE_CHUNK: usize = 16;

pub fn cut_at_stop(tokens: &[u32], stop: &[u32]) -> (usize, bool) {
    match tokens.iter().position(|t| stop.contains(t)) {
        Some(i) => (i, true),
        None => (tokens.len(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::cut_at_stop;

    #[test]
    fn generation_is_cut_before_the_first_stop_token() {
        assert_eq!(cut_at_stop(&[5, 6, 151645, 9, 9], &[151645, 151643]), (2, true));
        assert_eq!(cut_at_stop(&[5, 6], &[151645]), (2, false));
        assert_eq!(cut_at_stop(&[151643], &[151645, 151643]), (0, true));
        assert_eq!(cut_at_stop(&[1, 2, 3], &[]), (3, false));
    }
}
