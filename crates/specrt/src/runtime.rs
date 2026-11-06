use crate::codegen::{Module, PAGE};
use crate::jit::{Jit, default_cache_dir};
use crate::spec::Spec;
use crate::weights::{Matrix, Weights, find_safetensors};
use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
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
    gate_up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    k_cache: CudaSlice<f32>,
    v_cache: CudaSlice<f32>,
    block_table: CudaSlice<i32>,
    token: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    history: CudaSlice<i32>,
    count: CudaSlice<i32>,
    dummy: CudaSlice<f32>,
}

pub struct Generation {
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

pub struct Runtime {
    spec: Spec,
    max_ctx: usize,
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    embed: CudaSlice<u16>,
    norm: CudaSlice<f32>,
    lm_head: DeviceMatrix,
    layers: Vec<DeviceLayer>,
    buf: Buffers,
    pub cache_hit: bool,
    pub jit_key: String,
    pub arch: String,
    pub device_bytes: u64,
}

fn grid(n: usize, per_block: usize) -> u32 {
    n.div_ceil(per_block) as u32
}

fn cfg(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

const LINEAR_ROWS_PER_BLOCK: usize = 8;

impl Runtime {
    pub fn load(spec: &Spec, max_ctx: usize) -> Result<Self> {
        ensure!(max_ctx <= 4096, "max_ctx above 4096 does not fit the attention scratch");
        let ctx = CudaContext::new(0).context("cuda init")?;
        let free_before = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let jit = Jit::new(ctx.clone(), default_cache_dir())?;
        let module = Module::generate(spec, max_ctx)?;
        let loaded = jit.load(&module)?;
        let mut kernels = HashMap::new();
        for &name in &module.kernels {
            kernels.insert(name, loaded.function(name)?);
        }
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
        let cache_len =
            spec.layers.count * pages * spec.attention.kv_heads * PAGE * spec.attention.head_dim;
        let table: Vec<i32> = (0..pages as i32).rev().collect();
        let buf = Buffers {
            x: stream.alloc_zeros(d.hidden)?,
            xn: stream.alloc_zeros(d.hidden)?,
            qkv: stream.alloc_zeros(spec.q_dim() + 2 * spec.kv_dim())?,
            attn: stream.alloc_zeros(spec.q_dim())?,
            gate_up: stream.alloc_zeros(2 * d.intermediate)?,
            act: stream.alloc_zeros(d.intermediate)?,
            logits: stream.alloc_zeros(d.vocab)?,
            k_cache: stream.alloc_zeros(cache_len)?,
            v_cache: stream.alloc_zeros(cache_len)?,
            block_table: stream.clone_htod(&table)?,
            token: stream.alloc_zeros(1)?,
            pos: stream.alloc_zeros(1)?,
            history: stream.alloc_zeros(max_ctx)?,
            count: stream.alloc_zeros(1)?,
            dummy: stream.alloc_zeros(1)?,
        };
        let rt = Runtime {
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
        };
        rt.stream.synchronize()?;
        let free_after = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        let mut rt = rt;
        rt.device_bytes = free_before.saturating_sub(free_after) as u64;
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
            self.rmsnorm(&buf.x, &l.ln1, &buf.xn)?;
            self.linear("qkv_proj", &l.qkv, l.qkv_bias.as_ref(), &buf.xn, &buf.qkv, qkv_rows)?;
            let mut b = self.stream.launch_builder(self.kernel("qk_norm_rope"));
            b.arg(&buf.qkv)
                .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer);
            unsafe { b.launch(cfg((a.heads + 2 * a.kv_heads) as u32, a.head_dim.max(32) as u32)) }?;
            let mut b = self.stream.launch_builder(self.kernel("attention"));
            b.arg(&buf.qkv)
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&buf.block_table)
                .arg(&buf.pos)
                .arg(&layer)
                .arg(&buf.attn);
            unsafe { b.launch(cfg(a.heads as u32, 128)) }?;
            self.linear("o_proj", &l.o, None, &buf.attn, &buf.x, s.dims.hidden)?;
            self.rmsnorm(&buf.x, &l.ln2, &buf.xn)?;
            self.linear(
                "gate_up_proj",
                &l.gate_up,
                None,
                &buf.xn,
                &buf.gate_up,
                2 * s.dims.intermediate,
            )?;
            let mut b = self.stream.launch_builder(self.kernel("swiglu"));
            b.arg(&buf.gate_up).arg(&buf.act);
            unsafe { b.launch(cfg(grid(s.dims.intermediate, 256), 256)) }?;
            self.linear("down_proj", &l.down, None, &buf.act, &buf.x, s.dims.hidden)?;
        }
        self.rmsnorm(&buf.x, &self.norm, &buf.xn)?;
        self.linear("lm_head", &self.lm_head, None, &buf.xn, &buf.logits, s.dims.vocab)?;
        Ok(())
    }

    fn advance(&self) -> Result<()> {
        let buf = &self.buf;
        let mut b = self.stream.launch_builder(self.kernel("argmax_advance"));
        b.arg(&buf.logits).arg(&buf.token).arg(&buf.pos).arg(&buf.history).arg(&buf.count);
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
        let mut out = Vec::with_capacity(tokens.len() * self.spec.dims.vocab);
        for (pos, &t) in tokens.iter().enumerate() {
            self.set(t, pos)?;
            self.step()?;
            out.extend(self.stream.clone_dtoh(&self.buf.logits)?);
        }
        Ok(out)
    }

    fn capture(&self) -> Result<CudaGraph> {
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.step().and_then(|_| self.advance());
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        graph.context("empty graph")
    }

    pub fn generate(&mut self, prompt: &[u32], new_tokens: usize) -> Result<Generation> {
        ensure!(!prompt.is_empty() && new_tokens > 0, "empty prompt or nothing to generate");
        ensure!(
            prompt.len() + new_tokens <= self.max_ctx,
            "prompt plus generation exceeds max_ctx"
        );
        let started = Instant::now();
        self.stream.memcpy_htod(&[0i32], &mut self.buf.count)?;
        for (pos, &t) in prompt.iter().enumerate() {
            self.set(t, pos)?;
            self.step()?;
        }
        self.advance()?;
        self.stream.synchronize()?;
        let prefill_secs = started.elapsed().as_secs_f64();
        let graph = self.capture()?;
        graph.upload()?;
        self.stream.synchronize()?;
        let started = Instant::now();
        for _ in 1..new_tokens {
            graph.launch()?;
        }
        self.stream.synchronize()?;
        let decode_secs = started.elapsed().as_secs_f64();
        let history = self.stream.clone_dtoh(&self.buf.history)?;
        Ok(Generation {
            tokens: history[..new_tokens].iter().map(|&t| t as u32).collect(),
            prefill_secs,
            decode_secs,
        })
    }
}
