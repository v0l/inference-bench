use crate::codegen::{LA_CHUNK, LA_COLS, LA_KSPLIT, Module, PAGE};
use crate::jit::{Jit, default_cache_dir};
use crate::spec::{LinearAttention, Spec, TileChoice};
use crate::statepool::StatePool;
use crate::weights::{Checkpoint, Matrix, Mixer, peak_rss_bytes};
use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, CudaView, LaunchConfig,
    PushKernelArg,
};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::time::Instant;

mod fused;
mod mtp;
mod step;
pub use fused::PrefillPart;
pub use mtp::{SpecRow, SpecStats};
pub use step::{DecodeRow, GRAPH_SIZES, Stepper, pad_rows};

enum DeviceMatrix {
    Bf16(CudaSlice<u16>),
    Q8 { values: CudaSlice<i8>, scales: CudaSlice<f32> },
    Fp8 { values: CudaSlice<u8>, scales: CudaSlice<f32> },
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8> },
}

impl DeviceMatrix {
    fn upload(stream: &Arc<CudaStream>, m: &Matrix) -> Result<Self> {
        Ok(match m {
            Matrix::Bf16(w) => DeviceMatrix::Bf16(stream.clone_htod(w.as_ref())?),
            Matrix::Q8 { values, scales } => {
                DeviceMatrix::Q8 { values: stream.clone_htod(values)?, scales: stream.clone_htod(scales)? }
            }
            Matrix::Fp8 { values, scales } => {
                DeviceMatrix::Fp8 { values: stream.clone_htod(values.as_ref())?, scales: stream.clone_htod(scales)? }
            }
            Matrix::Fp4(m) => DeviceMatrix::Fp4 {
                packed: stream.clone_htod(m.packed.as_ref())?,
                scales: stream.clone_htod(&m.device_scales())?,
            },
        })
    }

    fn bytes(&self) -> u64 {
        (match self {
            DeviceMatrix::Bf16(w) => w.num_bytes(),
            DeviceMatrix::Q8 { values, scales } => values.num_bytes() + scales.num_bytes(),
            DeviceMatrix::Fp8 { values, scales } => values.num_bytes() + scales.num_bytes(),
            DeviceMatrix::Fp4 { packed, scales } => packed.num_bytes() + scales.num_bytes(),
        }) as u64
    }

    fn args<'a>(&'a self, b: &mut cudarc::driver::LaunchArgs<'a>, dummy: &'a CudaSlice<f32>) {
        match self {
            DeviceMatrix::Bf16(w) => b.arg(w).arg(dummy),
            DeviceMatrix::Q8 { values, scales } => b.arg(values).arg(scales),
            DeviceMatrix::Fp8 { values, scales } => b.arg(values).arg(scales),
            DeviceMatrix::Fp4 { packed, scales } => b.arg(packed).arg(scales),
        };
    }
}

struct DeviceAttn {
    qkv: DeviceMatrix,
    qkv_bias: Option<CudaSlice<f32>>,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    o: DeviceMatrix,
    kv: i32,
}

struct DeviceGdn {
    in_proj: DeviceMatrix,
    ba: DeviceMatrix,
    conv: CudaSlice<f32>,
    neg_a: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    out: DeviceMatrix,
    slot: i32,
}

enum DeviceMixer {
    Attention(DeviceAttn),
    Linear(DeviceGdn),
}

struct DeviceLayer {
    ln1: CudaSlice<f32>,
    ln2: CudaSlice<f32>,
    mixer: DeviceMixer,
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
}

fn opt_bytes(v: &Option<CudaSlice<f32>>) -> u64 {
    v.as_ref().map_or(0, |s| s.num_bytes() as u64)
}

impl DeviceLayer {
    fn bytes(&self) -> u64 {
        let mixer = match &self.mixer {
            DeviceMixer::Attention(a) => {
                a.qkv.bytes()
                    + a.o.bytes()
                    + opt_bytes(&a.qkv_bias)
                    + opt_bytes(&a.q_norm)
                    + opt_bytes(&a.k_norm)
            }
            DeviceMixer::Linear(g) => {
                g.in_proj.bytes()
                    + g.ba.bytes()
                    + g.out.bytes()
                    + (g.conv.num_bytes() + g.neg_a.num_bytes() + g.dt_bias.num_bytes() + g.norm.num_bytes()) as u64
            }
        };
        mixer
            + self.gate_up.bytes()
            + self.down.bytes()
            + (self.ln1.num_bytes() + self.ln2.num_bytes()) as u64
    }
}

struct GdnBuffers {
    la: LinearAttention,
    conv_pool: StatePool,
    ssm_pool: StatePool,
    conv_state: ManuallyDrop<CudaSlice<f32>>,
    ssm_state: ManuallyDrop<CudaSlice<f32>>,
    lin: CudaSlice<f32>,
    ba: CudaSlice<f32>,
    mix: CudaSlice<f32>,
    out: CudaSlice<f32>,
    pre_lin: CudaSlice<f32>,
    pre_ba: CudaSlice<f32>,
    pre_mix: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    g: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    w: CudaSlice<f32>,
    u: CudaSlice<f32>,
    aqk: CudaSlice<f32>,
    gc: CudaSlice<f32>,
    o: CudaSlice<f32>,
    planes: Planes,
}

impl GdnBuffers {
    fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        la: &LinearAttention,
        layers: usize,
        slots: usize,
        chunk: usize,
    ) -> Result<Self> {
        let d = la.key_head_dim;
        let conv_pool = StatePool::new(ctx, layers * la.conv_dim() * (la.conv_kernel - 1) * 4, slots)?;
        let ssm_pool = StatePool::new(ctx, layers * la.value_heads * d * d * 4, slots)?;
        let conv_state = ManuallyDrop::new(conv_pool.slice(stream));
        let ssm_state = ManuallyDrop::new(ssm_pool.slice(stream));
        let chunks = chunk.div_ceil(LA_CHUNK);
        let padded = chunks * LA_CHUNK;
        let zeros = |n: usize| stream.alloc_zeros::<f32>(n);
        Ok(GdnBuffers {
            la: la.clone(),
            conv_pool,
            ssm_pool,
            conv_state,
            ssm_state,
            lin: zeros(la.in_proj_rows())?,
            ba: zeros(la.ba_rows())?,
            mix: zeros(la.conv_dim())?,
            out: zeros(la.value_dim())?,
            pre_lin: zeros(chunk * la.in_proj_rows())?,
            pre_ba: zeros(chunk * la.ba_rows())?,
            pre_mix: zeros(chunk * la.conv_dim())?,
            qn: zeros(chunk * la.key_heads * d)?,
            kn: zeros(chunk * la.key_heads * d)?,
            g: zeros(chunk * la.value_heads)?,
            beta: zeros(chunk * la.value_heads)?,
            w: zeros(padded * la.value_heads * d)?,
            u: zeros(padded * la.value_heads * d)?,
            aqk: zeros(chunks * la.value_heads * LA_CHUNK * LA_CHUNK)?,
            gc: zeros(chunks * la.value_heads * LA_CHUNK)?,
            o: zeros(chunk * la.value_dim())?,
            planes: Planes::new(stream, chunk * la.value_dim())?,
        })
    }
}

impl Drop for GdnBuffers {
    fn drop(&mut self) {
        let conv = unsafe { ManuallyDrop::take(&mut self.conv_state) };
        let ssm = unsafe { ManuallyDrop::take(&mut self.ssm_state) };
        conv.leak();
        ssm.leak();
    }
}

pub struct Trace {
    pub layers: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct LinearStates {
    pub conv: Vec<f32>,
    pub state: Vec<f32>,
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
    zero_slots: CudaSlice<i32>,
    token: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    history: CudaSlice<i32>,
    count: CudaSlice<i32>,
    dummy: CudaSlice<f32>,
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

struct Fp8Act {
    q: CudaSlice<u8>,
    scale: CudaSlice<f32>,
}

struct Fp4Act {
    q: CudaSlice<u8>,
    scales: CudaSlice<u8>,
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
    pos: CudaSlice<i32>,
    slots: CudaSlice<i32>,
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
    gdn: Option<GdnBuffers>,
    fp8_act: Option<Fp8Act>,
    fp4_act: Option<Fp4Act>,
    f4b_part: CudaSlice<f32>,
    f8b_part: CudaSlice<f32>,
    chunk: usize,
    prefill_limit: usize,
    use_mma: bool,
    sms: u32,
    seq: i32,
    batch: usize,
    slots: usize,
    pages: usize,
    kv_blocks: usize,
    tables: Vec<i32>,
    pub cache_hit: bool,
    pub jit_key: String,
    pub arch: String,
    pub device_bytes: u64,
    pub weight_bytes: u64,
    pub weight_secs: f64,
    pub load_secs: f64,
    pub peak_host_rss_bytes: u64,
    ctx: Arc<CudaContext>,
    baseline_free: usize,
    mtp: Option<mtp::MtpState>,
}

fn grid(n: usize, per_block: usize) -> u32 {
    n.div_ceil(per_block) as u32
}

fn cfg(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

const LINEAR_ROWS_PER_BLOCK: usize = 8;
const FP4_GEMV_THREADS: u32 = 512;

fn gemv_cfg(name: &str, rows: usize) -> LaunchConfig {
    if name == "lm_head_norm_f4" {
        let per_block = FP4_GEMV_THREADS as usize / 32 * crate::codegen::fp4_head_rows(rows);
        cfg(grid(rows, per_block), FP4_GEMV_THREADS)
    } else if name.ends_with("_f4") {
        cfg(grid(rows, FP4_GEMV_THREADS as usize / 32), FP4_GEMV_THREADS)
    } else {
        cfg(grid(rows, LINEAR_ROWS_PER_BLOCK), 256)
    }
}
const PREFILL_CHUNK: usize = 2048;
const SERVING_RESIDENT_SLOTS: usize = 8;
const STATE_FREE_MARGIN: usize = 256 << 20;
const ATTN_SPLITS: usize = 64;
const ATTN_BLOCK: u32 = 128;
const GEMM_TILE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub max_ctx: usize,
    pub slots: usize,
    pub kv_blocks: usize,
    pub prefill_chunk: usize,
    pub kv_planes: usize,
    pub kv_headroom: usize,
    pub state_on_demand: bool,
}

impl Layout {
    pub fn contiguous(max_ctx: usize, slots: usize) -> Layout {
        Layout {
            max_ctx,
            slots,
            kv_blocks: slots * max_ctx.div_ceil(PAGE),
            prefill_chunk: PREFILL_CHUNK,
            kv_planes: 2,
            kv_headroom: 0,
            state_on_demand: false,
        }
    }

    pub fn serving(spec: &Spec, max_ctx: usize, slots: usize, kv_bytes: Option<usize>) -> Layout {
        Layout {
            max_ctx,
            slots,
            kv_blocks: kv_bytes.map_or(0, |b| (b / kv_block_bytes(spec, 1)).max(1)),
            prefill_chunk: PREFILL_CHUNK,
            kv_planes: 1,
            kv_headroom: (1 << 30) + slots.min(SERVING_RESIDENT_SLOTS) * crate::serve::slot_state_bytes(spec) as usize,
            state_on_demand: true,
        }
    }

    pub fn pages(&self) -> usize {
        self.max_ctx.div_ceil(PAGE)
    }
}

pub fn kv_block_bytes(spec: &Spec, planes: usize) -> usize {
    2 * spec.kv_layers() * spec.attention.kv_heads * PAGE * spec.attention.head_dim * planes * 2
}

impl Runtime {
    pub fn load(spec: &Spec, max_ctx: usize) -> Result<Self> {
        Self::load_batched(spec, max_ctx, 1)
    }

    pub fn load_batched(spec: &Spec, max_ctx: usize, batch: usize) -> Result<Self> {
        ensure!(batch >= 1, "batch must be at least 1");
        let mut rt = Self::load_layout(spec, Layout::contiguous(max_ctx, batch))?;
        rt.batch = batch;
        let pages = rt.pages;
        for slot in 0..batch {
            let blocks: Vec<u32> = (0..pages as u32).rev().map(|p| (slot * pages) as u32 + p).collect();
            rt.set_block_table(slot, &blocks)?;
        }
        Ok(rt)
    }

    pub fn load_layout(spec: &Spec, layout: Layout) -> Result<Self> {
        let resolved = spec.with_env_overrides();
        let spec = &resolved;
        ensure!(layout.slots >= 1, "need at least one slot");
        let max_ctx = layout.max_ctx;
        let batch = 1;
        let load_started = Instant::now();
        let ctx = CudaContext::new(0).context("cuda init")?;
        let free_before = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let jit = Jit::new(ctx.clone(), spec.runtime.cache_dir.clone().unwrap_or_else(default_cache_dir))?;
        let module = Module::generate_kv(spec, max_ctx, layout.kv_planes)?;
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
        let checkpoint = Checkpoint::for_repo(&spec.repo)?;
        let weights_started = Instant::now();
        let up_m = |m: &Matrix| DeviceMatrix::upload(&stream, m);
        let up_opt = |v: &Option<Vec<f32>>| -> Result<Option<CudaSlice<f32>>> {
            Ok(match v {
                Some(v) => Some(stream.clone_htod(v)?),
                None => None,
            })
        };
        let mut layers = Vec::with_capacity(spec.layers.count);
        let (mut kv, mut slot) = (0i32, 0i32);
        for i in 0..spec.layers.count {
            let l = checkpoint.layer(spec, i).with_context(|| format!("layer {i}"))?;
            let mixer = match &l.mixer {
                Mixer::Attention(a) => {
                    kv += 1;
                    DeviceMixer::Attention(DeviceAttn {
                        qkv: up_m(&a.qkv)?,
                        qkv_bias: up_opt(&a.qkv_bias)?,
                        q_norm: up_opt(&a.q_norm)?,
                        k_norm: up_opt(&a.k_norm)?,
                        o: up_m(&a.o)?,
                        kv: kv - 1,
                    })
                }
                Mixer::Linear(g) => {
                    slot += 1;
                    DeviceMixer::Linear(DeviceGdn {
                        in_proj: up_m(&g.in_proj)?,
                        ba: up_m(&g.ba)?,
                        conv: stream.clone_htod(&g.conv)?,
                        neg_a: stream.clone_htod(&g.neg_a)?,
                        dt_bias: stream.clone_htod(&g.dt_bias)?,
                        norm: stream.clone_htod(&g.norm)?,
                        out: up_m(&g.out)?,
                        slot: slot - 1,
                    })
                }
            };
            layers.push(DeviceLayer {
                ln1: stream.clone_htod(&l.ln1)?,
                ln2: stream.clone_htod(&l.ln2)?,
                mixer,
                gate_up: up_m(&l.gate_up)?,
                down: up_m(&l.down)?,
            });
            stream.synchronize()?;
            drop(l);
            checkpoint.release();
        }
        let outside = checkpoint.outside(spec)?;
        let embed = stream.clone_htod(outside.embed.as_ref())?;
        let norm = stream.clone_htod(&outside.norm)?;
        let lm_head = up_m(&outside.lm_head)?;
        stream.synchronize()?;
        drop(outside);
        checkpoint.release();
        let mtp_weights = mtp::DeviceMtp::load(&stream, &checkpoint, spec)?;
        checkpoint.release();
        let weight_secs = weights_started.elapsed().as_secs_f64();
        let weight_bytes = layers.iter().map(DeviceLayer::bytes).sum::<u64>()
            + (embed.num_bytes() + norm.num_bytes()) as u64
            + lm_head.bytes()
            + mtp_weights.as_ref().map_or(0, mtp::DeviceMtp::bytes);
        let d = &spec.dims;
        let pages = layout.pages();
        let block_elems = kv_block_bytes(spec, layout.kv_planes) / 4;
        let prefill_limit = max_ctx.min(layout.prefill_chunk).max(layout.slots);
        let chunk = prefill_limit + crate::runtime::step::row_capacity(layout.slots);
        let use_mma = ctx.compute_capability()?.0 >= 8 && !spec.runtime.gemm_fp32;
        ensure!(
            use_mma || spec.lm_head_quant() != crate::spec::Quant::Nvfp4G16,
            "head_quant = \"nvfp4_g16\" needs the tensor-core prefill path (an Ampere or newer GPU and [runtime] gemm_fp32 = false)"
        );
        let mma_only = |n: usize| if use_mma { 1 } else { n };
        let buf = Buffers {
            x: stream.alloc_zeros(d.hidden)?,
            xn: stream.alloc_zeros(d.hidden)?,
            qkv: stream.alloc_zeros(spec.qkv_rows())?,
            attn: stream.alloc_zeros(spec.q_dim())?,
            attn_part: stream.alloc_zeros(spec.attention.heads * ATTN_SPLITS * (spec.attention.head_dim + 2))?,
            gate_up: stream.alloc_zeros(2 * d.intermediate)?,
            act: stream.alloc_zeros(d.intermediate)?,
            logits: stream.alloc_zeros(d.vocab)?,
            k_cache: stream.alloc_zeros(1)?,
            v_cache: stream.alloc_zeros(1)?,
            block_table: stream.alloc_zeros((layout.slots + 1) * pages)?,
            zero_slots: stream.alloc_zeros(chunk)?,
            token: stream.alloc_zeros(1)?,
            pos: stream.alloc_zeros(1)?,
            history: stream.alloc_zeros(max_ctx)?,
            count: stream.alloc_zeros(1)?,
            dummy: stream.alloc_zeros(1)?,
        };
        let pre = PrefillBuffers {
            tokens: stream.alloc_zeros(chunk)?,
            x: stream.alloc_zeros(chunk * d.hidden)?,
            xn: stream.alloc_zeros(mma_only(chunk * d.hidden))?,
            qkv: stream.alloc_zeros(chunk * spec.qkv_rows())?,
            attn: stream.alloc_zeros(mma_only(chunk * spec.q_dim()))?,
            gate_up: stream.alloc_zeros(mma_only(chunk * 2 * d.intermediate))?,
            act: stream.alloc_zeros(mma_only(chunk * d.intermediate))?,
            xn_planes: Planes::new(&stream, chunk * d.hidden)?,
            attn_planes: Planes::new(&stream, chunk * spec.q_dim())?,
            act_planes: Planes::new(&stream, chunk * d.intermediate)?,
            dummy_u16: stream.alloc_zeros(8)?,
        };
        let gdn = match &spec.linear_attention {
            Some(la) => Some(GdnBuffers::new(&ctx, &stream, la, spec.linear_layers(), layout.slots + 1, chunk)?),
            None => None,
        };
        ensure!(use_mma || gdn.is_none(), "linear attention needs the tensor core path (sm_80 or newer)");
        let widest = [d.hidden, d.intermediate, spec.q_dim()]
            .into_iter()
            .chain(spec.linear_attention.as_ref().map(|l| l.value_dim()))
            .max()
            .unwrap_or(d.hidden);
        let fp8_act = if spec.quant.is_fp8() && crate::codegen::fp8_act_enabled(spec) {
            Some(Fp8Act {
                q: stream.alloc_zeros(chunk * widest)?,
                scale: stream.alloc_zeros(chunk * widest / crate::codegen::FP8_BLOCK)?,
            })
        } else {
            None
        };
        let fp4_act = if spec.mlp.quant == Some(crate::spec::Quant::Nvfp4G16) && crate::codegen::fp4_act_enabled(spec) {
            let widest = d.hidden.max(d.intermediate);
            Some(Fp4Act {
                q: stream.alloc_zeros(chunk * widest / 2)?,
                scales: stream.alloc_zeros(chunk * widest / crate::nvfp4::GROUP)?,
            })
        } else {
            None
        };
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let f4b_part = if spec.mlp.quant == Some(crate::spec::Quant::Nvfp4G16) {
            use crate::codegen::{F4B_MAX_TOKENS, f4b_splits};
            let (h, i) = (d.hidden, 2 * d.intermediate);
            let most = (f4b_splits(i, h, sms as usize) * i).max(f4b_splits(h, d.intermediate, sms as usize) * h);
            stream.alloc_zeros(most * F4B_MAX_TOKENS)?
        } else {
            stream.alloc_zeros(1)?
        };
        let f8b_part = if spec.quant.is_fp8() {
            use crate::codegen::{F8B_MAX_TOKENS, f8b_splits};
            let sms = sms as usize;
            let mut shapes = vec![
                (spec.qkv_rows(), d.hidden),
                (d.hidden, spec.q_dim()),
                (2 * d.intermediate, d.hidden),
                (d.hidden, d.intermediate),
            ];
            if let Some(la) = &spec.linear_attention {
                shapes.push((la.in_proj_rows(), d.hidden));
                shapes.push((d.hidden, la.value_dim()));
            }
            let most = shapes.into_iter().map(|(n, k)| f8b_splits(n, k, sms) * n).max().unwrap_or(1);
            stream.alloc_zeros(most * F8B_MAX_TOKENS)?
        } else {
            stream.alloc_zeros(1)?
        };
        let mut rt = Runtime {
            smem,
            pre,
            gdn,
            fp8_act,
            fp4_act,
            f4b_part,
            f8b_part,
            chunk,
            prefill_limit,
            seq: 0,
            batch,
            slots: layout.slots,
            pages,
            kv_blocks: layout.kv_blocks,
            tables: vec![0; (layout.slots + 1) * pages],
            sms,
            use_mma,
            spec: spec.clone(),
            max_ctx,
            embed,
            norm,
            lm_head,
            layers,
            buf,
            kernels,
            stream,
            cache_hit: loaded.cache_hit,
            jit_key: loaded.key,
            arch: jit.arch().to_string(),
            device_bytes: 0,
            weight_bytes,
            weight_secs,
            load_secs: 0.0,
            peak_host_rss_bytes: 0,
            ctx: ctx.clone(),
            baseline_free: free_before,
            mtp: None,
        };
        if let Some(w) = mtp_weights {
            rt.mtp = Some(mtp::MtpState::new(&rt, w)?);
        }
        rt.stream.synchronize()?;
        let kv_blocks = match layout.kv_blocks {
            0 => {
                let free = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(0);
                let spare = free.saturating_sub(layout.kv_headroom);
                ensure!(spare >= 2 * block_elems * 4, "no device memory left for the KV cache ({free} bytes free)");
                spare / (block_elems * 4)
            }
            n => n,
        };
        rt.buf.k_cache = rt.stream.alloc_zeros((kv_blocks + 1) * block_elems).context("KV cache")?;
        rt.buf.v_cache = rt.stream.alloc_zeros((kv_blocks + 1) * block_elems).context("KV cache")?;
        rt.kv_blocks = kv_blocks;
        let pad_row = layout.slots * pages;
        rt.tables[pad_row] = kv_blocks as i32;
        rt.stream.memcpy_htod(&[kv_blocks as i32], &mut rt.buf.block_table.slice_mut(pad_row..pad_row + 1))?;
        rt.acquire_state(layout.slots)?;
        if !layout.state_on_demand {
            for slot in 0..layout.slots {
                rt.acquire_state(slot)?;
            }
        }
        rt.stream.synchronize()?;
        let free_after = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        rt.device_bytes = free_before.saturating_sub(free_after) as u64;
        rt.load_secs = load_started.elapsed().as_secs_f64();
        rt.peak_host_rss_bytes = peak_rss_bytes().unwrap_or(0);
        tracing::info!(
            spec = %rt.spec.name,
            jit_key = %rt.jit_key,
            jit_cache_hit = rt.cache_hit,
            jit_secs = jit_secs,
            arch = %rt.arch,
            shards = checkpoint.files.len(),
            weight_bytes = rt.weight_bytes,
            weight_secs = rt.weight_secs,
            peak_host_rss_bytes = rt.peak_host_rss_bytes,
            vram_before_bytes = free_before,
            vram_after_bytes = free_after,
            device_bytes = rt.device_bytes,
            load_secs = rt.load_secs,
            "runtime loaded"
        );
        Ok(rt)
    }

    fn kernel(&self, name: &str) -> &CudaFunction {
        &self.kernels[name]
    }

    fn slot_table(&self, slot: usize) -> CudaView<'_, i32> {
        self.buf.block_table.slice(slot * self.pages..(slot + 1) * self.pages)
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    pub fn kv_blocks(&self) -> usize {
        self.kv_blocks
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn set_block_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        ensure!(blocks.len() <= self.pages, "{} blocks for a table of {}", blocks.len(), self.pages);
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "block outside the pool");
        let row = &mut self.tables[slot * self.pages..(slot + 1) * self.pages];
        let old = row[..blocks.len()].to_vec();
        for (dst, &b) in row.iter_mut().zip(blocks) {
            *dst = b as i32;
        }
        let changed = old.iter().zip(blocks).position(|(&o, &b)| o != b as i32);
        let Some(from) = changed else { return Ok(()) };
        let at = slot * self.pages + from;
        let host = self.tables[at..slot * self.pages + blocks.len()].to_vec();
        self.stream.memcpy_htod(&host, &mut self.buf.block_table.slice_mut(at..at + host.len()))?;
        Ok(())
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
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(x).arg(out);
        unsafe { b.launch(gemv_cfg(name, rows)) }?;
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
        w.args(&mut b, &self.buf.dummy);
        if !name.starts_with("gate_up_act") {
            b.arg(bias.unwrap_or(&self.buf.dummy));
        }
        b.arg(x).arg(norm).arg(out);
        unsafe { b.launch(gemv_cfg(name, rows)) }?;
        Ok(())
    }

    fn rmsnorm(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, out: &CudaSlice<f32>) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel("rmsnorm"));
        b.arg(x).arg(w).arg(out);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn step(&self) -> Result<()> {
        self.step_traced(None)
    }

    fn mlp_decode(&self, l: &DeviceLayer) -> Result<()> {
        let s = &self.spec;
        let buf = &self.buf;
        let fp4 = matches!(l.gate_up, DeviceMatrix::Fp4 { .. });
        let (act, down) = if fp4 { ("gate_up_act_f4", "down_proj_f4") } else { ("gate_up_act", "down_proj") };
        self.linear_norm(act, &l.gate_up, None, &buf.x, &l.ln2, &buf.act, s.dims.intermediate)?;
        self.linear(down, &l.down, None, &buf.act, &buf.x, s.dims.hidden)
    }

    fn trace_layer(&self, trace: Option<&CudaSlice<f32>>, layer: usize, tokens: usize) -> Result<()> {
        let Some(trace) = trace else { return Ok(()) };
        let n = tokens * self.spec.dims.hidden;
        let src = if tokens == 1 { self.buf.x.slice(..n) } else { self.pre.x.slice(..n) };
        let dst = trace.slice(layer * n..(layer + 1) * n);
        self.copy_rows(&src, &dst, n)
    }

    fn copy_rows(&self, src: &CudaView<f32>, dst: &CudaView<f32>, n: usize) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel("copy_rows"));
        let count = n as i32;
        b.arg(src).arg(dst).arg(&count);
        unsafe { b.launch(cfg(grid(n, 256).min(1024), 256)) }?;
        Ok(())
    }

    fn gdn_decode(&self, g: &DeviceGdn, ln1: &CudaSlice<f32>, x: &CudaSlice<f32>) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        self.linear_norm("lin_in_norm", &g.in_proj, None, x, ln1, &gb.lin, la.in_proj_rows())?;
        self.linear_norm("lin_ba_norm", &g.ba, None, x, ln1, &gb.ba, la.ba_rows())?;
        let zero = 0i32;
        let mut b = self.stream.launch_builder(self.kernel("gdn_conv_step"));
        b.arg(&gb.lin).arg(&g.conv).arg(&*gb.conv_state).arg(&gb.mix).arg(&g.slot).arg(&zero).arg(&self.buf.zero_slots);
        unsafe { b.launch(cfg(grid(la.conv_dim(), 256), 256)) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_step"));
        b.arg(&gb.mix)
            .arg(&gb.lin)
            .arg(&gb.ba)
            .arg(&g.neg_a)
            .arg(&g.dt_bias)
            .arg(&g.norm)
            .arg(&*gb.ssm_state)
            .arg(&gb.out)
            .arg(&g.slot)
            .arg(&zero)
            .arg(&self.buf.zero_slots);
        unsafe { b.launch(cfg(la.value_heads as u32, (la.value_head_dim * LA_KSPLIT) as u32)) }?;
        self.linear("lin_out_proj", &g.out, None, &gb.out, x, self.spec.dims.hidden)
    }

    fn step_traced(&self, trace: Option<&CudaSlice<f32>>) -> Result<()> {
        let s = &self.spec;
        let a = &s.attention;
        let buf = &self.buf;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&buf.token).arg(&buf.x);
        unsafe { b.launch(cfg(grid(s.dims.hidden, 256), 256)) }?;
        let qkv_rows = s.qkv_rows();
        for (i, layer) in self.layers.iter().enumerate() {
            let l = match &layer.mixer {
                DeviceMixer::Linear(g) => {
                    self.gdn_decode(g, &layer.ln1, &buf.x)?;
                    self.mlp_decode(layer)?;
                    self.trace_layer(trace, i, 1)?;
                    continue;
                }
                DeviceMixer::Attention(l) => l,
            };
            let kv_layer = l.kv;
            let table = self.slot_table(self.seq as usize);
            self.linear_norm(
                "qkv_proj_norm",
                &l.qkv,
                l.qkv_bias.as_ref(),
                &buf.x,
                &layer.ln1,
                &buf.qkv,
                qkv_rows,
            )?;
            let mut b = self.stream.launch_builder(self.kernel("qk_norm_rope"));
            b.arg(&buf.qkv)
                .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&table)
                .arg(&buf.pos)
                .arg(&buf.zero_slots)
                .arg(&kv_layer)
                .arg(&0i32);
            unsafe { b.launch(cfg((a.heads + 2 * a.kv_heads) as u32, a.head_dim.max(32) as u32)) }?;
            let mut b = self.stream.launch_builder(self.kernel("attention_split"));
            b.arg(&buf.qkv)
                .arg(&buf.k_cache)
                .arg(&buf.v_cache)
                .arg(&table)
                .arg(&buf.pos)
                .arg(&kv_layer)
                .arg(&buf.attn_part);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (a.heads as u32, ATTN_SPLITS as u32, 1),
                    block_dim: (ATTN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let mut b = self.stream.launch_builder(self.kernel("attention_reduce"));
            b.arg(&buf.attn_part).arg(&buf.pos).arg(&buf.qkv).arg(&buf.attn);
            unsafe { b.launch(cfg(a.heads as u32, a.head_dim.max(32) as u32)) }?;
            self.linear("o_proj", &l.o, None, &buf.attn, &buf.x, s.dims.hidden)?;
            self.mlp_decode(layer)?;
            self.trace_layer(trace, i, 1)?;
        }
        let head_kernel = if matches!(self.lm_head, DeviceMatrix::Fp4 { .. }) { "lm_head_norm_f4" } else { "lm_head_norm" };
        self.linear_norm(
            head_kernel,
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
        w.args(&mut b, &self.buf.dummy);
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
        let opts = &self.spec.runtime;
        let big = Tile::big(opts);
        let blocks = |t: Tile| grid(rows, t.bn) * grid(tokens, t.bm);
        let suffix = match opts.tile {
            TileChoice::Small => "",
            TileChoice::Big => "_big",
            TileChoice::Narrow => "_narrow",
            _ if tokens <= 16 => "_tiny",
            _ if blocks(big) >= 2 * self.sms => "_big",
            _ if blocks(Tile::SMALL) >= 2 * self.sms => "",
            _ => "_narrow",
        };
        let tile = Tile::variant(suffix, opts);
        let act_quant = name != "lm_head" && suffix != "_tiny";
        if let (Some(fa), DeviceMatrix::Fp8 { .. }, true) = (&self.fp8_act, w, act_quant) {
            return self.tc_fp8(fa, name, w, bias, x, out, rows, tokens, suffix, tile);
        }
        if let (Some(fa), DeviceMatrix::Fp4 { .. }, true) = (&self.fp4_act, w, act_quant) {
            return self.tc_fp4(fa, name, w, x, out, rows, tokens, suffix, tile);
        }
        let f8b = suffix == "_tiny" && tokens <= crate::codegen::F8B_MAX_TOKENS && crate::codegen::F8B_LINEARS.contains(&name);
        if let (DeviceMatrix::Fp8 { .. }, true) = (w, f8b) {
            return self.tc_f8b(name, w, bias, x, out, rows, tokens);
        }
        let batched_f4 = name != "lm_head" && suffix == "_tiny" && tokens <= crate::codegen::F4B_MAX_TOKENS;
        if let (DeviceMatrix::Fp4 { .. }, true) = (w, batched_f4) {
            return self.tc_f4b(name, w, x, out, rows, tokens);
        }
        let kernel = match w {
            DeviceMatrix::Fp4 { .. } => format!("{name}_f4w{suffix}"),
            _ => format!("{name}_mma{suffix}"),
        };
        let mut b = self.stream.launch_builder(self.kernel(&kernel));
        w.args(&mut b, &self.buf.dummy);
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

    #[allow(clippy::too_many_arguments)]
    fn tc_f8b(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        use crate::codegen::{F8B_REDUCE_BLOCK, F8B_ROWS, F8B_WARPS, f8b_smem, f8b_splits, f8b_tokens};
        let DeviceMatrix::Fp8 { values, .. } = w else { unreachable!("tc_f8b takes FP8 weights") };
        let width = values.len() / rows;
        let splits = f8b_splits(rows, width, self.sms as usize);
        let (n, s) = (tokens as i32, splits as i32);
        let kernel = format!("{name}_f8b{}", f8b_tokens(tokens));
        let mut b = self.stream.launch_builder(self.kernel(&kernel));
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        b.arg(&self.f8b_part).arg(&n).arg(&s);
        let config = LaunchConfig {
            grid_dim: (grid(rows, F8B_ROWS), splits as u32, 1),
            block_dim: ((F8B_WARPS * 32) as u32, 1, 1),
            shared_mem_bytes: f8b_smem(tokens),
        };
        unsafe { b.launch(config) }?;
        if splits == 1 {
            return Ok(());
        }
        let reduce = format!("{name}_f8b_reduce");
        let mut b = self.stream.launch_builder(self.kernel(&reduce));
        b.arg(&self.f8b_part).arg(bias.unwrap_or(&self.buf.dummy));
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let outs = match out {
            GemmOut::Planes(_) => tokens * rows / 2,
            GemmOut::F32(_) => tokens * rows,
        };
        b.arg(&n).arg(&s);
        unsafe { b.launch(cfg(grid(outs, F8B_REDUCE_BLOCK), F8B_REDUCE_BLOCK as u32)) }?;
        Ok(())
    }

    fn tc_f4b(
        &self,
        name: &str,
        w: &DeviceMatrix,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        use crate::codegen::{F4B_REDUCE_BLOCK, F4B_ROWS, f4b_smem, f4b_splits, f4b_tokens};
        let DeviceMatrix::Fp4 { packed, .. } = w else { unreachable!("tc_f4b takes NVFP4 weights") };
        let width = packed.len() * 2 / rows;
        let splits = f4b_splits(rows, width, self.sms as usize);
        let (n, s) = (tokens as i32, splits as i32);
        let kernel = format!("{name}_f4b{}", f4b_tokens(tokens));
        let mut b = self.stream.launch_builder(self.kernel(&kernel));
        w.args(&mut b, &self.buf.dummy);
        b.arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        b.arg(&self.f4b_part).arg(&n).arg(&s);
        let config = LaunchConfig {
            grid_dim: (grid(rows, F4B_ROWS), splits as u32, 1),
            block_dim: ((crate::codegen::F4B_WARPS * 32) as u32, 1, 1),
            shared_mem_bytes: f4b_smem(tokens),
        };
        unsafe { b.launch(config) }?;
        if splits == 1 {
            return Ok(());
        }
        let reduce = format!("{name}_f4b_reduce");
        let mut b = self.stream.launch_builder(self.kernel(&reduce));
        b.arg(&self.f4b_part);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let outs = match out {
            GemmOut::Planes(_) => tokens * rows / 2,
            GemmOut::F32(_) => tokens * rows,
        };
        b.arg(&n).arg(&s);
        unsafe { b.launch(cfg(grid(outs, F4B_REDUCE_BLOCK), F4B_REDUCE_BLOCK as u32)) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc_fp4(
        &self,
        fa: &Fp4Act,
        name: &str,
        w: &DeviceMatrix,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
        suffix: &str,
        tile: crate::codegen::Tile,
    ) -> Result<()> {
        let DeviceMatrix::Fp4 { packed, scales } = w else { unreachable!("tc_fp4 takes NVFP4 weights") };
        let width = packed.len() * 2 / rows;
        let width_i = width as i32;
        let groups = width / crate::nvfp4::GROUP;
        let mut b = self.stream.launch_builder(self.kernel("quant_rows_fp4"));
        b.arg(&x.0).arg(&x.1).arg(scales).arg(&fa.q).arg(&fa.scales).arg(&width_i);
        let config = LaunchConfig {
            grid_dim: (grid(groups, 128), tokens as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let kernel = format!("{name}_f4a{suffix}");
        let mut b = self.stream.launch_builder(self.kernel(&kernel));
        b.arg(packed).arg(scales).arg(&fa.q).arg(&fa.scales);
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

    #[allow(clippy::too_many_arguments)]
    fn tc_fp8(
        &self,
        fa: &Fp8Act,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
        suffix: &str,
        tile: crate::codegen::Tile,
    ) -> Result<()> {
        let width = match w {
            DeviceMatrix::Fp8 { values, .. } => values.len() / rows,
            _ => unreachable!("tc_fp8 takes FP8 weights"),
        };
        let width_i = width as i32;
        let mut b = self.stream.launch_builder(self.kernel("quant_rows_fp8"));
        b.arg(&x.0).arg(&x.1).arg(&fa.q).arg(&fa.scale).arg(&width_i);
        let config = LaunchConfig {
            grid_dim: ((width / crate::codegen::FP8_BLOCK) as u32, tokens as u32, 1),
            block_dim: ((crate::codegen::FP8_BLOCK / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let kernel = format!("{name}_f8{suffix}");
        let mut b = self.stream.launch_builder(self.kernel(&kernel));
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&fa.q).arg(&fa.scale);
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
        self.rms_split_into(x, w, &out.hi.as_view(), &out.lo.as_view(), tokens)
    }

    fn rms_split_into(
        &self,
        x: &CudaView<f32>,
        w: &CudaSlice<f32>,
        hi: &CudaView<u16>,
        lo: &CudaView<u16>,
        tokens: usize,
    ) -> Result<()> {
        let mut b = self.stream.launch_builder(self.kernel("rms_split"));
        b.arg(x).arg(w).arg(hi).arg(lo);
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
        self.prefill_chunk_traced(tokens, start, all_logits, None)
    }

    fn gdn_prefill(&self, g: &DeviceGdn, n: usize) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let pre = &self.pre;
        self.tc("lin_in", &g.in_proj, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_lin), la.in_proj_rows(), n)?;
        self.tc("lin_ba", &g.ba, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_ba), la.ba_rows(), n)?;
        self.gdn_prefill_rows(g, n, 0)?;
        self.tc("lin_out", &g.out, None, gb.planes.all(), GemmOut::F32(&pre.x), self.spec.dims.hidden, n)
    }

    fn gdn_prefill_rows(&self, g: &DeviceGdn, n: usize, at: usize) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let slot = g.slot + self.seq * self.spec.linear_layers() as i32;
        let d = la.key_head_dim;
        let lin = gb.pre_lin.slice(at * la.in_proj_rows()..);
        let ba = gb.pre_ba.slice(at * la.ba_rows()..);
        let mix = gb.pre_mix.slice(at * la.conv_dim()..);
        let o = gb.o.slice(at * la.value_dim()..);
        let hi = gb.planes.hi.slice(at * la.value_dim()..);
        let lo = gb.planes.lo.slice(at * la.value_dim()..);
        let launch = |name: &str, grid_dim: (u32, u32, u32), block: u32, smem: u32| LaunchConfig {
            grid_dim,
            block_dim: (block, 1, 1),
            shared_mem_bytes: if name.is_empty() { 0 } else { smem },
        };
        let mut b = self.stream.launch_builder(self.kernel("gdn_conv_prefill"));
        b.arg(&lin).arg(&g.conv).arg(&*gb.conv_state).arg(&mix).arg(&slot);
        unsafe { b.launch(launch("", (grid(la.conv_dim(), 256), n as u32, 1), 256, 0)) }?;
        let ntok = n as i32;
        let mut b = self.stream.launch_builder(self.kernel("gdn_conv_keep"));
        b.arg(&lin).arg(&*gb.conv_state).arg(&slot).arg(&ntok);
        unsafe { b.launch(cfg(grid(la.conv_dim(), 256), 256)) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_qk_prep"));
        b.arg(&mix).arg(&gb.qn).arg(&gb.kn);
        unsafe { b.launch(launch("", (la.key_heads as u32, n as u32, 1), d as u32, 0)) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_gates"));
        b.arg(&ba).arg(&g.neg_a).arg(&g.dt_bias).arg(&gb.g).arg(&gb.beta);
        unsafe { b.launch(cfg(n as u32, 64)) }?;
        let chunks = n.div_ceil(LA_CHUNK) as u32;
        let mut b = self.stream.launch_builder(self.kernel("gdn_chunk_prep"));
        b.arg(&gb.qn)
            .arg(&gb.kn)
            .arg(&mix)
            .arg(&gb.g)
            .arg(&gb.beta)
            .arg(&gb.w)
            .arg(&gb.u)
            .arg(&gb.aqk)
            .arg(&gb.gc)
            .arg(&ntok);
        let smem = self.smem["gdn_chunk_prep"];
        unsafe { b.launch(launch("prep", (la.value_heads as u32, chunks, 1), 256, smem)) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_chunk_scan"));
        b.arg(&gb.qn)
            .arg(&gb.kn)
            .arg(&gb.w)
            .arg(&gb.u)
            .arg(&gb.aqk)
            .arg(&gb.gc)
            .arg(&*gb.ssm_state)
            .arg(&o)
            .arg(&slot)
            .arg(&ntok);
        unsafe { b.launch(launch("", (la.value_heads as u32, (d / LA_COLS) as u32, 1), 256, 0)) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_gated_norm"));
        b.arg(&o).arg(&lin).arg(&g.norm).arg(&hi).arg(&lo);
        unsafe { b.launch(launch("", (la.value_heads as u32, n as u32, 1), (d / 2) as u32, 0)) }?;
        Ok(())
    }

    fn attn_prefill_rows(&self, l: &DeviceAttn, n: usize, at: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let qkv = pre.qkv.slice(at * self.spec.qkv_rows()..);
        let hi = pre.attn_planes.hi.slice(at * self.spec.q_dim()..);
        let lo = pre.attn_planes.lo.slice(at * self.spec.q_dim()..);
        let mut b = self.stream.launch_builder(self.kernel("kv_norm_rope"));
        b.arg(&qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(&buf.k_cache)
            .arg(&buf.v_cache)
            .arg(&table)
            .arg(&buf.pos)
            .arg(&l.kv);
        let config = LaunchConfig {
            grid_dim: ((2 * a.kv_heads) as u32, n as u32, 1),
            block_dim: (a.head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel("flash_mma"));
        let ntok = n as i32;
        b.arg(&qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(&buf.k_cache)
            .arg(&buf.v_cache)
            .arg(&table)
            .arg(&buf.pos)
            .arg(&l.kv)
            .arg(&hi)
            .arg(&lo)
            .arg(&ntok);
        let rows = crate::codegen::flash_rows(a.head_dim);
        let config = LaunchConfig {
            grid_dim: (grid(n, rows), a.heads as u32, 1),
            block_dim: (rows as u32 * 2, 1, 1),
            shared_mem_bytes: self.smem["flash_mma"],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn mlp_prefill(&self, l: &DeviceLayer, n: usize) -> Result<()> {
        let pre = &self.pre;
        let (hidden, inter) = (self.spec.dims.hidden, self.spec.dims.intermediate);
        self.rms_split(&pre.x.as_view(), &l.ln2, &pre.xn_planes, n)?;
        self.tc("gate_up", &l.gate_up, None, pre.xn_planes.all(), GemmOut::Planes(&pre.act_planes), 2 * inter, n)?;
        self.tc("down", &l.down, None, pre.act_planes.all(), GemmOut::F32(&pre.x), hidden, n)
    }

    fn prefill_chunk_traced(
        &mut self,
        tokens: &[u32],
        start: usize,
        all_logits: Option<&CudaSlice<f32>>,
        trace: Option<&CudaSlice<f32>>,
    ) -> Result<()> {
        let n = tokens.len();
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.stream.memcpy_htod(&ids, &mut self.pre.tokens.slice_mut(..n))?;
        self.stream.memcpy_htod(&[start as i32], &mut self.buf.pos)?;
        let s = &self.spec;
        let a = &s.attention;
        let hidden = s.dims.hidden;
        let qkv_rows = s.qkv_rows();
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
        for (i, layer) in self.layers.iter().enumerate() {
            let l = match &layer.mixer {
                DeviceMixer::Linear(g) => {
                    self.rms_split(&pre.x.as_view(), &layer.ln1, &pre.xn_planes, n)?;
                    self.gdn_prefill(g, n)?;
                    self.mlp_prefill(layer, n)?;
                    self.trace_layer(trace, i, n)?;
                    continue;
                }
                DeviceMixer::Attention(l) => l,
            };
            let kv_layer = l.kv;
            let table = self.slot_table(self.seq as usize);
            if self.use_mma {
                self.rms_split(&pre.x.as_view(), &layer.ln1, &pre.xn_planes, n)?;
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
                self.rmsnorm_rows(&pre.x, &layer.ln1, &pre.xn, n)?;
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
                .arg(&table)
                .arg(&buf.pos);
            if !self.use_mma {
                b.arg(&buf.zero_slots).arg(&kv_layer).arg(&0i32);
            } else {
                b.arg(&kv_layer);
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
                .arg(&table)
                .arg(&buf.pos)
                .arg(&kv_layer);
            if self.use_mma {
                b.arg(&pre.attn_planes.hi).arg(&pre.attn_planes.lo);
            } else {
                b.arg(&pre.attn);
            }
            b.arg(&ntok);
            let rows = if self.use_mma { crate::codegen::flash_rows(a.head_dim) } else { 16 };
            let config = LaunchConfig {
                grid_dim: (grid(n, rows), a.heads as u32, 1),
                block_dim: (if self.use_mma { rows as u32 * 2 } else { 128 }, 1, 1),
                shared_mem_bytes: if self.use_mma { self.smem["flash_mma"] } else { 0 },
            };
            unsafe { b.launch(config) }?;
            let inter = s.dims.intermediate;
            if self.use_mma {
                self.tc("o", &l.o, None, pre.attn_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                self.mlp_prefill(layer, n)?;
                self.trace_layer(trace, i, n)?;
                continue;
            }
            self.gemm("o_gemm", &l.o, None, &pre.attn.as_view(), &pre.x, hidden, n)?;
            self.rmsnorm_rows(&pre.x, &layer.ln2, &pre.xn, n)?;
            self.gemm(
                "gate_up_gemm",
                &layer.gate_up,
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
            self.gemm("down_gemm", &layer.down, None, &pre.act.as_view(), &pre.x, hidden, n)?;
            self.trace_layer(trace, i, n)?;
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
            Some(self.stream.alloc_zeros::<f32>(self.prefill_limit * vocab)?)
        } else {
            None
        };
        let mut out = Vec::new();
        for (c, piece) in tokens.chunks(self.prefill_limit).enumerate() {
            self.prefill_chunk(piece, c * self.prefill_limit, scratch.as_ref())?;
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
        self.reset_linear_state()?;
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
        self.reset_linear_state()?;
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

    fn gdn_batch(&self, g: &DeviceGdn, slots: &CudaSlice<i32>, n: usize) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let pre = &self.pre;
        self.tc("lin_in", &g.in_proj, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_lin), la.in_proj_rows(), n)?;
        self.tc("lin_ba", &g.ba, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_ba), la.ba_rows(), n)?;
        self.gdn_decode_rows(g, slots, n)?;
        self.tc("lin_out", &g.out, None, gb.planes.all(), GemmOut::F32(&pre.x), self.spec.dims.hidden, n)
    }

    fn gdn_decode_rows(&self, g: &DeviceGdn, slots: &CudaSlice<i32>, n: usize) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let seq_slots = self.spec.linear_layers() as i32;
        let mut b = self.stream.launch_builder(self.kernel("gdn_conv_step"));
        b.arg(&gb.pre_lin).arg(&g.conv).arg(&*gb.conv_state).arg(&gb.pre_mix).arg(&g.slot).arg(&seq_slots).arg(slots);
        let config = LaunchConfig {
            grid_dim: (grid(la.conv_dim(), 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_step"));
        b.arg(&gb.pre_mix)
            .arg(&gb.pre_lin)
            .arg(&gb.pre_ba)
            .arg(&g.neg_a)
            .arg(&g.dt_bias)
            .arg(&g.norm)
            .arg(&*gb.ssm_state)
            .arg(&gb.o)
            .arg(&g.slot)
            .arg(&seq_slots)
            .arg(slots);
        let config = LaunchConfig {
            grid_dim: (la.value_heads as u32, n as u32, 1),
            block_dim: ((la.value_head_dim * LA_KSPLIT) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel("split_rows"));
        let count = (n * la.value_dim()) as i32;
        b.arg(&gb.o).arg(&gb.planes.hi).arg(&gb.planes.lo).arg(&count);
        unsafe { b.launch(cfg(grid(n * la.value_dim(), 512), 256)) }?;
        Ok(())
    }

    fn reset_linear_state(&mut self) -> Result<()> {
        self.reset_slot(self.seq as usize)
    }

    pub fn state_slot_bytes(&self) -> usize {
        self.gdn.as_ref().map_or(0, |gb| gb.conv_pool.stride_bytes() + gb.ssm_pool.stride_bytes())
    }

    pub fn state_mapped_bytes(&self) -> usize {
        self.gdn.as_ref().map_or(0, |gb| gb.conv_pool.mapped_bytes() + gb.ssm_pool.mapped_bytes())
    }

    pub fn state_live_slots(&self) -> usize {
        self.gdn.as_ref().map_or(0, |gb| gb.ssm_pool.live_slots())
    }

    pub fn acquire_state(&mut self, slot: usize) -> Result<()> {
        ensure!(slot <= self.slots, "slot {slot} out of {}", self.slots);
        let need = self.state_slot_bytes();
        let free = self.ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
        let Some(gb) = self.gdn.as_mut() else { return Ok(()) };
        if gb.ssm_pool.is_live(slot) {
            return Ok(());
        }
        ensure!(
            free >= need + STATE_FREE_MARGIN,
            "{free} bytes of device memory free, a recurrent state slot needs {need} plus a margin"
        );
        gb.conv_pool.acquire(slot)?;
        if let Err(e) = gb.ssm_pool.acquire(slot) {
            gb.conv_pool.release(slot);
            return Err(e);
        }
        let conv = gb.conv_pool.stride_bytes() / 4;
        let ssm = gb.ssm_pool.stride_bytes() / 4;
        self.stream.memset_zeros(&mut gb.conv_state.slice_mut(slot * conv..(slot + 1) * conv))?;
        self.stream.memset_zeros(&mut gb.ssm_state.slice_mut(slot * ssm..(slot + 1) * ssm))?;
        Ok(())
    }

    pub fn release_state(&mut self, slot: usize) {
        if slot >= self.slots {
            return;
        }
        let _ = self.stream.synchronize();
        if let Some(gb) = self.gdn.as_mut() {
            gb.ssm_pool.release(slot);
            gb.conv_pool.release(slot);
        }
    }

    pub fn reset_slot(&mut self, seq: usize) -> Result<()> {
        ensure!(seq < self.slots, "slot {seq} out of {}", self.slots);
        self.mtp_forget(seq);
        if let Some(gb) = self.gdn.as_ref() {
            ensure!(gb.ssm_pool.is_live(seq), "recurrent state of slot {seq} is not allocated");
        }
        let slots = self.spec.linear_layers();
        let Some(gb) = self.gdn.as_mut() else { return Ok(()) };
        let conv = slots * gb.la.conv_dim() * (gb.la.conv_kernel - 1);
        let ssm = slots * gb.la.value_heads * gb.la.key_head_dim * gb.la.key_head_dim;
        self.stream.memset_zeros(&mut gb.conv_state.slice_mut(seq * conv..(seq + 1) * conv))?;
        self.stream.memset_zeros(&mut gb.ssm_state.slice_mut(seq * ssm..(seq + 1) * ssm))?;
        Ok(())
    }

    pub fn linear_states(&self) -> Result<LinearStates> {
        let gb = self.gdn.as_ref().context("no linear layers")?;
        let layers = self.spec.linear_layers();
        let conv = self.slots * layers * gb.la.conv_dim() * (gb.la.conv_kernel - 1);
        let ssm = self.slots * layers * gb.la.value_heads * gb.la.key_head_dim * gb.la.key_head_dim;
        Ok(LinearStates {
            conv: self.stream.clone_dtoh(&gb.conv_state.slice(..conv))?,
            state: self.stream.clone_dtoh(&gb.ssm_state.slice(..ssm))?,
        })
    }

    pub fn trace_prefill(&mut self, tokens: &[u32]) -> Result<Trace> {
        ensure!(tokens.len() <= self.prefill_limit, "trace prefill fits one chunk");
        let s = &self.spec;
        let n = tokens.len();
        let layers = self.stream.alloc_zeros::<f32>(s.layers.count * n * s.dims.hidden)?;
        let logits = self.stream.alloc_zeros::<f32>(n * s.dims.vocab)?;
        self.reset_linear_state()?;
        self.prefill_chunk_traced(tokens, 0, Some(&logits), Some(&layers))?;
        Ok(Trace { layers: self.stream.clone_dtoh(&layers)?, logits: self.stream.clone_dtoh(&logits)? })
    }

    pub fn trace_decode(&mut self, tokens: &[u32]) -> Result<Trace> {
        let s = self.spec.clone();
        let per_step = s.layers.count * s.dims.hidden;
        let layers = self.stream.alloc_zeros::<f32>(per_step)?;
        self.reset_linear_state()?;
        let mut trace = Trace { layers: Vec::new(), logits: Vec::new() };
        for (pos, &t) in tokens.iter().enumerate() {
            self.set(t, pos)?;
            self.step_traced(Some(&layers))?;
            trace.layers.extend(self.stream.clone_dtoh(&layers)?);
            trace.logits.extend(self.stream.clone_dtoh(&self.buf.logits)?);
        }
        Ok(trace)
    }

    pub fn prefill_then_decode(&mut self, prompt: &[u32], decode: &[u32]) -> Result<Vec<f32>> {
        ensure!(prompt.len() + decode.len() <= self.max_ctx, "longer than max_ctx");
        self.reset_linear_state()?;
        self.prefill(prompt, false)?;
        let mut logits = self.stream.clone_dtoh(&self.buf.logits)?;
        for (i, &t) in decode.iter().enumerate() {
            self.set(t, prompt.len() + i)?;
            self.step()?;
            logits.extend(self.stream.clone_dtoh(&self.buf.logits)?);
        }
        Ok(logits)
    }

    fn batch_step(&self, bb: &BatchBuffers, n: usize) -> Result<()> {
        self.decode_rows(bb, n)?;
        let stride = self.max_ctx as i32;
        let mut b = self.stream.launch_builder(self.kernel("argmax_rows"));
        b.arg(&bb.logits).arg(&bb.tokens).arg(&bb.history).arg(&self.buf.count).arg(&stride);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        let mut b = self.stream.launch_builder(self.kernel("bump"));
        b.arg(&bb.pos).arg(&self.buf.count);
        unsafe { b.launch(cfg(1, n as u32)) }?;
        Ok(())
    }

    fn attn_decode_rows(&self, l: &DeviceAttn, bb: &BatchBuffers, n: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let mut b = self.stream.launch_builder(self.kernel("qk_norm_rope"));
        b.arg(&pre.qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(&buf.k_cache)
            .arg(&buf.v_cache)
            .arg(&buf.block_table)
            .arg(&bb.pos)
            .arg(&bb.slots)
            .arg(&l.kv)
            .arg(&1i32);
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
            .arg(&bb.pos)
            .arg(&bb.slots)
            .arg(&l.kv)
            .arg(&bb.part_acc)
            .arg(&bb.part_ml);
        let config = LaunchConfig {
            grid_dim: (a.kv_heads as u32, n as u32, crate::codegen::BATCH_DECODE_SPLITS as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel("batch_decode_combine"));
        b.arg(&bb.part_acc).arg(&bb.part_ml).arg(&pre.qkv).arg(&pre.attn_planes.hi).arg(&pre.attn_planes.lo);
        let config = LaunchConfig {
            grid_dim: (a.heads as u32, n as u32, 1),
            block_dim: (a.head_dim as u32 / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn decode_rows(&self, bb: &BatchBuffers, n: usize) -> Result<()> {
        let s = &self.spec;
        let hidden = s.dims.hidden;
        let pre = &self.pre;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&bb.tokens).arg(&pre.x);
        let config = LaunchConfig {
            grid_dim: (grid(hidden, 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        for layer in &self.layers {
            self.rms_split(&pre.x.as_view(), &layer.ln1, &pre.xn_planes, n)?;
            let l = match &layer.mixer {
                DeviceMixer::Linear(g) => {
                    self.gdn_batch(g, &bb.slots, n)?;
                    self.mlp_prefill(layer, n)?;
                    continue;
                }
                DeviceMixer::Attention(l) => l,
            };
            let qkv_rows = s.qkv_rows();
            self.tc(
                "qkv",
                &l.qkv,
                l.qkv_bias.as_ref(),
                pre.xn_planes.all(),
                GemmOut::F32(&pre.qkv),
                qkv_rows,
                n,
            )?;
            self.attn_decode_rows(l, bb, n)?;
            self.tc("o", &l.o, None, pre.attn_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
            self.mlp_prefill(layer, n)?;
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
        Ok(())
    }

    fn batch_buffers(&self, rows: usize) -> Result<BatchBuffers> {
        let parts = rows * self.spec.attention.heads * crate::codegen::BATCH_DECODE_SPLITS;
        Ok(BatchBuffers {
            tokens: self.stream.alloc_zeros(rows)?,
            logits: self.stream.alloc_zeros(rows * self.spec.dims.vocab)?,
            history: self.stream.alloc_zeros(rows * self.max_ctx)?,
            part_acc: self.stream.alloc_zeros(parts * self.spec.attention.head_dim)?,
            part_ml: self.stream.alloc_zeros(parts * 2)?,
            pos: self.stream.alloc_zeros(rows)?,
            slots: self.stream.alloc_zeros(rows)?,
        })
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
        let mut bb = self.batch_buffers(self.batch)?;
        let started = Instant::now();
        self.stream.memcpy_htod(&[0i32], &mut self.buf.count)?;
        for (seq, p) in prompts.iter().enumerate() {
            self.seq = seq as i32;
            let prefilled = self.reset_linear_state().and_then(|_| self.prefill(p, false));
            self.seq = 0;
            prefilled?;
            self.first_token(&bb, seq)?;
        }
        let slots: Vec<i32> = (0..self.batch as i32).collect();
        self.stream.memcpy_htod(&slots, &mut bb.slots)?;
        self.stream.memcpy_htod(&vec![len as i32; self.batch], &mut bb.pos)?;
        self.stream.memcpy_htod(&[1i32], &mut self.buf.count)?;
        self.stream.synchronize()?;
        let prefill_secs = started.elapsed().as_secs_f64();
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.batch_step(&bb, self.batch);
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

fn argmax_host(row: &[f32]) -> u32 {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }) as u32
}

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
