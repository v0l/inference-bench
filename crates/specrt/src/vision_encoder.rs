use crate::codegen::Module;
use crate::jit::Jit;
use crate::vision::{Patches, VisionConfig};
use crate::weights::{Checkpoint, bf16_to_f32};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const VISION_TEMPLATE: &str = include_str!("../templates/vision.cu");

const KERNELS: &[&str] = &["vis_mma", "vis_split", "vis_layernorm", "vis_pos", "vis_rope_pack", "vis_flash", "vis_to_f16"];
const GEMM_BM: u32 = 128;
const GEMM_BN: u32 = 128;
const GEMM_THREADS: u32 = 256;
const GEMM_SMEM: u32 = 3 * (128 * 32 * 2 * 2 + 128 * 32 * 2);
const FLASH_BQ: u32 = 128;
const FLASH_THREADS: u32 = 256;

fn flash_smem(head_dim: usize) -> u32 {
    let stride = head_pad(head_dim) + 8;
    (2 * 128 * stride * 2).max(2 * 4 * 32 * stride * 2) as u32
}
pub const MAX_ENCODE_PATCHES: usize = 32768;

const LAYERNORM_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    GeluTanh,
    GeluErf,
}

impl Act {
    fn code(self) -> i32 {
        match self {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::GeluErf => 2,
        }
    }

    pub fn from_config(name: Option<&str>) -> Result<Self> {
        match name.unwrap_or("gelu_pytorch_tanh") {
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Ok(Act::GeluTanh),
            "gelu" => Ok(Act::GeluErf),
            other => bail!("vision hidden_act {other} is not supported"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VisionOp {
    PatchEmbed,
    Norm { block: usize, which: u8 },
    Qkv { block: usize },
    Rope,
    Attention,
    Proj { block: usize },
    MlpUp { block: usize },
    MlpDown { block: usize },
    Merger,
    DeepstackMerger { index: usize, after_block: usize },
}

impl VisionOp {
    pub fn label(&self) -> String {
        match self {
            VisionOp::PatchEmbed => "patch embed + pos embed".into(),
            VisionOp::Norm { block, which } => format!("block {block} norm{which}"),
            VisionOp::Qkv { block } => format!("block {block} qkv"),
            VisionOp::Rope => "2d rope".into(),
            VisionOp::Attention => "attention per frame".into(),
            VisionOp::Proj { block } => format!("block {block} proj"),
            VisionOp::MlpUp { block } => format!("block {block} mlp up"),
            VisionOp::MlpDown { block } => format!("block {block} mlp down"),
            VisionOp::Merger => "patch merger".into(),
            VisionOp::DeepstackMerger { index, after_block } => format!("deepstack merger {index} after block {after_block}"),
        }
    }
}

pub fn plan(cfg: &VisionConfig) -> Vec<VisionOp> {
    let mut ops = vec![VisionOp::PatchEmbed];
    for block in 0..cfg.depth {
        ops.push(VisionOp::Norm { block, which: 1 });
        ops.push(VisionOp::Qkv { block });
        ops.push(VisionOp::Rope);
        ops.push(VisionOp::Attention);
        ops.push(VisionOp::Proj { block });
        ops.push(VisionOp::Norm { block, which: 2 });
        ops.push(VisionOp::MlpUp { block });
        ops.push(VisionOp::MlpDown { block });
        if let Some(index) = cfg.deepstack_visual_indexes.iter().position(|&b| b == block) {
            ops.push(VisionOp::DeepstackMerger { index, after_block: block });
        }
    }
    ops.push(VisionOp::Merger);
    ops
}

struct Linear {
    w: CudaSlice<u16>,
    b: CudaSlice<u16>,
    rows: usize,
    cols: usize,
}

struct Norm {
    w: CudaSlice<u16>,
    b: CudaSlice<u16>,
}

struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    proj: Linear,
    up: Linear,
    down: Linear,
}

struct Merger {
    norm: Norm,
    up: Linear,
    down: Linear,
    post_shuffle_norm: bool,
}

pub struct VisionEncoder {
    stream: Arc<CudaStream>,
    mma: CudaFunction,
    split: CudaFunction,
    layernorm: CudaFunction,
    pos: CudaFunction,
    rope_pack: CudaFunction,
    flash: CudaFunction,
    cfg: VisionConfig,
    act: Act,
    patch: Linear,
    pos_embed: CudaSlice<f32>,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VisionPrecision {
    #[default]
    Fast,
    Exact,
}

pub struct VisionOutput {
    pub embeds: Vec<f32>,
    pub deepstack: Vec<Vec<f32>>,
    pub tokens: usize,
}

struct Planes {
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
}

impl Planes {
    fn new(s: &Arc<CudaStream>, n: usize) -> Result<Self> {
        Ok(Planes { hi: unsafe { s.alloc::<u16>(n) }?, lo: unsafe { s.alloc::<u16>(n) }? })
    }
}

enum Out<'a> {
    Store(&'a mut CudaSlice<f32>),
    AddInPlace(&'a mut CudaSlice<f32>),
    Split(&'a mut Planes),
}

fn head_pad(head_dim: usize) -> usize {
    head_dim.div_ceil(16) * 16
}

fn module(head_dim: usize, precise: bool) -> Result<Module> {
    let flag = |on: bool| if on { "true" } else { "false" }.to_string();
    let vars = BTreeMap::from([("VIS_HDP", head_pad(head_dim).to_string()), ("VIS_GEMM_SPLIT", flag(precise)), ("VIS_ATT_SPLIT", flag(precise))]);
    let source = crate::codegen::render(VISION_TEMPLATE, &vars)?;
    Ok(Module { source, kernels: KERNELS.to_vec(), smem: BTreeMap::new(), arch_features: false, kir_hash: String::new() })
}

fn prefix(ck: &Checkpoint) -> Result<&'static str> {
    for p in ["model.visual", "visual"] {
        if ck.has(&format!("{p}.patch_embed.proj.weight")) {
            return Ok(p);
        }
    }
    bail!("the checkpoint has no vision tensors")
}

fn upload(stream: &Arc<CudaStream>, ck: &Checkpoint, name: &str, shape: &[usize]) -> Result<CudaSlice<u16>> {
    let w = ck.bf16(name, shape).with_context(|| name.to_string())?;
    Ok(stream.clone_htod(w.as_ref())?)
}

fn linear(stream: &Arc<CudaStream>, ck: &Checkpoint, base: &str, rows: usize, cols: usize) -> Result<Linear> {
    ensure!(cols % 8 == 0 && rows % 2 == 0, "{base}: {rows}x{cols} is not a multiple of the mma tile steps");
    Ok(Linear {
        w: upload(stream, ck, &format!("{base}.weight"), &[rows, cols])?,
        b: upload(stream, ck, &format!("{base}.bias"), &[rows])?,
        rows,
        cols,
    })
}

fn norm(stream: &Arc<CudaStream>, ck: &Checkpoint, base: &str, width: usize) -> Result<Norm> {
    Ok(Norm { w: upload(stream, ck, &format!("{base}.weight"), &[width])?, b: upload(stream, ck, &format!("{base}.bias"), &[width])? })
}

fn merger(stream: &Arc<CudaStream>, ck: &Checkpoint, base: &str, cfg: &VisionConfig, post_shuffle_norm: bool) -> Result<Merger> {
    let merged = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
    Ok(Merger {
        norm: norm(stream, ck, &format!("{base}.norm"), if post_shuffle_norm { merged } else { cfg.hidden_size })?,
        up: linear(stream, ck, &format!("{base}.linear_fc1"), merged, merged)?,
        down: linear(stream, ck, &format!("{base}.linear_fc2"), cfg.out_hidden_size, merged)?,
        post_shuffle_norm,
    })
}

pub fn position_ids(grid: [usize; 3], merge: usize) -> Vec<(usize, usize)> {
    let [t, h, w] = grid;
    let mut frame = Vec::with_capacity(h * w);
    for bh in 0..h / merge {
        for bw in 0..w / merge {
            for mh in 0..merge {
                for mw in 0..merge {
                    frame.push((bh * merge + mh, bw * merge + mw));
                }
            }
        }
    }
    (0..t).flat_map(|_| frame.iter().copied()).collect()
}

fn axis_taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = if size > 1 { index as f32 * (side - 1) as f32 / (size - 1) as f32 } else { 0.0 };
    let floor = src.floor();
    let lo = (floor as usize).min(side - 1);
    let hi = (floor as usize + 1).min(side - 1);
    [(lo, (1.0 - (src - floor)).max(0.0)), (hi, (1.0 - (floor + 1.0 - src).abs()).max(0.0))]
}

pub fn position_taps(side: usize, grid: [usize; 3], merge: usize) -> (Vec<i32>, Vec<f32>) {
    let [_, h, w] = grid;
    let ids = position_ids(grid, merge);
    let mut taps = Vec::with_capacity(ids.len() * 4);
    let mut weights = Vec::with_capacity(ids.len() * 4);
    for &(row, col) in &ids {
        for (ri, rw) in axis_taps(row, h, side) {
            for (ci, cw) in axis_taps(col, w, side) {
                taps.push((ri * side + ci) as i32);
                weights.push(rw * cw);
            }
        }
    }
    (taps, weights)
}

pub fn interpolated_positions(table: &[f32], hidden: usize, side: usize, grid: [usize; 3], merge: usize) -> Vec<f32> {
    let (taps, weights) = position_taps(side, grid, merge);
    let mut out = vec![0f32; taps.len() / 4 * hidden];
    for (i, dst) in out.chunks_mut(hidden).enumerate() {
        for k in 0..4 {
            let at = taps[i * 4 + k] as usize * hidden;
            for (d, s) in dst.iter_mut().zip(&table[at..at + hidden]) {
                *d += weights[i * 4 + k] * s;
            }
        }
    }
    out
}

pub fn rope_tables(grid: [usize; 3], merge: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let ids = position_ids(grid, merge);
    let quarter = head_dim / 4;
    let inv: Vec<f32> = (0..quarter).map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f32 / (head_dim / 2) as f32)).collect();
    let mut cos = Vec::with_capacity(ids.len() * head_dim);
    let mut sin = Vec::with_capacity(ids.len() * head_dim);
    for &(h, w) in &ids {
        let freqs: Vec<f32> = inv.iter().map(|f| h as f32 * f).chain(inv.iter().map(|f| w as f32 * f)).collect();
        for _ in 0..2 {
            cos.extend(freqs.iter().map(|f| f.cos()));
            sin.extend(freqs.iter().map(|f| f.sin()));
        }
    }
    (cos, sin)
}

struct BatchLayout {
    tokens: usize,
    taps: Vec<i32>,
    tap_weights: Vec<f32>,
    hw: Vec<i32>,
    seg_lo: Vec<i32>,
    seg_hi: Vec<i32>,
    merged_rows: Vec<usize>,
}

impl VisionEncoder {
    pub fn load(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, jit: &Jit, ck: &Checkpoint, cfg: &VisionConfig) -> Result<Self> {
        Self::load_with(ctx, stream, jit, ck, cfg, VisionPrecision::default())
    }

    pub fn load_with(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, jit: &Jit, ck: &Checkpoint, cfg: &VisionConfig, precision: VisionPrecision) -> Result<Self> {
        let precise = precision == VisionPrecision::Exact;
        ensure!(cfg.hidden_size % cfg.num_heads == 0, "vision hidden_size must divide by num_heads");
        let head_dim = cfg.hidden_size / cfg.num_heads;
        ensure!(head_dim % 8 == 0 && head_dim <= 128, "vision head_dim {head_dim} is not supported");
        let act = Act::from_config(cfg.hidden_act.as_deref())?;
        let p = prefix(ck)?;
        let loaded = jit.load(&module(head_dim, precise)?)?;
        let _ = ctx;
        let channels = cfg.in_channels.unwrap_or(3);
        let patch_cols = channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        ensure!(patch_cols % 8 == 0, "patch of {patch_cols} values is not a multiple of 8");
        let mut patch = Linear {
            w: upload(stream, ck, &format!("{p}.patch_embed.proj.weight"), &[cfg.hidden_size, channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size])?,
            b: upload(stream, ck, &format!("{p}.patch_embed.proj.bias"), &[cfg.hidden_size])?,
            rows: cfg.hidden_size,
            cols: patch_cols,
        };
        let table = ck.bf16(&format!("{p}.pos_embed.weight"), &[cfg.num_position_embeddings, cfg.hidden_size])?;
        let table: Vec<f32> = table.iter().map(|&v| bf16_to_f32(v)).collect();
        let pos_embed = stream.clone_htod(&table)?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = format!("{p}.blocks.{i}");
            blocks.push(Block {
                norm1: norm(stream, ck, &format!("{b}.norm1"), cfg.hidden_size)?,
                norm2: norm(stream, ck, &format!("{b}.norm2"), cfg.hidden_size)?,
                qkv: linear(stream, ck, &format!("{b}.attn.qkv"), 3 * cfg.hidden_size, cfg.hidden_size)?,
                proj: linear(stream, ck, &format!("{b}.attn.proj"), cfg.hidden_size, cfg.hidden_size)?,
                up: linear(stream, ck, &format!("{b}.mlp.linear_fc1"), cfg.intermediate_size, cfg.hidden_size)?,
                down: linear(stream, ck, &format!("{b}.mlp.linear_fc2"), cfg.hidden_size, cfg.intermediate_size)?,
            });
        }
        let mut main = merger(stream, ck, &format!("{p}.merger"), cfg, false)?;
        let mut deepstack = Vec::new();
        for i in 0..cfg.deepstack_visual_indexes.len() {
            deepstack.push(merger(stream, ck, &format!("{p}.deepstack_merger_list.{i}"), cfg, true)?);
        }
        let mma = loaded.function("vis_mma")?;
        let flash = loaded.function("vis_flash")?;
        {
            use cudarc::driver::sys::CUfunction_attribute as A;
            mma.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, GEMM_SMEM as i32)?;
            flash.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, flash_smem(head_dim) as i32)?;
        }
        if !precise {
            let to_f16 = loaded.function("vis_to_f16")?;
            let mut convert = |w: &mut CudaSlice<u16>| -> Result<()> {
                let n = w.len() as i32;
                let mut b = stream.launch_builder(&to_f16);
                b.arg(w).arg(&n);
                unsafe { b.launch(LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
                Ok(())
            };
            convert(&mut patch.w)?;
            for blk in &mut blocks {
                for l in [&mut blk.qkv, &mut blk.proj, &mut blk.up, &mut blk.down] {
                    convert(&mut l.w)?;
                }
            }
            for m in std::iter::once(&mut main).chain(deepstack.iter_mut()) {
                convert(&mut m.up.w)?;
                convert(&mut m.down.w)?;
            }
        }
        stream.synchronize()?;
        Ok(VisionEncoder {
            stream: stream.clone(),
            mma,
            split: loaded.function("vis_split")?,
            layernorm: loaded.function("vis_layernorm")?,
            pos: loaded.function("vis_pos")?,
            rope_pack: loaded.function("vis_rope_pack")?,
            flash,
            cfg: cfg.clone(),
            act,
            patch,
            pos_embed,
            blocks,
            merger: main,
            deepstack,
        })
    }

    pub fn plan(&self) -> Vec<VisionOp> {
        plan(&self.cfg)
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    fn gemm(&self, l: &Linear, a: &Planes, out: Out<'_>, rows: usize, act: Act) -> Result<()> {
        let (m, n, k, code) = (rows as i32, l.rows as i32, l.cols as i32, act.code());
        let null = 0u64;
        let ptr = match &out {
            Out::AddInPlace(o) => o.device_ptr(&self.stream).0,
            _ => 0,
        };
        let cfg = LaunchConfig {
            grid_dim: ((l.rows as u32).div_ceil(GEMM_BN), (rows as u32).div_ceil(GEMM_BM), 1),
            block_dim: (GEMM_THREADS, 1, 1),
            shared_mem_bytes: GEMM_SMEM,
        };
        let mut b = self.stream.launch_builder(&self.mma);
        b.arg(&a.hi).arg(&a.lo).arg(&l.w).arg(&l.b);
        match out {
            Out::Store(o) => {
                b.arg(&null).arg(o).arg(&null).arg(&null);
            }
            Out::AddInPlace(_) => {
                b.arg(&ptr).arg(&ptr).arg(&null).arg(&null);
            }
            Out::Split(p) => {
                b.arg(&null).arg(&null).arg(&mut p.hi).arg(&mut p.lo);
            }
        }
        b.arg(&m).arg(&n).arg(&k).arg(&code);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn layernorm(&self, x: &CudaSlice<f32>, nm: &Norm, out: &mut Planes, rows: usize, width: usize) -> Result<()> {
        ensure!(width % 2 == 0, "layernorm width {width} is odd");
        let w = width as i32;
        let mut b = self.stream.launch_builder(&self.layernorm);
        b.arg(x).arg(&nm.w).arg(&nm.b).arg(&mut out.hi).arg(&mut out.lo).arg(&w).arg(&LAYERNORM_EPS);
        unsafe { b.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(())
    }

    fn run_merger(&self, m: &Merger, x: &CudaSlice<f32>, tokens: usize, scratch: &mut Planes, up: &mut Planes) -> Result<Vec<f32>> {
        let unit = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let merged_rows = tokens / unit;
        if m.post_shuffle_norm {
            self.layernorm(x, &m.norm, scratch, merged_rows, self.cfg.hidden_size * unit)?;
        } else {
            self.layernorm(x, &m.norm, scratch, tokens, self.cfg.hidden_size)?;
        }
        self.gemm(&m.up, scratch, Out::Split(up), merged_rows, Act::GeluErf)?;
        let mut out = unsafe { self.stream.alloc::<f32>(merged_rows * self.cfg.out_hidden_size) }?;
        self.gemm(&m.down, up, Out::Store(&mut out), merged_rows, Act::None)?;
        Ok(self.stream.clone_dtoh(&out)?)
    }

    fn layout(&self, items: &[&Patches]) -> Result<BatchLayout> {
        let cfg = &self.cfg;
        let merge = cfg.spatial_merge_size;
        let side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        let mut l = BatchLayout { tokens: 0, taps: Vec::new(), tap_weights: Vec::new(), hw: Vec::new(), seg_lo: Vec::new(), seg_hi: Vec::new(), merged_rows: Vec::new() };
        for p in items {
            let [t, h, w] = p.grid;
            ensure!(t > 0 && h > 0 && w > 0, "grid {t}x{h}x{w} is empty");
            ensure!(t * h * w <= MAX_ENCODE_PATCHES, "grid {t}x{h}x{w} is {} patches, the encoder takes at most {MAX_ENCODE_PATCHES}", t * h * w);
            ensure!(h % merge == 0 && w % merge == 0, "grid {h}x{w} is not a multiple of the merge size");
            let tokens = t * h * w;
            ensure!(p.data.len() == tokens * self.patch.cols, "patch tensor has {} values, grid needs {}", p.data.len(), tokens * self.patch.cols);
            let (taps, weights) = position_taps(side, p.grid, merge);
            l.taps.extend(taps);
            l.tap_weights.extend(weights);
            l.hw.extend(position_ids(p.grid, merge).into_iter().flat_map(|(a, b)| [a as i32, b as i32]));
            let frame = h * w;
            for i in 0..tokens {
                let lo = l.tokens + i / frame * frame;
                l.seg_lo.push(lo as i32);
                l.seg_hi.push((lo + frame) as i32);
            }
            l.tokens += tokens;
            l.merged_rows.push(tokens / (merge * merge));
        }
        ensure!(l.tokens > 0, "nothing to encode");
        Ok(l)
    }

    pub fn encode(&self, patches: &Patches) -> Result<VisionOutput> {
        let (mut embeds, deepstack) = self.encode_all(&[patches])?;
        let tokens = patches.tokens(self.cfg.spatial_merge_size);
        Ok(VisionOutput { embeds: embeds.remove(0), deepstack: deepstack.into_iter().map(|mut d| d.remove(0)).collect(), tokens })
    }

    pub fn encode_batch(&self, items: &[&Patches]) -> Result<Vec<Vec<f32>>> {
        Ok(self.encode_all(items)?.0)
    }

    fn encode_all(&self, items: &[&Patches]) -> Result<(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>)> {
        let cfg = &self.cfg;
        let l = self.layout(items)?;
        let tokens = l.tokens;
        let hidden = cfg.hidden_size;
        let heads = cfg.num_heads;
        let head_dim = hidden / heads;
        let hdp = head_pad(head_dim);
        let s = &self.stream;
        let data: Vec<f32> = items.iter().flat_map(|p| p.data.iter().copied()).collect();
        let input = s.clone_htod(&data)?;
        let taps = s.clone_htod(&l.taps)?;
        let tap_weights = s.clone_htod(&l.tap_weights)?;
        let hw = s.clone_htod(&l.hw)?;
        let seg_lo = s.clone_htod(&l.seg_lo)?;
        let seg_hi = s.clone_htod(&l.seg_hi)?;
        let mut x = unsafe { s.alloc::<f32>(tokens * hidden) }?;
        let mut qkv = unsafe { s.alloc::<f32>(tokens * 3 * hidden) }?;
        let mut packed = unsafe { s.alloc::<u16>(6 * heads * tokens * hdp) }?;
        let mut xn = Planes::new(s, tokens * hidden)?;
        let mut att = Planes::new(s, tokens * hidden)?;
        let mut mid = Planes::new(s, tokens * cfg.intermediate_size.max(self.patch.cols).max(hidden))?;
        let n = (tokens * self.patch.cols) as i32;
        let mut b = s.launch_builder(&self.split);
        b.arg(&input).arg(&mut mid.hi).arg(&mut mid.lo).arg(&n);
        unsafe { b.launch(LaunchConfig { grid_dim: ((n as u32 / 2).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        let h = hidden as i32;
        let mut b = s.launch_builder(&self.pos);
        b.arg(&self.pos_embed).arg(&taps).arg(&tap_weights).arg(&mut x).arg(&h);
        unsafe { b.launch(LaunchConfig { grid_dim: (tokens as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        self.gemm(&self.patch, &mid, Out::AddInPlace(&mut x), tokens, Act::None)?;
        let (t32, h32, hd32) = (tokens as i32, heads as i32, head_dim as i32);
        let scale_log2 = (1.0 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
        let mut deepstack = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            self.layernorm(&x, &blk.norm1, &mut xn, tokens, hidden)?;
            self.gemm(&blk.qkv, &xn, Out::Store(&mut qkv), tokens, Act::None)?;
            let mut b = s.launch_builder(&self.rope_pack);
            b.arg(&qkv).arg(&hw).arg(&mut packed).arg(&t32).arg(&h32).arg(&hd32).arg(&ROPE_THETA);
            unsafe { b.launch(LaunchConfig { grid_dim: (tokens as u32, heads as u32, 1), block_dim: (hdp as u32, 1, 1), shared_mem_bytes: 0 }) }?;
            let mut b = s.launch_builder(&self.flash);
            b.arg(&packed).arg(&seg_lo).arg(&seg_hi).arg(&mut att.hi).arg(&mut att.lo).arg(&t32).arg(&h32).arg(&hd32).arg(&scale_log2);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((tokens as u32).div_ceil(FLASH_BQ), heads as u32, 1),
                    block_dim: (FLASH_THREADS, 1, 1),
                    shared_mem_bytes: flash_smem(head_dim),
                })
            }?;
            self.gemm(&blk.proj, &att, Out::AddInPlace(&mut x), tokens, Act::None)?;
            self.layernorm(&x, &blk.norm2, &mut xn, tokens, hidden)?;
            self.gemm(&blk.up, &xn, Out::Split(&mut mid), tokens, self.act)?;
            self.gemm(&blk.down, &mid, Out::AddInPlace(&mut x), tokens, Act::None)?;
            if let Some(d) = cfg.deepstack_visual_indexes.iter().position(|&b| b == i) {
                let mut scratch = Planes::new(s, tokens * hidden)?;
                let mut up = Planes::new(s, tokens * hidden)?;
                let flat = self.run_merger(&self.deepstack[d], &x, tokens, &mut scratch, &mut up)?;
                deepstack.push(self.split_rows(flat, &l.merged_rows));
            }
        }
        let mut scratch = Planes::new(s, tokens * hidden)?;
        let mut up = Planes::new(s, tokens * hidden)?;
        let flat = self.run_merger(&self.merger, &x, tokens, &mut scratch, &mut up)?;
        Ok((self.split_rows(flat, &l.merged_rows), deepstack))
    }

    fn split_rows(&self, flat: Vec<f32>, rows: &[usize]) -> Vec<Vec<f32>> {
        let width = self.cfg.out_hidden_size;
        let mut out = Vec::with_capacity(rows.len());
        let mut at = 0;
        for &r in rows {
            out.push(flat[at..at + r * width].to_vec());
            at += r * width;
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TowerPart {
    pub name: String,
    pub count: usize,
    pub params: u64,
}

pub fn tower_parts(cfg: &VisionConfig) -> Vec<TowerPart> {
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let merged = h * (cfg.spatial_merge_size * cfg.spatial_merge_size) as u64;
    let channels = cfg.in_channels.unwrap_or(3) as u64;
    let patch = channels * (cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size) as u64;
    let merger = merged * merged + merged + merged * cfg.out_hidden_size as u64 + cfg.out_hidden_size as u64 + 2 * h;
    let mut parts = vec![
        TowerPart { name: "patch embed".into(), count: 1, params: h * patch + h },
        TowerPart { name: "position embeddings".into(), count: 1, params: cfg.num_position_embeddings as u64 * h },
        TowerPart { name: "attention (qkv, proj)".into(), count: cfg.depth, params: 3 * h * h + 3 * h + h * h + h },
        TowerPart { name: "mlp (up, down)".into(), count: cfg.depth, params: 2 * h * i + i + h },
        TowerPart { name: "layernorms".into(), count: cfg.depth, params: 4 * h },
        TowerPart { name: "patch merger".into(), count: 1, params: merger },
    ];
    if !cfg.deepstack_visual_indexes.is_empty() {
        parts.push(TowerPart { name: "deepstack mergers".into(), count: cfg.deepstack_visual_indexes.len(), params: merger - 2 * h + 2 * merged });
    }
    parts
}

pub fn tower_params(cfg: &VisionConfig) -> u64 {
    tower_parts(cfg).iter().map(|p| p.params * p.count as u64).sum()
}
