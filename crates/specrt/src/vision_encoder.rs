use crate::codegen::Module;
use crate::jit::Jit;
use crate::vision::{Patches, VisionConfig};
use crate::weights::{Checkpoint, bf16_to_f32};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const VISION_TEMPLATE: &str = include_str!("../templates/vision.cu");

const KERNELS: &[&str] = &["vis_gemm", "vis_layernorm", "vis_rope", "vis_attention"];
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
    gemm: CudaFunction,
    layernorm: CudaFunction,
    rope: CudaFunction,
    attention: CudaFunction,
    cfg: VisionConfig,
    act: Act,
    patch: Linear,
    pos_embed: Vec<f32>,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
}

pub struct VisionOutput {
    pub embeds: Vec<f32>,
    pub deepstack: Vec<Vec<f32>>,
    pub tokens: usize,
}

fn module() -> Module {
    Module { source: VISION_TEMPLATE.to_string(), kernels: KERNELS.to_vec(), smem: BTreeMap::new(), arch_features: false }
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

pub fn interpolated_positions(table: &[f32], hidden: usize, side: usize, grid: [usize; 3], merge: usize) -> Vec<f32> {
    let [_, h, w] = grid;
    let ids = position_ids(grid, merge);
    let mut out = vec![0f32; ids.len() * hidden];
    for (i, &(row, col)) in ids.iter().enumerate() {
        let dst = &mut out[i * hidden..(i + 1) * hidden];
        for (ri, rw) in axis_taps(row, h, side) {
            for (ci, cw) in axis_taps(col, w, side) {
                let weight = rw * cw;
                let src = &table[(ri * side + ci) * hidden..(ri * side + ci + 1) * hidden];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += weight * s;
                }
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

impl VisionEncoder {
    pub fn load(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, jit: &Jit, ck: &Checkpoint, cfg: &VisionConfig) -> Result<Self> {
        ensure!(cfg.hidden_size % cfg.num_heads == 0, "vision hidden_size must divide by num_heads");
        let head_dim = cfg.hidden_size / cfg.num_heads;
        ensure!(head_dim % 4 == 0 && head_dim <= 128, "vision head_dim {head_dim} is not supported");
        let act = Act::from_config(cfg.hidden_act.as_deref())?;
        let p = prefix(ck)?;
        let loaded = jit.load(&module())?;
        let _ = ctx;
        let channels = cfg.in_channels.unwrap_or(3);
        let patch_cols = channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let patch = Linear {
            w: upload(stream, ck, &format!("{p}.patch_embed.proj.weight"), &[cfg.hidden_size, channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size])?,
            b: upload(stream, ck, &format!("{p}.patch_embed.proj.bias"), &[cfg.hidden_size])?,
            rows: cfg.hidden_size,
            cols: patch_cols,
        };
        let table = ck.bf16(&format!("{p}.pos_embed.weight"), &[cfg.num_position_embeddings, cfg.hidden_size])?;
        let pos_embed = table.iter().map(|&v| bf16_to_f32(v)).collect();
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
        let main = merger(stream, ck, &format!("{p}.merger"), cfg, false)?;
        let mut deepstack = Vec::new();
        for i in 0..cfg.deepstack_visual_indexes.len() {
            deepstack.push(merger(stream, ck, &format!("{p}.deepstack_merger_list.{i}"), cfg, true)?);
        }
        Ok(VisionEncoder {
            stream: stream.clone(),
            gemm: loaded.function("vis_gemm")?,
            layernorm: loaded.function("vis_layernorm")?,
            rope: loaded.function("vis_rope")?,
            attention: loaded.function("vis_attention")?,
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

    fn gemm(&self, l: &Linear, a: &CudaSlice<f32>, resid: Option<&CudaSlice<f32>>, out: &mut CudaSlice<f32>, rows: usize, act: Act) -> Result<()> {
        let (n, m, k, code) = (rows as i32, l.rows as i32, l.cols as i32, act.code());
        let cfg = LaunchConfig { grid_dim: ((l.rows as u32).div_ceil(64), (rows as u32).div_ceil(64), 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.gemm);
        b.arg(a).arg(&l.w).arg(&l.b);
        match resid {
            Some(r) => b.arg(r),
            None => b.arg(&0u64),
        };
        b.arg(out).arg(&n).arg(&m).arg(&k).arg(&code);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn gemm_in_place_resid(&self, l: &Linear, a: &CudaSlice<f32>, x: &mut CudaSlice<f32>, rows: usize) -> Result<()> {
        let (n, m, k, code) = (rows as i32, l.rows as i32, l.cols as i32, 0i32);
        let cfg = LaunchConfig { grid_dim: ((l.rows as u32).div_ceil(64), (rows as u32).div_ceil(64), 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&self.gemm);
        let xp = &*x;
        b.arg(a).arg(&l.w).arg(&l.b).arg(xp).arg(xp).arg(&n).arg(&m).arg(&k).arg(&code);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn layernorm(&self, x: &CudaSlice<f32>, nm: &Norm, out: &mut CudaSlice<f32>, rows: usize, width: usize) -> Result<()> {
        let w = width as i32;
        let mut b = self.stream.launch_builder(&self.layernorm);
        b.arg(x).arg(&nm.w).arg(&nm.b).arg(out).arg(&w).arg(&LAYERNORM_EPS);
        unsafe { b.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(())
    }

    fn run_merger(&self, m: &Merger, x: &CudaSlice<f32>, tokens: usize, scratch: &mut CudaSlice<f32>, mid: &mut CudaSlice<f32>) -> Result<Vec<f32>> {
        let unit = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let merged_rows = tokens / unit;
        let merged = self.cfg.hidden_size * unit;
        if m.post_shuffle_norm {
            self.layernorm(x, &m.norm, scratch, merged_rows, merged)?;
        } else {
            self.layernorm(x, &m.norm, scratch, tokens, self.cfg.hidden_size)?;
        }
        let mut up = self.stream.alloc_zeros::<f32>(merged_rows * merged)?;
        self.gemm(&m.up, scratch, None, &mut up, merged_rows, Act::GeluErf)?;
        let mut out = self.stream.alloc_zeros::<f32>(merged_rows * self.cfg.out_hidden_size)?;
        self.gemm(&m.down, &up, None, &mut out, merged_rows, Act::None)?;
        let _ = mid;
        Ok(self.stream.clone_dtoh(&out)?)
    }

    pub fn encode(&self, patches: &Patches) -> Result<VisionOutput> {
        let cfg = &self.cfg;
        let [t, h, w] = patches.grid;
        let merge = cfg.spatial_merge_size;
        ensure!(h % merge == 0 && w % merge == 0, "grid {h}x{w} is not a multiple of the merge size");
        let tokens = t * h * w;
        let hidden = cfg.hidden_size;
        let head_dim = hidden / cfg.num_heads;
        let patch_dim = self.patch.cols;
        ensure!(patches.data.len() == tokens * patch_dim, "patch tensor has {} values, grid needs {}", patches.data.len(), tokens * patch_dim);
        let side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        let pos = interpolated_positions(&self.pos_embed, hidden, side, patches.grid, merge);
        let (cos, sin) = rope_tables(patches.grid, merge, head_dim);
        let frame = h * w;
        let seg_lo: Vec<i32> = (0..tokens).map(|i| (i / frame * frame) as i32).collect();
        let seg_hi: Vec<i32> = (0..tokens).map(|i| (i / frame * frame + frame) as i32).collect();
        let s = &self.stream;
        let input = s.clone_htod(&patches.data)?;
        let pos = s.clone_htod(&pos)?;
        let cos = s.clone_htod(&cos)?;
        let sin = s.clone_htod(&sin)?;
        let lo = s.clone_htod(&seg_lo)?;
        let hi = s.clone_htod(&seg_hi)?;
        let mut x = s.alloc_zeros::<f32>(tokens * hidden)?;
        let mut xn = s.alloc_zeros::<f32>(tokens * hidden)?;
        let mut qkv = s.alloc_zeros::<f32>(tokens * 3 * hidden)?;
        let mut att = s.alloc_zeros::<f32>(tokens * hidden)?;
        let mut mid = s.alloc_zeros::<f32>(tokens * cfg.intermediate_size)?;
        self.gemm(&self.patch, &input, Some(&pos), &mut x, tokens, Act::None)?;
        let (heads, hd, scale) = (cfg.num_heads as i32, head_dim as i32, 1.0 / (head_dim as f32).sqrt());
        let mut deepstack = Vec::new();
        let mut scratch = s.alloc_zeros::<f32>(tokens * hidden)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            self.layernorm(&x, &blk.norm1, &mut xn, tokens, hidden)?;
            self.gemm(&blk.qkv, &xn, None, &mut qkv, tokens, Act::None)?;
            let mut b = s.launch_builder(&self.rope);
            b.arg(&mut qkv).arg(&cos).arg(&sin).arg(&heads).arg(&hd);
            unsafe { b.launch(LaunchConfig { grid_dim: (tokens as u32, cfg.num_heads as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 }) }?;
            let mut b = s.launch_builder(&self.attention);
            b.arg(&qkv).arg(&mut att).arg(&lo).arg(&hi).arg(&heads).arg(&hd).arg(&scale);
            unsafe { b.launch(LaunchConfig { grid_dim: (tokens as u32, cfg.num_heads as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }) }?;
            self.gemm_in_place_resid(&blk.proj, &att, &mut x, tokens)?;
            self.layernorm(&x, &blk.norm2, &mut xn, tokens, hidden)?;
            self.gemm(&blk.up, &xn, None, &mut mid, tokens, self.act)?;
            self.gemm_in_place_resid(&blk.down, &mid, &mut x, tokens)?;
            if let Some(d) = cfg.deepstack_visual_indexes.iter().position(|&b| b == i) {
                let mut dummy = s.alloc_zeros::<f32>(1)?;
                deepstack.push(self.run_merger(&self.deepstack[d], &x, tokens, &mut scratch, &mut dummy)?);
            }
        }
        let mut dummy = s.alloc_zeros::<f32>(1)?;
        let embeds = self.run_merger(&self.merger, &x, tokens, &mut scratch, &mut dummy)?;
        Ok(VisionOutput { embeds, deepstack, tokens: tokens / (merge * merge) })
    }
}
