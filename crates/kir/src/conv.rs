use crate::{Language, Target, fnv};
use std::fmt;

pub const THREADS: u32 = 256;
pub const THREAD_ROWS: usize = 16;
pub const THREAD_COLS: usize = 16;
pub const PATCH_H: usize = 8;
pub const PATCH_W: usize = 16;
pub const K_SLICE: usize = 32;
pub const N_MAX: usize = 128;
pub const N_STEP: usize = 32;
pub const DIRECT_BELOW: usize = 8;
pub const MAX_STAGES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvShape {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub kt: usize,
    pub upsample: bool,
    pub interleave: bool,
    pub line: bool,
    pub stride: usize,
    pub phases: usize,
    pub pad: Option<usize>,
}

impl ConvShape {
    pub fn grid(cin: usize, cout: usize, k: usize, kt: usize, upsample: bool, interleave: bool) -> ConvShape {
        ConvShape { cin, cout, k, kt, upsample, interleave, line: false, stride: 1, phases: 1, pad: None }
    }

    pub fn line(cin: usize, cout: usize, taps: usize, stride: usize, phases: usize) -> ConvShape {
        ConvShape { cin, cout, k: 1, kt: taps, upsample: false, interleave: false, line: true, stride, phases, pad: None }
    }

    pub fn padded(cin: usize, cout: usize, taps: usize, stride: usize, pad: usize) -> ConvShape {
        ConvShape { pad: Some(pad), ..ConvShape::line(cin, cout, taps, stride, 1) }
    }

    pub fn context(&self) -> usize {
        if self.pad.is_some() { 0 } else { self.kt - self.stride }
    }

    pub fn out_len(&self, t: usize) -> usize {
        match self.pad {
            Some(p) => (t + 2 * p).saturating_sub(self.kt) / self.stride + usize::from(t + 2 * p >= self.kt),
            None => t.div_ceil(self.stride),
        }
    }

    pub fn taps(&self) -> usize {
        self.k * self.k * self.kt
    }

    pub fn flops_per_pixel(&self) -> u64 {
        2 * (self.cin * self.cout * self.taps()) as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Tile,
    Direct,
    Rows,
}

pub const ROWS_CHUNK: usize = 8;
pub const ROWS_MAX: usize = 64;
pub const ROWS_FEW_MAX: usize = 512;
pub const ROWS_FEW_BLOCKS: usize = 48;
pub const ROWS_MIN_REDUCTION: usize = 256;

pub fn rows_fit(s: &ConvShape, rows: usize) -> bool {
    let bn = s.cout.div_ceil(N_STEP).max(1).saturating_mul(N_STEP).min(N_MAX);
    let tile_blocks = rows.div_ceil(PATCH_H * PATCH_W) * s.cout.div_ceil(bn);
    s.line && s.cin * s.kt >= ROWS_MIN_REDUCTION && (rows <= ROWS_MAX || (rows <= ROWS_FEW_MAX && tile_blocks < ROWS_FEW_BLOCKS))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvTile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub ph: usize,
    pub pw: usize,
    pub tm: usize,
    pub tn: usize,
    pub stages: usize,
}

impl ConvTile {
    pub fn stage_bytes(&self) -> usize {
        (self.bm + self.bn) * self.bk * 4
    }
}

pub fn family(s: &ConvShape) -> Family {
    if s.cin < DIRECT_BELOW || s.cout < DIRECT_BELOW { Family::Direct } else { Family::Tile }
}

pub fn tile(t: &Target, s: &ConvShape) -> ConvTile {
    let (ph, pw) = if s.k == 1 { (1, PATCH_H * PATCH_W) } else { (PATCH_H, PATCH_W) };
    let bm = ph * pw;
    let bn = s.cout.div_ceil(N_STEP).max(1).saturating_mul(N_STEP).min(N_MAX);
    let bk = K_SLICE.min(s.cin.next_power_of_two()).max(1);
    let stage = (bm + bn) * bk * 4;
    let stages = (t.smem_limit as usize / stage.max(1)).clamp(1, MAX_STAGES);
    ConvTile { bm, bn, bk, ph, pw, tm: bm / THREAD_ROWS, tn: bn / THREAD_COLS, stages }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prologue {
    None,
    GroupApply { groups: usize, silu: bool },
    Act(&'static str),
}

pub const LINE_ACTS: &[&str] = &["gelu", "silu", "gelu_tanh", "elu"];

pub fn line_act_id(p: Prologue) -> usize {
    match p {
        Prologue::Act(name) => LINE_ACTS.iter().position(|a| *a == name).map_or(0, |i| i + 1),
        _ => 0,
    }
}

pub fn act_keeps_zero(name: &str) -> bool {
    crate::expr::Expr::parse(&format!("{name}(x)")).ok().and_then(|e| e.eval(&|n| (n == "x").then_some(0.0))) == Some(0.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Epilogue {
    pub residual: bool,
    pub stats: Option<usize>,
    pub gelu: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvKernel {
    pub target: Target,
    pub shape: ConvShape,
    pub family: Family,
    pub tile: ConvTile,
    pub prologue: Prologue,
    pub epilogue: Epilogue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvError(pub String);

impl fmt::Display for ConvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConvError {}

impl ConvKernel {
    pub fn new(t: &Target, shape: ConvShape, prologue: Prologue, epilogue: Epilogue) -> Result<ConvKernel, ConvError> {
        let k = ConvKernel { target: *t, shape, family: family(&shape), tile: tile(t, &shape), prologue, epilogue };
        k.validate()?;
        Ok(k)
    }

    pub fn rows(t: &Target, shape: ConvShape, epilogue: Epilogue) -> Result<ConvKernel, ConvError> {
        Self::rows_with(t, shape, Prologue::None, epilogue)
    }

    pub fn rows_with(t: &Target, shape: ConvShape, prologue: Prologue, epilogue: Epilogue) -> Result<ConvKernel, ConvError> {
        let k = ConvKernel { target: *t, shape, family: Family::Rows, tile: tile(t, &shape), prologue, epilogue };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), ConvError> {
        let bad = |s: String| Err(ConvError(format!("conv {}: {s}", self.describe_shape())));
        let s = &self.shape;
        if self.target.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", self.target.language));
        }
        if s.cin == 0 || s.cout == 0 || s.k.is_multiple_of(2) || s.kt == 0 {
            return bad("needs channels, an odd spatial kernel and at least one frame tap".into());
        }
        if s.interleave && (!s.cout.is_multiple_of(2) || s.upsample || s.phases != 1) {
            return bad("an interleaved output splits an even channel count and does not upsample".into());
        }
        if let Prologue::Act(name) = self.prologue
            && (!s.line || line_act_id(self.prologue) == 0 || crate::expr::unary_macro(name).is_none() || !act_keeps_zero(name))
        {
            return bad(format!("an input activation {name:?} is a line conv prologue over one of {LINE_ACTS:?} that maps 0 to 0, so the zero pad and tail need no activation"));
        }
        if s.line && (s.k != 1 || s.upsample || s.interleave || s.stride == 0 || s.stride > s.kt || s.phases == 0 || !s.cout.is_multiple_of(s.phases) || matches!(self.prologue, Prologue::GroupApply { .. }) || self.epilogue.stats.is_some()) {
            return bad(format!("a causal line conv over {} taps with stride {} and {} phases needs a 1x1 spatial kernel, whole phases and no group norm", s.kt, s.stride, s.phases));
        }
        if !s.line && (s.stride != 1 || s.phases != 1 || self.epilogue.gelu || s.pad.is_some()) {
            return bad("stride, phases, padding and gelu are line conv options".into());
        }
        if let Some(p) = s.pad
            && (s.phases != 1 || p >= s.kt)
        {
            return bad(format!("a padded line conv over {} taps pads fewer than its taps on each side and has one phase, not pad {p} and {} phases", s.kt, s.phases));
        }
        if self.family == Family::Rows && !s.line {
            return bad("the rows family is a line conv".into());
        }
        if let Prologue::GroupApply { groups, .. } = self.prologue
            && (groups == 0 || groups > MAX_GROUPS || !s.cin.is_multiple_of(groups) || s.kt > 1)
        {
            return bad(format!("a group norm prologue over {groups} groups needs whole groups, at most {MAX_GROUPS}, and a spatial conv"));
        }
        if let Some(g) = self.epilogue.stats
            && (g == 0 || g > MAX_GROUPS || !s.cout.is_multiple_of(g) || s.interleave)
        {
            return bad(format!("group statistics over {g} groups need whole groups, at most {MAX_GROUPS}"));
        }
        if self.smem_bytes() > self.target.smem_limit {
            return bad(format!("needs {} bytes of shared memory, the target allows {}", self.smem_bytes(), self.target.smem_limit));
        }
        if THREADS > self.target.max_threads {
            return bad("the target has too few threads per block".into());
        }
        if self.regs() > self.target.max_regs {
            return bad(format!("needs {} registers per thread", self.regs()));
        }
        Ok(())
    }

    fn describe_shape(&self) -> String {
        let s = &self.shape;
        format!(
            "{}x{}x{} {}->{}{}{}{}{}",
            s.kt,
            s.k,
            s.k,
            s.cin,
            s.cout,
            if s.upsample { " up" } else { "" },
            if s.interleave { " interleave" } else { "" },
            if s.line { format!(" line stride {} phases {}", s.stride, s.phases) } else { String::new() },
            s.pad.map_or(String::new(), |p| format!(" pad {p}"))
        )
    }

    pub fn groups_in(&self) -> usize {
        match self.prologue {
            Prologue::None | Prologue::Act(_) => 1,
            Prologue::GroupApply { groups, .. } => groups,
        }
    }

    pub fn smem_bytes(&self) -> u32 {
        let stats = 16 * self.epilogue.stats.unwrap_or(0) + 8 * self.groups_in();
        let tiles = match self.family {
            Family::Tile => self.tile.stage_bytes() * self.tile.stages,
            Family::Direct | Family::Rows => 0,
        };
        (tiles + stats) as u32
    }

    pub fn regs(&self) -> u32 {
        match self.family {
            Family::Tile => crate::heavy::BASE_REGS + (self.tile.tm * self.tile.tn + self.tile.tm + self.tile.tn) as u32,
            Family::Direct => crate::heavy::BASE_REGS + 8,
            Family::Rows => crate::heavy::BASE_REGS + ROWS_CHUNK as u32,
        }
    }

    pub fn name(&self) -> String {
        format!("{}_{}", if self.shape.line { "line_conv" } else { "vae_conv" }, &self.hash()[..12])
    }

    pub fn line_grid(&self, rows: usize) -> (u32, u32, u32) {
        match self.family {
            Family::Tile => (rows.div_ceil(self.tile.bm) as u32, self.shape.cout.div_ceil(self.tile.bn) as u32, 1),
            Family::Direct => ((rows * self.shape.cout).div_ceil(THREADS as usize).max(1) as u32, 1, 1),
            Family::Rows => (self.shape.cout as u32, 1, 1),
        }
    }

    pub fn grid(&self, frames: usize, h: usize, w: usize) -> (u32, u32, u32) {
        match self.family {
            Family::Tile => (w.div_ceil(self.tile.pw) as u32, h.div_ceil(self.tile.ph) as u32, (frames * self.shape.cout.div_ceil(self.tile.bn)) as u32),
            Family::Direct | Family::Rows => ((frames * h * w * self.shape.cout).div_ceil(THREADS as usize) as u32, 1, 1),
        }
    }

    pub fn describe(&self) -> String {
        let e = &self.epilogue;
        let epi = format!("Epilogue {{ residual: {}, stats: {:?} }}{}", e.residual, e.stats, if e.gelu { " gelu" } else { "" });
        format!("{}\nconv {} family={:?} tile={:?} prologue={:?} epilogue={epi}", self.target.describe(), self.describe_shape(), self.family, self.tile, self.prologue)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

pub const MAX_GROUPS: usize = 64;
pub const ATTENTION_THREADS: u32 = 128;
pub const ROW_THREADS: u32 = 256;

pub fn attention_smem(keys: usize) -> usize {
    (keys + 2 * 32) * 4
}

pub fn attention_fits(t: &Target, keys: usize) -> bool {
    attention_smem(keys) <= t.smem_limit as usize
}

pub const ROW_KERNELS: &[&str] = &["vae_zero", "vae_group_stats", "vae_group_apply", "vae_rms", "vae_upsample", "vae_add", "vae_clamp", "vae_attention"];

pub const LINE_ROW_KERNELS: &[&str] = &["line_rms", "line_dwconv", "line_shift", "line_layernorm", "line_dwup", "line_rope", "line_ring_attention", "line_ring_store", "line_advance"];

pub const RING_THREADS: u32 = 128;

pub fn ring_attention_smem(context: usize, head_dim: usize) -> u32 {
    ((context + head_dim + 64) * 4) as u32
}

pub fn ring_fits(t: &Target, context: usize, head_dim: usize) -> bool {
    ring_attention_smem(context, head_dim) <= t.smem_limit && head_dim.is_multiple_of(2) && context > 0
}
