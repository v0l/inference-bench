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
}

impl ConvShape {
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Epilogue {
    pub residual: bool,
    pub stats: Option<usize>,
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

    pub fn validate(&self) -> Result<(), ConvError> {
        let bad = |s: String| Err(ConvError(format!("conv {}: {s}", self.describe_shape())));
        let s = &self.shape;
        if self.target.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", self.target.language));
        }
        if s.cin == 0 || s.cout == 0 || s.k % 2 == 0 || s.kt == 0 {
            return bad("needs channels, an odd spatial kernel and at least one frame tap".into());
        }
        if s.interleave && (s.cout % 2 != 0 || s.upsample) {
            return bad("an interleaved output splits an even channel count and does not upsample".into());
        }
        if let Prologue::GroupApply { groups, .. } = self.prologue
            && (groups == 0 || s.cin % groups != 0 || s.kt > 1)
        {
            return bad(format!("a group norm prologue over {groups} groups needs whole groups and a spatial conv"));
        }
        if let Some(g) = self.epilogue.stats
            && (g == 0 || s.cout % g != 0 || s.interleave)
        {
            return bad(format!("group statistics over {g} groups need whole groups"));
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
        format!("{}x{}x{} {}->{}{}{}", s.kt, s.k, s.k, s.cin, s.cout, if s.upsample { " up" } else { "" }, if s.interleave { " interleave" } else { "" })
    }

    pub fn groups_in(&self) -> usize {
        match self.prologue {
            Prologue::None => 1,
            Prologue::GroupApply { groups, .. } => groups,
        }
    }

    pub fn smem_bytes(&self) -> u32 {
        let stats = 4 * (2 * self.epilogue.stats.unwrap_or(0) + 2 * self.groups_in());
        let tiles = match self.family {
            Family::Tile => self.tile.stage_bytes() * self.tile.stages,
            Family::Direct => 0,
        };
        (tiles + stats) as u32
    }

    pub fn regs(&self) -> u32 {
        match self.family {
            Family::Tile => crate::heavy::BASE_REGS + (self.tile.tm * self.tile.tn + self.tile.tm + self.tile.tn) as u32,
            Family::Direct => crate::heavy::BASE_REGS + 8,
        }
    }

    pub fn name(&self) -> String {
        format!("vae_conv_{}", &self.hash()[..12])
    }

    pub fn grid(&self, frames: usize, h: usize, w: usize) -> (u32, u32, u32) {
        match self.family {
            Family::Tile => (w.div_ceil(self.tile.pw) as u32, h.div_ceil(self.tile.ph) as u32, (frames * self.shape.cout.div_ceil(self.tile.bn)) as u32),
            Family::Direct => ((frames * h * w * self.shape.cout).div_ceil(THREADS as usize) as u32, 1, 1),
        }
    }

    pub fn describe(&self) -> String {
        format!("{}\nconv {} family={:?} tile={:?} prologue={:?} epilogue={:?}", self.target.describe(), self.describe_shape(), self.family, self.tile, self.prologue, self.epilogue)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

pub fn attention_rows(t: &Target, head_dim: usize, keys: usize) -> Option<usize> {
    [16usize, 8, 4, 2, 1].into_iter().find(|&bq| attention_smem(head_dim, bq, keys) <= t.smem_limit as usize)
}

pub fn attention_smem(head_dim: usize, rows: usize, keys: usize) -> usize {
    ((2 * rows + keys) * head_dim + rows * keys) * 4
}

pub const ATTENTION_KEYS: usize = 32;
pub const ATTENTION_THREADS: u32 = 128;
pub const ROW_THREADS: u32 = 256;
