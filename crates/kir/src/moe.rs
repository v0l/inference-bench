use crate::{Feature, Language, Target, fnv};

pub const MMA_BM: usize = 64;
pub const MMA_BM_SMALL: usize = 16;
pub const MMA_BN: usize = 64;
pub const MMA_STAGES: usize = 3;
pub const MMA_ROW: usize = 72;
pub const MMA_THREADS: u32 = 128;

pub use crate::ops::MoeShape;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Experts,
    ExpertTable,
    GroupedMma,
    GroupedMx,
    GroupedMmaFp8,
    GroupedMmaBf16,
}

impl Family {
    pub const fn kernels(self) -> &'static [&'static str] {
        match self {
            Family::Experts => &[
                "moe_route_logits",
                "moe_route",
                "moe_up_bf16",
                "moe_up_fp4",
                "moe_down_bf16",
                "moe_down_fp4",
                "moe_combine",
                "moe_scale_inputs",
                "moe_permute",
                "moe_group_up_bf16",
                "moe_group_up_fp4",
                "moe_group_down_bf16",
                "moe_group_down_fp4",
                "moe_up_fp8",
                "moe_down_fp8",
                "moe_group_up_fp8",
                "moe_group_down_fp8",
                "moe_up_mx",
                "moe_down_mx",
                "moe_group_up_mx",
                "moe_group_down_mx",
            ],
            Family::ExpertTable => &["moe_up_t_bf16", "moe_up_t_fp4", "moe_up_t_fp8", "moe_up_t_mx", "moe_down_t_bf16", "moe_down_t_fp4", "moe_down_t_fp8", "moe_down_t_mx"],
            Family::GroupedMma => &["moe_quant_f32", "moe_tiles", "moe_mma_up", "moe_mma_up_s", "moe_mma_down", "moe_mma_down_s"],
            Family::GroupedMmaBf16 => &["moe_split_bf16", "moe_tiles_b16", "moe_mma_up_b16", "moe_mma_up_b16_s", "moe_mma_down_b16", "moe_mma_down_b16_s"],
            Family::GroupedMmaFp8 => &["moe_quant_f8", "moe_tiles_f8", "moe_mma_up_f8", "moe_mma_up_f8_s", "moe_mma_down_f8", "moe_mma_down_f8_s"],
            Family::GroupedMx => &[
                "moe_quant_mx",
                "moe_tiles",
                "moe_mma_mx_up",
                "moe_mma_mx_up_s",
                "moe_mma_mx_down",
                "moe_mma_mx_down_s",
                "moe_mma_mx_up2",
                "moe_mma_mx_up2_s",
                "moe_mma_mx_down2",
                "moe_mma_mx_down2_s",
            ],
        }
    }

    pub fn needs(self) -> &'static [Feature] {
        match self {
            Family::Experts | Family::ExpertTable => &[Feature::WarpShuffle],
            Family::GroupedMma | Family::GroupedMx => &[Feature::BlockScaledFp4, Feature::CpAsync],
            Family::GroupedMmaFp8 => &[Feature::Fp8Mma, Feature::CpAsync],
            Family::GroupedMmaBf16 => &[Feature::Mma16816, Feature::CpAsync],
        }
    }
}

pub const MMA_STAGES_SMALL: usize = 6;

pub const EXPERT_THREADS: u32 = 256;
pub const EXPERT_WARPS: usize = EXPERT_THREADS as usize / 32;
pub const LOGIT_PER_WARP: usize = 4;
pub const LOGIT_TOKENS: usize = 4;
pub const ROUTE_THREADS: u32 = 32;
pub const PERMUTE_THREADS: u32 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: (u32, u32),
    pub threads: u32,
    pub smem: u32,
}

pub fn route_logits_smem(hidden: usize) -> u32 {
    (LOGIT_TOKENS * hidden * 4) as u32
}

pub fn route_logits_launch(experts: usize, tokens: usize, hidden: usize) -> Launch {
    Launch { grid: ((experts + 1).div_ceil(EXPERT_WARPS) as u32, tokens.div_ceil(LOGIT_TOKENS) as u32), threads: EXPERT_THREADS, smem: route_logits_smem(hidden) }
}

pub fn expert_rows_launch(slots: u32, rows: usize) -> Launch {
    Launch { grid: (slots, rows.div_ceil(EXPERT_WARPS) as u32), threads: EXPERT_THREADS, smem: 0 }
}

pub fn rows_launch(width: usize, rows: usize) -> Launch {
    Launch { grid: (width.div_ceil(EXPERT_THREADS as usize) as u32, rows as u32), threads: EXPERT_THREADS, smem: 0 }
}

pub fn mma_smem(rows_per_tile: usize, weights: usize) -> u32 {
    let stages = if rows_per_tile == MMA_BM_SMALL { MMA_STAGES_SMALL } else { MMA_STAGES };
    (stages * (rows_per_tile + weights * MMA_BN) * MMA_ROW) as u32
}

pub const MMA_BF16_ROW: usize = 64;

pub fn mma_bf16_smem(rows_per_tile: usize, weights: usize) -> u32 {
    (MMA_STAGES * (2 * rows_per_tile + weights * MMA_BN) * MMA_BF16_ROW) as u32
}

pub const MMA8_ROW: usize = 128;

pub fn mma8_smem(rows_per_tile: usize, weights: usize) -> u32 {
    (MMA_STAGES * (rows_per_tile + weights * MMA_BN) * MMA8_ROW) as u32
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeKernel {
    pub family: Family,
    pub target: Target,
    pub shape: MoeShape,
}

impl MoeKernel {
    pub fn new(t: &Target, family: Family, shape: MoeShape) -> Result<MoeKernel, String> {
        let k = MoeKernel { family, target: *t, shape };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> {
        let t = &self.target;
        let s = &self.shape;
        let bad = |m: String| Err(format!("moe {:?}: {m}", self.family));
        if t.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", t.language));
        }
        for &f in self.family.needs() {
            if !t.has(f) {
                return bad(format!("needs {f:?}, which the target lacks"));
            }
        }
        if s.experts == 0 || s.top_k == 0 || s.top_k > s.experts || s.top_k > 32 {
            return bad(format!("top_k {} of {} experts is out of range", s.top_k, s.experts));
        }
        if s.pairs != s.top_k + s.shared {
            return bad("pairs must be top_k plus the shared experts".into());
        }
        if s.inter == 0 {
            return bad("expert intermediate must be positive".into());
        }
        if self.family == Family::GroupedMx && s.block_group != 32 {
            return bad(format!("the block scaled mma reads groups of 32, not {}", s.block_group));
        }
        if self.family == Family::Experts && route_logits_smem(s.hidden) > t.smem_limit {
            return bad(format!("moe_route_logits needs {} bytes of shared memory, the target allows {}", route_logits_smem(s.hidden), t.smem_limit));
        }
        if matches!(self.family, Family::GroupedMma | Family::GroupedMx | Family::GroupedMmaFp8 | Family::GroupedMmaBf16) {
            if !s.inter.is_multiple_of(MMA_BN) {
                return bad(format!("expert intermediate {} must be a multiple of {MMA_BN}", s.inter));
            }
            let smem = match self.family {
                Family::GroupedMmaFp8 => mma8_smem(MMA_BM, 2),
                Family::GroupedMmaBf16 => mma_bf16_smem(MMA_BM, 2),
                _ => mma_smem(MMA_BM, 2),
            };
            if smem > t.smem_limit {
                return bad(format!("needs {smem} bytes of shared memory, the target allows {}", t.smem_limit));
            }
        }
        Ok(())
    }

    pub fn smem(&self) -> Vec<(&'static str, u32)> {
        let up = if self.shape.mapped { 1 } else { 2 };
        match self.family {
            Family::ExpertTable => vec![],
            Family::Experts => vec![("moe_route_logits", route_logits_smem(self.shape.hidden))],
            Family::GroupedMma => vec![
                ("moe_mma_up", mma_smem(MMA_BM, 2)),
                ("moe_mma_up_s", mma_smem(MMA_BM_SMALL, 2)),
                ("moe_mma_down", mma_smem(MMA_BM, 1)),
                ("moe_mma_down_s", mma_smem(MMA_BM_SMALL, 1)),
            ],
            Family::GroupedMmaBf16 => vec![
                ("moe_mma_up_b16", mma_bf16_smem(MMA_BM, up)),
                ("moe_mma_up_b16_s", mma_bf16_smem(MMA_BM_SMALL, up)),
                ("moe_mma_down_b16", mma_bf16_smem(MMA_BM, 1)),
                ("moe_mma_down_b16_s", mma_bf16_smem(MMA_BM_SMALL, 1)),
            ],
            Family::GroupedMmaFp8 => vec![
                ("moe_mma_up_f8", mma8_smem(MMA_BM, 2)),
                ("moe_mma_up_f8_s", mma8_smem(MMA_BM_SMALL, 2)),
                ("moe_mma_down_f8", mma8_smem(MMA_BM, 1)),
                ("moe_mma_down_f8_s", mma8_smem(MMA_BM_SMALL, 1)),
            ],
            Family::GroupedMx => vec![
                ("moe_mma_mx_up", mma_smem(MMA_BM, 2)),
                ("moe_mma_mx_up_s", mma_smem(MMA_BM_SMALL, 2)),
                ("moe_mma_mx_down", mma_smem(MMA_BM, 1)),
                ("moe_mma_mx_down_s", mma_smem(MMA_BM_SMALL, 1)),
                ("moe_mma_mx_up2", mma_smem(MMA_BM, 2)),
                ("moe_mma_mx_up2_s", mma_smem(MMA_BM_SMALL, 2)),
                ("moe_mma_mx_down2", mma_smem(MMA_BM, 1)),
                ("moe_mma_mx_down2_s", mma_smem(MMA_BM_SMALL, 1)),
            ],
        }
    }

    pub fn arch_features(&self) -> bool {
        matches!(self.family, Family::GroupedMma | Family::GroupedMx)
    }

    pub fn describe(&self) -> String {
        format!("{}\nmoe {:?}\n  {:?}", self.target.describe(), self.family, self.shape)
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

pub fn moe_hash(kernels: &[MoeKernel]) -> String {
    fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}
