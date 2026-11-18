use crate::{Feature, Language, Target, fnv};
use std::fmt;

pub const GEMM_BM: u32 = 128;
pub const GEMM_BN: u32 = 128;
pub const GEMM_THREADS: u32 = 256;
pub const GEMM_SMEM: u32 = 3 * (128 * 32 * 2 * 2 + 128 * 32 * 2);
pub const FLASH_BQ: u32 = 128;
pub const FLASH_THREADS: u32 = 256;
pub const MAX_HEAD_DIM: usize = 128;

pub const KERNELS: &[&str] = &["vis_mma", "vis_split", "vis_layernorm", "vis_pos", "vis_rope_pack", "vis_flash", "vis_to_f16", "vis_pool"];

#[derive(Clone, Debug, PartialEq)]
pub struct VisionKernel {
    pub target: Target,
    pub head_dim: usize,
    pub precise: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionError(pub String);

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VisionError {}

pub fn head_pad(head_dim: usize) -> usize {
    head_dim.div_ceil(16) * 16
}

impl VisionKernel {
    pub fn new(t: &Target, head_dim: usize, precise: bool) -> Result<VisionKernel, VisionError> {
        let k = VisionKernel { target: *t, head_dim, precise };
        k.validate()?;
        Ok(k)
    }

    pub fn head_pad(&self) -> usize {
        head_pad(self.head_dim)
    }

    pub fn gemm_smem(&self) -> u32 {
        GEMM_SMEM
    }

    pub fn flash_smem(&self) -> u32 {
        let stride = self.head_pad() + 8;
        (2 * 128 * stride * 2).max(2 * 4 * 32 * stride * 2) as u32
    }

    pub fn validate(&self) -> Result<(), VisionError> {
        let t = &self.target;
        let bad = |s: String| Err(VisionError(format!("vision: {s}")));
        if t.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", t.language));
        }
        for f in [Feature::CpAsync, Feature::Mma16816] {
            if !t.has(f) {
                return bad(format!("needs {f:?}, which the target lacks"));
            }
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(8) || self.head_dim > MAX_HEAD_DIM {
            return bad(format!("head_dim {} must be a multiple of 8 and at most {MAX_HEAD_DIM}", self.head_dim));
        }
        if GEMM_THREADS > t.max_threads || FLASH_THREADS > t.max_threads {
            return bad("the target has too few threads per block".into());
        }
        let smem = self.gemm_smem().max(self.flash_smem());
        if smem > t.smem_limit {
            return bad(format!("needs {smem} bytes of shared memory, the target allows {}", t.smem_limit));
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!("{}\nvision head_dim={} hdp={} precise={} gemm_smem={} flash_smem={}", self.target.describe(), self.head_dim, self.head_pad(), self.precise, self.gemm_smem(), self.flash_smem())
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}
