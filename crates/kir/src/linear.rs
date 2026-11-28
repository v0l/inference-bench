use crate::{Feature, Language, Target, fnv};
use std::fmt::{self, Write as _};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearTuning {
    pub chunk: usize,
    pub ksplit: usize,
    pub cols: usize,
}

impl LinearTuning {
    pub fn for_target(t: &Target) -> LinearTuning {
        let _ = t;
        LinearTuning { chunk: 64, ksplit: 4, cols: 32 }
    }

    pub fn chunk_threads(&self) -> u32 {
        (self.chunk / 16 * 32) as u32
    }
}

#[derive(Clone, PartialEq)]
pub struct LinearShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub tiled_value_heads: bool,
    pub gate_sigmoid: bool,
    pub chunk_single_bf16: bool,
}

impl fmt::Debug for LinearShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("LinearShape");
        d.field("key_heads", &self.key_heads).field("value_heads", &self.value_heads).field("head_dim", &self.head_dim).field("conv_kernel", &self.conv_kernel).field("tiled_value_heads", &self.tiled_value_heads);
        if self.gate_sigmoid {
            d.field("gate_sigmoid", &true);
        }
        if self.chunk_single_bf16 {
            d.field("chunk_single_bf16", &true);
        }
        d.finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Conv,
    Step,
    Chunk,
    Verify,
    Draft,
    DraftStreams,
}

impl Family {
    pub const ALL: [Family; 5] = [Family::Conv, Family::Step, Family::Chunk, Family::Verify, Family::Draft];

    pub fn block(self) -> &'static str {
        match self {
            Family::Conv => "GdnConv",
            Family::Step => "GdnStep",
            Family::Chunk => "GdnChunk",
            Family::Verify => "GdnVerify",
            Family::Draft => "MtpDraft",
            Family::DraftStreams => "MtpDraftStreams",
        }
    }

    pub fn kernels(self) -> &'static [&'static str] {
        match self {
            Family::Conv => &["gdn_conv_step", "gdn_conv_prefill", "gdn_conv_keep"],
            Family::Step => &["gdn_step", "gdn_qk_prep", "gdn_gates"],
            Family::Chunk => &["gdn_chunk_prep", "gdn_chunk_scan", "gdn_gated_norm"],
            Family::Verify => &["gdn_conv_verify", "gdn_conv_commit", "gdn_verify", "gdn_commit", "gdn_defer"],
            Family::Draft => &[
                "mtp_fc_in",
                "mtp_draft",
                "mtp_fill",
                "mtp_copy",
                "mtp_pick_rows",
                "mtp_round_in",
                "mtp_chain_rows",
                "spec_accept",
            ],
            Family::DraftStreams => &["mtp_streams_in", "mtp_copy_w", "mtp_state_save", "mtp_state_load", "mtp_replay_rows", "mtp_replay_seqs"],
        }
    }

    pub fn needs(self) -> &'static [Feature] {
        match self {
            Family::Chunk => &[Feature::WarpShuffle, Feature::Mma16816],
            _ => &[Feature::WarpShuffle],
        }
    }

    pub fn uses_shape(self) -> bool {
        !matches!(self, Family::Draft | Family::DraftStreams)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearKernel {
    pub family: Family,
    pub target: Target,
    pub shape: LinearShape,
    pub tuning: LinearTuning,
    pub threads: u32,
    pub smem_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearError(pub String);

impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LinearError {}

impl LinearKernel {
    pub fn new(t: &Target, family: Family, shape: LinearShape) -> Result<LinearKernel, LinearError> {
        let tuning = LinearTuning::for_target(t);
        let (threads, smem_bytes) = match family {
            Family::Step | Family::Verify => ((shape.head_dim * tuning.ksplit) as u32, 0),
            Family::Chunk => {
                let floats = tuning.chunk * (shape.head_dim + 1) + tuning.chunk * (tuning.chunk + 1) + 2 * tuning.chunk;
                (256.max(tuning.chunk_threads()), (floats * 4) as u32)
            }
            Family::Conv | Family::Draft | Family::DraftStreams => (256, 0),
        };
        let k = LinearKernel { family, target: *t, shape, tuning, threads, smem_bytes };
        k.validate()?;
        Ok(k)
    }

    pub fn set(t: &Target, shape: LinearShape) -> Result<Vec<LinearKernel>, LinearError> {
        Family::ALL.iter().map(|&f| LinearKernel::new(t, f, shape.clone())).collect()
    }

    pub fn validate(&self) -> Result<(), LinearError> {
        let t = &self.target;
        let bad = |s: String| Err(LinearError(format!("{}: {s}", self.family.block())));
        if t.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", t.language));
        }
        for &f in self.family.needs() {
            if !t.has(f) {
                return bad(format!("needs {f:?}, which the target lacks"));
            }
        }
        if self.threads > t.max_threads {
            return bad(format!("{} threads exceed the target maximum {}", self.threads, t.max_threads));
        }
        if self.smem_bytes > t.smem_limit {
            return bad(format!("needs {} bytes of shared memory, the target allows {}", self.smem_bytes, t.smem_limit));
        }
        if self.family.uses_shape() {
            let s = &self.shape;
            let u = &self.tuning;
            if s.key_heads == 0 || s.value_heads == 0 || !s.value_heads.is_multiple_of(s.key_heads) {
                return bad("value heads must be a multiple of key heads".into());
            }
            if s.head_dim == 0 || s.head_dim > 128 || !s.head_dim.is_multiple_of(u.cols) {
                return bad(format!("head_dim {} must be a multiple of {} and at most 128", s.head_dim, u.cols));
            }
            if u.ksplit == 0 || !s.head_dim.is_multiple_of(u.ksplit) {
                return bad(format!("head_dim {} must divide by ksplit {}", s.head_dim, u.ksplit));
            }
            if s.conv_kernel < 2 {
                return bad("conv kernel must be at least 2".into());
            }
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!(
            "{}\nlinear {} threads={} smem={} chunk={} ksplit={} cols={}\n  {:?}",
            self.target.describe(),
            self.family.block(),
            self.threads,
            self.smem_bytes,
            self.tuning.chunk,
            self.tuning.ksplit,
            self.tuning.cols,
            self.shape
        )
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

pub fn linear_hash(kernels: &[LinearKernel]) -> String {
    fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}

pub(crate) fn constants(s: &LinearShape, u: &LinearTuning) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "constexpr int LA_KH = {};", s.key_heads);
    let _ = writeln!(o, "constexpr int LA_VH = {};", s.value_heads);
    let _ = writeln!(o, "constexpr int LA_D = {};", s.head_dim);
    let _ = writeln!(o, "constexpr int LA_K = {};", s.conv_kernel);
    for l in [
        "constexpr int LA_KEY_DIM = LA_KH * LA_D;",
        "constexpr int LA_VALUE_DIM = LA_VH * LA_D;",
        "constexpr int LA_CONV_DIM = 2 * LA_KEY_DIM + LA_VALUE_DIM;",
        "constexpr int LA_IN_ROWS = LA_CONV_DIM + LA_VALUE_DIM;",
        "constexpr int LA_BA_ROWS = 2 * LA_VH;",
        "constexpr int LA_RATIO = LA_VH / LA_KH;",
        "constexpr int LA_V_OFF = 2 * LA_KEY_DIM;",
        "constexpr int LA_Z_OFF = LA_CONV_DIM;",
        "constexpr int LA_B_OFF = 0;",
        "constexpr int LA_A_OFF = LA_VH;",
        "constexpr int LA_STATE = LA_D * LA_D;",
    ] {
        let _ = writeln!(o, "{l}");
    }
    let _ = writeln!(o, "constexpr bool LA_TILED = {};", s.tiled_value_heads);
    let _ = writeln!(o, "__device__ __forceinline__ int la_kh(int h) {{");
    let _ = writeln!(o, "    return LA_TILED ? h % LA_KH : h / LA_RATIO;");
    let _ = writeln!(o, "}}");
    let _ = writeln!(o, "constexpr int LA_CHUNK = {};", u.chunk);
    let _ = writeln!(o, "constexpr int LA_KSPLIT = {};", u.ksplit);
    let _ = writeln!(o, "constexpr int LA_KS = LA_D + 1;");
    let _ = writeln!(o, "constexpr int LA_AS = LA_CHUNK + 1;");
    let _ = writeln!(o, "constexpr int LA_COLS = {};", u.cols);
    if s.chunk_single_bf16 {
        let _ = writeln!(o, "constexpr bool LA_LO = false;");
    } else {
        let _ = writeln!(o, "constexpr bool LA_LO = true;");
    }
    let _ = writeln!(o, "constexpr float LA_L2_EPS = 1e-6f;");
    let _ = writeln!(o, "static_assert(LA_D % LA_COLS == 0 && LA_D <= 128, \"linear head dim\");");
    let _ = writeln!(o, "static_assert(LA_VH % LA_KH == 0, \"linear head groups\");");
    o
}
