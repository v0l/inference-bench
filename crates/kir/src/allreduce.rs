use crate::{Language, Target, fnv, lit};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
}

impl Dtype {
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::Bf16 => "bf16",
        }
    }

    pub fn ctype(self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::Bf16 => "bf16_t",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    OneShot,
    TwoShot,
    AddRmsNorm { width: u32, eps: f32 },
    Gather { width: u32, each: u32 },
}

impl Mode {
    pub fn tag(&self) -> &'static str {
        match self {
            Mode::OneShot => "one_shot",
            Mode::TwoShot => "two_shot",
            Mode::AddRmsNorm { .. } => "add_rmsnorm",
            Mode::Gather { .. } => "gather",
        }
    }
}

pub fn gather_span(width: u32, each: u32, rank: u32) -> std::ops::Range<u32> {
    let lo = (rank * each).min(width);
    lo..(lo + each).min(width)
}

#[derive(Clone, Debug, PartialEq)]
pub struct AllReduceKernel {
    pub target: Target,
    pub ranks: u32,
    pub dtype: Dtype,
    pub mode: Mode,
    pub threads: u32,
    pub blocks: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllReduceError(pub String);

impl fmt::Display for AllReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AllReduceError {}

pub const MAX_RANKS: u32 = 8;
pub const MAX_BLOCKS: u32 = 64;
pub const NORM_THREADS: u32 = 256;

impl AllReduceKernel {
    pub fn new(t: &Target, ranks: u32, dtype: Dtype, mode: Mode, blocks: u32) -> Result<AllReduceKernel, AllReduceError> {
        let threads = match mode {
            Mode::AddRmsNorm { .. } => NORM_THREADS,
            _ => 512,
        }
        .min(t.max_threads);
        let k = AllReduceKernel { target: *t, ranks, dtype, mode, threads, blocks };
        k.validate()?;
        Ok(k)
    }

    pub fn name(&self) -> String {
        format!("ar_{}_{}", self.mode.tag(), self.dtype.tag())
    }

    pub fn validate(&self) -> Result<(), AllReduceError> {
        let bad = |s: String| Err(AllReduceError(format!("{}: {s}", self.name())));
        if self.target.language != Language::Cuda {
            return bad(format!("has no {:?} lowering", self.target.language));
        }
        if self.ranks < 2 || self.ranks > MAX_RANKS {
            return bad(format!("{} ranks is outside 2..={MAX_RANKS}", self.ranks));
        }
        if self.blocks == 0 || self.blocks > MAX_BLOCKS {
            return bad(format!("{} blocks is outside 1..={MAX_BLOCKS}", self.blocks));
        }
        if self.threads < self.ranks || self.threads > self.target.max_threads {
            return bad(format!("{} threads do not fit {} ranks and the target maximum {}", self.threads, self.ranks, self.target.max_threads));
        }
        if let Mode::AddRmsNorm { width, eps } = self.mode {
            if width == 0 || width % 4 != 0 || !eps.is_finite() || eps < 0.0 {
                return bad("needs a width that is a positive multiple of 4 and a finite eps".into());
            }
            if self.dtype != Dtype::F32 {
                return bad("the fused norm reduces f32 partials".into());
            }
        }
        if let Mode::Gather { width, each } = self.mode {
            if width == 0 || each == 0 || each * self.ranks < width || each * (self.ranks - 1) >= width {
                return bad(format!("a gather of {width} columns in slices of {each} leaves a rank with nothing or misses a column"));
            }
            if self.dtype != Dtype::F32 {
                return bad("the gather moves f32 logits".into());
            }
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        let mut s = format!(
            "{}\nallreduce {} ranks={} dtype={} threads={} blocks={}",
            self.target.describe(),
            self.mode.tag(),
            self.ranks,
            self.dtype.tag(),
            self.threads,
            self.blocks
        );
        if let Mode::AddRmsNorm { width, eps } = self.mode {
            s.push_str(&format!(" width={width} eps={}", lit(eps)));
        }
        if let Mode::Gather { width, each } = self.mode {
            s.push_str(&format!(" width={width} each={each}"));
        }
        s
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }
}

pub fn allreduce_hash(kernels: &[AllReduceKernel]) -> String {
    fnv(kernels.iter().flat_map(|k| k.hash().into_bytes().into_iter().chain(std::iter::once(0))))
}

pub fn flag_words() -> usize {
    (MAX_RANKS * MAX_BLOCKS + MAX_BLOCKS) as usize
}
