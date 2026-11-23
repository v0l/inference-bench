#[cfg(feature = "cuda")]
pub mod cuda;
pub mod allreduce;
pub mod attention;
pub mod compose;
pub mod desc;
pub mod expr;
pub mod gemm;
pub mod gemv;
pub mod ggml;
pub mod heavy;
pub mod layout;
pub mod linear;
pub mod moe;
pub mod ops;
pub mod prims;
pub mod recur;
pub mod rope;
pub mod rows;
pub mod scan;
pub mod target;
pub mod vision;
#[cfg(feature = "wgsl")]
pub mod wgsl;

pub use layout::{Dialect, Layout};
pub use target::{BlockSum, Feature, Language, Target};

use std::fmt::{self, Write as _};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    U32,
    Bf16,
    I32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Weight,
    Activation,
    Scratch,
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub ty: Scalar,
    pub role: Role,
    pub access: Access,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activation {
    Silu,
    GeluTanh,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pointwise {
    Scale(f32),
    SoftCap(f32),
    Fill(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    RmsNorm { width: u32, eps: f32, sum: BlockSum, offset: f32 },
    RmsNormSegment { width: u32, stride: u32, start: u32, eps: f32, sum: BlockSum },
    AddRmsNorm { width: u32, eps: f32, sum: BlockSum, offset: f32 },
    NormAdd { width: u32, eps: f32, sum: BlockSum, offset: f32 },
    Pointwise { width: u32, op: Pointwise },
    Add { width: u32, scale: f32 },
    GatedActivation { width: u32, act: Activation },
    Compose(compose::Compose),
    Scan(scan::Scan),
    Rows(rows::RowBody),
    Recurrence(recur::Recurrence),
    Seq(recur::SeqBody),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: &'static str,
    pub target: Target,
    pub params: Vec<Param>,
    pub threads: u32,
    pub smem_bytes: u32,
    pub body: Body,
}

fn param(name: &'static str, role: Role, access: Access) -> Param {
    Param { name, ty: Scalar::F32, role, access }
}

fn pick_threads(t: &Target, wanted: u32) -> u32 {
    let mut n = wanted.min(t.max_threads).max(1);
    while !n.is_power_of_two() {
        n &= n - 1;
    }
    n
}

impl Kernel {
    pub fn rmsnorm(t: &Target, width: u32, eps: f32, threads: u32) -> Result<Kernel, String> {
        Kernel::rmsnorm_offset(t, width, eps, 0.0, threads)
    }

    pub fn rmsnorm_offset(t: &Target, width: u32, eps: f32, offset: f32, threads: u32) -> Result<Kernel, String> {
        let threads = pick_threads(t, threads);
        let sum = BlockSum::for_target(t);
        let smem_bytes = match sum {
            BlockSum::WarpShuffle => 32 * 4,
            BlockSum::Subgroup | BlockSum::SharedTree => threads * 4,
        };
        let k = Kernel {
            name: "rmsnorm",
            target: *t,
            params: vec![
                param("x", Role::Activation, Access::Read),
                param("weight", Role::Weight, Access::Read),
                param("out", Role::Activation, Access::Write),
            ],
            threads,
            smem_bytes,
            body: Body::RmsNorm { width, eps, sum, offset },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn rmsnorm_segment(t: &Target, name: &'static str, width: u32, stride: u32, start: u32, eps: f32, threads: u32) -> Result<Kernel, String> {
        let threads = pick_threads(t, threads);
        let sum = BlockSum::for_target(t);
        let smem_bytes = match sum {
            BlockSum::WarpShuffle => 32 * 4,
            BlockSum::Subgroup | BlockSum::SharedTree => threads * 4,
        };
        let k = Kernel {
            name,
            target: *t,
            params: vec![
                param("x", Role::Activation, Access::Read),
                param("weight", Role::Weight, Access::Read),
                param("out", Role::Activation, Access::Write),
            ],
            threads,
            smem_bytes,
            body: Body::RmsNormSegment { width, stride, start, eps, sum },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn add_rmsnorm(t: &Target, width: u32, eps: f32, offset: f32, threads: u32) -> Result<Kernel, String> {
        let threads = pick_threads(t, threads);
        let sum = BlockSum::for_target(t);
        let smem_bytes = match sum {
            BlockSum::WarpShuffle => 32 * 4,
            BlockSum::Subgroup | BlockSum::SharedTree => threads * 4,
        };
        let k = Kernel {
            name: "add_rmsnorm",
            target: *t,
            params: vec![
                param("x", Role::Activation, Access::Read),
                param("delta", Role::Activation, Access::Read),
                param("weight", Role::Weight, Access::Read),
                param("sum", Role::Activation, Access::Write),
                param("out", Role::Activation, Access::Write),
            ],
            threads,
            smem_bytes,
            body: Body::AddRmsNorm { width, eps, sum, offset },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn add(t: &Target, name: &'static str, width: u32, threads: u32) -> Result<Kernel, String> {
        let k = Kernel {
            name,
            target: *t,
            params: vec![
                param("x", Role::Activation, Access::Read),
                param("delta", Role::Activation, Access::Read),
                param("out", Role::Activation, Access::Write),
            ],
            threads: pick_threads(t, threads),
            smem_bytes: 0,
            body: Body::Add { width, scale: 1.0 },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn add_scaled(t: &Target, name: &'static str, width: u32, scale: f32, threads: u32) -> Result<Kernel, String> {
        let mut k = Kernel::add(t, name, width, threads)?;
        k.body = Body::Add { width, scale };
        k.validate()?;
        Ok(k)
    }

    pub fn norm_add(t: &Target, width: u32, eps: f32, offset: f32, threads: u32) -> Result<Kernel, String> {
        let threads = pick_threads(t, threads);
        let sum = BlockSum::for_target(t);
        let smem_bytes = match sum {
            BlockSum::WarpShuffle => 32 * 4,
            BlockSum::Subgroup | BlockSum::SharedTree => threads * 4,
        };
        let k = Kernel {
            name: "norm_add",
            target: *t,
            params: vec![
                param("delta", Role::Activation, Access::Read),
                param("weight", Role::Weight, Access::Read),
                param("resid", Role::Activation, Access::Read),
                param("out", Role::Activation, Access::Write),
            ],
            threads,
            smem_bytes,
            body: Body::NormAdd { width, eps, sum, offset },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn pointwise(t: &Target, name: &'static str, width: u32, op: Pointwise, threads: u32) -> Result<Kernel, String> {
        let k = Kernel {
            name,
            target: *t,
            params: vec![param("x", Role::Activation, Access::Read), param("out", Role::Activation, Access::Write)],
            threads: pick_threads(t, threads),
            smem_bytes: 0,
            body: Body::Pointwise { width, op },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn gated_activation(t: &Target, name: &'static str, width: u32, act: Activation, threads: u32) -> Result<Kernel, String> {
        let k = Kernel {
            name,
            target: *t,
            params: vec![
                param("gate_up", Role::Activation, Access::Read),
                param("out", Role::Activation, Access::Write),
            ],
            threads: pick_threads(t, threads),
            smem_bytes: 0,
            body: Body::GatedActivation { width, act },
        };
        k.validate()?;
        Ok(k)
    }

    pub fn compose(t: &Target, name: &'static str, threads: u32, params: Vec<Param>, body: compose::Compose) -> Result<Kernel, String> {
        let threads = pick_threads(t, threads);
        let smem_bytes = body.smem_bytes(threads);
        let k = Kernel { name, target: *t, params, threads, smem_bytes, body: Body::Compose(body) };
        k.validate()?;
        Ok(k)
    }

    pub fn scan(t: &Target, name: &'static str, threads: u32, params: Vec<Param>, body: scan::Scan) -> Result<Kernel, String> {
        let k = Kernel { name, target: *t, params, threads: pick_threads(t, threads), smem_bytes: 0, body: Body::Scan(body) };
        k.validate()?;
        Ok(k)
    }

    pub fn rows(t: &Target, name: &'static str, threads: u32, params: Vec<Param>, body: rows::RowBody) -> Result<Kernel, String> {
        let k = Kernel { name, target: *t, params, threads: pick_threads(t, threads), smem_bytes: 0, body: Body::Rows(body) };
        k.validate()?;
        Ok(k)
    }

    pub fn recurrence(t: &Target, name: &'static str, threads: u32, params: Vec<Param>, body: recur::Recurrence) -> Result<Kernel, String> {
        let k = Kernel { name, target: *t, params, threads, smem_bytes: 0, body: Body::Recurrence(body) };
        k.validate()?;
        Ok(k)
    }

    pub fn seq(t: &Target, name: &'static str, threads: u32, params: Vec<Param>, body: recur::SeqBody) -> Result<Kernel, String> {
        let k = Kernel { name, target: *t, params, threads: pick_threads(t, threads), smem_bytes: 0, body: Body::Seq(body) };
        k.validate()?;
        Ok(k)
    }

    pub fn tile_scan(&self) -> bool {
        match &self.body {
            Body::Scan(s) => s.tile_eligible() && s.tile_names_free(&self.params) && self.target.has(Feature::Mma16816) && self.target.has(Feature::CpAsync) && self.target.language == Language::Cuda,
            _ => false,
        }
    }

    pub fn chunk_form(&self) -> Option<recur::ChunkForm> {
        match &self.body {
            Body::Recurrence(r) => r.chunk_form(self),
            _ => None,
        }
    }

    pub fn needs(&self) -> Vec<Feature> {
        match &self.body {
            Body::RmsNorm { sum, .. } | Body::RmsNormSegment { sum, .. } | Body::AddRmsNorm { sum, .. } | Body::NormAdd { sum, .. } => sum.needs().to_vec(),
            Body::Compose(c) => c.sum.map(|s| s.needs().to_vec()).unwrap_or_default(),
            Body::Scan(_) => vec![Feature::WarpShuffle],
            Body::Rows(_) => vec![],
            Body::Recurrence(_) | Body::Seq(_) => vec![Feature::WarpShuffle],
            Body::GatedActivation { .. } | Body::Pointwise { .. } | Body::Add { .. } => vec![],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let t = &self.target;
        if self.name.is_empty() || !self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("kernel name {:?} is not an identifier", self.name));
        }
        let warp_multiple = matches!(self.body, Body::Recurrence(_)) && self.threads.is_multiple_of(32);
        if self.threads == 0 || !(self.threads.is_power_of_two() || warp_multiple) || self.threads > t.max_threads {
            return Err(format!(
                "{}: threads {} must be a power of two up to the target limit {}",
                self.name, self.threads, t.max_threads
            ));
        }
        if self.smem_bytes > t.smem_limit {
            return Err(format!("{}: {} bytes of shared memory exceeds the target limit {}", self.name, self.smem_bytes, t.smem_limit));
        }
        for f in self.needs() {
            if !t.has(f) {
                return Err(format!("{}: target lacks {f:?}", self.name));
            }
        }
        let roles: Vec<(Role, Access)> = self.params.iter().map(|p| (p.role, p.access)).collect();
        let expected: &[(Role, Access)] = match &self.body {
            Body::AddRmsNorm { width, eps, sum, .. } => {
                if *width == 0 || !eps.is_finite() || *eps < 0.0 {
                    return Err(format!("{}: bad add_rmsnorm width or eps", self.name));
                }
                if sum.language() != t.language {
                    return Err(format!("{}: block sum {sum:?} cannot run on {:?}", self.name, t.language));
                }
                &[
                    (Role::Activation, Access::Read),
                    (Role::Activation, Access::Read),
                    (Role::Weight, Access::Read),
                    (Role::Activation, Access::Write),
                    (Role::Activation, Access::Write),
                ]
            }
            Body::NormAdd { width, eps, sum, .. } => {
                if *width == 0 || !eps.is_finite() || *eps < 0.0 {
                    return Err(format!("{}: bad norm_add width or eps", self.name));
                }
                if sum.language() != t.language {
                    return Err(format!("{}: block sum {sum:?} cannot run on {:?}", self.name, t.language));
                }
                &[(Role::Activation, Access::Read), (Role::Weight, Access::Read), (Role::Activation, Access::Read), (Role::Activation, Access::Write)]
            }
            Body::Add { width, scale } => {
                if *width == 0 || !scale.is_finite() {
                    return Err(format!("{}: width must be positive", self.name));
                }
                &[(Role::Activation, Access::Read), (Role::Activation, Access::Read), (Role::Activation, Access::Write)]
            }
            Body::Pointwise { width, op } => {
                let v = match op {
                    Pointwise::Scale(v) | Pointwise::SoftCap(v) | Pointwise::Fill(v) => *v,
                };
                if *width == 0 || !v.is_finite() || (matches!(op, Pointwise::SoftCap(_)) && v <= 0.0) {
                    return Err(format!("{}: bad pointwise width or parameter", self.name));
                }
                &[(Role::Activation, Access::Read), (Role::Activation, Access::Write)]
            }
            Body::RmsNorm { width, eps, sum, .. } => {
                if *width == 0 || !eps.is_finite() || *eps < 0.0 {
                    return Err(format!("{}: bad rmsnorm width or eps", self.name));
                }
                if sum.language() != t.language {
                    return Err(format!("{}: block sum {sum:?} cannot run on {:?}", self.name, t.language));
                }
                &[(Role::Activation, Access::Read), (Role::Weight, Access::Read), (Role::Activation, Access::Write)]
            }
            Body::RmsNormSegment { width, stride, start, eps, sum } => {
                if *width == 0 || start + width > *stride || !eps.is_finite() || *eps < 0.0 {
                    return Err(format!("{}: bad segment norm width, stride, start or eps", self.name));
                }
                if sum.language() != t.language {
                    return Err(format!("{}: block sum {sum:?} cannot run on {:?}", self.name, t.language));
                }
                &[(Role::Activation, Access::Read), (Role::Weight, Access::Read), (Role::Activation, Access::Write)]
            }
            Body::Compose(c) => {
                c.check(t, &self.params)?;
                &roles
            }
            Body::Scan(c) => {
                c.check(t, self.threads, &self.params)?;
                &roles
            }
            Body::Rows(c) => {
                c.check(t, &self.params)?;
                &roles
            }
            Body::Recurrence(c) => {
                c.check(t, self.threads, &self.params)?;
                &roles
            }
            Body::Seq(c) => {
                c.check(t, &self.params)?;
                &roles
            }
            Body::GatedActivation { width, .. } => {
                if *width == 0 {
                    return Err(format!("{}: width must be positive", self.name));
                }
                &[(Role::Activation, Access::Read), (Role::Activation, Access::Write)]
            }
        };
        if roles != expected {
            return Err(format!("{}: params {roles:?} do not match the body, want {expected:?}", self.name));
        }
        let wide = |p: &Param| match &self.body {
            Body::Scan(_) | Body::Rows(_) | Body::Recurrence(_) | Body::Seq(_) => true,
            Body::Compose(_) => p.ty == Scalar::I32,
            _ => false,
        };
        if self.params.iter().any(|p| p.ty != Scalar::F32 && !wide(p)) {
            return Err(format!("{}: only f32 params are supported", self.name));
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        let _ = write!(s, "{}\nkernel {} threads={} smem={}", self.target.describe(), self.name, self.threads, self.smem_bytes);
        for p in &self.params {
            let _ = write!(s, "\n  param {} {:?} {:?} {:?}", p.name, p.ty, p.role, p.access);
        }
        match &self.body {
            Body::RmsNorm { width, eps, sum, offset } => {
                let _ = write!(s, "\n  rmsnorm width={width} eps={eps:e} sum={sum:?}");
                if *offset != 0.0 {
                    let _ = write!(s, " offset={offset:e}");
                }
            }
            Body::RmsNormSegment { width, stride, start, eps, sum } => {
                let _ = write!(s, "\n  rmsnorm_segment width={width} stride={stride} start={start} eps={eps:e} sum={sum:?}");
            }
            Body::AddRmsNorm { width, eps, sum, offset } => {
                let _ = write!(s, "\n  add_rmsnorm width={width} eps={eps:e} sum={sum:?} offset={offset:e}");
            }
            Body::NormAdd { width, eps, sum, offset } => {
                let _ = write!(s, "\n  norm_add width={width} eps={eps:e} sum={sum:?} offset={offset:e}");
            }
            Body::Add { width, scale } => {
                let _ = write!(s, "\n  add width={width}");
                if *scale != 1.0 {
                    let _ = write!(s, " scale={scale:e}");
                }
            }
            Body::Pointwise { width, op } => {
                let _ = write!(s, "\n  pointwise width={width} op={op:?}");
            }
            Body::Compose(c) => {
                let _ = write!(s, "\n  {}", c.describe());
            }
            Body::Scan(c) => {
                let _ = write!(s, "\n  {}", c.describe());
            }
            Body::Rows(c) => {
                let _ = write!(s, "\n  {}", c.describe());
            }
            Body::Recurrence(c) => {
                let _ = write!(s, "\n  {}", c.describe());
            }
            Body::Seq(c) => {
                let _ = write!(s, "\n  {}", c.describe());
            }
            Body::GatedActivation { width, act } => {
                let _ = write!(s, "\n  gated_activation width={width} act={act:?}");
            }
        }
        s
    }

    pub fn hash(&self) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.describe().bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

pub fn module_hash(kernels: &[Kernel]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for k in kernels {
        for b in k.hash().bytes().chain(std::iter::once(0)) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

pub fn fold_hashes(hashes: impl IntoIterator<Item = String>) -> String {
    fnv(hashes.into_iter().flat_map(|h| h.into_bytes().into_iter().chain(std::iter::once(0))))
}

pub(crate) fn fnv(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

pub(crate) fn lit(v: f32) -> String {
    format!("{v:e}")
}
