use crate::compose::{Mat, Stage, Tile as StageTile};
use crate::expr::Expr;
use crate::gemm::{Epilogue, Fp4Gemv, Fp8Scale, GemmError, GemmShape, Kind, KernelInstance, Mainloop, Tile, Tuning, WeightFormat, ggml_gemv_rows};
use crate::gemv::{Gemv, GemvKernel};
use crate::layout::Layout;
use crate::{Activation, Target};
use std::fmt;

pub const ACC: &str = "acc";
pub const OUT: &str = "out";
pub const PLANES: &str = "planes";
pub const BIAS: &str = "bias";
pub const X: &str = "x";
pub const NORM: &str = "nw";
pub const STAGED: &str = "xs";
pub const BASE_REGS: u32 = 16;

pub fn acc_regs(elems: usize, threads: u32) -> u32 {
    BASE_REGS + (elems / threads.max(1) as usize) as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Tile { wf: WeightFormat, split: bool, token_major: bool },
    Fp8Act { scale: Fp8Scale },
    Fp4Act { stages: usize },
    Batch { wf: WeightFormat, tokens: usize },
    GgmlBatch { tokens: usize, split: bool },
    GgmlGemv { split: bool },
    BatchReduce { wf: WeightFormat },
    Gemv { wf: WeightFormat },
    Fp4Gemv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Problem {
    pub n: usize,
    pub k: usize,
    pub act: Activation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Prologue,
    Epilogue,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Heavy {
    pub name: String,
    pub family: Family,
    pub problem: Problem,
    pub tile: Option<Tile>,
    pub prologue: Vec<Stage>,
    pub epilogue: Vec<Stage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub threads: u32,
    pub regs: u32,
    pub smem: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeavyError {
    Lower { pass: String, kernel: String, reason: String },
    Gemm { pass: String, kernel: String, error: GemmError },
    Budget { pass: String, kernel: String, what: &'static str, need: u64, limit: u64 },
}

impl HeavyError {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, HeavyError::Gemm { error, .. } if error.is_unsupported())
    }
}

impl fmt::Display for HeavyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeavyError::Lower { pass, kernel, reason } => write!(f, "pass {pass}: kernel {kernel}: {reason}"),
            HeavyError::Gemm { pass, kernel, error } => write!(f, "pass {pass}: kernel {kernel}: {error}"),
            HeavyError::Budget { pass, kernel, what, need, limit } => write!(f, "pass {pass}: kernel {kernel} needs {need} {what}, the target has {limit}"),
        }
    }
}

impl std::error::Error for HeavyError {}

const EPILOGUES: [Epilogue; 4] = [Epilogue::Store, Epilogue::Residual, Epilogue::Act, Epilogue::Map];

fn expr(text: &str) -> Expr {
    Expr::parse(text).expect("built in epilogue expressions parse")
}

fn load(dst: &str, src: &str, tile: Option<StageTile>) -> Stage {
    Stage::TileLoad { dst: dst.into(), src: src.into(), tile, mat: Mat::default() }
}

fn compute(dst: &str, text: &str) -> Stage {
    Stage::Epilogue { dst: dst.into(), expr: expr(text), mat: Mat::default() }
}

fn store(dst: &str, value: &str) -> Stage {
    Stage::TileStore { dst: dst.into(), value: value.into(), tile: None }
}

fn pair(n: usize, half: &str) -> Result<StageTile, String> {
    let layout = Layout::flat(&[(n / 2) as u64, 2], &[2, 1])?;
    Ok(StageTile { layout, at: vec!["i".into(), half.into()] })
}

pub fn epilogue(epi: Epilogue, bias: bool, n: usize) -> Result<Vec<Stage>, String> {
    let mut out = Vec::new();
    let mut value = ACC;
    if bias {
        if matches!(epi, Epilogue::Act | Epilogue::Map) {
            return Err(format!("the {epi:?} epilogue takes no bias"));
        }
        out.push(load("b", BIAS, None));
        out.push(compute("v", "acc + b"));
        value = "v";
    }
    match epi {
        Epilogue::Store => out.push(store(OUT, value)),
        Epilogue::Residual => out.extend([load("r", OUT, None), compute("y", &format!("{value} + r")), store(OUT, "y")]),
        Epilogue::Act => {
            if n < 2 {
                return Err(format!("the gated epilogue needs at least 2 columns, not {n}"));
            }
            out.extend([load("g", ACC, Some(pair(n, "0")?)), load("u", ACC, Some(pair(n, "1")?)), compute("y", "gated(g, u)"), store(PLANES, "y")]);
        }
        Epilogue::Map => out.extend([compute("y", "mapped(acc)"), store(PLANES, "y")]),
    }
    Ok(out)
}

pub fn epilogue_into(epi: Epilogue, bias: bool, n: usize, dst: &str) -> Result<Vec<Stage>, String> {
    let mut out = epilogue(epi, bias, n)?;
    if let Some(Stage::TileStore { dst: d, .. }) = out.last_mut() {
        *d = dst.to_string();
    }
    Ok(out)
}

fn spelled_into(stages: &[Stage], n: usize, dst: Option<&str>) -> Result<(Epilogue, bool), String> {
    for epi in EPILOGUES {
        for bias in [false, true] {
            let spelt = match dst {
                Some(d) => epilogue_into(epi, bias, n, d),
                None => epilogue(epi, bias, n),
            };
            if spelt.is_ok_and(|s| s == stages) {
                return Ok((epi, bias));
            }
        }
    }
    Err(format!("no emitter spells the epilogue {stages:?}"))
}

pub fn spelled(stages: &[Stage], n: usize) -> Result<(Epilogue, bool), String> {
    spelled_into(stages, n, None)
}

pub fn norm_prologue() -> Vec<Stage> {
    vec![
        load("v", X, None),
        Stage::BlockSum { dst: "total".into(), expr: expr("v * v") },
        compute("inv", "rsqrt(total / width + eps)"),
        load("xv", X, None),
        load("w", NORM, None),
        compute("y", "w * (xv * inv)"),
        store(STAGED, "y"),
    ]
}

pub fn spelled_prologue(stages: &[Stage]) -> Result<bool, String> {
    if stages.is_empty() {
        return Ok(false);
    }
    if stages == norm_prologue() {
        return Ok(true);
    }
    Err(format!("no emitter spells the prologue {stages:?}"))
}

fn defined(s: &Stage) -> Option<&str> {
    match s {
        Stage::TileLoad { dst, .. } | Stage::Epilogue { dst, .. } | Stage::BlockSum { dst, .. } | Stage::Gather { dst, .. } | Stage::Lookup { dst, .. } => Some(dst),
        Stage::TileStore { .. } | Stage::Scatter { .. } => None,
    }
}

fn rename(s: Stage, from: &str, to: &str) -> Result<Stage, String> {
    let name = |n: String| if n == from { to.to_string() } else { n };
    let e = |x: Expr| x.subst(&|n| Some(Expr::Name(if n == from { to.to_string() } else { n.to_string() })));
    Ok(match s {
        Stage::TileLoad { dst, src, tile, mat } => Stage::TileLoad { dst, src: name(src), tile, mat },
        Stage::Epilogue { dst, expr, mat } => Stage::Epilogue { dst, expr: e(expr)?, mat },
        Stage::BlockSum { dst, expr } => Stage::BlockSum { dst, expr: e(expr)? },
        Stage::TileStore { dst, value, tile } => Stage::TileStore { dst, value: name(value), tile },
        Stage::Gather { dst, src, index } => Stage::Gather { dst, src: name(src), index },
        Stage::Lookup { dst, src, index } => Stage::Lookup { dst, src: name(src), index },
        Stage::Scatter { dst, value, index } => Stage::Scatter { dst, value: name(value), index },
    })
}

fn shape_of(k: &KernelInstance) -> Option<GemmShape> {
    match k.kind {
        Kind::Tile { shape, .. } | Kind::Fp8Act { shape, .. } | Kind::Fp4Act { shape, .. } | Kind::Batch { shape, .. } | Kind::GgmlBatch { shape, .. } | Kind::GgmlGemv { shape, .. } | Kind::BatchReduce { shape, .. } => Some(shape),
        Kind::QuantFp8 { .. } | Kind::QuantFp4 | Kind::Gemv { .. } | Kind::HeadNormFp4 { .. } => None,
    }
}

impl Heavy {
    pub fn new(name: &str, family: Family, problem: Problem, tile: Option<Tile>) -> Heavy {
        Heavy { name: name.to_string(), family, problem, tile, prologue: Vec::new(), epilogue: Vec::new() }
    }

    fn slotted(mut self, normed: bool, epilogue: Result<Vec<Stage>, String>) -> Result<Heavy, String> {
        if normed {
            self.prologue = norm_prologue();
        }
        self.epilogue = epilogue.map_err(|e| format!("{}: {e}", self.name))?;
        Ok(self)
    }

    fn lift_fp4(k: &KernelInstance) -> Result<Option<Heavy>, GemmError> {
        let gemv = |n: usize, kk: usize| Heavy::new(&k.name, Family::Fp4Gemv, Problem { n, k: kk, act: crate::Activation::Silu }, None);
        let h = match k.kind {
            Kind::Gemv { op: Fp4Gemv::GateUpAct, n, k: kk } => gemv(n, kk).slotted(true, epilogue_into(Epilogue::Act, false, n, OUT)),
            Kind::Gemv { op: Fp4Gemv::Down, n, k: kk } => gemv(n, kk).slotted(false, epilogue(Epilogue::Residual, false, n)),
            Kind::HeadNormFp4 { n, k: kk, .. } => gemv(n, kk).slotted(true, epilogue(Epilogue::Store, false, n)),
            _ => return Ok(None),
        };
        h.map(Some).map_err(GemmError::Invalid)
    }

    pub fn lift_gemv(k: &GemvKernel, hidden: usize, inter: usize) -> Result<Option<Heavy>, String> {
        let family = |decoder: &str| WeightFormat::from_decoder(decoder).map(|wf| Family::Gemv { wf }).ok_or_else(|| format!("{}: unknown decoder {decoder}", k.name));
        let gemv = |decoder: &str, n: usize, kk: usize| -> Result<Heavy, String> { Ok(Heavy::new(&k.name, family(decoder)?, Problem { n, k: kk, act: crate::Activation::Silu }, None)) };
        let h = match &k.op {
            Gemv::Linear { decoder, n, k: kk, bias, residual } => {
                let epi = if *residual { Epilogue::Residual } else { Epilogue::Store };
                gemv(decoder, *n, *kk)?.slotted(false, epilogue(epi, *bias, *n))?
            }
            Gemv::Norm { decoder, n, bias } => gemv(decoder, *n, hidden)?.slotted(true, epilogue(Epilogue::Store, *bias, *n))?,
            Gemv::GateUpAct { decoder } => gemv(decoder, 2 * inter, hidden)?.slotted(true, epilogue_into(Epilogue::Act, false, 2 * inter, OUT))?,
            Gemv::Tiled { .. } | Gemv::Bf16Rows { .. } => return Ok(None),
        };
        Ok(Some(h))
    }

    fn unspelled(&self, what: &str) -> String {
        format!("{}: no {what} emitter spells the prologue {:?} with the epilogue {:?}", self.name, self.prologue, self.epilogue)
    }

    pub fn lower_gemv(&self, t: &Target) -> Result<GemvKernel, String> {
        let Family::Gemv { wf } = self.family else { return Err(format!("{}: family {:?} does not lower to a gemv", self.name, self.family)) };
        let normed = spelled_prologue(&self.prologue).map_err(|e| format!("{}: {e}", self.name))?;
        let (epi, bias) = spelled_into(&self.epilogue, self.problem.n, Some(OUT)).map_err(|_| self.unspelled("gemv"))?;
        let decoder = wf.decoder().to_string();
        let (n, k) = (self.problem.n, self.problem.k);
        let op = match (normed, epi) {
            (false, Epilogue::Store | Epilogue::Residual) => Gemv::Linear { decoder, n, k, bias, residual: epi == Epilogue::Residual },
            (true, Epilogue::Store) => Gemv::Norm { decoder, n, bias },
            (true, Epilogue::Act) => Gemv::GateUpAct { decoder },
            _ => return Err(self.unspelled("gemv")),
        };
        GemvKernel::new(t, &self.name, op)
    }

    pub fn compile_gemv(&self, t: &Target, pass: &str) -> Result<GemvKernel, HeavyError> {
        self.budget(t, pass)?;
        self.lower_gemv(t).map_err(|reason| HeavyError::Lower { pass: pass.into(), kernel: self.name.clone(), reason })
    }

    fn lower_fp4(&self, t: &Target) -> Result<KernelInstance, GemmError> {
        let normed = spelled_prologue(&self.prologue).map_err(|e| GemmError::Invalid(format!("{}: {e}", self.name)))?;
        let spelt = spelled_into(&self.epilogue, self.problem.n, Some(OUT)).map_err(|_| GemmError::Invalid(self.unspelled("fp4 gemv")))?;
        let (n, k) = (self.problem.n, self.problem.k);
        match (normed, spelt) {
            (true, (Epilogue::Act, false)) => KernelInstance::fp4_gemv(t, &self.name, Fp4Gemv::GateUpAct, n, k),
            (false, (Epilogue::Residual, false)) => KernelInstance::fp4_gemv(t, &self.name, Fp4Gemv::Down, n, k),
            (true, (Epilogue::Store, false)) => KernelInstance::fp4_head_norm(t, &self.name, n, k),
            _ => Err(GemmError::Invalid(self.unspelled("fp4 gemv"))),
        }
    }

    pub fn lift(k: &KernelInstance) -> Result<Option<Heavy>, GemmError> {
        if matches!(k.kind, Kind::Gemv { .. } | Kind::HeadNormFp4 { .. }) {
            return Heavy::lift_fp4(k);
        }
        let Some(shape) = shape_of(k) else { return Ok(None) };
        let (family, tile) = match k.kind {
            Kind::Tile { wf, main: Mainloop::Mma(m), token_major, .. } => (Family::Tile { wf, split: m.split, token_major }, Some(m.tile)),
            Kind::Tile { wf, token_major, .. } => (Family::Tile { wf, split: false, token_major }, k.tile()),
            Kind::Fp8Act { tile, scale, .. } => (Family::Fp8Act { scale }, Some(tile)),
            Kind::Fp4Act { tile, stages, .. } => (Family::Fp4Act { stages }, Some(tile)),
            Kind::Batch { wf, batch, .. } => (Family::Batch { wf, tokens: batch.tokens }, None),
            Kind::GgmlBatch { batch, split, .. } => (Family::GgmlBatch { tokens: batch.tokens, split }, None),
            Kind::GgmlGemv { split, .. } => (Family::GgmlGemv { split }, None),
            Kind::BatchReduce { wf, .. } => (Family::BatchReduce { wf }, None),
            Kind::QuantFp8 { .. } | Kind::QuantFp4 | Kind::Gemv { .. } | Kind::HeadNormFp4 { .. } => return Ok(None),
        };
        let mut h = Heavy::new(&k.name, family, Problem { n: shape.n, k: shape.k, act: shape.act }, tile);
        h.epilogue = epilogue(shape.epi, shape.bias, shape.n).map_err(|e| GemmError::Invalid(format!("{}: {e}", k.name)))?;
        Ok(Some(h))
    }

    fn build(&self, t: &Target, shape: GemmShape) -> Result<KernelInstance, GemmError> {
        let tile = || self.tile.ok_or_else(|| GemmError::Invalid(format!("{}: family {:?} needs a tile", self.name, self.family)));
        let name = self.name.as_str();
        match self.family {
            Family::Tile { wf, split, token_major } => KernelInstance::gemm(t, name, shape, wf, tile()?, split, token_major),
            Family::Fp8Act { scale } => KernelInstance::fp8_act(t, name, shape, tile()?, scale),
            Family::Fp4Act { stages } => KernelInstance::fp4_act(t, name, shape, tile()?, &Tuning { fp4_stages: Some(stages), ..Tuning::default() }),
            Family::Batch { wf, tokens } => KernelInstance::batch(t, name, shape, wf, tokens),
            Family::GgmlBatch { tokens, split } => KernelInstance::ggml_batch(t, name, shape, tokens, split),
            Family::GgmlGemv { split } => KernelInstance::ggml_gemv(t, name, shape, split),
            Family::BatchReduce { wf } => KernelInstance::batch_reduce(t, name, shape, wf),
            Family::Fp4Gemv => KernelInstance::fp4_gemv(t, name, Fp4Gemv::Down, shape.n, shape.k),
            Family::Gemv { .. } => Err(GemmError::Invalid(format!("{name}: family {:?} lowers to a gemv kernel, not a gemm", self.family))),
        }
    }

    pub fn lower(&self, t: &Target) -> Result<KernelInstance, GemmError> {
        match self.family {
            Family::Fp4Gemv => return self.lower_fp4(t),
            Family::Gemv { .. } => return Err(GemmError::Invalid(format!("{}: family {:?} lowers to a gemv kernel, not a gemm", self.name, self.family))),
            _ => {}
        }
        if !self.prologue.is_empty() {
            return Err(GemmError::Invalid(format!("{}: family {:?} has no prologue emitter", self.name, self.family)));
        }
        let (epi, bias) = spelled(&self.epilogue, self.problem.n).map_err(|e| GemmError::Invalid(format!("{}: {e}", self.name)))?;
        self.build(t, GemmShape::new(self.problem.n, self.problem.k, epi, bias).with_act(self.problem.act))
    }

    fn rows_and_acc(&self, k: &KernelInstance) -> (usize, usize) {
        match k.kind {
            Kind::Batch { batch, .. } | Kind::GgmlBatch { batch, .. } => (batch.tokens, batch.tokens * batch.rows()),
            Kind::GgmlGemv { shape, .. } => (1, k.threads as usize / 32 * ggml_gemv_rows(shape.n)),
            Kind::BatchReduce { .. } => (1, k.threads as usize),
            _ => self.tile.or(k.tile()).map_or((1, k.threads as usize), |t| (t.bm, t.bm * t.bn)),
        }
    }

    fn mainloop(&self, t: &Target, pass: &str) -> Result<(u32, u32, usize, usize), HeavyError> {
        if let Family::Gemv { wf } = self.family {
            let g = GemvKernel::new(t, &self.name, Gemv::Linear { decoder: wf.decoder().into(), n: self.problem.n, k: self.problem.k, bias: false, residual: false })
                .map_err(|reason| HeavyError::Lower { pass: pass.into(), kernel: self.name.clone(), reason })?;
            return Ok((g.threads(), 0, 1, g.threads() as usize));
        }
        let bare = GemmShape::new(self.problem.n, self.problem.k, Epilogue::Store, false).with_act(self.problem.act);
        let main = self.build(t, bare).map_err(|error| HeavyError::Gemm { pass: pass.into(), kernel: self.name.clone(), error })?;
        let (rows, acc) = self.rows_and_acc(&main);
        Ok((main.threads, main.smem_bytes, rows, acc))
    }

    pub fn budget(&self, t: &Target, pass: &str) -> Result<Budget, HeavyError> {
        let (threads, smem, rows, acc) = self.mainloop(t, pass)?;
        let held = self.prologue.iter().chain(&self.epilogue).filter_map(defined).count() as u32;
        let staged = if self.prologue.is_empty() { 0 } else { rows * self.problem.k * 4 };
        let b = Budget { threads, regs: acc_regs(acc, threads) + held, smem: smem + staged as u32 };
        let over = |what: &'static str, need: u64, limit: u64| HeavyError::Budget { pass: pass.into(), kernel: self.name.clone(), what, need, limit };
        if b.threads > t.max_threads {
            return Err(over("threads", b.threads as u64, t.max_threads as u64));
        }
        if b.smem > t.smem_limit {
            return Err(over("bytes of shared memory", b.smem as u64, t.smem_limit as u64));
        }
        if b.regs > t.max_regs {
            return Err(over("registers per thread", b.regs as u64, t.max_regs as u64));
        }
        Ok(b)
    }

    pub fn compile(&self, t: &Target, pass: &str) -> Result<KernelInstance, HeavyError> {
        self.budget(t, pass)?;
        self.lower(t).map_err(|error| HeavyError::Gemm { pass: pass.into(), kernel: self.name.clone(), error })
    }

    fn check(&self, t: &Target, pass: &str) -> Result<(), HeavyError> {
        match self.family {
            Family::Gemv { .. } => self.compile_gemv(t, pass).map(|_| ()),
            _ => self.compile(t, pass).map(|_| ()),
        }
    }

    pub fn attach(mut self, t: &Target, pass: &str, slot: Slot, stages: Vec<Stage>) -> Result<Heavy, HeavyError> {
        let lower = |reason: String| HeavyError::Lower { pass: pass.into(), kernel: self.name.clone(), reason };
        match slot {
            Slot::Prologue => self.prologue.extend(stages),
            Slot::Epilogue => {
                let value = match self.epilogue.last() {
                    Some(Stage::TileStore { value, .. }) => value.clone(),
                    _ => return Err(lower("the epilogue slot does not end in a store".into())),
                };
                self.epilogue.pop();
                let mut renamed = Vec::new();
                for s in stages {
                    let s = rename(s, ACC, &value).map_err(lower)?;
                    if let Some(d) = defined(&s)
                        && self.epilogue.iter().chain(&renamed).filter_map(defined).any(|x| x == d)
                    {
                        return Err(lower(format!("stage name {d:?} is already defined in the epilogue")));
                    }
                    renamed.push(s);
                }
                self.epilogue.extend(renamed);
            }
        }
        self.check(t, pass)?;
        Ok(self)
    }
}
