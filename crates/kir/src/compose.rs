use crate::expr::{Const, Expr};
use crate::layout::{Dialect, Layout};
use crate::{Access, BlockSum, Kernel, Param, Target, lit};

pub const RESERVED: [&str; 16] = ["i", "ss", "base", "scratch", "t", "idx", "src", "dst", "p", "red", "li", "wid", "gid", "sg", "sgs", "lane"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iter {
    Block,
    Grid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    Advance,
    Base,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub iter: Iter,
    pub address: Address,
    pub per_row: Vec<(String, u32)>,
    pub stride: Vec<(String, u32)>,
    pub offset: Vec<(String, u32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mat {
    pub cuda: bool,
    pub wgsl: bool,
}

impl Mat {
    fn on(self, d: Dialect) -> bool {
        match d {
            Dialect::Cuda => self.cuda,
            Dialect::Wgsl => self.wgsl,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub layout: Layout,
    pub at: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    TileLoad { dst: String, src: String, tile: Option<Tile>, mat: Mat },
    Epilogue { dst: String, expr: Expr, mat: Mat },
    BlockSum { dst: String, expr: Expr },
    TileStore { dst: String, value: String, tile: Option<Tile> },
    SplitStore { hi: String, lo: String, value: String, tile: Option<Tile> },
    Index { dst: String },
    Gather { dst: String, src: String, index: String },
    Lookup { dst: String, src: String, index: String },
    Scatter { dst: String, value: String, index: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Compose {
    pub width: u32,
    pub rows: Rows,
    pub consts: Vec<Const>,
    pub sum: Option<BlockSum>,
    pub stages: Vec<Stage>,
}

#[derive(Debug, Default)]
struct Plan {
    passes: Vec<Vec<String>>,
    between: Vec<Vec<String>>,
    total: Option<String>,
}

impl Plan {
    fn final_pass(&self) -> &[String] {
        self.passes.last().map_or(&[], |p| p.as_slice())
    }
}

fn decl(d: Dialect) -> &'static str {
    match d {
        Dialect::Cuda => "float",
        Dialect::Wgsl => "let",
    }
}

fn name_ok(n: &str) -> bool {
    !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !n.starts_with(|c: char| c.is_ascii_digit()) && !RESERVED.contains(&n) && n != "width"
}

fn sums(stages: &[Stage]) -> usize {
    stages.iter().filter(|s| matches!(s, Stage::BlockSum { .. })).count()
}

fn scalar_flags(stages: &[Stage], consts: &[Const]) -> Vec<bool> {
    let mut scalars: Vec<String> = Vec::new();
    let mut after = false;
    stages
        .iter()
        .map(|s| match s {
            Stage::BlockSum { dst, .. } => {
                scalars.push(dst.clone());
                after = true;
                false
            }
            Stage::Epilogue { dst, expr, .. } if after => {
                let mut names = Vec::new();
                expr.names(&mut names);
                names.retain(|n| !consts.iter().any(|c| c.name == *n));
                let scalar = !names.is_empty() && names.iter().all(|n| scalars.contains(n));
                if scalar {
                    scalars.push(dst.clone());
                }
                scalar
            }
            _ => false,
        })
        .collect()
}

impl Compose {
    pub fn has_sum(&self) -> bool {
        self.stages.iter().any(|s| matches!(s, Stage::BlockSum { .. }))
    }

    pub fn describe(&self) -> String {
        let mut s = format!("compose width={} rows={:?} sum={:?}", self.width, self.rows, self.sum);
        for c in &self.consts {
            s.push_str(&format!("\n    const {} = {:e} wgsl_const={} fold={}", c.name, c.value, c.wgsl_const, c.fold));
        }
        for st in &self.stages {
            s.push_str(&format!("\n    {st:?}"));
        }
        s
    }

    pub fn smem_bytes(&self, threads: u32) -> u32 {
        match self.sum {
            None => 0,
            Some(BlockSum::WarpShuffle) => 32 * 4,
            Some(BlockSum::Subgroup | BlockSum::SharedTree) => threads * 4,
        }
    }

    pub fn check(&self, t: &Target, params: &[Param]) -> Result<(), String> {
        if self.width == 0 {
            return Err("compose width must be positive".into());
        }
        if params.is_empty() {
            return Err("compose needs params".into());
        }
        if sums(&self.stages) > 1 && t.language != crate::Language::Cuda {
            return Err("compose has more than one block_sum, which only the CUDA lowering runs".into());
        }
        if (!self.rows.stride.is_empty() || !self.rows.offset.is_empty()) && t.language != crate::Language::Cuda {
            return Err("compose row strides and offsets have only a CUDA lowering".into());
        }
        for (n, _) in &self.rows.offset {
            if !params.iter().any(|p| p.name == n) {
                return Err(format!("rows offset names unknown param {n:?}"));
            }
        }
        for (n, s) in &self.rows.stride {
            if !params.iter().any(|p| p.name == n) || *s == 0 || self.rows.per_row.iter().any(|(m, _)| m == n) {
                return Err(format!("rows stride names unknown param {n:?}, a zero stride or a param that also advances per row"));
            }
        }
        if self.has_sum() != self.sum.is_some() {
            return Err("compose block sum and its reduction strategy disagree".into());
        }
        if self.has_sum() && self.rows.iter == Iter::Grid {
            return Err("a block_sum needs block iteration".into());
        }
        if let Some(sum) = self.sum
            && sum.language() != t.language
        {
            return Err(format!("block sum {sum:?} cannot run on {:?}", t.language));
        }
        for (i, p) in params.iter().enumerate() {
            if !name_ok(p.name) || params[..i].iter().any(|q| q.name == p.name) {
                return Err(format!("param name {:?} is not usable", p.name));
            }
        }
        for (i, c) in self.consts.iter().enumerate() {
            if !name_ok(&c.name) || params.iter().any(|p| p.name == c.name) || self.consts[..i].iter().any(|o| o.name == c.name) || !c.value.is_finite() {
                return Err(format!("constant {:?} is not usable", c.name));
            }
        }
        for (n, m) in &self.rows.per_row {
            if !params.iter().any(|p| p.name == n) || *m == 0 {
                return Err(format!("rows name unknown param {n:?} or a zero multiple"));
            }
            if self.rows.address == Address::Base && *m != 1 {
                return Err("base addressing needs unit row multiples".into());
            }
        }
        let d = if t.language == crate::Language::Cuda { Dialect::Cuda } else { Dialect::Wgsl };
        self.plan(d, params)?;
        Ok(())
    }

    fn named(&self) -> Vec<String> {
        self.consts.iter().filter(|c| c.wgsl_const).map(|c| c.name.clone()).collect()
    }

    fn index(&self, d: Dialect, params: &[Param], p: &str, tile: Option<&Tile>) -> Result<String, String> {
        let u = if d == Dialect::Wgsl { "u" } else { "" };
        let local = match tile {
            Some(t) => t.layout.expr(&t.at.iter().map(String::as_str).collect::<Vec<_>>(), d)?,
            None => "i".to_string(),
        };
        if self.rows.stride.iter().any(|(n, _)| n == p) {
            return Ok(local);
        }
        let Some((_, m)) = self.rows.per_row.iter().find(|(n, _)| n == p) else {
            return Ok(local);
        };
        Ok(match (d, self.rows.address, self.rows.iter) {
            (Dialect::Cuda, Address::Advance, _) => local,
            (Dialect::Cuda, Address::Base, _) => format!("base + {local}"),
            (Dialect::Wgsl, _, Iter::Block) => {
                let write = params.iter().any(|q| q.name == p && q.access == Access::Write);
                format!("{} + {local}", if write { "dst" } else { "src" })
            }
            (Dialect::Wgsl, _, Iter::Grid) if tile.is_none() && *m == 1 => "idx".to_string(),
            (Dialect::Wgsl, _, Iter::Grid) if *m == 1 => format!("t * WIDTH + {local}"),
            (Dialect::Wgsl, _, Iter::Grid) => format!("t * {m}{u} * WIDTH + {local}"),
        })
    }

    fn row_index(&self, d: Dialect, params: &[Param], table: &str, index: &str, wide: bool) -> Result<String, String> {
        if d != Dialect::Cuda {
            return Err("gather, lookup and scatter have no WGSL lowering".into());
        }
        let table_param = params.iter().find(|p| p.name == table).ok_or_else(|| format!("indexed access to unknown param {table:?}"))?;
        let index_param = params.iter().find(|p| p.name == index).ok_or_else(|| format!("index param {index:?} is unknown"))?;
        if index_param.ty != crate::Scalar::I32 || index_param.access != Access::Read {
            return Err(format!("index param {index:?} must be a read i32 buffer"));
        }
        if table_param.ty != crate::Scalar::F32 {
            return Err(format!("indexed param {table:?} must be f32"));
        }
        if self.rows.per_row.iter().any(|(n, _)| n == table || n == index) {
            return Err(format!("{table:?} and {index:?} are indexed, so they cannot also advance per row"));
        }
        let _ = wide;
        let row = match self.rows.iter {
            Iter::Block => "x",
            Iter::Grid => "y",
        };
        Ok(format!("(size_t){index}[blockIdx.{row}]"))
    }

    fn plan(&self, d: Dialect, params: &[Param]) -> Result<Plan, String> {
        let named = self.named();
        let flags = scalar_flags(&self.stages, &self.consts);
        let sum_text = if d == Dialect::Cuda { "kir_block_sum(ss, scratch)" } else { "" };
        let mut env: Vec<(String, Expr)> = Vec::new();
        let mut plan = Plan::default();
        let mut defined: Vec<String> = Vec::new();
        let mut scalar_names: Vec<String> = Vec::new();
        let mut define = |n: &str| -> Result<(), String> {
            if !name_ok(n) || params.iter().any(|p| p.name == n) || self.consts.iter().any(|c| c.name == n) || defined.iter().any(|e| e == n) {
                return Err(format!("stage name {n:?} is not usable or is defined twice"));
            }
            defined.push(n.to_string());
            Ok(())
        };
        let total_sums = sums(&self.stages);
        plan.passes.push(Vec::new());
        for (st, scalar) in self.stages.iter().zip(flags.iter()) {
            macro_rules! emit {
                ($line:expr) => {
                    plan.passes.last_mut().expect("a pass").push($line)
                };
            }
            match st {
                Stage::TileLoad { dst, src, tile, mat } => {
                    define(dst)?;
                    if !params.iter().any(|p| p.name == src) {
                        return Err(format!("load from unknown param {src:?}"));
                    }
                    let text = format!("{src}[{}]", self.index(d, params, src, tile.as_ref())?);
                    if mat.on(d) {
                        emit!(format!("{} {dst} = {text};", decl(d)));
                        env.push((dst.clone(), Expr::Raw(dst.clone())));
                    } else {
                        env.push((dst.clone(), Expr::Raw(text)));
                    }
                }
                Stage::Index { dst } => {
                    define(dst)?;
                    if d != Dialect::Cuda {
                        return Err("an index stage has only a CUDA lowering".into());
                    }
                    let row = match self.rows.iter {
                        Iter::Block => "x",
                        Iter::Grid => "y",
                    };
                    let text = match dst.as_str() {
                        "col" => "(float)i".to_string(),
                        "row" => format!("(float)blockIdx.{row}"),
                        other => return Err(format!("index stage {other:?} is not col or row")),
                    };
                    env.push((dst.clone(), Expr::Raw(text)));
                }
                Stage::Epilogue { dst, expr, mat } => {
                    define(dst)?;
                    let e = expr.clone().bind(&self.consts).subst(&|n| env.iter().find(|(k, _)| k == n).map(|(_, x)| x.clone()))?;
                    if *scalar {
                        plan.between.last_mut().expect("a gap").push(format!("{} {dst} = {};", decl(d), e.lower(d, &named)?));
                        env.push((dst.clone(), Expr::Raw(dst.clone())));
                        scalar_names.push(dst.clone());
                    } else if mat.on(d) {
                        emit!(format!("{} {dst} = {};", decl(d), e.lower(d, &named)?));
                        env.push((dst.clone(), Expr::Raw(dst.clone())));
                    } else {
                        env.push((dst.clone(), e));
                    }
                }
                Stage::BlockSum { dst, expr } => {
                    let e = expr.clone().bind(&self.consts).subst(&|n| env.iter().find(|(k, _)| k == n).map(|(_, x)| x.clone()))?;
                    emit!(format!("ss += {};", e.lower(d, &named)?));
                    define(dst)?;
                    let mut uses = 0;
                    let mut wide = 0;
                    for (s2, f2) in self.stages.iter().zip(flags.iter()) {
                        if let Stage::Epilogue { expr, .. } = s2 {
                            let n = expr.count(dst);
                            if *f2 { uses += n } else { wide += n }
                        }
                    }
                    plan.total = Some(dst.clone());
                    env.retain(|(n, x)| scalar_names.contains(n) || !matches!(x, Expr::Raw(r) if r == n));
                    plan.between.push(Vec::new());
                    let gap = plan.between.last_mut().expect("a gap");
                    if d == Dialect::Cuda && total_sums == 1 && uses <= 1 && wide == 0 {
                        env.push((dst.clone(), Expr::Raw(sum_text.to_string())));
                    } else {
                        if d == Dialect::Cuda {
                            gap.insert(0, format!("float {dst} = {sum_text};"));
                        }
                        env.push((dst.clone(), Expr::Raw(dst.clone())));
                    }
                    plan.passes.push(Vec::new());
                }
                Stage::Gather { dst, src, index } | Stage::Lookup { dst, src, index } => {
                    define(dst)?;
                    let row = self.row_index(d, params, src, index, matches!(st, Stage::Gather { .. }))?;
                    let wide = matches!(st, Stage::Gather { .. });
                    let at = if wide { format!("{row} * WIDTH + i") } else { row };
                    emit!(format!("float {dst} = {src}[{at}];"));
                    env.push((dst.clone(), Expr::Raw(dst.clone())));
                }
                Stage::Scatter { dst, value, index } => {
                    match params.iter().find(|p| p.name == dst) {
                        Some(p) if p.access == Access::Write => {}
                        _ => return Err(format!("scatter to {dst:?}, which is not a written param")),
                    }
                    let row = self.row_index(d, params, dst, index, true)?;
                    let e = env.iter().find(|(k, _)| k == value).map(|(_, x)| x.clone()).ok_or_else(|| format!("scatter of unknown value {value:?}"))?;
                    emit!(format!("{dst}[{row} * WIDTH + i] = {};", e.lower(d, &named)?));
                }
                Stage::TileStore { dst, value, tile } => {
                    match params.iter().find(|p| p.name == dst) {
                        Some(p) if p.access == Access::Write => {}
                        _ => return Err(format!("store to {dst:?}, which is not a written param")),
                    }
                    let e = env.iter().find(|(k, _)| k == value).map(|(_, x)| x.clone()).ok_or_else(|| format!("store of unknown value {value:?}"))?;
                    emit!(format!("{dst}[{}] = {};", self.index(d, params, dst, tile.as_ref())?, e.lower(d, &named)?));
                }
                Stage::SplitStore { hi, lo, value, tile } => {
                    if d != Dialect::Cuda {
                        return Err("a split store has only a CUDA lowering".into());
                    }
                    for p in [hi, lo] {
                        match params.iter().find(|q| q.name == p) {
                            Some(q) if q.access == Access::Write && q.ty == crate::Scalar::Bf16 => {}
                            _ => return Err(format!("split store to {p:?}, which is not a written bf16 param")),
                        }
                    }
                    let e = env.iter().find(|(k, _)| k == value).map(|(_, x)| x.clone()).ok_or_else(|| format!("store of unknown value {value:?}"))?;
                    let at_hi = self.index(d, params, hi, tile.as_ref())?;
                    let at_lo = self.index(d, params, lo, tile.as_ref())?;
                    emit!(format!("kir_split_bf16({}, {hi}[{at_hi}], {lo}[{at_lo}]);", e.lower(d, &named)?));
                }
            }
        }
        if !self.stages.iter().any(|s| matches!(s, Stage::TileStore { .. } | Stage::Scatter { .. } | Stage::SplitStore { .. })) {
            return Err("compose has no tile_store".into());
        }
        Ok(plan)
    }

    pub fn splits(&self) -> bool {
        self.stages.iter().any(|s| matches!(s, Stage::SplitStore { .. }))
    }

    pub fn emit_cuda(&self, k: &Kernel) -> Result<String, String> {
        let plan = self.plan(Dialect::Cuda, &k.params)?;
        let sig: Vec<String> = k
            .params
            .iter()
            .map(|p| {
                let ty = match p.ty {
                    crate::Scalar::I32 => "int",
                    crate::Scalar::Bf16 => "unsigned short",
                    _ => "float",
                };
                format!("{}{ty}* {}", if p.access == Access::Write { "" } else { "const " }, p.name)
            })
            .collect();
        let (row, loop_head) = match self.rows.iter {
            Iter::Block => ("x", "for (int i = threadIdx.x; i < WIDTH; i += blockDim.x)"),
            Iter::Grid => ("y", "for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WIDTH; i += gridDim.x * blockDim.x)"),
        };
        let mut s = format!("extern \"C\" __global__ void __launch_bounds__({}) {}({}) {{\n    constexpr int WIDTH = {};\n", k.threads, k.name, sig.join(", "), self.width);
        if self.has_sum() {
            s.push_str("    __shared__ float scratch[32];\n");
        }
        match self.rows.address {
            Address::Advance => {
                for (n, m) in &self.rows.per_row {
                    let mul = if *m == 1 { String::new() } else { format!("{m} * ") };
                    s.push_str(&format!("    {n} += (size_t)blockIdx.{row} * {mul}WIDTH;\n"));
                }
            }
            Address::Base => s.push_str(&format!("    size_t base = (size_t)blockIdx.{row} * WIDTH;\n")),
        }
        for (n, stride) in &self.rows.stride {
            s.push_str(&format!("    {n} += (size_t)blockIdx.{row} * {stride};\n"));
        }
        for (n, off) in self.rows.offset.iter().filter(|(_, o)| *o > 0) {
            s.push_str(&format!("    {n} += {off};\n"));
        }
        let looped = |lines: &[String]| -> String {
            match lines {
                [one] => format!("    {loop_head} {one}\n"),
                many => format!("    {loop_head} {{\n{}    }}\n", many.iter().map(|l| format!("        {l}\n")).collect::<String>()),
            }
        };
        if self.has_sum() {
            let sums = plan.passes.len() - 1;
            for (n, pass) in plan.passes[..sums].iter().enumerate() {
                s.push_str(if n == 0 { "    float ss = 0.0f;\n" } else { "    ss = 0.0f;\n" });
                s.push_str(&looped(pass));
                for l in &plan.between[n] {
                    s.push_str(&format!("    {l}\n"));
                }
            }
        }
        s.push_str(&looped(plan.final_pass()));
        s.push_str("}\n\n");
        Ok(s)
    }

    pub fn emit_wgsl(&self, k: &Kernel, header: &str) -> Result<String, String> {
        let plan = self.plan(Dialect::Wgsl, &k.params)?;
        let t = k.threads;
        let mut s = String::from(header);
        s.push_str(&format!("const WIDTH: u32 = {}u;\n", self.width));
        for c in self.consts.iter().filter(|c| c.wgsl_const) {
            s.push_str(&format!("const {}: f32 = {};\n", c.name.to_uppercase(), lit(c.value)));
        }
        match self.rows.iter {
            Iter::Block => {
                let sum = self.sum.ok_or_else(|| format!("{}: WGSL block rows need a block_sum", k.name))?;
                if self.rows.address != Address::Advance {
                    return Err(format!("{}: WGSL block rows need advance addressing", k.name));
                }
                let total = plan.total.clone().unwrap_or_default();
                let (builtins, scratch, reduce) = match sum {
                    BlockSum::SharedTree => (
                        "@builtin(local_invocation_index) li: u32".to_string(),
                        format!("var<workgroup> red: array<f32, {t}>;\n"),
                        format!(
                            "    red[li] = ss;
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s >>= 1u) {{
        if (li < s) {{
            red[li] += red[li + s];
        }}
        workgroupBarrier();
    }}
    let {total} = red[0];
",
                            half = t / 2
                        ),
                    ),
                    BlockSum::Subgroup => {
                        if t > 128 * k.target.warp_size.max(1) {
                            return Err(format!("{}: too many subgroups for the scratch array", k.name));
                        }
                        (
                            "@builtin(local_invocation_index) li: u32, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(num_subgroups) sgs: u32".to_string(),
                            "var<workgroup> red: array<f32, 128>;\n".to_string(),
                            format!(
                                "    let part = subgroupAdd(ss);
    if (lane == 0u) {{
        red[sg] = part;
    }}
    workgroupBarrier();
    var {total} = 0.0;
    for (var i = 0u; i < sgs; i++) {{
        {total} += red[i];
    }}
"
                            ),
                        )
                    }
                    BlockSum::WarpShuffle => return Err(format!("{}: block sum {sum:?} has no WGSL snippet", k.name)),
                };
                s.push_str(&format!(
                    "{scratch}
@compute @workgroup_size({t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, {builtins}) {{
    let t = wid.x;
    let src = (p.a.x + t) * WIDTH;
    let dst = (p.a.y + t) * WIDTH;
    var ss = 0.0;
"
                ));
                let looped = |lines: &[String]| -> String {
                    format!("    for (var i = li; i < WIDTH; i += {t}u) {{\n{}    }}\n", lines.iter().map(|l| format!("        {l}\n")).collect::<String>())
                };
                s.push_str(&looped(&plan.passes[0]));
                s.push_str(&reduce);
                for l in plan.between.first().into_iter().flatten() {
                    s.push_str(&format!("    {l}\n"));
                }
                s.push_str(&looped(plan.final_pass()));
                s.push_str("}\n");
            }
            Iter::Grid => {
                if self.has_sum() {
                    return Err(format!("{}: a block_sum needs block iteration", k.name));
                }
                s.push_str(&format!(
                    "
@compute @workgroup_size({t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x + gid.y * 8388608u;
    if (idx >= p.a.x * WIDTH) {{
        return;
    }}
    let t = idx / WIDTH;
    let i = idx % WIDTH;
"
                ));
                for l in plan.final_pass() {
                    s.push_str(&format!("    {l}\n"));
                }
                s.push_str("}\n");
            }
        }
        Ok(s)
    }
}
