use crate::expr::{Const, Expr};
use crate::layout::Dialect;
use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

pub const SEQUENCE: [(&str, Scalar, Role, Access); 5] = [
    ("slots", Scalar::I32, Role::Activation, Access::Read),
    ("layer", Scalar::I32, Role::Value, Access::Read),
    ("slot_base", Scalar::I32, Role::Value, Access::Read),
    ("slot_step", Scalar::I32, Role::Value, Access::Read),
    ("rows_per_seq", Scalar::I32, Role::Value, Access::Read),
];

pub const LAUNCH: [&str; 5] = ["slots", "layer", "slot_base", "slot_step", "rows_per_seq"];

const GENERATED: [&str; 14] = ["head", "seq", "col", "part", "slot", "st", "s", "j", "t", "row", "d", "kr", "acc", "o"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Key,
    Value,
    Scalar,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    pub name: String,
    pub src: String,
    pub space: Space,
    pub row: u32,
    pub head: u32,
    pub per: u32,
    pub offset: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    Key { dst: String, expr: Expr },
    KeySum { dst: String, expr: Expr },
    Scalar { dst: String, expr: Expr },
    Value { dst: String, expr: Expr },
    Decay { expr: Expr },
    Update { key: Expr, value: Expr },
    Read { dst: String, key: Expr },
    Store { dst: String, value: Expr, row: u32, head: u32, offset: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Recurrence {
    pub state: String,
    pub keys: u32,
    pub values: u32,
    pub ksplit: u32,
    pub layers: u32,
    pub consts: Vec<Const>,
    pub inputs: Vec<Input>,
    pub stages: Vec<Stage>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SeqBody {
    Conv { x_row: u32, x_offset: u32, channels: u32, taps: u32, y_row: u32, y_offset: u32, layers: u32, expr: Expr },
    GroupNorm { x_row: u32, x_offset: u32, g_row: u32, g_offset: u32, width: u32, w_group: u32, out_row: u32, eps: f32, input: Expr, output: Expr },
}

fn ident(n: &str) -> bool {
    !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !n.starts_with(|c: char| c.is_ascii_digit())
}

fn sig(params: &[Param]) -> String {
    params
        .iter()
        .map(|p| {
            let ty = match p.ty {
                Scalar::F32 => "float",
                Scalar::U32 => "unsigned",
                Scalar::Bf16 => "bf16_t",
                Scalar::I32 => "int",
            };
            match (p.role, p.access) {
                (Role::Value, _) => format!("{ty} {}", p.name),
                (_, Access::Read) => format!("const {ty}* {}", p.name),
                (_, Access::Write) => format!("{ty}* {}", p.name),
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn signature(params: &[Param]) -> String {
    sig(params)
}

fn address(row: u32, head: u32, per: u32, offset: u32, index: &str) -> String {
    let mut terms = Vec::new();
    if row != 0 {
        terms.push(format!("(size_t)row * {row}"));
    }
    if head != 0 {
        terms.push(if per > 1 { format!("(size_t)(head / {per}) * {head}") } else { format!("(size_t)head * {head}") });
    }
    if offset != 0 {
        terms.push(offset.to_string());
    }
    terms.push(index.to_string());
    terms.join(" + ")
}

#[derive(Clone, Copy, PartialEq)]
enum Ctx {
    KeyAt,
    KeyReg,
    Scalar,
    Value,
}

struct Env {
    names: Vec<(String, Space)>,
    values: Vec<String>,
}

impl Env {
    fn lookup(&self, n: &str, ctx: Ctx) -> Option<Expr> {
        if let Some((_, space)) = self.names.iter().find(|(k, _)| k == n) {
            let text = match (space, ctx) {
                (Space::Key, Ctx::KeyAt) => format!("r_{n}_s[d]"),
                (Space::Key, Ctx::KeyReg) => format!("r_{n}_s[kr]"),
                (Space::Key, _) => return None,
                (Space::Value, Ctx::Value) => format!("r_{n}"),
                (Space::Value, _) => return None,
                (Space::Scalar, _) => format!("r_{n}"),
            };
            return Some(Expr::Raw(text));
        }
        self.values.iter().find(|v| *v == n).map(|v| Expr::Raw(format!("(float){v}")))
    }
}

fn find(params: &[Param], name: &str) -> Option<Param> {
    params.iter().find(|p| p.name == name).cloned()
}

fn want(params: &[Param], name: &str, ty: Scalar, role: Role, access: Access, family: &str) -> Result<(), String> {
    match find(params, name) {
        Some(p) if p.ty == ty && p.role == role && p.access == access => Ok(()),
        Some(p) => Err(format!("{family} param {name:?} must be {ty:?} {role:?} {access:?}, not {:?} {:?} {:?}", p.ty, p.role, p.access)),
        None => Err(format!("{family} needs a param named {name:?}")),
    }
}

fn lower_with(e: &Expr, names: &[(&str, String)], params: &[Param]) -> Result<String, String> {
    let env = |n: &str| -> Option<Expr> {
        names.iter().find(|(k, _)| *k == n).map(|(_, v)| Expr::Raw(v.clone())).or_else(|| params.iter().find(|p| p.name == n && p.role == Role::Value).map(|p| Expr::Raw(format!("(float){}", p.name))))
    };
    e.subst(&env)?.lower(Dialect::Cuda, &[])
}

impl SeqBody {
    pub fn family(&self) -> &'static str {
        match self {
            SeqBody::Conv { .. } => "seq_conv",
            SeqBody::GroupNorm { .. } => "group_norm",
        }
    }

    pub fn describe(&self) -> String {
        format!("{self:?}")
    }

    fn planes(params: &[Param]) -> bool {
        params.iter().any(|p| p.name == "out_hi")
    }

    fn allowed(&self, params: &[Param]) -> Vec<&'static str> {
        match self {
            SeqBody::Conv { .. } => {
                let mut v = vec!["x", "w", "bias", "y", "state"];
                v.extend(LAUNCH);
                v
            }
            SeqBody::GroupNorm { .. } if Self::planes(params) => vec!["x", "g", "w", "out_hi", "out_lo"],
            SeqBody::GroupNorm { .. } => vec!["x", "g", "w", "y"],
        }
    }

    pub fn check(&self, t: &Target, params: &[Param]) -> Result<(), String> {
        let family = self.family();
        if t.language != Language::Cuda {
            return Err(format!("a {family} body has only a CUDA lowering"));
        }
        let allowed = self.allowed(params);
        if let Some(p) = params.iter().find(|p| !allowed.contains(&p.name) && p.role != Role::Value) {
            return Err(format!("{family} has no buffer param {:?}, it takes {allowed:?}", p.name));
        }
        for (i, p) in params.iter().enumerate() {
            if params[..i].iter().any(|q| q.name == p.name) || (p.role == Role::Value && !matches!(p.ty, Scalar::I32 | Scalar::F32)) {
                return Err(format!("{family} param {:?} is not usable", p.name));
            }
        }
        let f = |n: &str, role: Role, access: Access| want(params, n, Scalar::F32, role, access, family);
        match self {
            SeqBody::Conv { channels, taps, layers, expr, .. } => {
                f("x", Role::Activation, Access::Read)?;
                f("w", Role::Weight, Access::Read)?;
                f("y", Role::Activation, Access::Write)?;
                f("state", Role::Activation, Access::Write)?;
                if find(params, "bias").is_some() {
                    f("bias", Role::Weight, Access::Read)?;
                }
                for &(name, ty, role, access) in &SEQUENCE {
                    want(params, name, ty, role, access, family)?;
                }
                if *channels == 0 || *layers == 0 || !(2..=8).contains(taps) {
                    return Err("seq_conv needs positive channels and layers and 2 to 8 taps".into());
                }
                lower_with(expr, &[("v", "v".into())], params)?;
            }
            SeqBody::GroupNorm { width, eps, input, output, .. } => {
                f("x", Role::Activation, Access::Read)?;
                f("g", Role::Activation, Access::Read)?;
                f("w", Role::Weight, Access::Read)?;
                if Self::planes(params) {
                    want(params, "out_hi", Scalar::Bf16, Role::Activation, Access::Write, family)?;
                    want(params, "out_lo", Scalar::Bf16, Role::Activation, Access::Write, family)?;
                } else {
                    f("y", Role::Activation, Access::Write)?;
                }
                if *width == 0 || !eps.is_finite() || *eps < 0.0 {
                    return Err("group_norm needs a positive width and eps".into());
                }
                lower_with(input, &[("x", "xv[i]".into()), ("g", "gv[i]".into())], params)?;
                lower_with(output, &[("n", "nv".into()), ("w", "wv[i]".into()), ("g", "gv[i]".into()), ("x", "xv[i]".into())], params)?;
            }
        }
        Ok(())
    }

    pub fn emit_cuda(&self, k: &Kernel) -> Result<String, String> {
        let head = format!("extern \"C\" __global__ void __launch_bounds__({}) {}({}) {{\n", k.threads, k.name, sig(&k.params));
        let body = match self {
            SeqBody::Conv { x_row, x_offset, channels, taps, y_row, y_offset, layers, expr } => {
                let act = lower_with(expr, &[("v", "v".into())], &k.params)?;
                let bias = if find(&k.params, "bias").is_some() { "bias[c]" } else { "0.0f" };
                format!(
                    "    constexpr int CH = {channels};
    constexpr int TAPS = {taps};
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    const int seq = blockIdx.y;
    const int slot = slot_step ? slots[seq] : slot_base;
    float* st = state + ((size_t)slot * {layers} + layer) * ((size_t)CH * (TAPS - 1)) + (size_t)c * (TAPS - 1);
    float win[TAPS];
#pragma unroll
    for (int j = 0; j < TAPS - 1; j++) win[j] = st[j];
    for (int t = 0; t < rows_per_seq; t++) {{
        const int row = seq * rows_per_seq + t;
        win[TAPS - 1] = x[(size_t)row * {x_row} + {x_offset} + c];
        float v = {bias};
#pragma unroll
        for (int j = 0; j < TAPS; j++) v += w[c * TAPS + j] * win[j];
        y[(size_t)row * {y_row} + {y_offset} + c] = {act};
#pragma unroll
        for (int j = 0; j < TAPS - 1; j++) win[j] = win[j + 1];
    }}
#pragma unroll
    for (int j = 0; j < TAPS - 1; j++) st[j] = win[j];
"
                )
            }
            SeqBody::GroupNorm { x_row, x_offset, g_row, g_offset, width, w_group, out_row, eps, input, output } => {
                let pre = lower_with(input, &[("x", "xv[i]".into()), ("g", "gv[i]".into())], &k.params)?;
                let post = lower_with(output, &[("n", "nv".into()), ("w", "wv[i]".into()), ("g", "gv[i]".into()), ("x", "xv[i]".into())], &k.params)?;
                let store = if Self::planes(&k.params) {
                    "        unsigned hi, lo;\n        split_hi_lo(o, hi, lo);\n        out_hi[at] = (bf16_t)hi;\n        out_lo[at] = (bf16_t)lo;\n".to_string()
                } else {
                    "        y[at] = o;\n".to_string()
                };
                format!(
                    "    constexpr int GW = {width};
    __shared__ float red[32];
    const int grp = blockIdx.x;
    const int row = blockIdx.y;
    const float* xv = x + (size_t)row * {x_row} + {x_offset} + (size_t)grp * GW;
    const float* gv = g + (size_t)row * {g_row} + {g_offset} + (size_t)grp * GW;
    const float* wv = w + (size_t)grp * {w_group};
    float ss = 0.0f;
    for (int i = threadIdx.x; i < GW; i += blockDim.x) {{
        const float a = {pre};
        ss += a * a;
    }}
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)((blockDim.x + 31) >> 5); i++) total += red[i];
    const float inv = rsqrtf(total / GW + {eps}f);
    for (int i = threadIdx.x; i < GW; i += blockDim.x) {{
        const float nv = ({pre}) * inv;
        const float o = {post};
        const size_t at = (size_t)row * {out_row} + (size_t)grp * GW + i;
{store}    }}
",
                    eps = crate::lit(*eps),
                )
            }
        };
        Ok(format!("{head}{body}}}\n\n"))
    }
}

impl Recurrence {
    pub fn describe(&self) -> String {
        let mut s = format!("recurrence state={} keys={} values={} ksplit={} layers={}", self.state, self.keys, self.values, self.ksplit, self.layers);
        for c in &self.consts {
            s.push_str(&format!("\n    const {} = {:e}", c.name, c.value));
        }
        for i in &self.inputs {
            s.push_str(&format!("\n    {i:?}"));
        }
        for st in &self.stages {
            s.push_str(&format!("\n    {st:?}"));
        }
        s
    }

    pub fn check(&self, t: &Target, threads: u32, params: &[Param]) -> Result<(), String> {
        if t.language != Language::Cuda {
            return Err("a recurrence body has only a CUDA lowering".into());
        }
        if self.keys == 0 || self.values == 0 || self.layers == 0 || self.keys > 1024 {
            return Err("a recurrence needs positive keys (at most 1024), values and layers".into());
        }
        if !self.ksplit.is_power_of_two() || self.ksplit > 32 || !self.keys.is_multiple_of(self.ksplit) {
            return Err(format!("ksplit {} must be a power of two up to 32 that divides keys {}", self.ksplit, self.keys));
        }
        if threads != self.values * self.ksplit || !threads.is_multiple_of(32) {
            return Err(format!("recurrence threads {threads} must equal values * ksplit = {} and be a multiple of 32", self.values * self.ksplit));
        }
        for &(name, ty, role, access) in &SEQUENCE {
            match params.iter().find(|p| p.name == name) {
                Some(p) if p.ty == ty && p.role == role && p.access == access => {}
                Some(p) => return Err(format!("recurrence param {name:?} must be {ty:?} {role:?} {access:?}, not {:?} {:?} {:?}", p.ty, p.role, p.access)),
                None => return Err(format!("recurrence needs a param named {name:?}")),
            }
        }
        match params.iter().find(|p| p.name == self.state) {
            Some(p) if p.ty == Scalar::F32 && p.role == Role::Activation && p.access == Access::Write => {}
            _ => return Err(format!("state {:?} must be a written f32 activation param", self.state)),
        }
        for (i, p) in params.iter().enumerate() {
            if !ident(p.name) || GENERATED.contains(&p.name) || params[..i].iter().any(|q| q.name == p.name) || p.name.starts_with("r_") {
                return Err(format!("recurrence param name {:?} is not usable", p.name));
            }
            if p.role == Role::Value && !matches!(p.ty, Scalar::I32 | Scalar::F32) {
                return Err(format!("recurrence value param {:?} must be i32 or f32", p.name));
            }
            if p.role != Role::Value && p.name != "slots" && p.ty != Scalar::F32 {
                return Err(format!("recurrence buffer param {:?} must be f32", p.name));
            }
        }
        for c in &self.consts {
            if !ident(&c.name) || !c.value.is_finite() || params.iter().any(|p| p.name == c.name) {
                return Err(format!("recurrence constant {:?} is not usable", c.name));
            }
        }
        for i in &self.inputs {
            match params.iter().find(|p| p.name == i.src) {
                Some(p) if p.access == Access::Read && p.ty == Scalar::F32 && matches!(p.role, Role::Activation | Role::Weight) => {}
                _ => return Err(format!("input {:?} reads {:?}, which is not an f32 read param", i.name, i.src)),
            }
            if i.per == 0 {
                return Err(format!("input {:?} has a zero head divisor", i.name));
            }
        }
        if !self.stages.iter().any(|s| matches!(s, Stage::Store { .. })) {
            return Err("a recurrence needs a store stage".into());
        }
        let smem = 4 * (self.keys * self.key_arrays().len() as u32 + self.sums().max(1));
        if smem > t.smem_limit.min(48 * 1024) {
            return Err(format!("recurrence key arrays need {smem} bytes of shared memory"));
        }
        self.body(params)?;
        Ok(())
    }

    fn key_arrays(&self) -> Vec<String> {
        let mut out: Vec<String> = self.inputs.iter().filter(|i| i.space == Space::Key).map(|i| i.name.clone()).collect();
        out.extend(self.stages.iter().filter_map(|s| match s {
            Stage::Key { dst, .. } => Some(dst.clone()),
            _ => None,
        }));
        out
    }

    fn sums(&self) -> u32 {
        self.stages.iter().filter(|s| matches!(s, Stage::KeySum { .. })).count() as u32
    }

    fn body(&self, params: &[Param]) -> Result<(String, String), String> {
        let mut env = Env { names: Vec::new(), values: params.iter().filter(|p| p.role == Role::Value).map(|p| p.name.to_string()).collect() };
        let define = |env: &mut Env, n: &str, space: Space| -> Result<(), String> {
            if !ident(n) || env.names.iter().any(|(k, _)| k == n) || self.consts.iter().any(|c| c.name == n) || params.iter().any(|p| p.name == n) {
                return Err(format!("recurrence name {n:?} is not usable or is defined twice"));
            }
            env.names.push((n.to_string(), space));
            Ok(())
        };
        let lower = |e: &Expr, env: &Env, ctx: Ctx| -> Result<String, String> {
            e.clone().bind(&self.consts).subst(&|n| env.lookup(n, ctx)).map_err(|e| format!("{e} in this space"))?.lower(Dialect::Cuda, &[])
        };
        let mut loads = String::new();
        let mut regs = String::new();
        for i in &self.inputs {
            define(&mut env, &i.name, i.space)?;
            match i.space {
                Space::Key => loads.push_str(&format!("            r_{}_s[d] = {}[{}];\n", i.name, i.src, address(i.row, i.head, i.per, i.offset, "d"))),
                Space::Value => regs.push_str(&format!("        const float r_{} = {}[{}];\n", i.name, i.src, address(i.row, i.head, i.per, i.offset, "col"))),
                Space::Scalar => regs.push_str(&format!("        const float r_{} = {}[{}];\n", i.name, i.src, address(i.row, i.head, i.per, i.offset, "0"))),
            }
        }
        let mut out = String::new();
        if !loads.is_empty() {
            out.push_str(&format!("        for (int d = threadIdx.x; d < RK; d += blockDim.x) {{\n{loads}        }}\n        __syncthreads();\n"));
        }
        out.push_str(&regs);
        let mut sum = 0;
        for (n, st) in self.stages.iter().enumerate() {
            match st {
                Stage::Key { dst, expr } => {
                    let e = lower(expr, &env, Ctx::KeyAt)?;
                    define(&mut env, dst, Space::Key)?;
                    out.push_str(&format!("        for (int d = threadIdx.x; d < RK; d += blockDim.x) r_{dst}_s[d] = {e};\n        __syncthreads();\n"));
                }
                Stage::KeySum { dst, expr } => {
                    let e = lower(expr, &env, Ctx::KeyAt)?;
                    define(&mut env, dst, Space::Scalar)?;
                    out.push_str(&format!(
                        "        if (threadIdx.x < 32) {{
            float acc = 0.0f;
            for (int d = threadIdx.x; d < RK; d += 32) acc += {e};
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (threadIdx.x == 0) r_sums_s[{sum}] = acc;
        }}
        __syncthreads();
        const float r_{dst} = r_sums_s[{sum}];
"
                    ));
                    sum += 1;
                }
                Stage::Scalar { dst, expr } => {
                    let e = lower(expr, &env, Ctx::Scalar)?;
                    define(&mut env, dst, Space::Scalar)?;
                    out.push_str(&format!("        const float r_{dst} = {e};\n"));
                }
                Stage::Value { dst, expr } => {
                    let e = lower(expr, &env, Ctx::Value)?;
                    define(&mut env, dst, Space::Value)?;
                    out.push_str(&format!("        const float r_{dst} = {e};\n"));
                }
                Stage::Decay { expr } => {
                    let e = lower(expr, &env, Ctx::KeyReg)?;
                    out.push_str(&format!("#pragma unroll\n        for (int j = 0; j < KJ; j++) {{\n            const int kr = part + KS * j;\n            s[j] *= {e};\n        }}\n"));
                }
                Stage::Update { key, value } => {
                    let k = lower(key, &env, Ctx::KeyReg)?;
                    let v = lower(value, &env, Ctx::Value)?;
                    out.push_str(&format!(
                        "        const float r_update{n} = {v};\n#pragma unroll\n        for (int j = 0; j < KJ; j++) {{\n            const int kr = part + KS * j;\n            s[j] += ({k}) * r_update{n};\n        }}\n"
                    ));
                }
                Stage::Read { dst, key } => {
                    let k = lower(key, &env, Ctx::KeyReg)?;
                    define(&mut env, dst, Space::Value)?;
                    out.push_str(&format!(
                        "        float r_{dst} = 0.0f;\n#pragma unroll\n        for (int j = 0; j < KJ; j++) {{\n            const int kr = part + KS * j;\n            r_{dst} += ({k}) * s[j];\n        }}\n        for (int o = KS / 2; o > 0; o >>= 1) r_{dst} += __shfl_xor_sync(0xffffffffu, r_{dst}, o);\n"
                    ));
                }
                Stage::Store { dst, value, row, head, offset } => {
                    match params.iter().find(|p| p.name == dst) {
                        Some(p) if p.access == Access::Write && p.role == Role::Activation && *dst != self.state => {}
                        _ => return Err(format!("store to {dst:?}, which is not a written activation param")),
                    }
                    let v = lower(value, &env, Ctx::Value)?;
                    out.push_str(&format!("        if (part == 0) {dst}[{}] = {v};\n", address(*row, *head, 1, *offset, "col")));
                }
            }
        }
        let mut shared = String::new();
        for a in self.key_arrays() {
            shared.push_str(&format!("    __shared__ float r_{a}_s[RK];\n"));
        }
        if sum > 0 {
            shared.push_str(&format!("    __shared__ float r_sums_s[{sum}];\n"));
        }
        Ok((shared, out))
    }

    pub fn emit_cuda(&self, k: &Kernel) -> Result<String, String> {
        let (shared, body) = self.body(&k.params)?;
        Ok(format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}({sig}) {{
    constexpr int RK = {keys};
    constexpr int RV = {values};
    constexpr int KS = {ksplit};
    constexpr int KJ = RK / KS;
{shared}    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int col = threadIdx.x / KS;
    const int part = threadIdx.x % KS;
    const int slot = slot_step ? slots[seq] : slot_base;
    float* st = {state} + ((size_t)slot * {layers} + layer) * ((size_t)gridDim.x * RK * RV) + (size_t)head * RK * RV;
    float s[KJ];
#pragma unroll
    for (int j = 0; j < KJ; j++) s[j] = st[(size_t)(part + KS * j) * RV + col];
    for (int t = 0; t < rows_per_seq; t++) {{
        const int row = seq * rows_per_seq + t;
        __syncthreads();
{body}    }}
#pragma unroll
    for (int j = 0; j < KJ; j++) st[(size_t)(part + KS * j) * RV + col] = s[j];
}}

",
            threads = k.threads,
            name = k.name,
            sig = sig(&k.params),
            keys = self.keys,
            values = self.values,
            ksplit = self.ksplit,
            state = self.state,
            layers = self.layers,
        ))
    }
}
