use crate::expr::{Const, Expr};
use crate::layout::Dialect;
use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

pub const WARP: u32 = 32;

pub const BUILTINS: [&str; 6] = ["s", "scale", "q_pos", "k_pos", "head", "kv_head"];

const GENERATED: [&str; 30] = [
    "dot", "m", "l", "m_new", "corr", "p", "t", "q", "acc", "lane", "warp", "row", "table", "top", "denom", "total", "kr", "vr", "ksc", "vsc", "part", "d", "j", "at", "hi", "lo", "v", "w", "q_pos_i", "sink_logit",
];

pub const LAUNCH: [&str; 12] = ["qkv", "k_cache", "v_cache", "block_table", "pos", "slots", "out_hi", "out_lo", "layer", "pages", "pos_step", "row_step"];

const REQUIRED: [(&str, Scalar, Role, Access); 12] = [
    ("qkv", Scalar::F32, Role::Activation, Access::Read),
    ("k_cache", Scalar::Bf16, Role::Activation, Access::Read),
    ("v_cache", Scalar::Bf16, Role::Activation, Access::Read),
    ("block_table", Scalar::I32, Role::Activation, Access::Read),
    ("pos", Scalar::I32, Role::Activation, Access::Read),
    ("slots", Scalar::I32, Role::Activation, Access::Read),
    ("out_hi", Scalar::Bf16, Role::Activation, Access::Write),
    ("out_lo", Scalar::Bf16, Role::Activation, Access::Write),
    ("layer", Scalar::I32, Role::Value, Access::Read),
    ("pages", Scalar::I32, Role::Value, Access::Read),
    ("pos_step", Scalar::I32, Role::Value, Access::Read),
    ("row_step", Scalar::I32, Role::Value, Access::Read),
];

#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    Epilogue { dst: String, expr: Expr },
    Mask { expr: Expr },
    Softmax { value: String, sink: Option<String> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scan {
    pub consts: Vec<Const>,
    pub stages: Vec<Stage>,
}

fn ident(n: &str) -> bool {
    !GENERATED.contains(&n) && !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !n.starts_with(|c: char| c.is_ascii_digit())
}

impl Scan {
    pub fn describe(&self) -> String {
        let mut s = String::from("scan");
        for c in &self.consts {
            s.push_str(&format!("\n    const {} = {:e}", c.name, c.value));
        }
        for st in &self.stages {
            s.push_str(&format!("\n    {st:?}"));
        }
        s
    }

    pub fn sink(&self) -> Option<&str> {
        self.stages.iter().find_map(|s| match s {
            Stage::Softmax { sink, .. } => sink.as_deref(),
            _ => None,
        })
    }

    pub fn check(&self, t: &Target, threads: u32, params: &[Param]) -> Result<(), String> {
        if t.language != Language::Cuda {
            return Err("a scan body has only a CUDA lowering".into());
        }
        if !threads.is_multiple_of(WARP) || threads == 0 {
            return Err(format!("scan threads {threads} must be a multiple of {WARP}"));
        }
        for (name, ty, role, access) in REQUIRED {
            match params.iter().find(|p| p.name == name) {
                Some(p) if p.ty == ty && p.role == role && p.access == access => {}
                Some(p) => return Err(format!("scan param {name:?} must be {ty:?} {role:?} {access:?}, not {:?} {:?} {:?}", p.ty, p.role, p.access)),
                None => return Err(format!("scan needs a param named {name:?}")),
            }
        }
        for (i, p) in params.iter().enumerate() {
            if !ident(p.name) || params[..i].iter().any(|q| q.name == p.name) || BUILTINS.contains(&p.name) {
                return Err(format!("scan param name {:?} is not usable", p.name));
            }
            if p.role == Role::Value && !matches!(p.ty, Scalar::I32 | Scalar::F32) {
                return Err(format!("scan value param {:?} must be i32 or f32", p.name));
            }
        }
        for c in &self.consts {
            if !ident(&c.name) || BUILTINS.contains(&c.name.as_str()) || params.iter().any(|p| p.name == c.name) || !c.value.is_finite() {
                return Err(format!("scan constant {:?} is not usable", c.name));
            }
        }
        let softmaxes = self.stages.iter().filter(|s| matches!(s, Stage::Softmax { .. })).count();
        if softmaxes != 1 || !matches!(self.stages.last(), Some(Stage::Softmax { .. })) {
            return Err("a scan needs exactly one online_softmax, as its last stage".into());
        }
        if let Some(sink) = self.sink() {
            match params.iter().find(|p| p.name == sink) {
                Some(p) if p.ty == Scalar::F32 && p.role == Role::Weight && p.access == Access::Read => {}
                _ => return Err(format!("sink {sink:?} must be an f32 weight param")),
            }
        }
        self.lines(params)?;
        Ok(())
    }

    fn lines(&self, params: &[Param]) -> Result<(Vec<String>, String), String> {
        let mut env: Vec<(String, Expr)> = vec![
            ("s".into(), Expr::Raw("dot".into())),
            ("scale".into(), Expr::Raw("QUERY_SCALE".into())),
            ("q_pos".into(), Expr::Raw("(float)q_pos_i".into())),
            ("k_pos".into(), Expr::Raw("(float)t".into())),
            ("head".into(), Expr::Raw("(float)head".into())),
            ("kv_head".into(), Expr::Raw("(float)kv_head".into())),
        ];
        for p in params.iter().filter(|p| p.role == Role::Value) {
            env.push((p.name.to_string(), Expr::Raw(format!("(float){}", p.name))));
        }
        let mut lines = Vec::new();
        let mut value = String::new();
        let lower = |e: &Expr, env: &[(String, Expr)]| -> Result<String, String> {
            e.clone().bind(&self.consts).subst(&|n| env.iter().find(|(k, _)| k == n).map(|(_, x)| x.clone()))?.lower(Dialect::Cuda, &[])
        };
        for st in &self.stages {
            match st {
                Stage::Epilogue { dst, expr } => {
                    if !ident(dst) || env.iter().any(|(k, _)| k == dst) || self.consts.iter().any(|c| c.name == *dst) {
                        return Err(format!("stage name {dst:?} is not usable or is defined twice"));
                    }
                    lines.push(format!("float {dst} = {};", lower(expr, &env)?));
                    env.push((dst.clone(), Expr::Raw(dst.clone())));
                }
                Stage::Mask { expr } => lines.push(format!("if (!({})) continue;", lower(expr, &env)?)),
                Stage::Softmax { value: v, .. } => {
                    if !env.iter().any(|(k, _)| k == v) {
                        return Err(format!("online_softmax of unknown value {v:?}"));
                    }
                    value = v.clone();
                }
            }
        }
        Ok((lines, value))
    }

    pub fn emit_cuda(&self, k: &Kernel) -> Result<String, String> {
        let (lines, value) = self.lines(&k.params)?;
        let sink = self.sink();
        let sig: Vec<String> = k
            .params
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
            .collect();
        let warps = k.threads / WARP;
        let body: String = lines.iter().map(|l| format!("        {l}\n")).collect();
        let (sink_decl, top_init, sink_denom) = match sink {
            Some(n) => (format!("    float sink_logit = {n}[head];\n"), "sink_logit", "expf(sink_logit - top)"),
            None => (String::new(), "NEG_INF", "0.0f"),
        };
        Ok(format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}({sig}) {{
    constexpr int SCAN_WARPS = {warps};
    constexpr int SCAN_LANE_DIMS = (HEAD_DIM + 31) / 32;
    __shared__ float acc_s[SCAN_WARPS][HEAD_DIM];
    __shared__ float m_s[SCAN_WARPS];
    __shared__ float l_s[SCAN_WARPS];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int kv_head = head / GROUP;
    const int q_pos_i = pos[row * pos_step] + row * row_step;
    const int* table = seq_table(block_table, slots, row, pages);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float q[SCAN_LANE_DIMS];
#pragma unroll
    for (int j = 0; j < SCAN_LANE_DIMS; j++) {{
        int d = lane + 32 * j;
        q[j] = d < HEAD_DIM ? qkv[(size_t)row * QKV_ROW + head * HEAD_DIM + d] : 0.0f;
    }}
    float m = NEG_INF;
    float l = 0.0f;
    float acc[SCAN_LANE_DIMS];
#pragma unroll
    for (int j = 0; j < SCAN_LANE_DIMS; j++) acc[j] = 0.0f;
    for (int t = warp; t <= q_pos_i; t += SCAN_WARPS) {{
        const bf16_t* kr = kv_row(k_cache, table, layer, kv_head, t);
        float ksc = kv_scale(k_cache, table, layer, kv_head, t);
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < SCAN_LANE_DIMS; j++) {{
            int d = lane + 32 * j;
            if (d < HEAD_DIM) part += q[j] * kv_at(kr, d, ksc, kv_q(layer));
        }}
        float dot = warp_sum(part);
{body}        float m_new = fmaxf(m, {value});
        float corr = m == NEG_INF ? 0.0f : expf(m - m_new);
        float p = expf({value} - m_new);
        l = l * corr + p;
        const bf16_t* vr = kv_row(v_cache, table, layer, kv_head, t);
        float vsc = kv_scale(v_cache, table, layer, kv_head, t);
#pragma unroll
        for (int j = 0; j < SCAN_LANE_DIMS; j++) {{
            int d = lane + 32 * j;
            if (d < HEAD_DIM) acc[j] = acc[j] * corr + p * kv_at(vr, d, vsc, kv_q(layer));
        }}
        m = m_new;
    }}
    if (lane == 0) {{
        m_s[warp] = m;
        l_s[warp] = l;
    }}
#pragma unroll
    for (int j = 0; j < SCAN_LANE_DIMS; j++) {{
        int d = lane + 32 * j;
        if (d < HEAD_DIM) acc_s[warp][d] = acc[j];
    }}
    __syncthreads();
{sink_decl}    float top = {top_init};
    for (int w = 0; w < SCAN_WARPS; w++) top = fmaxf(top, m_s[w]);
    float denom = {sink_denom};
    for (int w = 0; w < SCAN_WARPS; w++) denom += m_s[w] == NEG_INF ? 0.0f : l_s[w] * expf(m_s[w] - top);
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {{
        float total = 0.0f;
        for (int w = 0; w < SCAN_WARPS; w++) total += m_s[w] == NEG_INF ? 0.0f : acc_s[w][d] * expf(m_s[w] - top);
        float v = total / denom * out_gate(qkv, row, head * HEAD_DIM + d);
        unsigned hi, lo;
        split_hi_lo(v, hi, lo);
        size_t at = (size_t)row * Q_DIM + head * HEAD_DIM + d;
        out_hi[at] = (bf16_t)hi;
        out_lo[at] = (bf16_t)lo;
    }}
}}

",
            threads = k.threads,
            name = k.name,
            sig = sig.join(", "),
        ))
    }
}
