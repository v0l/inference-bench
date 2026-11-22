use crate::expr::{Const, Expr};
use crate::layout::Dialect;
use crate::{Access, Kernel, Language, Param, Role, Scalar, Target};

pub const WARP: u32 = 32;

pub const BUILTINS: [&str; 6] = ["s", "scale", "q_pos", "k_pos", "head", "kv_head"];

const GENERATED: &[&str] = &[
    "dot", "m", "l", "m_new", "corr", "p", "t", "q", "acc", "lane", "warp", "row", "table", "top", "denom", "total", "kr", "vr", "ksc", "vsc", "part", "d", "j", "at", "hi", "lo", "v", "w", "q_pos_i", "sink_logit", "visible", "e", "r", "slot",
];

const TILE_NAMES: &[&str] = &[
    "key", "live", "tokens", "base", "pos0", "pos1", "row0", "row1", "tid", "g", "kt", "k0", "nb", "np", "kb", "m0", "m1", "l0", "l1", "n0", "n1", "a0", "a1", "i0", "i1", "ph", "pl", "qh", "ql", "bh", "bl", "mx0", "mx1", "sum0", "sum1", "nm", "tiles", "key_end", "warp_last", "sink_l2", "mf0", "mf1", "LOG2E",
    "o", "dst", "src",
    "which",
    "dot", "m", "l", "m_new", "corr", "p", "t", "q", "acc", "lane", "warp", "row", "table", "top", "denom", "total", "kr", "vr", "ksc", "vsc", "part", "d", "j", "at", "hi", "lo", "v", "w", "q_pos_i", "sink_logit", "visible", "e", "r", "slot",
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
    Select { index: String, row: u32 },
    Softmax { value: String, sink: Option<String> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub src: String,
    pub row: u32,
    pub head: u32,
    pub offset: u32,
    pub width: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Blocks {
    pub cache: String,
    pub layers: u32,
    pub ratio: String,
    pub topk: u32,
    pub index: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub blocks: Option<Blocks>,
    pub query: Vec<Segment>,
    pub layers: u32,
    pub kv_rows: u32,
    pub k_width: u32,
    pub v_stride: u32,
    pub v_offset: u32,
    pub v_width: u32,
    pub out_row: u32,
    pub out_head: u32,
    pub out_offset: u32,
}

const STATE_REQUIRED: [(&str, Scalar, Role, Access); 10] = [
    ("k_cache", Scalar::Bf16, Role::Activation, Access::Read),
    ("v_cache", Scalar::Bf16, Role::Activation, Access::Read),
    ("block_table", Scalar::I32, Role::Activation, Access::Read),
    ("pos", Scalar::I32, Role::Activation, Access::Read),
    ("slots", Scalar::I32, Role::Activation, Access::Read),
    ("out", Scalar::F32, Role::Activation, Access::Write),
    ("layer", Scalar::I32, Role::Value, Access::Read),
    ("pages", Scalar::I32, Role::Value, Access::Read),
    ("pos_step", Scalar::I32, Role::Value, Access::Read),
    ("row_step", Scalar::I32, Role::Value, Access::Read),
];

#[derive(Clone, Debug, PartialEq)]
pub struct Scan {
    pub consts: Vec<Const>,
    pub stages: Vec<Stage>,
    pub geometry: Option<Geometry>,
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
        if let Some(g) = &self.geometry {
            s.push_str(&format!("\n    {g:?}"));
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
        let required: &[(&str, Scalar, Role, Access)] = if self.geometry.is_some() { &STATE_REQUIRED } else { &REQUIRED };
        for &(name, ty, role, access) in required {
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
        for st in &self.stages {
            if let Stage::Select { index, row } = st {
                match params.iter().find(|p| p.name == index) {
                    Some(p) if p.ty == Scalar::I32 && p.role == Role::Activation && p.access == Access::Read => {}
                    _ => return Err(format!("select_from index {index:?} must be an i32 activation read param")),
                }
                if *row == 0 {
                    return Err("select_from needs a positive row width".into());
                }
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
        if let Some(g) = &self.geometry {
            let width: u32 = g.query.iter().map(|q| q.width).sum();
            if g.query.is_empty() || width != g.k_width {
                return Err(format!("the query segments are {width} wide, the key is {}", g.k_width));
            }
            for q in &g.query {
                match params.iter().find(|p| p.name == q.src) {
                    Some(p) if p.ty == Scalar::F32 && p.role == Role::Activation && p.access == Access::Read => {}
                    _ => return Err(format!("query source {:?} must be an f32 activation param", q.src)),
                }
            }
            if g.v_offset + g.v_width > g.v_stride || g.k_width == 0 || g.v_width == 0 || g.layers == 0 || g.kv_rows == 0 {
                return Err("the value span must sit inside its state row".into());
            }
            if g.k_width.max(g.v_width) > 1024 {
                return Err("scan head widths are at most 1024".into());
            }
        }
        if let Some(g) = &self.geometry
            && let Some(b) = &g.blocks
        {
            match params.iter().find(|p| p.name == b.cache) {
                Some(p) if p.ty == Scalar::Bf16 && p.role == Role::Activation && p.access == Access::Read => {}
                _ => return Err(format!("block cache {:?} must be a bf16 activation param", b.cache)),
            }
            if let Some(ix) = &b.index {
                match params.iter().find(|p| p.name == *ix) {
                    Some(p) if p.ty == Scalar::I32 && p.role == Role::Activation && p.access == Access::Read => {}
                    _ => return Err(format!("block index {ix:?} must be a read i32 activation param")),
                }
                if b.topk == 0 {
                    return Err("an indexed block source needs a positive topk".into());
                }
            }
            match params.iter().find(|p| p.name == b.ratio) {
                Some(p) if p.ty == Scalar::I32 && p.role == Role::Value && p.access == Access::Read => {}
                _ => return Err(format!("block ratio {:?} must be an i32 value param", b.ratio)),
            }
            if b.layers == 0 || g.v_width != g.k_width || g.v_offset != 0 || g.v_stride != g.k_width {
                return Err("a block source needs positive layers and keys equal to values".into());
            }
        }
        self.lines(params, true, false)?;
        self.lines(params, false, false)?;
        if self.tile_eligible() && self.tile_names_free(params) {
            self.lines(params, true, true)?;
        }
        Ok(())
    }

    pub fn tile_eligible(&self) -> bool {
        self.geometry.is_none() && !self.stages.iter().any(|s| matches!(s, Stage::Select { .. }))
    }

    pub fn tile_names_free(&self, params: &[Param]) -> bool {
        let used = |n: &str| params.iter().any(|p| p.name == n) || self.consts.iter().any(|c| c.name == n) || self.stages.iter().any(|s| matches!(s, Stage::Epilogue { dst, .. } if dst == n));
        !TILE_NAMES.iter().any(|n| used(n))
    }

    fn lines(&self, params: &[Param], masks: bool, tile: bool) -> Result<(Vec<String>, String), String> {
        let mut env: Vec<(String, Expr)> = vec![
            ("s".into(), Expr::Raw(if tile { "s[nb][e]" } else { "dot" }.into())),
            ("scale".into(), Expr::Raw("QUERY_SCALE".into())),
            ("q_pos".into(), Expr::Raw("(float)q_pos_i".into())),
            ("k_pos".into(), Expr::Raw(if tile { "(float)key" } else { "(float)t" }.into())),
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
                Stage::Mask { expr } => {
                    let text = lower(expr, &env)?;
                    if tile {
                        lines.push(format!("live = live && ({text});"));
                    } else if masks {
                        lines.push(format!("if (!({text})) continue;"));
                    }
                }
                Stage::Select { .. } if tile => return Err("select_from has no tile lowering".into()),
                Stage::Select { index, row } => lines.push(format!(
                    "{{ bool sel_hit = false; for (int sel_j = lane; sel_j < {row}; sel_j += 32) sel_hit |= {index}[(size_t)row * {row} + sel_j] == t; if (!__any_sync(0xffffffff, sel_hit)) continue; }}"
                )),
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
        if let Some(g) = &self.geometry {
            return self.emit_state_cuda(k, g);
        }
        let (lines, value) = self.lines(&k.params, true, false)?;
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
        let tile = if self.tile_eligible() && self.tile_names_free(&k.params) && k.target.has(crate::Feature::Mma16816) && k.target.has(crate::Feature::CpAsync) {
            let (tile_lines, tile_value) = self.lines(&k.params, true, true)?;
            let t = crate::cuda::TileScan { name: k.name, sig: &sig.join(", "), lines: &tile_lines, value: &tile_value, sink };
            format!("{}{}", crate::cuda::scan_tile_prefill(&t), crate::cuda::scan_tile_decode(&t))
        } else {
            String::new()
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

{tile}",
            threads = k.threads,
            name = k.name,
            sig = sig.join(", "),
        ))
    }

    fn emit_state_cuda(&self, k: &Kernel, g: &Geometry) -> Result<String, String> {
        let (lines, value) = self.lines(&k.params, true, false)?;
        let (block_lines, _) = self.lines(&k.params, false, false)?;
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
        let block_body: String = block_lines.iter().map(|l| format!("            {l}\n")).collect();
        let blocks = match &g.blocks {
            None => String::new(),
            Some(b) => {
                let (count, entry) = match &b.index {
                    Some(ix) => (format!("{}", b.topk), format!("{ix}[(size_t)row * {} + r]", b.topk)),
                    None => ("visible".to_string(), "r".to_string()),
                };
                format!(
                    "    {{
        const int visible = {ratio} > 0 ? (q_pos_i + 1) / {ratio} : 0;
        for (int r = warp; r < {count}; r += SCAN_WARPS) {{
            const int e = {entry};
            if (e < 0 || e >= visible) continue;
            const int t = e * {ratio} + {ratio} - 1;
            size_t slot = (((size_t)table[t / PAGE] * {layers} + layer) * KV_ROWS + 0) * PAGE + t % PAGE;
            const bf16_t* kr = {cache} + slot * QK * KV_PLANES;
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < QK_LANES; j++) {{
                int d = lane + 32 * j;
                if (d < QK) part += q[j] * (attn_float(kr[d]) + (KV_PLANES == 2 ? attn_float(kr[QK + d]) : 0.0f));
            }}
            float dot = warp_sum(part);
{block_body}            float m_new = fmaxf(m, {value});
            float corr = m == NEG_INF ? 0.0f : expf(m - m_new);
            float p = expf({value} - m_new);
            l = l * corr + p;
#pragma unroll
            for (int j = 0; j < V_LANES; j++) {{
                int d = lane + 32 * j;
                if (d < VD) acc[j] = acc[j] * corr + p * (attn_float(kr[d]) + (KV_PLANES == 2 ? attn_float(kr[QK + d]) : 0.0f));
            }}
            m = m_new;
        }}
    }}
",
                    ratio = b.ratio,
                    layers = b.layers,
                    cache = b.cache,
                    value = value,
                )
            }
        };
        let (sink_decl, top_init, sink_denom) = match self.sink() {
            Some(n) => (format!("    float sink_logit = {n}[head];\n"), "sink_logit", "expf(sink_logit - top)"),
            None => (String::new(), "NEG_INF", "0.0f"),
        };
        let mut at = 0;
        let mut query = String::new();
        for q in &g.query {
            query.push_str(&format!(
                "    for (int d = threadIdx.x; d < {w}; d += blockDim.x) q_s[{at} + d] = {src}[(size_t)row * {row} + (size_t)head * {head} + {off} + d];\n",
                w = q.width,
                src = q.src,
                row = q.row,
                head = q.head,
                off = q.offset,
            ));
            at += q.width;
        }
        let kv_heads = g.kv_rows;
        Ok(format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}({sig}) {{
    constexpr int SCAN_WARPS = {warps};
    constexpr int QK = {qk};
    constexpr int VD = {vd};
    constexpr int KV_ROWS = {kv_heads};
    constexpr int QK_LANES = (QK + 31) / 32;
    constexpr int V_LANES = (VD + 31) / 32;
    __shared__ float q_s[QK];
    __shared__ float acc_s[SCAN_WARPS][VD];
    __shared__ float m_s[SCAN_WARPS];
    __shared__ float l_s[SCAN_WARPS];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int kv_head = head / ({heads_per_row});
    const int q_pos_i = pos[row * pos_step] + row * row_step;
    const int* table = seq_table(block_table, slots, row, pages);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
{query}    __syncthreads();
    float q[QK_LANES];
#pragma unroll
    for (int j = 0; j < QK_LANES; j++) {{
        int d = lane + 32 * j;
        q[j] = d < QK ? q_s[d] : 0.0f;
    }}
    float m = NEG_INF;
    float l = 0.0f;
    float acc[V_LANES];
#pragma unroll
    for (int j = 0; j < V_LANES; j++) acc[j] = 0.0f;
    for (int t = warp; t <= q_pos_i; t += SCAN_WARPS) {{
        size_t slot = (((size_t)table[t / PAGE] * {layers} + layer) * KV_ROWS + kv_head) * PAGE + t % PAGE;
        const bf16_t* kr = k_cache + slot * QK * KV_PLANES;
        const bf16_t* vr = v_cache + slot * {v_stride} * KV_PLANES;
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < QK_LANES; j++) {{
            int d = lane + 32 * j;
            if (d < QK) part += q[j] * (attn_float(kr[d]) + (KV_PLANES == 2 ? attn_float(kr[QK + d]) : 0.0f));
        }}
        float dot = warp_sum(part);
{body}        float m_new = fmaxf(m, {value});
        float corr = m == NEG_INF ? 0.0f : expf(m - m_new);
        float p = expf({value} - m_new);
        l = l * corr + p;
#pragma unroll
        for (int j = 0; j < V_LANES; j++) {{
            int d = lane + 32 * j;
            if (d < VD) acc[j] = acc[j] * corr + p * (attn_float(vr[{v_offset} + d]) + (KV_PLANES == 2 ? attn_float(vr[{v_stride} + {v_offset} + d]) : 0.0f));
        }}
        m = m_new;
    }}
{blocks}    if (lane == 0) {{
        m_s[warp] = m;
        l_s[warp] = l;
    }}
#pragma unroll
    for (int j = 0; j < V_LANES; j++) {{
        int d = lane + 32 * j;
        if (d < VD) acc_s[warp][d] = acc[j];
    }}
    __syncthreads();
{sink_decl}    float top = {top_init};
    for (int w = 0; w < SCAN_WARPS; w++) top = fmaxf(top, m_s[w]);
    float denom = {sink_denom};
    for (int w = 0; w < SCAN_WARPS; w++) denom += m_s[w] == NEG_INF ? 0.0f : l_s[w] * expf(m_s[w] - top);
    for (int d = threadIdx.x; d < VD; d += blockDim.x) {{
        float total = 0.0f;
        for (int w = 0; w < SCAN_WARPS; w++) total += m_s[w] == NEG_INF ? 0.0f : acc_s[w][d] * expf(m_s[w] - top);
        out[(size_t)row * {out_row} + (size_t)head * {out_head} + {out_offset} + d] = total / denom;
    }}
}}

",
            threads = k.threads,
            name = k.name,
            sig = sig.join(", "),
            qk = g.k_width,
            vd = g.v_width,
            heads_per_row = "gridDim.x / KV_ROWS",
            layers = g.layers,
            v_stride = g.v_stride,
            v_offset = g.v_offset,
            out_row = g.out_row,
            out_head = g.out_head,
            out_offset = g.out_offset,
        ))
    }
}
