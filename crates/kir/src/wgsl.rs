use crate::layout::Dialect;
use crate::{Access, Activation, BlockSum, Body, Kernel, Language, lit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    Read,
    Write,
    Uniform,
}

pub fn bindings(k: &Kernel) -> Vec<Binding> {
    let mut b: Vec<Binding> =
        k.params.iter().map(|p| if p.access == Access::Write { Binding::Write } else { Binding::Read }).collect();
    b.push(Binding::Uniform);
    b
}

fn header(k: &Kernel, enable: &str) -> String {
    let mut s = String::from(enable);
    for (i, p) in k.params.iter().enumerate() {
        let mode = if p.access == Access::Write { "read_write" } else { "read" };
        s.push_str(&format!("@group(0) @binding({i}) var<storage, {mode}> {}: array<f32>;\n", p.name));
    }
    s.push_str("struct P { a: vec4<u32>, b: vec4<u32> };\n");
    s.push_str(&format!("@group(0) @binding({}) var<uniform> p: P;\n\n", k.params.len()));
    s
}

fn rmsnorm(k: &Kernel, width: u32, eps: f32, sum: BlockSum, offset: f32) -> String {
    let t = k.threads;
    let (x, w, out) = (k.params[0].name, k.params[1].name, k.params[2].name);
    let (enable, builtins, scratch, reduce) = match sum {
        BlockSum::SharedTree => (
            "",
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
    let total = red[0];
",
                half = t / 2
            ),
        ),
        BlockSum::Subgroup => (
            "",
            "@builtin(local_invocation_index) li: u32, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(num_subgroups) sgs: u32".to_string(),
            "var<workgroup> red: array<f32, 128>;\n".to_string(),
            "    let part = subgroupAdd(ss);
    if (lane == 0u) {
        red[sg] = part;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < sgs; i++) {
        total += red[i];
    }
"
            .to_string(),
        ),
        BlockSum::WarpShuffle => unreachable!(),
    };
    let mut s = header(k, enable);
    s.push_str(&format!(
        "const WIDTH: u32 = {width}u;
const EPS: f32 = {eps};
{scratch}
@compute @workgroup_size({t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, {builtins}) {{
    let t = wid.x;
    let src = (p.a.x + t) * WIDTH;
    let dst = (p.a.y + t) * WIDTH;
    var ss = 0.0;
    for (var i = li; i < WIDTH; i += {t}u) {{
        let v = {x}[src + i];
        ss += v * v;
    }}
{reduce}    let inv = inverseSqrt(total / f32(WIDTH) + EPS);
    for (var i = li; i < WIDTH; i += {t}u) {{
        {out}[dst + i] = {wi} * ({x}[src + i] * inv);
    }}
}}
",
        eps = lit(eps),
        wi = if offset == 0.0 { format!("{w}[i]") } else { format!("({w}[i] + {})", lit(offset)) },
    ));
    s
}

pub fn emit_kernel(k: &Kernel) -> Result<String, String> {
    k.validate()?;
    if k.target.language != Language::Wgsl {
        return Err(format!("{}: the WGSL emitter cannot target {:?}", k.name, k.target.language));
    }
    Ok(match &k.body {
        Body::RmsNormSegment { .. } | Body::AddRmsNorm { .. } | Body::NormAdd { .. } | Body::Pointwise { .. } | Body::Add { .. } => return Err(format!("{}: no WGSL emitter for this body yet", k.name)),
        Body::RmsNorm { width, eps, sum, offset } => {
            if *sum == BlockSum::WarpShuffle {
                return Err(format!("{}: block sum {sum:?} has no WGSL snippet", k.name));
            }
            if *sum == BlockSum::Subgroup && k.threads > 128 * k.target.warp_size.max(1) {
                return Err(format!("{}: too many subgroups for the scratch array", k.name));
            }
            rmsnorm(k, *width, *eps, *sum, *offset)
        }
        Body::GatedActivation { width, act } => {
            let (gu, out) = (k.params[0].name, k.params[1].name);
            let a = match act {
                Activation::Silu => "g / (1.0 + exp(-g))",
                Activation::GeluTanh => "0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))",
            };
            let mut s = header(k, "");
            s.push_str(&format!(
                "const WIDTH: u32 = {width}u;

@compute @workgroup_size({t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x + gid.y * 8388608u;
    if (idx >= p.a.x * WIDTH) {{
        return;
    }}
    let t = idx / WIDTH;
    let i = idx % WIDTH;
    let g = {gu}[t * 2u * WIDTH + {g}];
    let a = {a};
    {out}[idx] = a * {gu}[t * 2u * WIDTH + {u}];
}}
",
                t = k.threads,
                g = crate::cuda::gate_up_tile_in(*width, "i", "0", Dialect::Wgsl)?,
                u = crate::cuda::gate_up_tile_in(*width, "i", "1", Dialect::Wgsl)?,
            ));
            s
        }
    })
}

pub fn gemm_bindings(_k: &crate::gemm::KernelInstance) -> Vec<Binding> {
    vec![Binding::Read, Binding::Read, Binding::Read, Binding::Read, Binding::Write, Binding::Uniform]
}

fn dequant_fn(wf: crate::gemm::WeightFormat, k: usize) -> Result<String, String> {
    use crate::gemm::WeightFormat as W;
    Ok(match wf {
        W::Bf16 => format!(
            "fn wv(n: u32, c: u32) -> f32 {{
    let i = n * {k}u + c;
    let word = w[i / 2u];
    return bitcast<f32>(select(word << 16u, word & 0xffff0000u, (i & 1u) == 1u));
}}
"
        ),
        W::Fp8 => format!(
            "fn e4m3(b: u32) -> f32 {{
    return bitcast<f32>(((b & 0x80u) << 24u) | ((b & 0x7fu) << 20u)) * bitcast<f32>(0x7b800000u);
}}
fn byte_at(i: u32) -> u32 {{
    return (w[i / 4u] >> (8u * (i & 3u))) & 0xffu;
}}
fn wv(n: u32, c: u32) -> f32 {{
    let base = s[n];
    return e4m3(byte_at(n * {k}u + c)) * bitcast<f32>(s[base + c / {block}u]);
}}
",
            block = crate::gemm::FP8_BLOCK
        ),
        W::Fp4 => format!(
            "fn e4m3(b: u32) -> f32 {{
    return bitcast<f32>(((b & 0x80u) << 24u) | ((b & 0x7fu) << 20u)) * bitcast<f32>(0x7b800000u);
}}
fn e2m1(v: u32) -> f32 {{
    var mag = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    return select(mag[v & 7u], -mag[v & 7u], (v & 8u) != 0u);
}}
fn sbyte(i: u32) -> u32 {{
    return (s[i / 4u] >> (8u * (i & 3u))) & 0xffu;
}}
fn wv(n: u32, c: u32) -> f32 {{
    let i = n * {k}u + c;
    let byte = (w[i / 8u] >> (8u * ((i / 2u) & 3u))) & 0xffu;
    let nib = select(byte & 15u, byte >> 4u, (i & 1u) == 1u);
    let sc = e4m3(sbyte({header}u + n * {kg}u + c / {group}u));
    return e2m1(nib) * sc * bitcast<f32>(s[0]);
}}
",
            header = 16,
            kg = k / crate::gemm::FP4_GROUP,
            group = crate::gemm::FP4_GROUP
        ),
        W::Q8 | W::Ggml => return Err(format!("block DequantF16 has no {} decoder", wf.label())),
    })
}

pub fn emit_gemm(k: &crate::gemm::KernelInstance) -> Result<String, String> {
    use crate::gemm::{Epilogue, Kind, Mainloop};
    k.validate().map_err(|e| e.to_string())?;
    if k.target.language != Language::Wgsl {
        return Err(format!("{}: the WGSL emitter cannot target {:?}", k.name, k.target.language));
    }
    let Kind::Tile { shape, wf, main: Mainloop::DequantF16 { tile }, .. } = k.kind else {
        return Err(format!("{}: block {} has no WGSL lowering", k.name, k.kind.block()));
    };
    let (n, kk, t) = (shape.n, shape.k, tile);
    let act = match shape.act {
        Activation::Silu => "g / (1.0 + exp(-g))",
        Activation::GeluTanh => "0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))",
    };
    let store = match shape.epi {
        Epilogue::Store | Epilogue::Residual => format!(
            "    if (m < tokens && col < {n}u) {{
        var v = acc{bias};
        let at = m * {n}u + col;
        {res}out[at] = v;
    }}
",
            bias = if shape.bias { " + bias[col]" } else { "" },
            res = if shape.epi == Epilogue::Residual { "v += out[at];\n        " } else { "" },
        ),
        Epilogue::Act => format!(
            "    res[ly][lx] = acc;
    workgroupBarrier();
    if (m < tokens && col < {n}u && (lx & 1u) == 0u) {{
        let g = res[ly][lx];
        out[m * {half}u + col / 2u] = ({act}) * res[ly][lx + 1u];
    }}
",
            half = n / 2,
        ),
    };
    let mut s = String::from("enable f16;\n");
    s.push_str("@group(0) @binding(0) var<storage, read> w: array<u32>;\n");
    s.push_str("@group(0) @binding(1) var<storage, read> s: array<u32>;\n");
    s.push_str("@group(0) @binding(2) var<storage, read> bias: array<f32>;\n");
    s.push_str("@group(0) @binding(3) var<storage, read> x: array<f32>;\n");
    s.push_str("@group(0) @binding(4) var<storage, read_write> out: array<f32>;\n");
    s.push_str("struct P { a: vec4<u32>, b: vec4<u32> };\n@group(0) @binding(5) var<uniform> p: P;\n\n");
    s.push_str(&dequant_fn(wf, kk)?);
    s.push_str(&format!(
        "
var<workgroup> xs: array<array<f16, {t}>, {t}>;
var<workgroup> ws: array<array<f16, {t}>, {t}>;
var<workgroup> res: array<array<f32, {t}>, {t}>;

@compute @workgroup_size({t}, {t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tokens = p.a.x;
    let lx = lid.x;
    let ly = lid.y;
    let col = wid.x * {t}u + lx;
    let m = wid.y * {t}u + ly;
    let wrow = min(wid.x * {t}u + ly, {last}u);
    let xrow = min(m, max(tokens, 1u) - 1u);
    var acc = 0.0;
    for (var k0 = 0u; k0 < {kk}u; k0 += {t}u) {{
        xs[ly][lx] = f16(x[xrow * {kk}u + k0 + lx]);
        ws[ly][lx] = f16(wv(wrow, k0 + lx));
        workgroupBarrier();
        for (var i = 0u; i < {t}u; i++) {{
            acc += f32(xs[ly][i]) * f32(ws[lx][i]);
        }}
        workgroupBarrier();
    }}
{store}}}
",
        last = n - 1,
    ));
    Ok(s)
}

pub fn ggml_decode() -> String {
    crate::ggml::wgsl_decoder()
}

pub fn ggml_bindings() -> Vec<Binding> {
    vec![Binding::Read, Binding::Read, Binding::Write, Binding::Uniform]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    Embed,
    Rope,
    Attention,
    Bias,
    Argmax,
    Dequant,
}

pub fn ops_header(s: &crate::ops::WgslShape) -> String {
    use crate::ops::wgsl_float as f;
    format!(
        "const HIDDEN: u32 = {}u;
const INTER: u32 = {}u;
const VOCAB: u32 = {}u;
const HEADS: u32 = {}u;
const KV_HEADS: u32 = {}u;
const HEAD_DIM: u32 = {}u;
const GROUP: u32 = {}u;
const QK_NORM: bool = {};
const EPS: f32 = {};
const THETA: f32 = {};
const ROPE_KIND: u32 = {}u;
const ROPE_FACTOR: f32 = {};
const ROPE_LOW_FREQ: f32 = {};
const ROPE_HIGH_FREQ: f32 = {};
const ROPE_ORIG_POS: f32 = {};
const ROPE_MSCALE: f32 = {};
const ACT_KIND: u32 = {}u;
const MAX_CTX: u32 = {}u;
const Q_DIM: u32 = {}u;
const QKV_ROW: u32 = {}u;
const ROT: u32 = {}u;
const NEG_INF: f32 = -3.0e38;
",
        s.hidden,
        s.inter,
        s.vocab,
        s.heads,
        s.kv_heads,
        s.head_dim,
        s.heads / s.kv_heads,
        s.qk_norm,
        f(s.eps),
        f(s.theta),
        s.rope_kind,
        f(s.rope_factor),
        f(s.rope_low),
        f(s.rope_high),
        f(s.rope_original),
        f(s.rope_mscale),
        if s.gated_gelu { 1 } else { 0 },
        s.max_ctx,
        s.q_dim,
        s.qkv_row,
        s.rotary_dim,
    )
}

pub fn emit_op(kind: OpKind, s: &crate::ops::WgslShape) -> String {
    use crate::prims::{Prim, PrimKernel, RowSource};
    let t = crate::Target::wgpu_baseline();
    let prim = |p: Prim| PrimKernel::new(&t, "main", p).and_then(|k| k.emit()).expect("wgsl prim");
    let mut out = ops_header(s);
    out.push_str(&match kind {
        OpKind::Embed => prim(Prim::Embed { width: s.hidden, scale: 1.0, source: RowSource::Ggml }),
        OpKind::Rope => rope(s),
        OpKind::Attention => attention(),
        OpKind::Bias => prim(Prim::AddBias),
        OpKind::Argmax => prim(Prim::Sample { vocab: s.vocab }),
        OpKind::Dequant => prim(Prim::DequantRows),
    });
    out
}

pub fn emit_linear_op(s: &crate::ops::WgslShape, tokens: usize, workgroup: usize) -> String {
    let mut out = ops_header(s);
    out.push_str(&linear_rows(tokens, workgroup));
    out.push_str(&ggml_decode());
    out
}

const PARAMS: &str = "struct P { a: vec4<u32>, b: vec4<u32> };\n";

fn storage(list: &[(&str, bool, &str)]) -> String {
    let mut s = String::new();
    for (i, (name, write, ty)) in list.iter().enumerate() {
        let mode = if *write { "read_write" } else { "read" };
        s.push_str(&format!("@group(0) @binding({i}) var<storage, {mode}> {name}: array<{ty}>;\n"));
    }
    s.push_str(PARAMS);
    s.push_str(&format!("@group(0) @binding({}) var<uniform> p: P;\n", list.len()));
    s
}

fn tree_sum(var: &str, n: u32, max: bool, pad: &str) -> String {
    let body = if max { format!("{var}[li] = max({var}[li], {var}[li + h]);") } else { format!("{var}[li] += {var}[li + h];") };
    format!(
        "{pad}for (var h = {}u; h > 0u; h >>= 1u) {{
{pad}    if (li < h) {{
{pad}        {body}
{pad}    }}
{pad}    workgroupBarrier();
{pad}}}
",
        n / 2
    )
}

pub const ROPE_THREADS: u32 = 128;
pub const ATTENTION_TILE: u32 = 128;

fn rope(_s: &crate::ops::WgslShape) -> String {
    let t = ROPE_THREADS;
    format!(
        "{bind}
var<workgroup> red: array<f32, {t}>;
var<workgroup> vec_s: array<f32, HEAD_DIM>;

{freq}
@compute @workgroup_size({t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let head = wid.x;
    let row = wid.y;
    let pos = p.a.x + row;
    let is_q = head < HEADS;
    let is_k = !is_q && head < HEADS + KV_HEADS;
    let at = row * QKV_ROW + head * HEAD_DIM;
    var ss = 0.0;
    for (var d = li; d < HEAD_DIM; d += {t}u) {{
        let v = qkv[at + d];
        vec_s[d] = v;
        ss += v * v;
    }}
    red[li] = ss;
    workgroupBarrier();
{sum}    let inv = inverseSqrt(red[0] / f32(HEAD_DIM) + EPS);
    workgroupBarrier();
    if (QK_NORM && (is_q || is_k)) {{
        for (var d = li; d < HEAD_DIM; d += {t}u) {{
            let w = select(k_norm[d], q_norm[d], is_q);
            vec_s[d] = w * (vec_s[d] * inv);
        }}
        workgroupBarrier();
    }}
    let hr = ROT / 2u;
    for (var d = li; d < HEAD_DIM; d += {t}u) {{
        var v = vec_s[d];
        if ((is_q || is_k) && d < ROT) {{
            let i = select(d - hr, d, d < hr);
            let angle = f32(pos) * inv_freq(i);
            let c = cos(angle) * ROPE_MSCALE;
            let s = sin(angle) * ROPE_MSCALE;
            if (d < hr) {{
                v = vec_s[d] * c - vec_s[d + hr] * s;
            }} else {{
                v = vec_s[d] * c + vec_s[d - hr] * s;
            }}
        }}
        if (is_q) {{
            qkv[at + d] = v;
        }} else {{
            let kvh = select(head - HEADS - KV_HEADS, head - HEADS, is_k);
            let slot = (kvh * MAX_CTX + pos) * HEAD_DIM + d;
            if (is_k) {{
                k_cache[slot] = v;
            }} else {{
                v_cache[slot] = v;
            }}
        }}
    }}
}}
",
        bind = storage(&[("qkv", true, "f32"), ("q_norm", false, "f32"), ("k_norm", false, "f32"), ("k_cache", true, "f32"), ("v_cache", true, "f32")]),
        freq = crate::rope::inv_freq_fn(crate::layout::Dialect::Wgsl, &[1, 2, 3]),
        sum = tree_sum("red", t, false, "    "),
    )
}

fn attention() -> String {
    let t = ATTENTION_TILE;
    format!(
        "{bind}
const TILE: u32 = {t}u;
var<workgroup> qs: array<f32, HEAD_DIM>;
var<workgroup> ps: array<f32, {t}>;
var<workgroup> red: array<f32, {t}>;

@compute @workgroup_size({t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let head = wid.x;
    let t = wid.y;
    let kvh = head / GROUP;
    let len = p.a.x + t + 1u;
    let head_base = kvh * MAX_CTX * HEAD_DIM;
    for (var d = li; d < HEAD_DIM; d += TILE) {{
        qs[d] = qkv[t * QKV_ROW + head * HEAD_DIM + d];
    }}
    workgroupBarrier();
    let scale = inverseSqrt(f32(HEAD_DIM));
    var m = NEG_INF;
    var l = 0.0;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var t0 = 0u; t0 < len; t0 += TILE) {{
        let key = t0 + li;
        var s = NEG_INF;
        if (key < len) {{
            var dot = 0.0;
            let kb = head_base + key * HEAD_DIM;
            for (var d = 0u; d < HEAD_DIM; d++) {{
                dot += qs[d] * k_cache[kb + d];
            }}
            s = dot * scale;
        }}
        red[li] = s;
        workgroupBarrier();
{max}        let m_new = max(m, red[0]);
        let corr = exp(m - m_new);
        ps[li] = select(0.0, exp(s - m_new), key < len);
        workgroupBarrier();
        let n = min(TILE, len - t0);
        var psum = 0.0;
        var a0 = 0.0;
        var a1 = 0.0;
        for (var i = 0u; i < n; i++) {{
            let pi = ps[i];
            psum += pi;
            let vb = head_base + (t0 + i) * HEAD_DIM;
            a0 += pi * v_cache[vb + min(li, HEAD_DIM - 1u)];
            if (HEAD_DIM > TILE) {{
                a1 += pi * v_cache[vb + min(li + TILE, HEAD_DIM - 1u)];
            }}
        }}
        l = l * corr + psum;
        acc0 = acc0 * corr + a0;
        acc1 = acc1 * corr + a1;
        m = m_new;
        workgroupBarrier();
    }}
    let ob = t * HEADS * HEAD_DIM + head * HEAD_DIM;
    if (li < HEAD_DIM) {{
        out[ob + li] = acc0 / l;
    }}
    if (HEAD_DIM > TILE && li + TILE < HEAD_DIM) {{
        out[ob + li + TILE] = acc1 / l;
    }}
}}
",
        bind = storage(&[("qkv", false, "f32"), ("k_cache", false, "f32"), ("v_cache", false, "f32"), ("out", true, "f32")]),
        max = tree_sum("red", t, true, "        "),
    )
}

pub const ROW_SPAN: u32 = 32768;

fn linear_rows(tokens: usize, workgroup: usize) -> String {
    let wg = workgroup;
    format!(
        "{bind}
const TT: u32 = {tokens}u;
const WG: u32 = {wg}u;
var<workgroup> red: array<f32, {reduce}>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let nrows = p.a.x;
    let cols = p.a.y;
    let tokens = p.a.z;
    let x_row0 = p.a.w;
    let y_row0 = p.b.x;
    let accumulate = p.b.y;
    let r = min(wid.x + wid.y * {ROW_SPAN}u, nrows - 1u);
    let valid = (wid.x + wid.y * {ROW_SPAN}u) < nrows;
    let t0 = wid.z * TT;
    let entry = rows[r];
    let base = (entry >> 3u) * 16u;
    let code = entry & 7u;
    var acc: array<f32, {tokens}>;
    for (var c = li; c < cols; c += WG) {{
        let w = dequant_at(base, code, c);
        for (var j = 0u; j < TT; j++) {{
            let t = min(t0 + j, tokens - 1u);
            acc[j] += w * x[(x_row0 + t) * cols + c];
        }}
    }}
    for (var j = 0u; j < TT; j++) {{
        red[j * WG + li] = acc[j];
    }}
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {{
        if (li < s) {{
            for (var j = 0u; j < TT; j++) {{
                red[j * WG + li] += red[j * WG + li + s];
            }}
        }}
        workgroupBarrier();
    }}
    if (li < TT && valid && t0 + li < tokens) {{
        let at = (y_row0 + t0 + li) * nrows + r;
        let v = red[li * WG];
        if (accumulate == 1u) {{
            y[at] = y[at] + v;
        }} else {{
            y[at] = v;
        }}
    }}
}}
",
        bind = storage(&[("x", false, "f32"), ("y", true, "f32"), ("wdata", false, "u32"), ("rows", false, "u32")]),
        reduce = tokens * workgroup,
    )
}
