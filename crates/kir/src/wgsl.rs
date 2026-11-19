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
        Body::Compose(c) => c.emit_wgsl(k, &header(k, ""))?,
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

pub const TILE_TOKENS: u32 = 64;
pub const TILE_ROWS: u32 = 64;
pub const TILE_K: u32 = 16;

pub fn emit_linear_tile_op(s: &crate::ops::WgslShape) -> String {
    let mut out = ops_header(s);
    out.push_str(&linear_tile());
    out.push_str(&ggml_decode());
    out
}

fn linear_tile() -> String {
    format!(
        "{bind}
const BM: u32 = {TILE_TOKENS}u;
const BN: u32 = {TILE_ROWS}u;
const BK: u32 = {TILE_K}u;
var<workgroup> xs: array<f32, {xs}>;
var<workgroup> ws: array<f32, {ws}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let nrows = p.a.x;
    let cols = p.a.y;
    let tokens = p.a.z;
    let x_row0 = p.a.w;
    let y_row0 = p.b.x;
    let accumulate = p.b.y;
    let r0 = wid.x * BN;
    let t0 = wid.z * BM;
    let tx = li % 16u;
    let ty = li / 16u;
    var acc: array<vec4<f32>, 4>;
    for (var k0 = 0u; k0 < cols; k0 += BK) {{
        for (var n = 0u; n < 4u; n++) {{
            let idx = li + n * 256u;
            let k = idx % BK;
            let m = idx / BK;
            let c = k0 + k;
            let t = t0 + m;
            var xv = 0.0;
            if (t < tokens && c < cols) {{
                xv = x[(x_row0 + t) * cols + c];
            }}
            xs[k * BM + m] = xv;
            let r = min(r0 + m, nrows - 1u);
            var wv = 0.0;
            if (c < cols) {{
                let entry = rows[r];
                wv = dequant_at((entry >> 3u) * 16u, entry & 7u, c);
            }}
            ws[k * BN + m] = wv;
        }}
        workgroupBarrier();
        for (var k = 0u; k < BK; k++) {{
            let a = vec4<f32>(xs[k * BM + ty * 4u], xs[k * BM + ty * 4u + 1u], xs[k * BM + ty * 4u + 2u], xs[k * BM + ty * 4u + 3u]);
            let b = vec4<f32>(ws[k * BN + tx * 4u], ws[k * BN + tx * 4u + 1u], ws[k * BN + tx * 4u + 2u], ws[k * BN + tx * 4u + 3u]);
            acc[0] += a.x * b;
            acc[1] += a.y * b;
            acc[2] += a.z * b;
            acc[3] += a.w * b;
        }}
        workgroupBarrier();
    }}
    for (var i = 0u; i < 4u; i++) {{
        let t = t0 + ty * 4u + i;
        if (t >= tokens) {{
            continue;
        }}
        for (var j = 0u; j < 4u; j++) {{
            let r = r0 + tx * 4u + j;
            if (r < nrows) {{
                let at = (y_row0 + t) * nrows + r;
                if (accumulate == 1u) {{
                    y[at] = y[at] + acc[i][j];
                }} else {{
                    y[at] = acc[i][j];
                }}
            }}
        }}
    }}
}}
",
        bind = storage(&[("x", false, "f32"), ("y", true, "f32"), ("wdata", false, "u32"), ("rows", false, "u32")]),
        xs = TILE_K * TILE_TOKENS,
        ws = TILE_K * TILE_ROWS,
    )
}

pub const COOP_THREADS: u32 = 128;
pub const COOP_TOKENS: u32 = 64;
pub const COOP_ROWS: u32 = 64;
pub const COOP_K: u32 = 32;

pub fn emit_linear_coop_op(s: &crate::ops::WgslShape) -> String {
    let mut out = String::from("enable wgpu_cooperative_matrix;\n");
    out.push_str(&ops_header(s));
    out.push_str(&linear_coop());
    out.push_str(&ggml_decode());
    out
}

fn linear_coop() -> String {
    format!(
        "{bind}
const BM: u32 = {COOP_TOKENS}u;
const BN: u32 = {COOP_ROWS}u;
const BK: u32 = {COOP_K}u;
const WOFF: u32 = {woff}u;
var<workgroup> sh: array<f32, {os}>;

@compute @workgroup_size({COOP_THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let nrows = p.a.x;
    let cols = p.a.y;
    let tokens = p.a.z;
    let x_row0 = p.a.w;
    let y_row0 = p.b.x;
    let accumulate = p.b.y;
    let ntt = (tokens + BM - 1u) / BM;
    let nrt = (nrows + BN - 1u) / BN;
    let lin = wid.x + wid.z * nrt;
    let r0 = (lin / ntt) * BN;
    let t0 = (lin % ntt) * BM;
    let sg = li / 32u;
    let sm = (sg / 2u) * 32u;
    let sn = (sg % 2u) * 32u;
{acc_init}    var ents: array<u32, {quads}>;
    for (var n = 0u; n < {quads}u; n++) {{
        ents[n] = rows[min(r0 + (li + n * {COOP_THREADS}u) / (BK / 4u), nrows - 1u)];
    }}
    let cols4 = cols / 4u;
    for (var k0 = 0u; k0 < cols; k0 += BK) {{
        for (var n = 0u; n < {quads}u; n++) {{
            let idx = li + n * {COOP_THREADS}u;
            let kq = idx % (BK / 4u);
            let m = idx / (BK / 4u);
            let c = k0 + kq * 4u;
            let t = t0 + m;
            var xv = vec4<f32>(0.0);
            var w0 = 0.0;
            var w1 = 0.0;
            var w2 = 0.0;
            var w3 = 0.0;
            if (t < tokens && c < cols) {{
                xv = x[(x_row0 + t) * cols4 + c / 4u];
            }}
            if (c < cols) {{
                let entry = ents[n];
                let base = (entry >> 3u) * 16u;
                let code = entry & 7u;
                if (code == 4u) {{
                    let wa = wdata[(base + 2u * c) >> 2u];
                    let wb = wdata[((base + 2u * c) >> 2u) + 1u];
                    w0 = bitcast<f32>(wa << 16u);
                    w1 = bitcast<f32>(wa & 0xffff0000u);
                    w2 = bitcast<f32>(wb << 16u);
                    w3 = bitcast<f32>(wb & 0xffff0000u);
                }} else if (code == 0u) {{
                    let sb = base + (c / 256u) * 144u;
                    let i = c % 256u;
                    let j = i / 64u;
                    let hi = (i % 64u) >= 32u;
                    let sm = scale_min_k4(2u * j + select(0u, 1u, hi), sb);
                    let d1 = wf16(sb) * f32(sm.x);
                    let mm = wf16(sb + 2u) * f32(sm.y);
                    var q = wdata[(sb + 16u + 32u * j + (i % 32u)) >> 2u];
                    if (hi) {{
                        q = q >> 4u;
                    }}
                    w0 = d1 * f32(q & 0xfu) - mm;
                    w1 = d1 * f32((q >> 8u) & 0xfu) - mm;
                    w2 = d1 * f32((q >> 16u) & 0xfu) - mm;
                    w3 = d1 * f32((q >> 24u) & 0xfu) - mm;
                }} else if (code == 2u) {{
                    let sb = base + (c / 256u) * 224u;
                    let i = c % 256u;
                    let hr = i / 128u;
                    let r = i % 128u;
                    let seg = r / 32u;
                    let l = r % 32u;
                    let ql = wdata[(sb + 64u * hr + (seg & 1u) * 32u + l) >> 2u] >> (4u * (seg >> 1u));
                    let qh = wdata[(sb + 128u + 32u * hr + l) >> 2u] >> (2u * seg);
                    let v = (ql & 0x0f0f0f0fu) | ((qh & 0x03030303u) << 4u);
                    let dd = wf16(sb + 208u) * wi8(sb + 192u + 8u * hr + r / 16u);
                    w0 = dd * (f32(v & 0xffu) - 32.0);
                    w1 = dd * (f32((v >> 8u) & 0xffu) - 32.0);
                    w2 = dd * (f32((v >> 16u) & 0xffu) - 32.0);
                    w3 = dd * (f32(v >> 24u) - 32.0);
                }} else {{
                    w0 = dequant_at(base, code, c);
                    w1 = dequant_at(base, code, c + 1u);
                    w2 = dequant_at(base, code, c + 2u);
                    w3 = dequant_at(base, code, c + 3u);
                }}
            }}
            let so = ((m / 8u) * (BK / 8u) + (kq * 4u) / 8u) * 64u + (m % 8u) * 8u + (kq * 4u) % 8u;
            sh[so] = xv.x;
            sh[so + 1u] = xv.y;
            sh[so + 2u] = xv.z;
            sh[so + 3u] = xv.w;
            sh[WOFF + so] = w0;
            sh[WOFF + so + 1u] = w1;
            sh[WOFF + so + 2u] = w2;
            sh[WOFF + so + 3u] = w3;
        }}
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk += 8u) {{
{mma}        }}
        workgroupBarrier();
    }}
    if (accumulate == 0u && r0 + BN <= nrows && t0 + BM <= tokens) {{
{direct}        return;
    }}
{store}    workgroupBarrier();
    for (var e = li; e < BM * BN; e += {COOP_THREADS}u) {{
        let m = e / BN;
        let n = e % BN;
        let t = t0 + m;
        let r = r0 + n;
        if (t < tokens && r < nrows) {{
            let at = (y_row0 + t) * nrows + r;
            if (accumulate == 1u) {{
                y[at] = y[at] + sh[e];
            }} else {{
                y[at] = sh[e];
            }}
        }}
    }}
}}
",
        bind = storage(&[("x", false, "vec4<f32>"), ("y", true, "f32"), ("wdata", false, "u32"), ("rows", false, "u32")]),
        woff = COOP_K * COOP_TOKENS,
        os = (COOP_TOKENS * COOP_ROWS).max(COOP_K * (COOP_TOKENS + COOP_ROWS)),
        quads = COOP_K * COOP_TOKENS / 4 / COOP_THREADS,
        acc_init = (0..16).map(|n| format!("    var acc{n} = coop_mat8x8<f32, C>();\n")).collect::<String>(),
        mma = coop_mma(),
        direct = (0..16).map(|n| format!("        coopStoreT(acc{n}, &y[(y_row0 + t0 + sm + {}u) * nrows + r0 + sn + {}u], nrows);\n", (n / 4) * 8, (n % 4) * 8)).collect::<String>(),
        store = (0..16).map(|n| format!("    coopStoreT(acc{n}, &sh[(sm + {}u) * BN + sn + {}u], BN);\n", (n / 4) * 8, (n % 4) * 8)).collect::<String>(),
    )
}

fn coop_mma() -> String {
    let mut s = String::new();
    for i in 0..4 {
        s.push_str(&format!("            let a{i} = coopLoadT<coop_mat8x8<f32, A>>(&sh[((sm + {}u) / 8u * (BK / 8u) + kk / 8u) * 64u], 8u);\n", i * 8));
        s.push_str(&format!("            let b{i} = coopLoad<coop_mat8x8<f32, B>>(&sh[WOFF + ((sn + {}u) / 8u * (BK / 8u) + kk / 8u) * 64u], 8u);\n", i * 8));
    }
    for n in 0..16 {
        s.push_str(&format!("            acc{n} = coopMultiplyAdd(a{}, b{}, acc{n});\n", n / 4, n % 4));
    }
    s
}

pub const SG_ROWS: u32 = 4;
pub const SG_LANE_ELEMS: u32 = 8;

pub fn emit_linear_sg_op(s: &crate::ops::WgslShape) -> String {
    let mut out = ops_header(s);
    out.push_str(&linear_sg());
    out.push_str(&ggml_decode());
    out
}

fn linear_sg() -> String {
    format!(
        "{bind}
fn nib_lo(w: u32) -> vec4<f32> {{
    return vec4<f32>(vec4<u32>(w & 0xfu, (w >> 8u) & 0xfu, (w >> 16u) & 0xfu, (w >> 24u) & 0xfu));
}}

fn nib_hi(w: u32) -> vec4<f32> {{
    return vec4<f32>(vec4<u32>((w >> 4u) & 0xfu, (w >> 12u) & 0xfu, (w >> 20u) & 0xfu, (w >> 28u) & 0xfu));
}}

fn xs(i: u32) -> f32 {{
    return x[i >> 2u][i & 3u];
}}

@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {{
    let nrows = p.a.x;
    let cols = p.a.y;
    let x_row0 = p.a.w;
    let y_row0 = p.b.x;
    let accumulate = p.b.y;
    let r = (wid.x + wid.y * {ROW_SPAN}u) * {SG_ROWS}u + sg;
    if (r >= nrows) {{
        return;
    }}
    let entry = rows[r];
    let base = (entry >> 3u) * 16u;
    let code = entry & 7u;
    var acc = 0.0;
    let xo = x_row0 * cols;
    if (code == 4u) {{
        let b4 = base >> 2u;
        for (var q = lane; q < cols / 2u; q += 32u) {{
            let w = wdata[b4 + q];
            acc += bitcast<f32>(w << 16u) * xs(xo + 2u * q) + bitcast<f32>(w & 0xffff0000u) * xs(xo + 2u * q + 1u);
        }}
    }} else if (code == 0u) {{
        let ix = lane / 8u;
        let iq = (lane % 8u) / 4u;
        let ir = lane % 4u;
        let sh = 16u * iq;
        for (var sb = ix; sb < cols / 256u; sb += 4u) {{
            let b = base + sb * 144u;
            let b4 = b >> 2u;
            let xb = (xo + sb * 256u + 64u * iq + 8u * ir) / 4u;
            let ya0 = x[xb];
            let ya1 = x[xb + 1u];
            let yb0 = x[xb + 8u];
            let yb1 = x[xb + 9u];
            let yc0 = x[xb + 32u];
            let yc1 = x[xb + 33u];
            let yd0 = x[xb + 40u];
            let yd1 = x[xb + 41u];
            let sy = vec4<f32>(dot(ya0 + ya1, vec4<f32>(1.0)), dot(yb0 + yb1, vec4<f32>(1.0)), dot(yc0 + yc1, vec4<f32>(1.0)), dot(yd0 + yd1, vec4<f32>(1.0)));
            let dd = unpack2x16float(wdata[b4]);
            let s0 = wdata[b4 + 1u];
            let s1 = wdata[b4 + 2u];
            let s2 = wdata[b4 + 3u];
            let scl = (s0 & 0x3f3f3f3fu) >> sh;
            let mnl = (s1 & 0x3f3f3f3fu) >> sh;
            let sch = ((s2 & 0x0f0f0f0fu) | (((s0 >> 6u) & 0x03030303u) << 4u)) >> sh;
            let mnh = (((s2 >> 4u) & 0x0f0f0f0fu) | (((s1 >> 6u) & 0x03030303u) << 4u)) >> sh;
            let q = (b + 16u + 32u * iq + 8u * ir) >> 2u;
            let qa = wdata[q];
            let qb = wdata[q + 1u];
            let qc = wdata[q + 16u];
            let qd = wdata[q + 17u];
            let d0 = dot(nib_lo(qa), ya0) + dot(nib_lo(qb), ya1);
            let d1 = dot(nib_hi(qa), yb0) + dot(nib_hi(qb), yb1);
            let d2 = dot(nib_lo(qc), yc0) + dot(nib_lo(qd), yc1);
            let d3 = dot(nib_hi(qc), yd0) + dot(nib_hi(qd), yd1);
            let sc = vec4<f32>(f32(scl & 0xffu), f32((scl >> 8u) & 0xffu), f32(sch & 0xffu), f32((sch >> 8u) & 0xffu));
            let mn = vec4<f32>(f32(mnl & 0xffu), f32((mnl >> 8u) & 0xffu), f32(mnh & 0xffu), f32((mnh >> 8u) & 0xffu));
            acc += dd.x * dot(sc, vec4<f32>(d0, d1, d2, d3)) - dd.y * dot(mn, sy);
        }}
    }} else if (code == 2u) {{
        let hr = (lane / 8u) % 2u;
        let l = (lane % 8u) * 4u;
        let isc = l / 16u;
        for (var sb = lane / 16u; sb < cols / 256u; sb += 2u) {{
            let b = base + sb * 224u;
            let d = wf16(b + 208u);
            let ql0 = wdata[(b + 64u * hr + l) >> 2u];
            let ql1 = wdata[(b + 64u * hr + l + 32u) >> 2u];
            let qh = wdata[(b + 128u + 32u * hr + l) >> 2u];
            let sw = (b + 192u + 8u * hr) >> 2u;
            let sc = vec4<f32>(vec4<i32>(bitcast<i32>(wdata[sw] << (24u - 8u * isc)) >> 24u, bitcast<i32>(wdata[sw] << (8u - 8u * isc)) >> 24u, bitcast<i32>(wdata[sw + 1u] << (24u - 8u * isc)) >> 24u, bitcast<i32>(wdata[sw + 1u] << (8u - 8u * isc)) >> 24u));
            let xb = (xo + sb * 256u + hr * 128u + l) / 4u;
            var part = 0.0;
            for (var seg = 0u; seg < 4u; seg++) {{
                let qw = select(ql0, ql1, (seg & 1u) == 1u) >> (4u * (seg >> 1u));
                let v = (qw & 0x0f0f0f0fu) | (((qh >> (2u * seg)) & 0x03030303u) << 4u);
                let f = vec4<f32>(vec4<u32>(v & 0xffu, (v >> 8u) & 0xffu, (v >> 16u) & 0xffu, v >> 24u)) - vec4<f32>(32.0);
                part += sc[seg] * dot(f, x[xb + seg * 8u]);
            }}
            acc += d * part;
        }}
    }} else if (code == 3u) {{
        let blk = lane / 4u;
        let off = (lane % 4u) * 8u;
        for (var sb = 0u; sb < cols / 256u; sb++) {{
            let b = base + sb * 272u + blk * 34u;
            let d = wf16(b);
            let xb = xo + sb * 256u + blk * 32u + off;
            var part = 0.0;
            for (var i = 0u; i < 8u; i++) {{
                part += wi8(b + 2u + off + i) * xs(xb + i);
            }}
            acc += d * part;
        }}
    }} else {{
        for (var c0 = 0u; c0 < cols; c0 += {span}u) {{
            let e0 = c0 + lane * {SG_LANE_ELEMS}u;
            for (var i = 0u; i < {SG_LANE_ELEMS}u; i++) {{
                let e = e0 + i;
                if (e < cols) {{
                    acc += dequant_at(base, code, e) * xs(xo + e);
                }}
            }}
        }}
    }}
    let total = subgroupAdd(acc);
    if (lane == 0u) {{
        let at = y_row0 * nrows + r;
        if (accumulate == 1u) {{
            y[at] = y[at] + total;
        }} else {{
            y[at] = total;
        }}
    }}
}}
",
        bind = storage(&[("x", false, "vec4<f32>"), ("y", true, "f32"), ("wdata", false, "u32"), ("rows", false, "u32")]),
        threads = 32 * SG_ROWS,
        span = 32 * SG_LANE_ELEMS,
    )
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

pub const FLASH_GROUPS: u32 = 4;
pub const FLASH_TOKENS: u32 = FLASH_GROUPS;

pub fn emit_flash_op(s: &crate::ops::WgslShape) -> String {
    let mut out = ops_header(s);
    out.push_str(&flash());
    out
}

fn flash() -> String {
    format!(
        "{bind}
const HD4: u32 = HEAD_DIM / 4u;
const EPL: u32 = (HD4 + 31u) / 32u;
var<workgroup> qs: array<vec4<f32>, {qs}>;

@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let head = wid.x;
    let g = li / 32u;
    let t = wid.y * {FLASH_GROUPS}u + g;
    let lane = li % 32u;
    let live_row = t < p.a.y;
    let tt = min(t, p.a.y - 1u);
    let kvh = head / GROUP;
    let len = p.a.x + tt + 1u;
    let head_base = kvh * MAX_CTX * HEAD_DIM;
    let scale = inverseSqrt(f32(HEAD_DIM));
    for (var d = lane; d < HD4; d += 32u) {{
        let qb = tt * QKV_ROW + head * HEAD_DIM + d * 4u;
        qs[g * HD4 + d] = vec4<f32>(qkv[qb], qkv[qb + 1u], qkv[qb + 2u], qkv[qb + 3u]) * scale;
    }}
    workgroupBarrier();
    var acc: array<vec4<f32>, EPL>;
    var m = NEG_INF;
    var l = 0.0;
    for (var key0 = 0u; key0 < len; key0 += 32u) {{
        let key = key0 + lane;
        var s = NEG_INF;
        if (key < len) {{
            let kb = (head_base + key * HEAD_DIM) / 4u;
            var dot = vec4<f32>(0.0);
            for (var d = 0u; d < HD4; d++) {{
                dot += qs[g * HD4 + d] * k_cache[kb + d];
            }}
            s = dot.x + dot.y + dot.z + dot.w;
        }}
        let m_new = max(m, subgroupMax(s));
        let corr = exp(m - m_new);
        let pr = select(0.0, exp(s - m_new), key < len);
        l = l * corr + subgroupAdd(pr);
        for (var i = 0u; i < EPL; i++) {{
            acc[i] *= corr;
        }}
        let n = min(32u, len - key0);
        for (var j = 0u; j < n; j++) {{
            let pj = subgroupShuffle(pr, j);
            let vb = (head_base + (key0 + j) * HEAD_DIM) / 4u;
            for (var i = 0u; i < EPL; i++) {{
                let d = lane + 32u * i;
                if (d < HD4) {{
                    acc[i] += pj * v_cache[vb + d];
                }}
            }}
        }}
        m = m_new;
    }}
    if (live_row) {{
        let ob = t * HEADS * HEAD_DIM + head * HEAD_DIM;
        for (var i = 0u; i < EPL; i++) {{
            let d = lane + 32u * i;
            if (d < HD4) {{
                let o = acc[i] / l;
                out[ob + d * 4u] = o.x;
                out[ob + d * 4u + 1u] = o.y;
                out[ob + d * 4u + 2u] = o.z;
                out[ob + d * 4u + 3u] = o.w;
            }}
        }}
    }}
}}
",
        bind = storage(&[("qkv", false, "f32"), ("k_cache", false, "vec4<f32>"), ("v_cache", false, "vec4<f32>"), ("out", true, "f32")]),
        threads = 32 * FLASH_GROUPS,
        qs = FLASH_GROUPS * 64,
    )
}

pub const FLASH_COOP_THREADS: u32 = 128;
pub const FLASH_COOP_QUERIES: u32 = 32;

pub fn emit_flash_coop_op(s: &crate::ops::WgslShape) -> String {
    let mut out = String::from("enable wgpu_cooperative_matrix;\n");
    out.push_str(&ops_header(s));
    out.push_str(&flash_coop(s.head_dim as u32 / 8));
    out
}

fn flash_coop(nd: u32) -> String {
    let q_frags: String = (0..nd).map(|i| format!("    let qa{i} = coopLoadT<coop_mat8x8<f32, A>>(&qkv[qb0 * QKV_ROW + head * HEAD_DIM + {}u], QKV_ROW);\n", i * 8)).collect();
    let o_init: String = (0..nd).map(|i| format!("    var o{i} = coop_mat8x8<f32, C>();\n")).collect();
    let s_init: String = (0..4).map(|k| format!("        var s{k} = coop_mat8x8<f32, C>();\n")).collect();
    let s_mma: String = (0..4)
        .flat_map(|k| (0..nd).map(move |i| format!("        s{k} = coopMultiplyAdd(qa{i}, coopLoad<coop_mat8x8<f32, B>>(&k_cache[head_base + (key0 + {}u) * HEAD_DIM + {}u], HEAD_DIM), s{k});\n", k * 8, i * 8)))
        .collect();
    let s_store: String = (0..4).map(|k| format!("        coopStoreT(s{k}, &sbuf[g * 256u + {}u], 32u);\n", k * 8)).collect();
    let o_mma: String = (0..4)
        .map(|ks| {
            let mut b = format!("        let pa{ks} = coopLoadT<coop_mat8x8<f32, A>>(&sbuf[g * 256u + {}u], 32u);\n", ks * 8);
            for i in 0..nd {
                b.push_str(&format!("        o{i} = coopMultiplyAdd(pa{ks}, coopLoadT<coop_mat8x8<f32, B>>(&v_cache[head_base + (key0 + {}u) * HEAD_DIM + {}u], HEAD_DIM), o{i});\n", ks * 8, i * 8));
            }
            b
        })
        .collect();
    let o_store: String = (0..nd).map(|i| format!("{{pad}}coopStoreT(o{i}, &opart[g * 8u * HEAD_DIM + {}u], HEAD_DIM);\n", i * 8)).collect();
    let o_load: String = (0..nd).map(|i| format!("{{pad}}o{i} = coopLoadT<coop_mat8x8<f32, C>>(&opart[g * 8u * HEAD_DIM + {}u], HEAD_DIM);\n", i * 8)).collect();
    let store_inner = o_store.replace("{pad}", "            ");
    let load_inner = o_load.replace("{pad}", "            ");
    let store_final = o_store.replace("{pad}", "    ");
    format!(
        "{bind}
const HD4: u32 = HEAD_DIM / 4u;
const EPL: u32 = (HD4 + 31u) / 32u;
const LAZY: f32 = 6.0;
var<workgroup> sbuf: array<f32, 1024>;
var<workgroup> opart: array<f32, {opart}>;
var<workgroup> flags: array<u32, 4>;

@compute @workgroup_size({FLASH_COOP_THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let head = wid.x;
    let g = li / 32u;
    let lane = li % 32u;
    let kvh = head / GROUP;
    let head_base = kvh * MAX_CTX * HEAD_DIM;
    let pos0 = p.a.x;
    let rows = p.a.y;
    let q0 = wid.y * {FLASH_COOP_QUERIES}u;
    let wg_len = pos0 + min(q0 + {FLASH_COOP_QUERIES}u - 1u, rows - 1u) + 1u;
    let qrow0 = q0 + g * 8u;
    let qb0 = min(qrow0, rows - 1u);
    let scale = inverseSqrt(f32(HEAD_DIM));
{q_frags}{o_init}    var m: array<f32, 8>;
    var l: array<f32, 8>;
    var lens: array<u32, 8>;
    for (var r = 0u; r < 8u; r++) {{
        m[r] = NEG_INF;
        l[r] = 0.0;
        lens[r] = pos0 + min(qrow0 + r, rows - 1u) + 1u;
    }}
    for (var key0 = 0u; key0 < wg_len; key0 += 32u) {{
{s_init}{s_mma}{s_store}        workgroupBarrier();
        var smax: array<f32, 8>;
        var stale = 0.0;
        for (var r = 0u; r < 8u; r++) {{
            var s = NEG_INF;
            if (key0 + lane < lens[r]) {{
                s = sbuf[g * 256u + r * 32u + lane] * scale;
            }}
            smax[r] = subgroupMax(s);
            if (smax[r] > m[r] + LAZY) {{
                stale = 1.0;
            }}
        }}
        if (lane == 0u) {{
            flags[g] = select(0u, 1u, stale > 0.0);
        }}
        workgroupBarrier();
        if ((flags[0] | flags[1] | flags[2] | flags[3]) != 0u) {{
            if (stale > 0.0) {{
                for (var r = 0u; r < 8u; r++) {{
                    let m_new = max(m[r], smax[r]);
                    let c = exp(m[r] - m_new);
                    l[r] *= c;
                    m[r] = m_new;
                    smax[r] = c;
                }}
            }} else {{
                for (var r = 0u; r < 8u; r++) {{
                    smax[r] = 1.0;
                }}
            }}
{store_inner}            workgroupBarrier();
            for (var r = 0u; r < 8u; r++) {{
                for (var e = 0u; e < EPL; e++) {{
                    let d = lane + 32u * e;
                    if (d < HD4) {{
                        let at = g * 8u * HEAD_DIM + r * HEAD_DIM + d * 4u;
                        opart[at] *= smax[r];
                        opart[at + 1u] *= smax[r];
                        opart[at + 2u] *= smax[r];
                        opart[at + 3u] *= smax[r];
                    }}
                }}
            }}
            workgroupBarrier();
{load_inner}        }}
        for (var r = 0u; r < 8u; r++) {{
            var pr = 0.0;
            if (key0 + lane < lens[r]) {{
                pr = exp(sbuf[g * 256u + r * 32u + lane] * scale - m[r]);
            }}
            l[r] += subgroupAdd(pr);
            sbuf[g * 256u + r * 32u + lane] = pr;
        }}
        workgroupBarrier();
{o_mma}        workgroupBarrier();
    }}
{store_final}    workgroupBarrier();
    for (var r = 0u; r < 8u; r++) {{
        let t = qrow0 + r;
        if (t < rows) {{
            let ob = t * HEADS * HEAD_DIM + head * HEAD_DIM;
            for (var e = 0u; e < EPL; e++) {{
                let d = lane + 32u * e;
                if (d < HD4) {{
                    let at = g * 8u * HEAD_DIM + r * HEAD_DIM + d * 4u;
                    let o = vec4<f32>(opart[at], opart[at + 1u], opart[at + 2u], opart[at + 3u]) / l[r];
                    out[ob + d * 4u] = o.x;
                    out[ob + d * 4u + 1u] = o.y;
                    out[ob + d * 4u + 2u] = o.z;
                    out[ob + d * 4u + 3u] = o.w;
                }}
            }}
        }}
    }}
}}
",
        bind = storage(&[("qkv", false, "f32"), ("k_cache", false, "f32"), ("v_cache", false, "f32"), ("out", true, "f32")]),
        opart = 4 * 8 * nd * 8,
    )
}

pub const DECODE_GROUPS: u32 = 8;

pub fn emit_flash_decode_op(s: &crate::ops::WgslShape) -> String {
    let mut out = ops_header(s);
    out.push_str(&flash_decode());
    out
}

fn flash_decode() -> String {
    format!(
        "{bind}
const HD4: u32 = HEAD_DIM / 4u;
const HPL: u32 = HD4 / 4u;
const NSG: u32 = {DECODE_GROUPS}u;
var<workgroup> sm_m: array<f32, NSG>;
var<workgroup> sm_l: array<f32, NSG>;
var<workgroup> sm_acc: array<f32, {acc}>;

@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let head = wid.x;
    let g = li / 32u;
    let lane = li % 32u;
    let slot = lane / 4u;
    let part = lane % 4u;
    let kvh = head / GROUP;
    let len = p.a.x + 1u;
    let head_base4 = kvh * MAX_CTX * HD4;
    let scale = inverseSqrt(f32(HEAD_DIM));
    var q: array<vec4<f32>, HPL>;
    for (var i = 0u; i < HPL; i++) {{
        let qb = head * HEAD_DIM + (part * HPL + i) * 4u;
        q[i] = vec4<f32>(qkv[qb], qkv[qb + 1u], qkv[qb + 2u], qkv[qb + 3u]) * scale;
    }}
    var acc: array<vec4<f32>, HPL>;
    var m = NEG_INF;
    var l = 0.0;
    for (var key0 = g * 8u; key0 < len; key0 += NSG * 8u) {{
        let key = key0 + slot;
        let live = key < len;
        let kb = head_base4 + select(0u, key, live) * HD4 + part * HPL;
        var dots = vec4<f32>(0.0);
        for (var i = 0u; i < HPL; i++) {{
            dots += q[i] * k_cache[kb + i];
        }}
        var s = dots.x + dots.y + dots.z + dots.w;
        s += subgroupShuffleXor(s, 1u);
        s += subgroupShuffleXor(s, 2u);
        if (!live) {{
            s = NEG_INF;
        }}
        let m_new = max(m, subgroupMax(s));
        let corr = exp(m - m_new);
        let pr = select(0.0, exp(s - m_new), live);
        l = l * corr + pr;
        for (var i = 0u; i < HPL; i++) {{
            acc[i] = acc[i] * corr + pr * v_cache[kb + i];
        }}
        m = m_new;
    }}
    for (var i = 0u; i < HPL; i++) {{
        var a = acc[i];
        a += subgroupShuffleXor(a, 4u);
        a += subgroupShuffleXor(a, 8u);
        a += subgroupShuffleXor(a, 16u);
        acc[i] = a;
    }}
    let l_total = subgroupAdd(l) * 0.25;
    if (slot == 0u) {{
        for (var i = 0u; i < HPL; i++) {{
            let o = g * HEAD_DIM + (part * HPL + i) * 4u;
            sm_acc[o] = acc[i].x;
            sm_acc[o + 1u] = acc[i].y;
            sm_acc[o + 2u] = acc[i].z;
            sm_acc[o + 3u] = acc[i].w;
        }}
    }}
    if (lane == 0u) {{
        sm_m[g] = m;
        sm_l[g] = l_total;
    }}
    workgroupBarrier();
    if (li < HEAD_DIM) {{
        var big = NEG_INF;
        for (var j = 0u; j < NSG; j++) {{
            big = max(big, sm_m[j]);
        }}
        var num = 0.0;
        var den = 0.0;
        for (var j = 0u; j < NSG; j++) {{
            let w = exp(sm_m[j] - big);
            num += w * sm_acc[j * HEAD_DIM + li];
            den += w * sm_l[j];
        }}
        out[head * HEAD_DIM + li] = num / den;
    }}
}}
",
        bind = storage(&[("qkv", false, "f32"), ("k_cache", false, "vec4<f32>"), ("v_cache", false, "vec4<f32>"), ("out", true, "f32")]),
        threads = 32 * DECODE_GROUPS,
        acc = DECODE_GROUPS * 256,
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
