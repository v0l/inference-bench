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
        Body::AddRmsNorm { .. } | Body::Pointwise { .. } => return Err(format!("{}: no WGSL emitter for this body yet", k.name)),
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
