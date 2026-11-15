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

fn rmsnorm(k: &Kernel, width: u32, eps: f32, sum: BlockSum) -> String {
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
            "enable subgroups;\n",
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
        {out}[dst + i] = {w}[i] * ({x}[src + i] * inv);
    }}
}}
",
        eps = lit(eps),
    ));
    s
}

pub fn emit_kernel(k: &Kernel) -> Result<String, String> {
    k.validate()?;
    if k.target.language != Language::Wgsl {
        return Err(format!("{}: the WGSL emitter cannot target {:?}", k.name, k.target.language));
    }
    Ok(match &k.body {
        Body::RmsNorm { width, eps, sum } => {
            if *sum == BlockSum::WarpShuffle {
                return Err(format!("{}: block sum {sum:?} has no WGSL snippet", k.name));
            }
            if *sum == BlockSum::Subgroup && k.threads > 128 * k.target.warp_size.max(1) {
                return Err(format!("{}: too many subgroups for the scratch array", k.name));
            }
            rmsnorm(k, *width, *eps, *sum)
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
