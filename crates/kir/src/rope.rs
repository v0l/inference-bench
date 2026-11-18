use crate::layout::Dialect;

struct Syntax {
    f: &'static str,
    fl: fn(f64) -> String,
    int_to_f: fn(&str) -> String,
    pow: &'static str,
    clamp: fn(&str) -> String,
}

fn cuda_lit(v: f64) -> String {
    format!("{v:?}f")
}

fn wgsl_lit(v: f64) -> String {
    format!("{v:?}")
}

const CUDA: Syntax = Syntax { f: "float", fl: cuda_lit, int_to_f: |e| format!("(float)({e})"), pow: "powf", clamp: |e| format!("fminf(fmaxf({e}, 0.0f), 1.0f)") };
const WGSL: Syntax = Syntax { f: "f32", fl: wgsl_lit, int_to_f: |e| format!("f32({e})"), pow: "pow", clamp: |e| format!("clamp({e}, 0.0, 1.0)") };

pub fn inv_freq_fn(d: Dialect, kinds: &[u32]) -> String {
    let s = match d {
        Dialect::Cuda => &CUDA,
        Dialect::Wgsl => &WGSL,
    };
    let one = (s.fl)(1.0);
    let ramp_i = match d {
        Dialect::Cuda => "(float)i".to_string(),
        Dialect::Wgsl => "f32(i)".to_string(),
    };
    let two_i = (s.int_to_f)(match d {
        Dialect::Cuda => "2 * i",
        Dialect::Wgsl => "2u * i",
    });
    let rot = (s.int_to_f)("ROT");
    let two_pi = (s.fl)(std::f64::consts::TAU);
    let mut branches = Vec::new();
    for &k in kinds {
        let body = match k {
            1 => "        f = f / ROPE_FACTOR;\n".to_string(),
            2 => format!(
                "        let_ wavelen = {two_pi} / f;
        let_ low = ROPE_ORIG_POS / ROPE_LOW_FREQ;
        let_ high = ROPE_ORIG_POS / ROPE_HIGH_FREQ;
        if (wavelen > low) {{
            f = f / ROPE_FACTOR;
        }} else if (wavelen >= high) {{
            let_ sm = (ROPE_ORIG_POS / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
            f = ({one} - sm) * f / ROPE_FACTOR + sm * f;
        }}
"
            ),
            3 => format!(
                "        let_ ramp = {};
        f = ramp * f / ROPE_FACTOR + ({one} - ramp) * f;
",
                (s.clamp)(&format!("({ramp_i} - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ)"))
            ),
            4 if d == Dialect::Cuda => "        f = f / ROPE_FREQ_FACTORS[i];\n".to_string(),
            _ => continue,
        };
        branches.push((k, body));
    }
    let (sig, local, lit_kind) = match d {
        Dialect::Cuda => (
            "__device__ __forceinline__ float rope_inv_freq(int i, int layer) {\n    if (LOCAL_ROPE[layer]) return 1.0f / powf(LOCAL_THETA, (float)(2 * i) / (float)ROT);\n".to_string(),
            "float",
            "",
        ),
        Dialect::Wgsl => ("fn inv_freq(i: u32) -> f32 {\n".to_string(), "var", "u"),
    };
    let mut out = sig;
    out.push_str(&format!("    {local} f = {one} / {}(THETA, {two_i} / {rot});\n", s.pow));
    for (n, (k, body)) in branches.iter().enumerate() {
        let kw = if n == 0 { "    if" } else { " else if" };
        out.push_str(&format!("{kw} (ROPE_KIND == {k}{lit_kind}) {{\n{body}    }}"));
    }
    if !branches.is_empty() {
        out.push('\n');
    }
    out.push_str("    return f;\n}\n");
    let decl = match d {
        Dialect::Cuda => format!("{} ", s.f),
        Dialect::Wgsl => "let ".to_string(),
    };
    out.replace("let_ ", &decl)
}
