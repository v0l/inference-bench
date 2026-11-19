pub fn zero_frag(arr: &str, outer: (&str, &str), inner: (&str, &str), indent: usize) -> String {
    let s = " ".repeat(indent);
    format!(
        "#pragma unroll
{s}for (int {i} = 0; {i} < {oc}; {i}++)
#pragma unroll
{s}    for (int {j} = 0; {j} < {ic}; {j}++)
#pragma unroll
{s}        for (int e = 0; e < 4; e++) {arr}[{i}][{j}][e] = 0.0f;
",
        i = outer.0,
        oc = outer.1,
        j = inner.0,
        ic = inner.1,
    )
}

pub fn pipeline_prologue(count: &str, stage: &str) -> String {
    format!(
        "#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {{
        if (st < {count}) {stage};
        cp_async_commit();
    }}
"
    )
}

pub fn batch_helpers(p: &str) -> String {
    format!(
        r#"
__device__ __forceinline__ void {p}_cp_async8(void* smem, const void* gmem, bool valid) {{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(s), "l"(gmem), "r"(n));
}}

__device__ __forceinline__ uint4 {p}_load16(const void* p) {{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {{%0,%1,%2,%3}}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}}
"#
    )
}

pub fn batch_store(p: &str) -> String {
    format!(
        "template <int N, int EPI>
__device__ __forceinline__ void {p}_store(int m, int r, float v, float u, float* out, bf16_t* out_hi, bf16_t* out_lo) {{
    if (EPI == EPI_ACT) {{
        unsigned hi, lo;
        split_hi_lo(activation(v) * u, hi, lo);
        size_t at = (size_t)m * (N / 2) + r / 2;
        out_hi[at] = (bf16_t)hi;
        out_lo[at] = (bf16_t)lo;
    }} else {{
        float* o = out + (size_t)m * N + r;
        *o = EPI == EPI_RESIDUAL ? *o + v : v;
    }}
}}
"
    )
}

pub fn pack_bf16x8(indent: usize) -> String {
    format!(
        "make_uint4(bf16_bits_rn(v[0]) | (bf16_bits_rn(v[1]) << 16), bf16_bits_rn(v[2]) | (bf16_bits_rn(v[3]) << 16),
{s}bf16_bits_rn(v[4]) | (bf16_bits_rn(v[5]) << 16), bf16_bits_rn(v[6]) | (bf16_bits_rn(v[7]) << 16))",
        s = " ".repeat(indent)
    )
}

pub fn device(ret: &str, name: &str, args: &str, body: &str) -> String {
    format!("__device__ __forceinline__ {ret} {name}({args}) {{\n{body}}}\n\n")
}

pub fn mma_fp4(name: &str) -> String {
    device(
        "void",
        name,
        "float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa, unsigned sfb",
        r#"    unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
"#,
    )
}

pub fn e2m1x2(name: &str) -> String {
    device(
        "unsigned",
        name,
        "float lo, float hi",
        "    unsigned short r;\n    asm(\"{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }\" : \"=h\"(r) : \"f\"(hi), \"f\"(lo));\n    return r & 0xffu;\n",
    )
}

pub fn e2m1_to_f32(name: &str) -> String {
    device("float", name, "unsigned n", "    return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22)) * __uint_as_float(0x7e800000u);\n")
}

pub fn cp8_zfill(name: &str) -> String {
    device(
        "void",
        name,
        "void* smem, const void* gmem, bool valid",
        "    unsigned s = (unsigned)__cvta_generic_to_shared(smem);\n    int n = valid ? 8 : 0;\n    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8, %2;\" :: \"r\"(s), \"l\"(gmem), \"r\"(n));\n",
    )
}

pub fn swizzle(name: &str, row_bytes: &str) -> String {
    device("int", name, "int r, int c", &format!("    return r * {row_bytes} + ((c ^ (r >> 1)) & 3) * 16;\n"))
}

pub fn warp_ids() -> &'static str {
    "    int warp = threadIdx.x >> 5;\n    int lane = threadIdx.x & 31;\n"
}

pub fn mainloop_head() -> String {
    "    for (int kt = 0; kt < KT; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT) load_stage(next % STAGES, next);
        cp_async_commit();
"
    .to_string()
}

pub fn batch_loop_head() -> String {
    "    for (int it = 0; it < nc; it++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < nc) stage((it + STAGES - 1) % STAGES, c_begin + it + STAGES - 1);
        cp_async_commit();
"
    .to_string()
}

pub fn batch_epilogue(p: &str, value: &str, stored: &str) -> String {
    format!(
        "    cp_async_wait<0>();
{pre}#pragma unroll
    for (int i = 0; i < MJ; i++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {{
                int r = row0 + i * 16 + g + 8 * (e >> 1);
                int m = n * 8 + 2 * t + (e & 1);
                float v = {value};
                float u = __shfl_down_sync(0xffffffff, v, 4);
                if (r >= N || m >= tokens) continue;
                if (splits > 1) part[((size_t)split * tokens + m) * N + r] = v;
                else if (EPI != EPI_ACT || (g & 1) == 0) {p}_store<N, EPI>(m, r, {stored}, u, out, out_hi, out_lo);
            }}
}}
",
        pre = if value.contains("inv") { "    float inv = fp4_inv_global(scales);\n" } else { "" },
    )
}

pub fn batch_reduce(p: &str, bias: bool) -> String {
    let (sig, add) = if bias {
        (
            "template <int N, int EPI, bool BIAS>\n__device__ __forceinline__ void P_reduce(const float* part, const float* bias, float* out, bf16_t* out_hi, bf16_t* out_lo,\n                                           int tokens, int splits) {",
            "    if (BIAS) v += bias[r];\n",
        )
    } else {
        (
            "template <int N, int EPI>\n__device__ __forceinline__ void P_reduce(const float* part, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens, int splits) {",
            "",
        )
    };
    format!(
        "{sig}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int OUT = EPI == EPI_ACT ? N / 2 : N;
    if (i >= tokens * OUT) return;
    int m = i / OUT;
    int r = EPI == EPI_ACT ? 2 * (i % OUT) : i % OUT;
    float v = 0.0f, u = 0.0f;
    for (int sp = 0; sp < splits; sp++) {{
        const float* p = part + ((size_t)sp * tokens + m) * N + r;
        v += p[0];
        if (EPI == EPI_ACT) u += p[1];
    }}
{add}    P_store<N, EPI>(m, r, v, u, out, out_hi, out_lo);
}}

",
        sig = sig.replace("P_reduce", &format!("{p}_reduce")),
        add = add,
    )
    .replace("P_store", &format!("{p}_store"))
}

pub fn device_t(template: &str, ret: &str, name: &str, args: &str, body: &str) -> String {
    format!("template <{template}>\n__device__ __forceinline__ {ret} {name}({args}) {{\n{body}}}\n\n")
}

pub fn thread_ids(with_t: bool) -> String {
    format!(
        "    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
{t}    int wm = warp / WN;
    int wn = warp % WN;
",
        t = if with_t { "    int t = lane & 3;\n" } else { "" }
    )
}

pub fn load_iters(iters: &str, total: &str, body: &str) -> String {
    let lines: String = body.lines().map(|l| if l.is_empty() { "\n".to_string() } else { format!("            {l}\n") }).collect();
    format!(
        "#pragma unroll
        for (int i = 0; i < {iters}; i++) {{
            int idx = tid + i * THREADS;
            if ({iters} * THREADS != {total} && idx >= {total}) break;
{lines}        }}
"
    )
}

pub fn unroll_nest(indent: usize, levels: &[(&str, &str)], block: bool, body: &str) -> String {
    let mut out = String::new();
    for (k, (v, b)) in levels.iter().enumerate() {
        out.push_str(&format!("#pragma unroll\n{}for (int {v} = 0; {v} < {b}; {v}++)", " ".repeat(indent + 4 * k)));
        if k + 1 < levels.len() {
            out.push('\n');
        }
    }
    if !block {
        out.push_str(&format!(" {body}\n"));
        return out;
    }
    out.push_str(" {\n");
    let pad = " ".repeat(indent + 4 * levels.len());
    for l in body.lines() {
        if l.is_empty() {
            out.push('\n');
        } else {
            out.push_str(&format!("{pad}{l}\n"));
        }
    }
    out.push_str(&format!("{}}}\n", " ".repeat(indent + 4 * (levels.len() - 1))));
    out
}
