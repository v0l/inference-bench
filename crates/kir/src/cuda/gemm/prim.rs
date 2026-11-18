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
