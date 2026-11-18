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
