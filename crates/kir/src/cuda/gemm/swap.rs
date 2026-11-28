use super::prim;

pub fn common() -> String {
    [
        prim::batch_helpers("swp", prim::L2_PREFETCH_128),
        prim::batch_store("swp"),
        r##"__device__ __forceinline__ uint2 swp_load8(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}

"##
        .to_string(),
        r##"__device__ __forceinline__ unsigned swp_load4(const void* p) {
    unsigned r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

"##
        .to_string(),
    ]
    .concat()
}

fn head(scale_rows: &str) -> String {
    format!(
        r##"    constexpr int CHUNKS = K / SWP_KC;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int split = blockIdx.y;
    int c_begin = split * CHUNKS / splits;
    int nc = (split + 1) * CHUNKS / splits - c_begin;
    int row0 = (blockIdx.x * WARPS + warp) * (MJ * 16);
    int last = max(tokens - 1, 0);
    const unsigned char* wrow[MJ][2];
{scale_rows}"##
    )
}

fn zero_acc() -> String {
    prim::zero_frag("acc", ("i", "MJ"), ("n", "NT"), 4)
}

fn depth_loop(body: &str) -> String {
    let lines: String = body.lines().map(|l| if l.is_empty() { "\n".to_string() } else { format!("                {l}\n") }).collect();
    format!(
        r##"#pragma unroll
    for (int d = 0; d < D; d++) load(d, d);
    for (int c0 = 0; c0 < nc; c0 += D) {{
#pragma unroll
        for (int d = 0; d < D; d++) {{
            int c = c0 + d;
            if (c < nc) {{
{lines}                load(d, c + D);
            }}
        }}
    }}
"##
    )
}

pub fn fp4() -> String {
    [
        prim::mma_fp4("swp_mma_fp4"),
        r##"template <int N, int K, int EPI, int NT, int WARPS, int MJ, int D>
__device__ __forceinline__ void swap_f4(const void* w, const unsigned char* scales, const unsigned char* xq, const unsigned char* xs,
                                        float* out, bf16_t* out_hi, bf16_t* out_lo, float* part, int tokens, int splits) {
    constexpr int KG = K / FP4_GROUP;
    static_assert(SWP_KC == 128, "four lanes of 16 bytes span one chunk of a row");
    static_assert(K % SWP_KC == 0, "K must be a multiple of the chunk");
"##
        .to_string(),
        head("    const unsigned char* srow[MJ];\n    int p = t & 1;\n"),
        prim::unroll_nest(
            4,
            &[("i", "MJ")],
            true,
            r##"#pragma unroll
for (int h = 0; h < 2; h++) wrow[i][h] = (const unsigned char*)w + (size_t)min(row0 + i * 16 + g + 8 * h, N - 1) * (K / 2) + t * 16 + (size_t)c_begin * (SWP_KC / 2);
srow[i] = scales + FP4_HEADER + (size_t)min(row0 + i * 16 + g + 8 * p, N - 1) * KG + (size_t)c_begin * (SWP_KC / FP4_GROUP);
"##,
        ),
        r##"    const unsigned char* xrow[NT];
    const unsigned char* xsrow[NT];
#pragma unroll
    for (int n = 0; n < NT; n++) {
        size_t m = min(n * 8 + g, last);
        xrow[n] = xq + m * (K / 2) + (size_t)c_begin * (SWP_KC / 2) + (t >> 1) * 32 + 4 * p;
        xsrow[n] = xs + m * KG + (size_t)c_begin * (SWP_KC / FP4_GROUP);
    }
    float acc[MJ][NT][4];
"##
        .to_string(),
        zero_acc(),
        r##"    uint4 wq[D][MJ][2];
    uint2 ws[D][MJ];
    auto load = [&](int d, int c) {
        if (c >= nc) return;
#pragma unroll
        for (int i = 0; i < MJ; i++) {
#pragma unroll
            for (int h = 0; h < 2; h++) wq[d][i][h] = swp_load16(wrow[i][h] + (size_t)c * (SWP_KC / 2));
            ws[d][i] = swp_load8(srow[i] + (size_t)c * (SWP_KC / FP4_GROUP));
        }
    };
"##
        .to_string(),
        depth_loop(
            r##"unsigned b[NT][2][2], sfb[NT][2];
#pragma unroll
for (int n = 0; n < NT; n++) {
    const unsigned char* xr = xrow[n] + (size_t)c * (SWP_KC / 2);
    b[n][0][0] = __ldg((const unsigned*)xr);
    b[n][1][0] = __ldg((const unsigned*)(xr + 8));
    b[n][0][1] = __ldg((const unsigned*)(xr + 16));
    b[n][1][1] = __ldg((const unsigned*)(xr + 24));
    uint2 s8 = __ldg((const uint2*)(xsrow[n] + (size_t)c * (SWP_KC / FP4_GROUP)));
    sfb[n][0] = __byte_perm(s8.x, s8.y, 0x6240);
    sfb[n][1] = __byte_perm(s8.x, s8.y, 0x7351);
}
#pragma unroll
for (int i = 0; i < MJ; i++) {
    unsigned a[2][4];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint4 v = wq[d][i][h];
        unsigned r1 = __shfl_xor_sync(0xffffffff, p ? v.x : v.y, 1);
        unsigned r2 = __shfl_xor_sync(0xffffffff, p ? v.z : v.w, 1);
        a[0][h] = p ? r1 : v.x;
        a[0][2 + h] = p ? v.y : r1;
        a[1][h] = p ? r2 : v.z;
        a[1][2 + h] = p ? v.w : r2;
    }
    unsigned sfa0 = __byte_perm(ws[d][i].x, ws[d][i].y, 0x6240);
    unsigned sfa1 = __byte_perm(ws[d][i].x, ws[d][i].y, 0x7351);
#pragma unroll
    for (int n = 0; n < NT; n++) {
        swp_mma_fp4(acc[i][n], a[0], b[n][0][0], b[n][0][1], sfa0, sfb[n][0]);
        swp_mma_fp4(acc[i][n], a[1], b[n][1][0], b[n][1][1], sfa1, sfb[n][1]);
    }
}
"##,
        ),
        r##"    float alpha = fp4_inv_global(scales) / __ldg((const float*)scales + 1);
"##
        .to_string(),
        prim::unroll_nest(4, &[("i", "MJ"), ("n", "NT"), ("e", "4")], false, "acc[i][n][e] *= alpha;"),
        prim::batch_epilogue("swp", "acc[i][n][e]", "v"),
    ]
    .concat()
}

pub fn fp8() -> String {
    [
        prim::mma_e4m3("swp_mma_e4m3", false),
        r##"__device__ __forceinline__ unsigned swp_word(const uint4& a, const uint4& b, int i) {
    const uint4& q = i < 4 ? a : b;
    int j = i & 3;
    return j == 0 ? q.x : j == 1 ? q.y : j == 2 ? q.z : q.w;
}

template <int N, int K, int EPI, bool BIAS, bool PER_TOKEN, int NT, int WARPS, int MJ, int D>
__device__ __forceinline__ void swap_f8(const void* w, const float* scales, const float* bias, const unsigned char* xq, const float* xs,
                                        float* out, bf16_t* out_hi, bf16_t* out_lo, float* part, int tokens, int splits) {
    static_assert(SWP_KC == FP8_BLOCK, "one scale per row and chunk");
    static_assert(K % SWP_KC == 0, "K must be a multiple of the chunk");
"##
        .to_string(),
        head("    const float* srow[MJ][2];\n"),
        prim::unroll_nest(
            4,
            &[("i", "MJ"), ("h", "2")],
            true,
            r##"int r = min(row0 + i * 16 + g + 8 * h, N - 1);
wrow[i][h] = (const unsigned char*)w + (size_t)r * K + t * 32 + (size_t)c_begin * SWP_KC;
srow[i][h] = fp8_row_scales(scales, r);
"##,
        ),
        r##"    const unsigned char* xrow[NT];
#pragma unroll
    for (int n = 0; n < NT; n++) xrow[n] = xq + (size_t)min(n * 8 + g, last) * K + t * 32 + (size_t)c_begin * SWP_KC;
    float acc[MJ][NT][4];
"##
        .to_string(),
        zero_acc(),
        r##"    uint4 wq[D][MJ][2][2];
    auto load = [&](int d, int c) {
        if (c >= nc) return;
#pragma unroll
        for (int i = 0; i < MJ; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                wq[d][i][h][0] = swp_load16(wrow[i][h] + (size_t)c * SWP_KC);
                wq[d][i][h][1] = swp_load16(wrow[i][h] + (size_t)c * SWP_KC + 16);
            }
    };
"##
        .to_string(),
        depth_loop(
            r##"uint4 xv[NT][2];
#pragma unroll
for (int n = 0; n < NT; n++) {
    xv[n][0] = __ldg((const uint4*)(xrow[n] + (size_t)c * SWP_KC));
    xv[n][1] = __ldg((const uint4*)(xrow[n] + (size_t)c * SWP_KC + 16));
}
float cc[MJ][NT][4];
#pragma unroll
for (int i = 0; i < MJ; i++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) cc[i][n][e] = 0.0f;
#pragma unroll
for (int ks = 0; ks < SWP_KC / 32; ks++) {
#pragma unroll
    for (int i = 0; i < MJ; i++) {
        unsigned a[4] = {swp_word(wq[d][i][0][0], wq[d][i][0][1], 2 * ks), swp_word(wq[d][i][1][0], wq[d][i][1][1], 2 * ks),
                         swp_word(wq[d][i][0][0], wq[d][i][0][1], 2 * ks + 1), swp_word(wq[d][i][1][0], wq[d][i][1][1], 2 * ks + 1)};
#pragma unroll
        for (int n = 0; n < NT; n++) swp_mma_e4m3(PER_TOKEN ? acc[i][n] : cc[i][n], a, swp_word(xv[n][0], xv[n][1], 2 * ks), swp_word(xv[n][0], xv[n][1], 2 * ks + 1));
    }
}
if (!PER_TOKEN) {
    int kt = c_begin + c;
    float sw[MJ][2];
#pragma unroll
    for (int i = 0; i < MJ; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) sw[i][h] = __uint_as_float(swp_load4(srow[i][h] + kt));
#pragma unroll
    for (int n = 0; n < NT; n++) {
        float s0 = __ldg(xs + (size_t)min(n * 8 + 2 * t, last) * CHUNKS + kt);
        float s1 = __ldg(xs + (size_t)min(n * 8 + 2 * t + 1, last) * CHUNKS + kt);
#pragma unroll
        for (int i = 0; i < MJ; i++) {
            acc[i][n][0] += cc[i][n][0] * s0 * sw[i][0];
            acc[i][n][1] += cc[i][n][1] * s1 * sw[i][0];
            acc[i][n][2] += cc[i][n][2] * s0 * sw[i][1];
            acc[i][n][3] += cc[i][n][3] * s1 * sw[i][1];
        }
    }
}
"##,
        ),
        r##"    if (PER_TOKEN) {
#pragma unroll
        for (int i = 0; i < MJ; i++)
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][n][e] *= __ldg(xs + min(n * 8 + 2 * t + (e & 1), last)) * __ldg(srow[i][e >> 1]);
    }
"##
        .to_string(),
        prim::batch_epilogue("swp", "acc[i][n][e]", "BIAS ? v + bias[r] : v"),
    ]
    .concat()
}
