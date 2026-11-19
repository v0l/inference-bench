use super::prim;

pub fn body() -> String {
    [
        prim::mma_fp4("mma_fp4"),
        prim::e2m1x2("f32x2_to_e2m1x2"),
        r##"__device__ __forceinline__ unsigned f32_to_e4m3_bits(float v) {
    return f32x2_to_e4m3x2(v, 0.0f) & 0xffu;
}

"##.to_string(),
        prim::cp8_zfill("cp_async8_zfill"),
        r##"constexpr int F4A_BK = 128;
constexpr int F4A_DATA = F4A_BK / 2;
constexpr int F4A_SCALES = F4A_BK / FP4_GROUP;
constexpr int F4A_ROW = F4A_DATA + F4A_SCALES;

"##.to_string(),
        prim::swizzle("f4a_chunk", "F4A_DATA"),
        r##"
template <int N, int K, int BM, int BN, int WM, int WN, int STAGES, int EPI>
__device__ __forceinline__ void gemm_f4a(const void* w, const unsigned char* wscales, const unsigned char* xq,
                                         const unsigned char* xs, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr int BK = F4A_BK;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 64;
    constexpr int STAGE_BYTES = (BM + BN) * F4A_ROW;
    constexpr int SCALE_OFF = (BM + BN) * F4A_DATA;
    constexpr int KT = K / BK;
    constexpr int KG = K / FP4_GROUP;
    constexpr int CHUNKS = 5;
    constexpr int A_ITERS = (BM * CHUNKS + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of 128");
    static_assert(KG % 8 == 0, "scale rows are copied 8 bytes at a time");
    static_assert(NJ % 2 == 0, "B fragments come two n tiles per ldmatrix");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
"##.to_string(),
        prim::thread_ids(false),
        r##"    int2 tile = grouped_tile<N, BN>();
    int m0 = tile.x * BM;
    int n0 = tile.y * BN;
    float acc[MI][NJ][4];
"##.to_string(),
        prim::zero_frag("acc", ("i", "MI"), ("j", "NJ"), 4),
        r##"    int last_row = max(tokens - 1, 0);
    const unsigned char* wsc = wscales + FP4_HEADER;
    auto load_stage = [&](int buf, int kt) {
        unsigned char* as = gemm_smem + buf * STAGE_BYTES;
        unsigned char* ws = as + BM * F4A_DATA;
        unsigned char* ass = as + SCALE_OFF;
        unsigned char* wss = ass + BM * F4A_SCALES;
"##.to_string(),
        prim::cp_async_stage(
            "A_ITERS",
            "BM * CHUNKS",
            "CHUNKS",
            "c",
            None,
            "int m = m0 + r;\nsize_t row = (size_t)min(m, last_row);\n",
            &prim::split_copy(
                "c < 4",
                &prim::cp_async_call("cp_async16_zfill", &["as + f4a_chunk(r, c)", "xq + row * (K / 2) + kt * (BK / 2) + c * 16", "m < tokens"]),
                &prim::cp_async_call("cp_async8_zfill", &["ass + r * F4A_SCALES", "xs + row * KG + kt * (BK / FP4_GROUP)", "m < tokens"]),
            ),
        ),
        r##""##.to_string(),
        prim::cp_async_stage(
            "W_ITERS",
            "BN * CHUNKS",
            "CHUNKS",
            "c",
            None,
            "size_t n = (size_t)min(n0 + r, N - 1);\n",
            &prim::split_copy(
                "c < 4",
                &prim::cp_async_call("cp_async16", &["ws + f4a_chunk(r, c)", "(const unsigned char*)w + n * (K / 2) + kt * (BK / 2) + c * 16"]),
                &prim::cp_async_call("cp_async8_zfill", &["wss + r * F4A_SCALES", "wsc + n * KG + kt * (BK / FP4_GROUP)", "true"]),
            ),
        ),
        r##"    };
"##.to_string(),
        prim::pipeline_prologue("KT", "load_stage(st, st)"),
        prim::mainloop_head(),
        r##"        const unsigned char* as = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const unsigned char* ws = as + BM * F4A_DATA;
        const unsigned char* ass = as + SCALE_OFF;
        const unsigned char* wss = ass + BM * F4A_SCALES;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned a[MI][4], sfa[MI];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 15);
                ldsm_x4(a[i], as + f4a_chunk(r, ks * 2 + (lane >> 4)));
                sfa[i] = *(const unsigned*)(ass + (wm * TM + i * 16 + g + 8 * (lane & 1)) * F4A_SCALES + ks * 4);
            }
#pragma unroll
            for (int j = 0; j < NJ; j += 2) {
                unsigned b[4];
                int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, ws + f4a_chunk(r, ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
                for (int jj = 0; jj < 2; jj++) {
                    unsigned sfb = *(const unsigned*)(wss + (wn * TN + (j + jj) * 8 + g) * F4A_SCALES + ks * 4);
"##.to_string(),
        prim::mma_fp4_tile(),
        r##"                }
            }
        }
    }
    cp_async_wait<0>();
    float alpha = fp4_inv_global(wscales) / __ldg((const float*)wscales + 1);
"##.to_string(),
        prim::unroll_nest(4, &[("i", "MI"), ("j", "NJ"), ("e", "4")], false, "acc[i][j][e] *= alpha;"),
        r##"    gemm_epilogue_at<N, BM, BN, WM, WN, EPI, false>(acc, nullptr, out, out_hi, out_lo, tokens, m0, n0);
}

"##.to_string(),
    ]
    .concat()
}
