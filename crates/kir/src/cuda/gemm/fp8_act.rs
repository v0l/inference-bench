use super::prim;

pub fn body() -> String {
    [
        r##"template <int N, int K, int BM, int BN, int WM, int WN, int STAGES, int EPI, bool BIAS, bool PER_TOKEN = false>
__device__ __forceinline__ void gemm_fp8(const void* w, const float* wscale, const float* bias, const unsigned char* xq,
                                         const float* xscale, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr int BK = FP8_BLOCK;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 32;
    constexpr int ROW = BK;
    constexpr int STAGE_BYTES = (BM + BN) * ROW;
    constexpr int KT = K / BK;
    constexpr int CHUNKS = BK / 16;
    constexpr int A_ITERS = (BM * CHUNKS + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of the FP8 block");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    int2 tile = grouped_tile<N, BN>();
    int m0 = tile.x * BM;
    int n0 = tile.y * BN;
    float acc[MI][NJ][4];
"##.to_string(),
        prim::zero_frag("acc", ("i", "MI"), ("j", "NJ"), 4),
        r##"    float c[MI][NJ][4];
"##.to_string(),
        prim::zero_frag("c", ("i", "MI"), ("j", "NJ"), 4),
        r##"    int last_row = max(tokens - 1, 0);
    unsigned srow0[NJ], srow1[NJ];
#pragma unroll
    for (int j = 0; j < NJ; j++) {
        int col = n0 + wn * TN + j * 8 + 2 * t;
        srow0[j] = __float_as_uint(__ldg(wscale + min(col, N - 1)));
        srow1[j] = __float_as_uint(__ldg(wscale + min(col + 1, N - 1)));
    }
    auto load_stage = [&](int buf, int kt) {
        unsigned char* as = gemm_smem + buf * STAGE_BYTES;
        unsigned char* ws = as + BM * ROW;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != BM * CHUNKS && idx >= BM * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            int m = m0 + r;
            cp_async16_zfill(as + r * ROW + ((c ^ (r & 7)) << 4), xq + (size_t)min(m, last_row) * K + kt * BK + c * 16, m < tokens);
        }
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != BN * CHUNKS && idx >= BN * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            int n = min(n0 + r, N - 1);
            cp_async16(ws + r * ROW + ((c ^ (r & 7)) << 4), (const unsigned char*)w + (size_t)n * K + kt * BK + c * 16);
        }
    };
"##.to_string(),
        prim::pipeline_prologue("KT", "load_stage(st, st)"),
        prim::mainloop_head(),
        r##"        const unsigned char* as = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const unsigned char* ws = as + BM * ROW;
        float sw0[NJ], sw1[NJ], sa[MI][2];
        if (!PER_TOKEN) {
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                sw0[j] = wscale[srow0[j] + kt];
                sw1[j] = wscale[srow1[j] + kt];
            }
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int m = min(m0 + wm * TM + i * 16 + g + h * 8, last_row);
                    sa[i][h] = xscale[(size_t)m * KT + kt];
                }
"##.to_string(),
        prim::zero_frag("c", ("i", "MI"), ("j", "NJ"), 12),
        r##"        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned a[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int ch = ks * 2 + (lane >> 4);
                ldsm_x4(a[i], as + r * ROW + ((ch ^ (r & 7)) << 4));
            }
#pragma unroll
            for (int jj = 0; jj < NJ; jj += 2) {
                unsigned b[4];
                int r = wn * TN + jj * 8 + (lane & 7) + (lane >> 4) * 8;
                int ch = ks * 2 + ((lane >> 3) & 1);
                ldsm_x4(b, ws + r * ROW + ((ch ^ (r & 7)) << 4));
#pragma unroll
                for (int i = 0; i < MI; i++) {
                    mma_e4m3(c[i][jj], a[i], b[0], b[1]);
                    mma_e4m3(c[i][jj + 1], a[i], b[2], b[3]);
                }
            }
        }
        if (!PER_TOKEN) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    acc[i][j][0] += c[i][j][0] * sa[i][0] * sw0[j];
                    acc[i][j][1] += c[i][j][1] * sa[i][0] * sw1[j];
                    acc[i][j][2] += c[i][j][2] * sa[i][1] * sw0[j];
                    acc[i][j][3] += c[i][j][3] * sa[i][1] * sw1[j];
                }
        }
    }
    if (PER_TOKEN) {
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = min(m0 + wm * TM + i * 16 + g + h * 8, last_row);
                float sa = xscale[m];
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    acc[i][j][2 * h] = c[i][j][2 * h] * sa * wscale[srow0[j]];
                    acc[i][j][2 * h + 1] = c[i][j][2 * h + 1] * sa * wscale[srow1[j]];
                }
            }
    }
    cp_async_wait<0>();
    gemm_epilogue_at<N, BM, BN, WM, WN, EPI, BIAS>(acc, bias, out, out_hi, out_lo, tokens, m0, n0);
}
"##.to_string(),
    ]
    .concat()
}
