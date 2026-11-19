use super::prim;

pub fn body() -> String {
    [
        r##"constexpr int EPI_STORE = 0;
constexpr int EPI_RESIDUAL = 1;
constexpr int EPI_ACT = 2;

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS, bool MFAST = false>
__device__ __forceinline__ void gemm_epilogue(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                              bf16_t* out_hi, bf16_t* out_lo, int tokens);

template <int N, int K, int BM, int BN, int WM, int WN, int BK, int STAGES, int EPI, bool BIAS, typename D, bool SPLIT, bool MFAST = false>
__device__ __forceinline__ void gemm_tc(const void* w, const void* scales, const float* bias, const bf16_t* x_hi,
                                        const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr bool SCALED = D::ACC == ACC_SCALED;
    constexpr bool DIRECT = D::ACC == ACC_DIRECT;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 16;
    constexpr int CPR = BK / 8;
    constexpr int PLANES = SPLIT ? 2 : 1;
    constexpr int A_TILE = BM * BK;
    constexpr int A_BYTES = A_TILE * 2 * PLANES;
    constexpr int WROW = D::template wrow<BK>;
    constexpr int STAGE_BYTES = A_BYTES + BN * WROW;
    constexpr int KT = K / BK;
    constexpr int A_CHUNKS = BK / 8;
    constexpr int A_TOTAL = BM * A_CHUNKS * PLANES;
    constexpr int A_ITERS = (A_TOTAL + THREADS - 1) / THREADS;
    constexpr int W_CHUNKS = D::template w_chunks<BK>;
    constexpr int W_ITERS = (BN * W_CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of BK");
    static_assert(NJ % 2 == 0, "warp n tile");
    static_assert(STAGES >= 2, "pipeline depth");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    int m0 = (MFAST ? blockIdx.x : blockIdx.y) * BM;
    int n0 = (MFAST ? blockIdx.y : blockIdx.x) * BN;
    float acc[MI][NJ][4];
"##.to_string(),
        prim::zero_frag("acc", ("i", "MI"), ("j", "NJ"), 4),
        r##"    int last_row = max(tokens - 1, 0);
    unsigned srow0[SCALED ? NJ : 1], srow1[SCALED ? NJ : 1];
    if (SCALED) {
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = n0 + wn * TN + j * 8 + 2 * t;
            srow0[j] = D::template col<K>(scales, min(col, N - 1));
            srow1[j] = D::template col<K>(scales, min(col + 1, N - 1));
        }
    }
    auto load_stage = [&](int buf, int kt) {
        unsigned char* base = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != A_TOTAL && idx >= A_TOTAL) break;
            int p = idx / (BM * A_CHUNKS);
            int rem = idx % (BM * A_CHUNKS);
            int r = rem / A_CHUNKS;
            int c = rem % A_CHUNKS;
            int m = m0 + r;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last_row) * K + kt * BK + c * 8;
            cp_async16_zfill(base + p * A_TILE * 2 + swz<CPR>(r, c) * 16, src, m < tokens);
        }
        unsigned char* ws = base + A_BYTES;
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != BN * W_CHUNKS && idx >= BN * W_CHUNKS) break;
            int r = idx / W_CHUNKS;
            int ck = idx % W_CHUNKS;
            int n = min(n0 + r, N - 1);
            D::template load<K, BK, CPR, WROW>(ws, w, scales, r, n, kt, ck);
        }
    };
"##.to_string(),
        prim::pipeline_prologue("KT", "load_stage(st, st)"),
        prim::mainloop_head(),
        r##"        const unsigned char* base = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const bf16_t* a_hi = (const bf16_t*)base;
        const bf16_t* a_lo = a_hi + A_TILE;
        const unsigned char* ws = base + A_BYTES;
        float part[SPLIT ? MI : 1][SPLIT ? NJ : 1][4];
        if (SPLIT) {
"##.to_string(),
        prim::zero_frag("part", ("i", "MI"), ("j", "NJ"), 12),
        r##"        }
        float s0[NJ], s1[NJ];
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            s0[j] = 1.0f;
            s1[j] = 1.0f;
            if (SCALED) {
                s0[j] = D::template scale<K, BK>(scales, srow0[j], kt);
                s1[j] = D::template scale<K, BK>(scales, srow1[j], kt);
            }
        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned bw[NJ][2];
            if constexpr (D::LDSM) {
#pragma unroll
                for (int jp = 0; jp < NJ / 2; jp++) {
                    unsigned r[4];
                    int row = wn * TN + jp * 16 + (lane >> 4) * 8 + (lane & 7);
                    ldsm_x4(r, ws + swz<CPR>(row, ks * 2 + ((lane >> 3) & 1)) * 16);
                    bw[2 * jp][0] = r[0];
                    bw[2 * jp][1] = r[1];
                    bw[2 * jp + 1][0] = r[2];
                    bw[2 * jp + 1][1] = r[3];
                }
            } else {
#pragma unroll
                for (int j = 0; j < NJ; j++) D::frag(bw[j], ws + (wn * TN + j * 8 + g) * WROW, ks, t);
            }
            unsigned ah[MI][4], al[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int row = wm * TM + i * 16 + (lane & 15);
                int at = swz<CPR>(row, ks * 2 + (lane >> 4)) * 8;
                ldsm_x4(ah[i], &a_hi[at]);
                if (SPLIT) ldsm_x4(al[i], &a_lo[at]);
            }
            if (DIRECT) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        if (SPLIT) mma_bf16(acc[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(acc[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                    }
            } else if (SPLIT && SCALED) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16(c, al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(c, ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                        acc[i][j][0] += c[0] * s0[j];
                        acc[i][j][1] += c[1] * s1[j];
                        acc[i][j][2] += c[2] * s0[j];
                        acc[i][j][3] += c[3] * s1[j];
                    }
            } else if (SPLIT) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        mma_bf16(part[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(part[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                    }
            } else if (!SCALED) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) mma_bf16(acc[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
            } else {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16(c, ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                        acc[i][j][0] += c[0] * s0[j];
                        acc[i][j][1] += c[1] * s1[j];
                        acc[i][j][2] += c[2] * s0[j];
                        acc[i][j][3] += c[3] * s1[j];
                    }
            }
        }
        if (SPLIT && !SCALED && !DIRECT) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] += part[i][j][e];
        }
    }
    cp_async_wait<0>();
    D::finish(acc, scales);
    gemm_epilogue<N, BM, BN, WM, WN, EPI, BIAS, MFAST>(acc, bias, out, out_hi, out_lo, tokens);
}

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS>
__device__ __forceinline__ void gemm_epilogue_at(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                                 bf16_t* out_hi, bf16_t* out_lo, int tokens, int m0, int n0);

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS, bool MFAST>
__device__ __forceinline__ void gemm_epilogue(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                              bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    int m0 = (MFAST ? blockIdx.x : blockIdx.y) * BM;
    int n0 = (MFAST ? blockIdx.y : blockIdx.x) * BN;
    gemm_epilogue_at<N, BM, BN, WM, WN, EPI, BIAS>(acc, bias, out, out_hi, out_lo, tokens, m0, n0);
}

constexpr int RASTER_GROUP = 32;
constexpr int RASTER_MIN_COLS = 256;

template <int N, int BN>
__device__ __forceinline__ int2 grouped_tile() {
    constexpr int GROUP = (N + BN - 1) / BN >= RASTER_MIN_COLS ? RASTER_GROUP : 1;
    if (GROUP == 1) return make_int2(blockIdx.y, blockIdx.x);
    int num_m = gridDim.y;
    int num_n = gridDim.x;
    int pid = blockIdx.y * num_n + blockIdx.x;
    int per_group = GROUP * num_n;
    int first_m = pid / per_group * GROUP;
    int size = min(num_m - first_m, GROUP);
    int in_group = pid % per_group;
    return make_int2(first_m + in_group % size, in_group / size);
}

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS>
__device__ __forceinline__ void gemm_epilogue_at(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                                 bf16_t* out_hi, bf16_t* out_lo, int tokens, int m0, int n0) {
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    if (EPI == EPI_ACT) {
        static_assert(EPI != EPI_ACT || N % 4 == 0, "gate/up pairs");
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                int col = n0 + wn * TN + j * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int m = m0 + wm * TM + i * 16 + g + h * 8;
                    float a = activation(acc[i][j][h * 2]) * acc[i][j][h * 2 + 1];
                    float b = __shfl_xor_sync(0xffffffff, a, 1);
                    if ((t & 1) == 0 && col < N && m < tokens)
                        split_store2(out_hi, out_lo, (size_t)m * (N / 2) + col / 2, a, b);
                }
            }
        return;
    }
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = n0 + wn * TN + j * 8 + 2 * t;
            if (col >= N) continue;
            float b0 = 0.0f, b1 = 0.0f;
            if (BIAS) {
                b0 = bias[col];
                b1 = col + 1 < N ? bias[col + 1] : 0.0f;
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * TM + i * 16 + g + h * 8;
                if (m >= tokens) continue;
                float v0 = acc[i][j][h * 2] + b0;
                float v1 = acc[i][j][h * 2 + 1] + b1;
                {
                    float* o = out + (size_t)m * N + col;
                    if (EPI == EPI_RESIDUAL) {
                        v0 += o[0];
                        if (col + 1 < N) v1 += o[1];
                    }
                    o[0] = v0;
                    if (col + 1 < N) o[1] = v1;
                }
            }
        }
}

__device__ __forceinline__ void mma_e4m3(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
"##.to_string(),
    ]
    .concat()
}
