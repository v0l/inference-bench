
__device__ __forceinline__ void store_act2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    if (GEMM_SPLIT) {
        split_store2(hi, lo, at, a, b);
    } else {
        *(unsigned*)(hi + at) = pack_f16_hi(a, b);
    }
}

extern "C" __global__ void vis_to_f16(bf16_t* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = (bf16_t)f16_bits(bf(w[i]));
}

__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(PENDING));
}

__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16_hi_lo(float* c, const unsigned* ah, const unsigned* al, unsigned bh0, unsigned bh1,
                                              unsigned bl0, unsigned bl1) {
    mma_f16(c, al, bh0, bh1);
    mma_f16(c, ah, bl0, bl1);
    mma_f16(c, ah, bh0, bh1);
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ int swz4(int r, int c) { return r * 4 + (c ^ ((r >> 1) & 3)); }

constexpr int GM_BM = 128;
constexpr int GM_BN = 128;
constexpr int GM_BK = 32;
constexpr int GM_CPR = GM_BK / 8;
constexpr int GM_PLANES = GEMM_SPLIT ? 2 : 1;
constexpr int GM_WM = 2;
constexpr int GM_WN = 4;
constexpr int GM_STAGES = 3;
constexpr int GM_THREADS = GM_WM * GM_WN * 32;
constexpr int GM_TM = GM_BM / GM_WM;
constexpr int GM_TN = GM_BN / GM_WN;
constexpr int GM_MI = GM_TM / 16;
constexpr int GM_NJ = GM_TN / 8;
constexpr int GM_A_TILE = GM_BM * GM_BK;
constexpr int GM_A_BYTES = GM_A_TILE * 2 * GM_PLANES;
constexpr int GM_STAGE_BYTES = GM_A_BYTES + GM_BN * GM_BK * 2;

__device__ __forceinline__ int gswz(int r, int c) {
    if (GM_CPR == 8) return r * 8 + (c ^ (r & 7));
    return r * 4 + (c ^ ((r >> 1) & 3));
}

extern "C" __global__ void __launch_bounds__(GM_THREADS) vis_mma(const bf16_t* a_hi, const bf16_t* a_lo, const bf16_t* w,
                                                                const bf16_t* bias, const float* resid, float* out,
                                                                bf16_t* out_hi, bf16_t* out_lo, int M, int N, int K, int act) {
    extern __shared__ __align__(16) unsigned char gm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / GM_WN;
    int wn = warp % GM_WN;
    int m0 = blockIdx.y * GM_BM;
    int n0 = blockIdx.x * GM_BN;
    int kt_count = (K + GM_BK - 1) / GM_BK;
    float acc[GM_MI][GM_NJ][4];
#pragma unroll
    for (int i = 0; i < GM_MI; i++)
#pragma unroll
        for (int j = 0; j < GM_NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    auto load_stage = [&](int buf, int kt) {
        unsigned char* base = gm_smem + buf * GM_STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < (GM_PLANES * GM_BM * GM_CPR) / GM_THREADS; i++) {
            int idx = tid + i * GM_THREADS;
            int p = idx / (GM_BM * GM_CPR);
            int rem = idx % (GM_BM * GM_CPR);
            int r = rem / GM_CPR;
            int c = rem % GM_CPR;
            int m = m0 + r;
            int k = kt * GM_BK + c * 8;
            const bf16_t* src = (p ? a_lo : a_hi) + (size_t)min(m, M - 1) * K + min(k, K - 8);
            cp_async16_zfill(base + p * GM_A_TILE * 2 + gswz(r, c) * 16, src, m < M && k < K);
        }
        unsigned char* ws = base + GM_A_BYTES;
#pragma unroll
        for (int i = 0; i < (GM_BN * GM_CPR) / GM_THREADS; i++) {
            int idx = tid + i * GM_THREADS;
            int r = idx / GM_CPR;
            int c = idx % GM_CPR;
            int n = n0 + r;
            int k = kt * GM_BK + c * 8;
            const bf16_t* src = w + (size_t)min(n, N - 1) * K + min(k, K - 8);
            cp_async16_zfill(ws + gswz(r, c) * 16, src, n < N && k < K);
        }
    };
#pragma unroll
    for (int st = 0; st < GM_STAGES - 1; st++) {
        if (st < kt_count) load_stage(st, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < kt_count; kt++) {
        cp_async_wait<GM_STAGES - 2>();
        __syncthreads();
        int next = kt + GM_STAGES - 1;
        if (next < kt_count) load_stage(next % GM_STAGES, next);
        cp_async_commit();
        const unsigned char* base = gm_smem + (kt % GM_STAGES) * GM_STAGE_BYTES;
        const bf16_t* sa_hi = (const bf16_t*)base;
        const bf16_t* sa_lo = sa_hi + GM_A_TILE;
        const unsigned char* ws = base + GM_A_BYTES;
#pragma unroll
        for (int ks = 0; ks < GM_BK / 16; ks++) {
            unsigned bw[GM_NJ][2];
#pragma unroll
            for (int jp = 0; jp < GM_NJ / 2; jp++) {
                unsigned r[4];
                int row = wn * GM_TN + jp * 16 + (lane >> 4) * 8 + (lane & 7);
                ldsm_x4(r, ws + gswz(row, ks * 2 + ((lane >> 3) & 1)) * 16);
                bw[2 * jp][0] = r[0];
                bw[2 * jp][1] = r[1];
                bw[2 * jp + 1][0] = r[2];
                bw[2 * jp + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < GM_MI; i++) {
                unsigned ah[4], al[4];
                int row = wm * GM_TM + i * 16 + (lane & 15);
                int at = gswz(row, ks * 2 + (lane >> 4)) * 8;
                ldsm_x4(ah, &sa_hi[at]);
                if (GEMM_SPLIT) ldsm_x4(al, &sa_lo[at]);
#pragma unroll
                for (int j = 0; j < GM_NJ; j++) {
                    if (GEMM_SPLIT) {
                        mma_bf16(acc[i][j], al, bw[j][0], bw[j][1]);
                        mma_bf16(acc[i][j], ah, bw[j][0], bw[j][1]);
                    } else {
                        mma_f16(acc[i][j], ah, bw[j][0], bw[j][1]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int i = 0; i < GM_MI; i++)
#pragma unroll
        for (int j = 0; j < GM_NJ; j++) {
            int col = n0 + wn * GM_TN + j * 8 + 2 * t;
            if (col >= N) continue;
            float b0 = bias ? bf(bias[col]) : 0.0f;
            float b1 = bias ? bf(bias[col + 1]) : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * GM_TM + i * 16 + g + h * 8;
                if (m >= M) continue;
                float v0 = activate(acc[i][j][h * 2] + b0, act);
                float v1 = activate(acc[i][j][h * 2 + 1] + b1, act);
                size_t at = (size_t)m * N + col;
                if (out_hi) {
                    store_act2(out_hi, out_lo, at, v0, v1);
                } else {
                    if (resid) {
                        v0 += resid[at];
                        v1 += resid[at + 1];
                    }
                    *(float2*)(out + at) = make_float2(v0, v1);
                }
            }
        }
}
