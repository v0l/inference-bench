constexpr int F8B_WARPS = {{F8B_WARPS}};
constexpr int F8B_MJ = {{F8B_MJ}};
constexpr int F8B_ROWS = F8B_WARPS * F8B_MJ * 16;
constexpr int F8B_KC = {{F8B_KC}};
constexpr int F8B_STAGES = {{F8B_STAGES}};

__device__ __forceinline__ void f8b_cp_async8(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ uint4 f8b_load16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ unsigned f8b_word(const uint4& a, const uint4& b, int i) {
    const uint4& q = i < 4 ? a : b;
    int j = i & 3;
    return j == 0 ? q.x : j == 1 ? q.y : j == 2 ? q.z : q.w;
}

template <int N, int EPI>
__device__ __forceinline__ void f8b_store(int m, int r, float v, float u, float* out, bf16_t* out_hi, bf16_t* out_lo) {
    if (EPI == EPI_ACT) {
        unsigned hi, lo;
        split_hi_lo(activation(v) * u, hi, lo);
        size_t at = (size_t)m * (N / 2) + r / 2;
        out_hi[at] = (bf16_t)hi;
        out_lo[at] = (bf16_t)lo;
    } else {
        float* o = out + (size_t)m * N + r;
        *o = EPI == EPI_RESIDUAL ? *o + v : v;
    }
}

template <int N, int K, int EPI, bool BIAS, int NT>
__device__ __forceinline__ void gemv_f8b(const void* w, const float* scales, const float* bias, const bf16_t* x_hi,
                                         const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, float* part,
                                         int tokens, int splits) {
    constexpr int M = NT * 8;
    constexpr int PLANE = M * F8B_KC;
    constexpr int STAGE = PLANE * 2;
    constexpr int CHUNKS = K / F8B_KC;
    constexpr int PIECES = F8B_KC / 4;
    constexpr int LOADS = 2 * M * PIECES;
    static_assert(F8B_KC == 128, "four lanes of 32 bytes span one chunk of a row");
    static_assert(F8B_KC == FP8_BLOCK, "one scale per row and chunk");
    static_assert(K % F8B_KC == 0, "K must be a multiple of the chunk");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    bf16_t* xs = (bf16_t*)gemm_smem;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int split = blockIdx.y;
    int c_begin = split * CHUNKS / splits;
    int nc = (split + 1) * CHUNKS / splits - c_begin;
    int row0 = blockIdx.x * F8B_ROWS + warp * (F8B_MJ * 16);
    int last = max(tokens - 1, 0);
    const unsigned char* wb = (const unsigned char*)w;
    const unsigned char* wrow[F8B_MJ][2];
    const float* srow[F8B_MJ][2];
#pragma unroll
    for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int r = min(row0 + i * 16 + g + 8 * h, N - 1);
            wrow[i][h] = wb + (size_t)r * K + t * 32;
            srow[i][h] = fp8_row_scales(scales, r);
        }
    auto stage = [&](int buf, int c) {
        bf16_t* base = xs + buf * STAGE;
        for (int i = tid; i < LOADS; i += F8B_WARPS * 32) {
            int p = i / (M * PIECES);
            int m = (i / PIECES) % M;
            int piece = i % PIECES;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last) * K + c * F8B_KC + piece * 4;
            f8b_cp_async8(base + p * PLANE + (((piece & 7) * M + m) * 4 + (piece >> 3)) * 4, src, m < tokens);
        }
    };
    auto load_w = [&](int c, uint4 (&q)[F8B_MJ][2][2], float (&s)[F8B_MJ][2]) {
#pragma unroll
        for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                q[i][h][0] = f8b_load16(wrow[i][h] + (size_t)c * F8B_KC);
                q[i][h][1] = f8b_load16(wrow[i][h] + (size_t)c * F8B_KC + 16);
                s[i][h] = __ldg(srow[i][h] + c);
            }
    };
    float acc[F8B_MJ][NT][4];
#pragma unroll
    for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][n][e] = 0.0f;
#pragma unroll
    for (int st = 0; st < F8B_STAGES - 1; st++) {
        if (st < nc) stage(st, c_begin + st);
        cp_async_commit();
    }
    uint4 q[F8B_MJ][2][2];
    float s[F8B_MJ][2];
    if (nc > 0) load_w(c_begin, q, s);
    for (int it = 0; it < nc; it++) {
        cp_async_wait<F8B_STAGES - 2>();
        __syncthreads();
        if (it + F8B_STAGES - 1 < nc) stage((it + F8B_STAGES - 1) % F8B_STAGES, c_begin + it + F8B_STAGES - 1);
        cp_async_commit();
        uint4 nq[F8B_MJ][2][2];
        float ns[F8B_MJ][2];
        if (it + 1 < nc) load_w(c_begin + it + 1, nq, ns);
        const bf16_t* xh = xs + (it % F8B_STAGES) * STAGE;
        const bf16_t* xl = xh + PLANE;
        float c[F8B_MJ][NT][4];
#pragma unroll
        for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) c[i][n][e] = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint2 bh[NT], bl[NT];
#pragma unroll
            for (int n = 0; n < NT; n++) {
                bh[n] = *(const uint2*)(xh + ((k * M + n * 8 + g) * 4 + t) * 4);
                bl[n] = *(const uint2*)(xl + ((k * M + n * 8 + g) * 4 + t) * 4);
            }
#pragma unroll
            for (int i = 0; i < F8B_MJ; i++) {
                unsigned w0 = f8b_word(q[i][0][0], q[i][0][1], k);
                unsigned w1 = f8b_word(q[i][1][0], q[i][1][1], k);
                unsigned a0 = e4m3_pair_bf16(w0 & 0xffffu);
                unsigned a1 = e4m3_pair_bf16(w1 & 0xffffu);
                unsigned a2 = e4m3_pair_bf16(w0 >> 16);
                unsigned a3 = e4m3_pair_bf16(w1 >> 16);
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    mma_bf16(c[i][n], a0, a1, a2, a3, bl[n].x, bl[n].y);
                    mma_bf16(c[i][n], a0, a1, a2, a3, bh[n].x, bh[n].y);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][n][e] += c[i][n][e] * s[i][e >> 1];
        if (it + 1 < nc) {
#pragma unroll
            for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    q[i][h][0] = nq[i][h][0];
                    q[i][h][1] = nq[i][h][1];
                    s[i][h] = ns[i][h];
                }
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int i = 0; i < F8B_MJ; i++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = row0 + i * 16 + g + 8 * (e >> 1);
                int m = n * 8 + 2 * t + (e & 1);
                float v = acc[i][n][e];
                float u = __shfl_down_sync(0xffffffff, v, 4);
                if (r >= N || m >= tokens) continue;
                if (splits > 1) part[((size_t)split * tokens + m) * N + r] = v;
                else if (EPI != EPI_ACT || (g & 1) == 0) f8b_store<N, EPI>(m, r, BIAS ? v + bias[r] : v, u, out, out_hi, out_lo);
            }
}

template <int N, int EPI, bool BIAS>
__device__ __forceinline__ void f8b_reduce(const float* part, const float* bias, float* out, bf16_t* out_hi, bf16_t* out_lo,
                                           int tokens, int splits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int OUT = EPI == EPI_ACT ? N / 2 : N;
    if (i >= tokens * OUT) return;
    int m = i / OUT;
    int r = EPI == EPI_ACT ? 2 * (i % OUT) : i % OUT;
    float v = 0.0f, u = 0.0f;
    for (int sp = 0; sp < splits; sp++) {
        const float* p = part + ((size_t)sp * tokens + m) * N + r;
        v += p[0];
        if (EPI == EPI_ACT) u += p[1];
    }
    if (BIAS) v += bias[r];
    f8b_store<N, EPI>(m, r, v, u, out, out_hi, out_lo);
}

{{F8B_KERNELS}}
