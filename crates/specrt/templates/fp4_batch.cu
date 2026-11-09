constexpr int F4B_KC = {{F4B_KC}};

__device__ __forceinline__ void f4b_cp_async8(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ uint4 f4b_load16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ unsigned f4b_byte(const uint4& q, int i) {
    unsigned word = (i >> 2) == 0 ? q.x : (i >> 2) == 1 ? q.y : (i >> 2) == 2 ? q.z : q.w;
    return (word >> (8 * (i & 3))) & 0xffu;
}

template <int N, int EPI>
__device__ __forceinline__ void f4b_store(int m, int r, float v, float u, float* out, bf16_t* out_hi, bf16_t* out_lo) {
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

template <int N, int K, int EPI, int NT, int WARPS, int STAGES, int MJ>
__device__ __forceinline__ void gemv_f4b(const void* w, const unsigned char* scales, const bf16_t* x_hi, const bf16_t* x_lo,
                                         float* out, bf16_t* out_hi, bf16_t* out_lo, float* part, int tokens, int splits) {
    constexpr int M = NT * 8;
    constexpr int PLANE = M * F4B_KC;
    constexpr int STAGE = PLANE * 2;
    constexpr int CHUNKS = K / F4B_KC;
    constexpr int KG = K / FP4_GROUP;
    constexpr int PIECES = F4B_KC / 4;
    constexpr int LOADS = 2 * M * PIECES;
    static_assert(F4B_KC == 128, "four lanes of 16 bytes span one chunk of a row");
    static_assert(K % F4B_KC == 0, "K must be a multiple of the chunk");
    static_assert(N % 16 == 0, "rows come in mma tiles of 16");
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
    int row0 = blockIdx.x * (WARPS * MJ * 16) + warp * (MJ * 16);
    int last = max(tokens - 1, 0);
    const unsigned char* wb = (const unsigned char*)w;
    const unsigned char* sb = scales + FP4_HEADER;
    const unsigned char* wrow[MJ][2];
    const unsigned char* srow[MJ][2];
#pragma unroll
    for (int i = 0; i < MJ; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int r = min(row0 + i * 16 + g + 8 * h, N - 1);
            wrow[i][h] = wb + (size_t)r * (K / 2) + t * 16;
            srow[i][h] = sb + (size_t)r * KG + 2 * t;
        }
    auto stage = [&](int buf, int c) {
        bf16_t* base = xs + buf * STAGE;
        for (int i = tid; i < LOADS; i += WARPS * 32) {
            int p = i / (M * PIECES);
            int m = (i / PIECES) % M;
            int piece = i % PIECES;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last) * K + c * F4B_KC + piece * 4;
            f4b_cp_async8(base + p * PLANE + (((piece & 7) * M + m) * 4 + (piece >> 3)) * 4, src, m < tokens);
        }
    };
    auto load_w = [&](int c, uint4 (&q)[MJ][2], unsigned (&s)[MJ][2]) {
#pragma unroll
        for (int i = 0; i < MJ; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                q[i][h] = f4b_load16(wrow[i][h] + (size_t)c * (F4B_KC / 2));
                s[i][h] = __ldg((const unsigned short*)(srow[i][h] + c * (F4B_KC / FP4_GROUP)));
            }
    };
    float acc[MJ][NT][4];
#pragma unroll
    for (int i = 0; i < MJ; i++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][n][e] = 0.0f;
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < nc) stage(st, c_begin + st);
        cp_async_commit();
    }
    uint4 q[MJ][2];
    unsigned s[MJ][2];
    if (nc > 0) load_w(c_begin, q, s);
    for (int it = 0; it < nc; it++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < nc) stage((it + STAGES - 1) % STAGES, c_begin + it + STAGES - 1);
        cp_async_commit();
        uint4 nq[MJ][2];
        unsigned ns[MJ][2];
        if (it + 1 < nc) load_w(c_begin + it + 1, nq, ns);
        const bf16_t* xh = xs + (it % STAGES) * STAGE;
        const bf16_t* xl = xh + PLANE;
        float sc[MJ][2][2];
#pragma unroll
        for (int i = 0; i < MJ; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                sc[i][h][0] = e4m3_to_f32(s[i][h] & 0xffu);
                sc[i][h][1] = e4m3_to_f32((s[i][h] >> 8) & 0xffu);
            }
#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint2 bh[NT], bl[NT];
#pragma unroll
            for (int n = 0; n < NT; n++) {
                bh[n] = *(const uint2*)(xh + ((k * M + n * 8 + g) * 4 + t) * 4);
                bl[n] = *(const uint2*)(xl + ((k * M + n * 8 + g) * 4 + t) * 4);
            }
#pragma unroll
            for (int i = 0; i < MJ; i++) {
                float s0 = sc[i][0][k >> 2];
                float s1 = sc[i][1][k >> 2];
                unsigned a0 = fp4_pair_bf16(f4b_byte(q[i][0], 2 * k), s0);
                unsigned a1 = fp4_pair_bf16(f4b_byte(q[i][1], 2 * k), s1);
                unsigned a2 = fp4_pair_bf16(f4b_byte(q[i][0], 2 * k + 1), s0);
                unsigned a3 = fp4_pair_bf16(f4b_byte(q[i][1], 2 * k + 1), s1);
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    mma_bf16(acc[i][n], a0, a1, a2, a3, bl[n].x, bl[n].y);
                    mma_bf16(acc[i][n], a0, a1, a2, a3, bh[n].x, bh[n].y);
                }
            }
        }
        if (it + 1 < nc) {
#pragma unroll
            for (int i = 0; i < MJ; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    q[i][h] = nq[i][h];
                    s[i][h] = ns[i][h];
                }
        }
    }
    cp_async_wait<0>();
    float inv = fp4_inv_global(scales);
#pragma unroll
    for (int i = 0; i < MJ; i++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = row0 + i * 16 + g + 8 * (e >> 1);
                int m = n * 8 + 2 * t + (e & 1);
                float v = acc[i][n][e] * inv;
                float u = __shfl_down_sync(0xffffffff, v, 4);
                if (r >= N || m >= tokens) continue;
                if (splits > 1) part[((size_t)split * tokens + m) * N + r] = v;
                else if (EPI != EPI_ACT || (g & 1) == 0) f4b_store<N, EPI>(m, r, v, u, out, out_hi, out_lo);
            }
}

template <int N, int EPI>
__device__ __forceinline__ void f4b_reduce(const float* part, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens, int splits) {
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
    f4b_store<N, EPI>(m, r, v, u, out, out_hi, out_lo);
}

{{F4B_KERNELS}}
