use super::prim;

pub fn body() -> String {
    [
        prim::batch_helpers("ggr"),
        r##"
struct GgrRaw {
    uint4 v[4];
    unsigned meta;
};

__device__ __forceinline__ unsigned ggr_pack(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ unsigned ggr_word(const uint4& a, const uint4& b, int j) {
    const uint4& q = j < 4 ? a : b;
    int i = j & 3;
    return i == 0 ? q.x : i == 1 ? q.y : i == 2 ? q.z : q.w;
}

__device__ __forceinline__ void ggr_fetch(const unsigned char* row, int type, int c, int t, GgrRaw& r) {
    const unsigned char* sb = row + (size_t)(c >> 1) * gg_bytes(type);
    int h = c & 1;
    r.meta = 0u;
    if (type == GG_Q4K) {
        const unsigned char* q = sb + 16 + 64 * h + 32 * (t >> 1);
        r.v[0] = ggr_load16(sb);
        r.v[1] = ggr_load16(q);
        r.v[2] = ggr_load16(q + 16);
    } else if (type == GG_Q6K) {
        const unsigned char* ql = sb + 64 * h + 32 * (t & 1);
        const unsigned char* qh = sb + 128 + 32 * h;
        r.v[0] = ggr_load16(ql);
        r.v[1] = ggr_load16(ql + 16);
        r.v[2] = ggr_load16(qh);
        r.v[3] = ggr_load16(qh + 16);
        r.meta = (unsigned)__ldg((const unsigned short*)(sb + 192 + 8 * h + 2 * t)) | ((unsigned)__ldg((const unsigned short*)(sb + 208)) << 16);
    }
}

__device__ __forceinline__ void ggr_convert(const unsigned char* row, int type, int c, int t, const GgrRaw& r, unsigned* a) {
    int h = c & 1;
    if (type == GG_Q4K) {
        float d = gg_f16((unsigned short)(r.v[0].x & 0xffffu));
        float dmin = gg_f16((unsigned short)(r.v[0].x >> 16));
        unsigned s[3] = {r.v[0].y, r.v[0].z, r.v[0].w};
        float sc, m;
        gg_scale_min(s, 4 * h + t, sc, m);
        float dl = __fmul_rn(d, sc), ml = __fmul_rn(dmin, m);
        int shift = 4 * (t & 1);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned nibs = (ggr_word(r.v[1], r.v[2], i >> 1) >> shift) & 0x0f0f0f0fu;
            unsigned pair = nibs >> (16 * (i & 1));
            float w0 = __fsub_rn(__fmul_rn(dl, (float)(pair & 15u)), ml);
            float w1 = __fsub_rn(__fmul_rn(dl, (float)((pair >> 8) & 15u)), ml);
            a[i] = ggr_pack(w0, w1);
        }
    } else if (type == GG_Q6K) {
        float d = gg_f16((unsigned short)(r.meta >> 16));
        float ds0 = __fmul_rn(d, (float)(signed char)(r.meta & 0xffu));
        float ds1 = __fmul_rn(d, (float)(signed char)((r.meta >> 8) & 0xffu));
        int shift = 4 * (t >> 1);
        int hs = 2 * t;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned lo = (ggr_word(r.v[0], r.v[1], i >> 1) >> (shift + 16 * (i & 1))) & 0x0f0fu;
            unsigned hi = (ggr_word(r.v[2], r.v[3], i >> 1) >> (hs + 16 * (i & 1))) & 0x0303u;
            int q0 = (int)((lo & 15u) | ((hi & 3u) << 4)) - 32;
            int q1 = (int)(((lo >> 8) & 15u) | (((hi >> 8) & 3u) << 4)) - 32;
            float ds = i < 8 ? ds0 : ds1;
            a[i] = ggr_pack(__fmul_rn(ds, (float)q0), __fmul_rn(ds, (float)q1));
        }
    } else {
        const unsigned char* sb = row + (size_t)(c >> 1) * gg_bytes(type);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            float w[8];
            gg_dequant8(sb, type, h * 16 + t * 4 + q, w);
#pragma unroll
            for (int i = 0; i < 4; i++) a[4 * q + i] = ggr_pack(w[2 * i], w[2 * i + 1]);
        }
    }
}
"##.to_string(),
        prim::batch_store("ggr"),
        r##"
template <int N, int K, int EPI, bool BIAS, bool SPLIT, int NT, int WARPS, int STAGES>
__device__ __forceinline__ void gemv_ggr(const void* w, const unsigned* table, const float* bias, const bf16_t* x_hi,
                                         const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, float* part,
                                         int tokens, int splits) {
    constexpr int M = NT * 8;
    constexpr int PLANE = M * GGR_KC;
    constexpr int STAGE = PLANE * 2;
    constexpr int CHUNKS = K / GGR_KC;
    constexpr int PIECES = GGR_KC / 4;
    constexpr int LOADS = (SPLIT ? 2 : 1) * M * PIECES;
    static_assert(GGR_KC == 128, "one chunk is half a ggml super block");
    static_assert(K % GGR_KC == 0, "K must be a multiple of the chunk");
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
    int row0 = blockIdx.x * (WARPS * 16) + warp * 16;
    int last = max(tokens - 1, 0);
    const unsigned char* wrow[2];
    int wtype[2];
#pragma unroll
    for (int h = 0; h < 2; h++) wrow[h] = gg_row(w, table, min(row0 + g + 8 * h, N - 1), wtype[h]);
    auto stage = [&](int buf, int c) {
        bf16_t* base = xs + buf * STAGE;
        for (int i = tid; i < LOADS; i += WARPS * 32) {
            int p = i / (M * PIECES);
            int m = (i / PIECES) % M;
            int piece = i % PIECES;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last) * K + c * GGR_KC + piece * 4;
            ggr_cp_async8(base + p * PLANE + (((piece & 7) * M + m) * 4 + (piece >> 3)) * 4, src, m < tokens);
        }
    };
    auto fetch = [&](int c, GgrRaw (&dst)[2]) {
#pragma unroll
        for (int h = 0; h < 2; h++) ggr_fetch(wrow[h], wtype[h], c, t, dst[h]);
    };
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = 0.0f;
"##.to_string(),
        prim::pipeline_prologue("nc", "stage(st, c_begin + st)"),
        r##"    GgrRaw q[2];
    if (nc > 0) fetch(c_begin, q);
    for (int it = 0; it < nc; it++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < nc) stage((it + STAGES - 1) % STAGES, c_begin + it + STAGES - 1);
        cp_async_commit();
        GgrRaw nq[2];
        if (it + 1 < nc) fetch(c_begin + it + 1, nq);
        unsigned a[2][16];
#pragma unroll
        for (int h = 0; h < 2; h++) ggr_convert(wrow[h], wtype[h], c_begin + it, t, q[h], a[h]);
        const bf16_t* xh = xs + (it % STAGES) * STAGE;
        const bf16_t* xl = xh + PLANE;
#pragma unroll
        for (int k = 0; k < 8; k++) {
#pragma unroll
            for (int n = 0; n < NT; n++) {
                uint2 bh = *(const uint2*)(xh + ((k * M + n * 8 + g) * 4 + t) * 4);
                if (SPLIT) {
                    uint2 bl = *(const uint2*)(xl + ((k * M + n * 8 + g) * 4 + t) * 4);
                    mma_bf16(acc[n], a[0][2 * k], a[1][2 * k], a[0][2 * k + 1], a[1][2 * k + 1], bl.x, bl.y);
                }
                mma_bf16(acc[n], a[0][2 * k], a[1][2 * k], a[0][2 * k + 1], a[1][2 * k + 1], bh.x, bh.y);
            }
        }
        if (it + 1 < nc) {
#pragma unroll
            for (int h = 0; h < 2; h++) q[h] = nq[h];
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int r = row0 + g + 8 * (e >> 1);
            int m = n * 8 + 2 * t + (e & 1);
            float v = acc[n][e];
            float u = __shfl_down_sync(0xffffffff, v, 4);
            if (r >= N || m >= tokens) continue;
            if (splits > 1) part[((size_t)split * tokens + m) * N + r] = v;
            else if (EPI != EPI_ACT || (g & 1) == 0) ggr_store<N, EPI>(m, r, BIAS ? v + bias[r] : v, u, out, out_hi, out_lo);
        }
}

template <int N, int EPI, bool BIAS>
__device__ __forceinline__ void ggr_reduce(const float* part, const float* bias, float* out, bf16_t* out_hi, bf16_t* out_lo,
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
    ggr_store<N, EPI>(m, r, v, u, out, out_hi, out_lo);
}

"##.to_string(),
    ]
    .concat()
}
