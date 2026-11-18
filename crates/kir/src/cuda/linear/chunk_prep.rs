pub fn body() -> String {
    r##"extern "C" __global__ void __launch_bounds__(256) gdn_chunk_prep(const float* qn, const float* kn, const float* mix,
                                                                 const float* g, const float* beta, float* w,
                                                                 float* u, float* aqk, float* gc, float* qe, float* kt, int tokens) {
    constexpr int C = LA_CHUNK;
    extern __shared__ __align__(16) unsigned char la_smem[];
    float* ks = (float*)la_smem;
    float* A = ks + C * LA_KS;
    float* gs = A + C * LA_AS;
    float* bs = gs + C;
    int h = blockIdx.x;
    int kh = la_kh(h);
    int c = blockIdx.y;
    int t0 = c * C;
    int tid = threadIdx.x;
    if (tid < C) {
        int t = t0 + tid;
        gs[tid] = t < tokens ? g[(size_t)t * LA_VH + h] : 0.0f;
        bs[tid] = t < tokens ? beta[(size_t)t * LA_VH + h] : 0.0f;
    }
    for (int i = tid; i < C * LA_D; i += blockDim.x) {
        int r = i / LA_D;
        int d = i % LA_D;
        int t = t0 + r;
        ks[r * LA_KS + d] = t < tokens ? kn[((size_t)t * LA_KH + kh) * LA_D + d] : 0.0f;
    }
    __syncthreads();
    if (tid == 0)
        for (int i = 1; i < C; i++) gs[i] += gs[i - 1];
    __syncthreads();
    constexpr int WARPS = 8;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g8 = lane >> 2;
    int t4 = lane & 3;
    auto k_row = [&](int m, int k) { return ks[m * LA_KS + k]; };
    auto k_col = [&](int k, int n) { return ks[n * LA_KS + k]; };
    for (int tile = warp; tile < (C / 16) * (C / 8); tile += WARPS) {
        int m0 = (tile / (C / 8)) * 16;
        int n0 = (tile % (C / 8)) * 8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (n0 <= m0 + 15) mma_split_tile<LA_D / 16>(acc, m0, n0, lane, k_row, k_col);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4 + (e & 1);
            A[i * LA_AS + j] = j < i ? -(bs[i] * acc[e]) * expf(gs[i] - gs[j]) : 0.0f;
        }
    }
    const float* qc = qn + ((size_t)t0 * LA_KH + kh) * LA_D;
    float* ac = aqk + (size_t)(c * LA_VH + h) * C * C;
    int live = min(C, tokens - t0);
    for (int tile = warp; tile < (C / 16) * (C / 8); tile += WARPS) {
        int m0 = (tile / (C / 8)) * 16;
        int n0 = (tile % (C / 8)) * 8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (n0 <= m0 + 15)
            mma_split_tile<LA_D / 16>(
                acc, m0, n0, lane, [&](int m, int k) { return m < live ? qc[(size_t)m * LA_KH * LA_D + k] : 0.0f; },
                k_col);
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4;
            float v0 = j <= i && i < live ? acc[e] * expf(gs[i] - gs[j]) : 0.0f;
            float v1 = j + 1 <= i && i < live ? acc[e + 1] * expf(gs[i] - gs[j + 1]) : 0.0f;
            gdn_frag_put<C / 16>((unsigned*)ac, i, j, v0, v1);
        }
    }
    __syncthreads();
    constexpr int LEAF = 16;
    for (int i = 1; i < LEAF; i++) {
        int b = (tid / LEAF) * LEAF;
        int j = tid % LEAF;
        float next = 0.0f;
        if (tid < C && j < i) {
            next = A[(b + i) * LA_AS + b + j];
            for (int m = j + 1; m < i; m++) next += A[(b + i) * LA_AS + b + m] * A[(b + m) * LA_AS + b + j];
        }
        __syncthreads();
        if (tid < C && j < i) A[(b + i) * LA_AS + b + j] = next;
        __syncthreads();
    }
    for (int sz = LEAF; sz < C; sz *= 2) {
        constexpr int PER = C * C / 4 / 256;
        int per = sz * sz * (C / (2 * sz));
        float acc[PER];
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            acc[p] = 0.0f;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                float y = A[(base + sz + r) * LA_AS + base + c2];
                for (int m = c2 + 1; m < sz; m++)
                    y += A[(base + sz + r) * LA_AS + base + m] * A[(base + m) * LA_AS + base + c2];
                acc[p] = y;
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                A[(base + sz + r) * LA_AS + base + c2] = acc[p];
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                float x = A[(base + sz + r) * LA_AS + base + c2];
                for (int m = 0; m < r; m++)
                    x += A[(base + sz + r) * LA_AS + base + sz + m] * A[(base + sz + m) * LA_AS + base + c2];
                acc[p] = x;
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                A[(base + sz + r) * LA_AS + base + c2] = acc[p];
            }
        }
        __syncthreads();
    }
    if (tid < C) A[tid * LA_AS + tid] = 1.0f;
    __syncthreads();
    const float* vc = mix + (size_t)t0 * LA_CONV_DIM + LA_V_OFF + h * LA_D;
    auto t_at = [&](int m, int k) { return k <= m ? A[m * LA_AS + k] : 0.0f; };
    for (int tile = warp; tile < (C / 16) * (LA_D / 8); tile += WARPS) {
        int m0 = (tile / (LA_D / 8)) * 16;
        int n0 = (tile % (LA_D / 8)) * 8;
        float uu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ww[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_split_tile<C / 16>(uu, m0, n0, lane, t_at,
                               [&](int k, int n) { return k < live ? vc[(size_t)k * LA_CONV_DIM + n] * bs[k] : 0.0f; });
        mma_split_tile<C / 16>(ww, m0, n0, lane, t_at,
                               [&](int k, int n) { return ks[k * LA_KS + n] * bs[k] * expf(gs[k]); });
        unsigned* wp = (unsigned*)w + (size_t)(c * LA_VH + h) * C * LA_D;
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
            int i = m0 + g8 + (e >> 1) * 8;
            int d = n0 + 2 * t4;
            size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
            u[at] = uu[e];
            u[at + 1] = uu[e + 1];
            gdn_frag_put<LA_D / 16>(wp, i, d, ww[e], ww[e + 1]);
        }
    }
    if (tid < C) gc[(size_t)(c * LA_VH + h) * C + tid] = gs[tid];
    float glast = gs[C - 1];
    unsigned* qp = (unsigned*)qe + (size_t)(c * LA_VH + h) * C * LA_D;
    unsigned* kp = (unsigned*)kt + (size_t)(c * LA_VH + h) * C * LA_D;
    for (int p = tid; p < C * LA_D / 2; p += blockDim.x) {
        int m = p / (LA_D / 2);
        int k = (p % (LA_D / 2)) * 2;
        float2 q = t0 + m < tokens ? *(const float2*)(qn + ((size_t)(t0 + m) * LA_KH + kh) * LA_D + k) : make_float2(0.0f, 0.0f);
        float eg = t0 + m < tokens ? expf(gs[m]) : 0.0f;
        gdn_frag_put<LA_D / 16>(qp, m, k, q.x * eg, q.y * eg);
    }
    for (int p = tid; p < C * LA_D / 2; p += blockDim.x) {
        int d = p % LA_D;
        int k = (p / LA_D) * 2;
        float v0 = 0.0f, v1 = 0.0f;
        if (t0 + k < tokens) v0 = kn[((size_t)(t0 + k) * LA_KH + kh) * LA_D + d] * expf(glast - gs[k]);
        if (t0 + k + 1 < tokens) v1 = kn[((size_t)(t0 + k + 1) * LA_KH + kh) * LA_D + d] * expf(glast - gs[k + 1]);
        gdn_frag_put<C / 16>(kp, d, k, v0, v1);
    }
}
"##
    .to_string()
}
