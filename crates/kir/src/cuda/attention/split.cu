constexpr int ATTN_SPLITS = 64;
constexpr int ATTN_TILE = 16;
constexpr int ATTN_MIN_CHUNK = 16;
constexpr int ATTN_BLOCK = 128;
constexpr int ATTN_V4 = (HEAD_DIM / 4 + 31) / 32;
constexpr int ATTN_PART = HEAD_DIM + 2;

__device__ __forceinline__ int attn_chunk(int len) {
    return max(ATTN_MIN_CHUNK, (len + ATTN_SPLITS - 1) / ATTN_SPLITS);
}

constexpr int ATTN_WARPS = ATTN_BLOCK / 32;
constexpr int ATTN_UNROLL = 4;

extern "C" __global__ void __launch_bounds__(ATTN_BLOCK) attention_split(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                     const int* block_table, const int* pos_ptr, int layer, float* part, int window) {
    __shared__ float4 acc_s[ATTN_WARPS][GROUP][HEAD_DIM / 4];
    __shared__ float m_s[ATTN_WARPS][GROUP];
    __shared__ float l_s[ATTN_WARPS][GROUP];
    int kv_head = blockIdx.x;
    int split = blockIdx.y;
    int len = pos_ptr[0] + 1;
    int lo = window_lo(len, window);
    int chunk = attn_chunk(len - lo);
    int start = lo + split * chunk;
    if (start >= len) return;
    int end = min(len, start + chunk);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    static_assert(HEAD_DIM % 4 == 0 && HEAD_DIM <= 256, "unsupported head_dim");
    float scale = QUERY_SCALE;
    float4 q[GROUP][ATTN_V4];
#pragma unroll
    for (int g = 0; g < GROUP; g++)
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++) {
            int c = lane + 32 * j;
            q[g][j] = c < HEAD_DIM / 4 ? *(const float4*)(qkv + (kv_head * GROUP + g) * HEAD_DIM + 4 * c)
                                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    float m[GROUP], l[GROUP];
    float4 acc[GROUP][ATTN_V4];
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        m[g] = NEG_INF;
        l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++) acc[g][j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int tb = start + warp; tb < end; tb += ATTN_WARPS * ATTN_UNROLL) {
        float4 kr[ATTN_UNROLL][ATTN_V4], vr[ATTN_UNROLL][ATTN_V4];
#pragma unroll
        for (int u = 0; u < ATTN_UNROLL; u++) {
            int t = min(tb + u * ATTN_WARPS, end - 1);
            const bf16_t* k = kv_row(k_cache, block_table, layer, kv_head, t);
            const bf16_t* v = kv_row(v_cache, block_table, layer, kv_head, t);
            float ksc = kv_scale(k_cache, block_table, layer, kv_head, t);
            float vsc = kv_scale(v_cache, block_table, layer, kv_head, t);
#pragma unroll
            for (int j = 0; j < ATTN_V4; j++) {
                int c = lane + 32 * j;
                bool on = c < HEAD_DIM / 4;
                kr[u][j] = on ? kv_at4(k, 4 * c, ksc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vr[u][j] = on ? kv_at4(v, 4 * c, vsc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int u = 0; u < ATTN_UNROLL; u++) {
            if (tb + u * ATTN_WARPS >= end) break;
#pragma unroll
            for (int g = 0; g < GROUP; g++) {
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < ATTN_V4; j++)
                    dot += q[g][j].x * kr[u][j].x + q[g][j].y * kr[u][j].y + q[g][j].z * kr[u][j].z + q[g][j].w * kr[u][j].w;
                dot = cap_score(warp_sum(dot) * scale);
                float nm = fmaxf(m[g], dot);
                float corr = expf(m[g] - nm);
                float p = expf(dot - nm);
                l[g] = l[g] * corr + p;
#pragma unroll
                for (int j = 0; j < ATTN_V4; j++) {
                    acc[g][j].x = acc[g][j].x * corr + p * vr[u][j].x;
                    acc[g][j].y = acc[g][j].y * corr + p * vr[u][j].y;
                    acc[g][j].z = acc[g][j].z * corr + p * vr[u][j].z;
                    acc[g][j].w = acc[g][j].w * corr + p * vr[u][j].w;
                }
                m[g] = nm;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        if (lane == 0) {
            m_s[warp][g] = m[g];
            l_s[warp][g] = l[g];
        }
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++)
            if (lane + 32 * j < HEAD_DIM / 4) acc_s[warp][g][lane + 32 * j] = acc[g][j];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < GROUP * HEAD_DIM; idx += ATTN_BLOCK) {
        int g = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        float M = NEG_INF;
        for (int w = 0; w < ATTN_WARPS; w++) M = fmaxf(M, m_s[w][g]);
        float L = 0.0f, A = 0.0f;
        for (int w = 0; w < ATTN_WARPS; w++) {
            float c = m_s[w][g] == NEG_INF ? 0.0f : expf(m_s[w][g] - M);
            L += l_s[w][g] * c;
            A += ((const float*)acc_s[w][g])[d] * c;
        }
        float* dst = part + ((size_t)(kv_head * GROUP + g) * ATTN_SPLITS + split) * ATTN_PART;
        dst[d] = A;
        if (d == 0) {
            dst[HEAD_DIM] = M;
            dst[HEAD_DIM + 1] = L;
        }
    }
}

extern "C" __global__ void attention_reduce(const float* part, const int* pos_ptr, const float* qkv, float* out, int window) {
    int head = blockIdx.x;
    int len = pos_ptr[0] + 1 - window_lo(pos_ptr[0] + 1, window);
    int chunk = attn_chunk(len);
    int active = (len + chunk - 1) / chunk;
    const float* base = part + (size_t)head * ATTN_SPLITS * ATTN_PART;
    __shared__ float w_s[ATTN_SPLITS];
    int d = threadIdx.x;
    float ms = d < active ? base[d * ATTN_PART + HEAD_DIM] : NEG_INF;
    float ls = d < active ? base[d * ATTN_PART + HEAD_DIM + 1] : 0.0f;
    float m = warp_max(ms);
    if (ATTN_SPLITS > 32) {
        __shared__ float wm[ATTN_SPLITS / 32 + 1];
        if ((d & 31) == 0 && d < ATTN_SPLITS) wm[d >> 5] = m;
        __syncthreads();
        m = NEG_INF;
        for (int i = 0; i <= (ATTN_SPLITS - 1) / 32; i++) m = fmaxf(m, wm[i]);
    }
    float w = d < active ? expf(ms - m) : 0.0f;
    if (d < ATTN_SPLITS) w_s[d] = w;
    __syncthreads();
    float l = 0.0f;
    for (int i = 0; i < active; i++) l += w_s[i] * base[i * ATTN_PART + HEAD_DIM + 1];
    float total = 0.0f;
    if (d < HEAD_DIM) {
#pragma unroll 8
        for (int i = 0; i < active; i++) total += w_s[i] * base[i * ATTN_PART + d];
        out[head * HEAD_DIM + d] = total / l * out_gate(qkv, 0, head * HEAD_DIM + d);
    }
}
