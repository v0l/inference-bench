constexpr int AT_STRIDE = AT_HDP + 8;
constexpr int AT_WARPS = 8;
constexpr int AT_BQ = 16 * AT_WARPS;
constexpr int AT_BK = ATT_SPLIT ? 32 : 64;
constexpr int AT_KB = AT_HDP / 16;
constexpr int AT_NB = AT_HDP / 8;
constexpr int AT_CHUNKS = AT_HDP / 8;
constexpr int AT_TILE = AT_BK * AT_STRIDE;
constexpr int AT_STAGE = (ATT_SPLIT ? 4 : 2) * AT_TILE;

extern "C" __global__ void vis_rope_pack(const float* qkv, const int* hw, bf16_t* planes, int tokens, int heads, int head_dim, float theta) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int half = head_dim / 2;
    int quarter = head_dim / 4;
    size_t width = (size_t)3 * heads * head_dim;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    size_t dst = ((size_t)h * tokens + tok) * AT_HDP;
    int ph = hw[2 * tok], pw = hw[2 * tok + 1];
    for (int part = 0; part < 3; part++) {
        const float* src = qkv + tok * width + (size_t)part * heads * head_dim + (size_t)h * head_dim;
        bf16_t* hi = planes + (2 * part) * plane + dst;
        bf16_t* lo = planes + (2 * part + 1) * plane + dst;
        for (int d = threadIdx.x; d < AT_HDP; d += blockDim.x) {
            float v = 0.0f;
            if (d < head_dim) {
                v = src[d];
                if (part < 2 && theta > 0.0f) {
                    int i = d < half ? d : d - half;
                    int fi = i < quarter ? i : i - quarter;
                    float inv = exp2f(-(2.0f * fi / (float)half) * log2f(theta));
                    float angle = (float)(i < quarter ? ph : pw) * inv;
                    float s, c;
                    sincosf(angle, &s, &c);
                    float other = d < half ? src[d + half] : src[d - half];
                    v = d < half ? v * c - other * s : v * c + other * s;
                }
            }
            unsigned b = f16_bits(v);
            hi[d] = (bf16_t)b;
            if (ATT_SPLIT) lo[d] = (bf16_t)f16_bits(v - f16_float(b));
        }
    }
}

extern "C" __global__ void __launch_bounds__(32 * AT_WARPS) vis_flash(const bf16_t* planes, const int* seg_lo, const int* seg_hi,
                                                                     bf16_t* out_hi, bf16_t* out_lo, int tokens, int heads,
                                                                     int head_dim, float scale_log2) {
    extern __shared__ __align__(16) unsigned char at_smem[];
    bf16_t* sm = (bf16_t*)at_smem;
    int head = blockIdx.y;
    int q0 = blockIdx.x * AT_BQ;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    const bf16_t* head_base = planes + (size_t)head * tokens * AT_HDP;
    for (int i = tid; i < 2 * AT_BQ * AT_CHUNKS; i += 32 * AT_WARPS) {
        int p = i / (AT_BQ * AT_CHUNKS);
        int rem = i % (AT_BQ * AT_CHUNKS);
        int r = rem / AT_CHUNKS;
        int c = rem % AT_CHUNKS;
        int q = q0 + r;
        const bf16_t* src = head_base + p * plane + (size_t)min(q, tokens - 1) * AT_HDP + c * 8;
        cp_async16_zfill(sm + (p * AT_BQ + r) * AT_STRIDE + c * 8, src, q < tokens);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    unsigned qh[AT_KB][4], ql[AT_KB][4];
#pragma unroll
    for (int kb = 0; kb < AT_KB; kb++) {
        int row = warp * 16 + (lane & 15);
        int col = kb * 16 + (lane >> 4) * 8;
        ldsm_x4(qh[kb], sm + row * AT_STRIDE + col);
        if (ATT_SPLIT) ldsm_x4(ql[kb], sm + (AT_BQ + row) * AT_STRIDE + col);
    }
    __syncthreads();
    int row0 = q0 + warp * 16 + g;
    int row1 = row0 + 8;
    int lo0 = row0 < tokens ? seg_lo[row0] : 0, hi0 = row0 < tokens ? seg_hi[row0] : 0;
    int lo1 = row1 < tokens ? seg_lo[row1] : 0, hi1 = row1 < tokens ? seg_hi[row1] : 0;
    int kstart = seg_lo[q0];
    int kend = seg_hi[min(q0 + AT_BQ, tokens) - 1];
    int tiles = (kend - kstart + AT_BK - 1) / AT_BK;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = sm + buf * AT_STAGE;
        constexpr int KV_LOADS = ATT_SPLIT ? 4 : 2;
        for (int i = tid; i < KV_LOADS * AT_BK * AT_CHUNKS; i += 32 * AT_WARPS) {
            int slot = i / (AT_BK * AT_CHUNKS);
            int p = ATT_SPLIT ? slot : slot * 2;
            int rem = i % (AT_BK * AT_CHUNKS);
            int r = rem / AT_CHUNKS;
            int c = rem % AT_CHUNKS;
            int key = k0 + r;
            const bf16_t* src = head_base + (size_t)(2 + p) * plane + (size_t)min(key, tokens - 1) * AT_HDP + c * 8;
            cp_async16_zfill(st + slot * AT_TILE + r * AT_STRIDE + c * 8, src, key < kend);
        }
    };
    float o[AT_NB][4];
#pragma unroll
    for (int i = 0; i < AT_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    const float NEG = -3.0e38f;
    float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
    if (tiles > 0) load_kv(0, kstart);
    cp_async_commit();
    for (int kt = 0; kt < tiles; kt++) {
        int k0 = kstart + kt * AT_BK;
        if (kt + 1 < tiles) load_kv((kt + 1) & 1, k0 + AT_BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16_t* kh = sm + (kt & 1) * AT_STAGE;
        const bf16_t* kl = kh + AT_TILE;
        const bf16_t* vh = ATT_SPLIT ? kl + AT_TILE : kh + AT_TILE;
        const bf16_t* vl = vh + AT_TILE;
        constexpr int SB = AT_BK / 8;
        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < AT_KB; kb++) {
                unsigned bh[4], bl[4];
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = kr * AT_STRIDE + kb * 16 + ((lane >> 3) & 1) * 8;
                ldsm_x4(bh, kh + at);
                if (ATT_SPLIT) {
                    ldsm_x4(bl, kl + at);
                    mma_f16_hi_lo(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                    mma_f16_hi_lo(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
                } else {
                    mma_f16(s[2 * np], qh[kb], bh[0], bh[1]);
                    mma_f16(s[2 * np + 1], qh[kb], bh[2], bh[3]);
                }
            }
        float mx0 = NEG, mx1 = NEG;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                bool ok = e < 2 ? (key >= lo0 && key < hi0) : (key >= lo1 && key < hi1);
                float v = ok ? s[nb][e] * scale_log2 : NEG;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == NEG ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG ? 0.0f : fast_exp2(s[nb][e] - nm);
                s[nb][e] = p;
                if (e < 2) sum0 += p; else sum1 += p;
            }
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < AT_NB; i++) {
            o[i][0] *= a0;
            o[i][1] *= a0;
            o[i][2] *= a1;
            o[i][3] *= a1;
        }
#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
            ph[0] = pack_f16_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_f16_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_f16_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_f16_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            pl[0] = pack_f16_lo(s[2 * kb][0], s[2 * kb][1]);
            pl[1] = pack_f16_lo(s[2 * kb][2], s[2 * kb][3]);
            pl[2] = pack_f16_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            pl[3] = pack_f16_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
#pragma unroll
            for (int np = 0; np < AT_NB / 2; np++) {
                unsigned bh[4], bl[4];
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = kr * AT_STRIDE + np * 16 + (lane >> 4) * 8;
                ldsm_x4_t(bh, vh + at);
                if (ATT_SPLIT) {
                    ldsm_x4_t(bl, vl + at);
                    mma_f16_hi_lo(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                    mma_f16_hi_lo(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
                } else {
                    mma_f16(o[2 * np], ph, bh[0], bh[1]);
                    mma_f16(o[2 * np + 1], ph, bh[2], bh[3]);
                }
            }
        }
        __syncthreads();
    }
    float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    float i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    size_t width = (size_t)heads * head_dim;
#pragma unroll
    for (int nb = 0; nb < AT_NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (d >= head_dim) continue;
        size_t col = (size_t)head * head_dim + d;
        if (row0 < tokens) store_act2(out_hi, out_lo, row0 * width + col, o[nb][0] * i0, o[nb][1] * i0);
        if (row1 < tokens) store_act2(out_hi, out_lo, row1 * width + col, o[nb][2] * i1, o[nb][3] * i1);
    }
}
