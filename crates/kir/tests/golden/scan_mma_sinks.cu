extern "C" __global__ void __launch_bounds__(32 * FM_WARPS) attn_sinks_mma(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache, const int* block_table, const int* pos, const int* slots, bf16_t* out_hi, bf16_t* out_lo, int layer, int pages, int pos_step, int row_step, const float* sinks, int window, float softcap, int tokens) {
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = FM_BK * HEAD_DIM;
    constexpr float LOG2E = 1.4426950408889634f;
    static_assert(!KV_FP8, "a tile scan reads the unquantised cache");
    extern __shared__ __align__(16) unsigned char fm_smem[];
    bf16_t* kvs = (bf16_t*)fm_smem;
    const int head = blockIdx.y;
    const int kv_head = head / GROUP;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * FM_BQ;
    const int base = pos[0];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row0 = q0 + warp * 16 + g;
    const int row1 = row0 + 8;
    const int pos0 = base + row0;
    const int pos1 = base + row1;
    const int* table = seq_table(block_table, slots, 0, pages);
    float sink_l2 = sinks[head] * LOG2E;
    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
#pragma unroll
    for (int kb = 0; kb < FM_KB; kb++) {
        int d = kb * 16 + 2 * t;
        float2 a = row0 < tokens ? *(const float2*)&qkv[(size_t)row0 * QKV_ROW + head * HEAD_DIM + d] : make_float2(0.0f, 0.0f);
        float2 b = row1 < tokens ? *(const float2*)&qkv[(size_t)row1 * QKV_ROW + head * HEAD_DIM + d] : make_float2(0.0f, 0.0f);
        float2 c = row0 < tokens ? *(const float2*)&qkv[(size_t)row0 * QKV_ROW + head * HEAD_DIM + d + 8] : make_float2(0.0f, 0.0f);
        float2 e = row1 < tokens ? *(const float2*)&qkv[(size_t)row1 * QKV_ROW + head * HEAD_DIM + d + 8] : make_float2(0.0f, 0.0f);
        qh[kb][0] = pack_hi(a.x, a.y);
        qh[kb][1] = pack_hi(b.x, b.y);
        qh[kb][2] = pack_hi(c.x, c.y);
        qh[kb][3] = pack_hi(e.x, e.y);
        if (KV_PLANES == 2) {
            ql[kb][0] = pack_lo(a.x, a.y);
            ql[kb][1] = pack_lo(b.x, b.y);
            ql[kb][2] = pack_lo(c.x, c.y);
            ql[kb][3] = pack_lo(e.x, e.y);
        }
    }
    float o[FM_NB][4];
#pragma unroll
    for (int i = 0; i < FM_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = sink_l2, m1 = sink_l2, l0 = 1.0f, l1 = 1.0f;
    const int last = min(q0 + FM_BQ, tokens) - 1;
    const int key_end = base + last + 1;
    const int warp_last = base + q0 + warp * 16 + 15;
    const int tiles = (key_end + FM_BK - 1) / FM_BK;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
        constexpr int ROW_CHUNKS = KV_PLANES * CPR;
        constexpr int TOTAL = 2 * FM_BK * ROW_CHUNKS;
#pragma unroll
        for (int i = tid; i < TOTAL; i += 32 * FM_WARPS) {
            int which = i / (FM_BK * ROW_CHUNKS);
            int rem = i % (FM_BK * ROW_CHUNKS);
            int r = rem / ROW_CHUNKS;
            int c = rem % ROW_CHUNKS;
            int tk = k0 + r;
            const bf16_t* src = kv_row(which ? v_cache : k_cache, table, layer, kv_head, min(tk, key_end - 1));
            bf16_t* dst = st + (which * KV_PLANES + c / CPR) * PLANE + swz<CPR>(r, c % CPR) * 8;
            cp_async16_zfill(dst, src + c * 8, tk < key_end);
        }
    };
    __syncthreads();
    load_kv(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < tiles; kt++) {
        int k0 = kt * FM_BK;
        if (kt + 1 < tiles) load_kv((kt + 1) % FM_STAGES, k0 + FM_BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16_t* k_hi = kvs + (kt % FM_STAGES) * 2 * KV_PLANES * PLANE;
        const bf16_t* k_lo = k_hi + PLANE;
        const bf16_t* v_hi = k_hi + KV_PLANES * PLANE;
        const bf16_t* v_lo = v_hi + PLANE;
        if (k0 > warp_last) {
            __syncthreads();
            continue;
        }
        constexpr int SB = FM_BK / 8;
        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < FM_KB; kb++) {
                unsigned bh[4], bl[4] = {0, 0, 0, 0};
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = swz<CPR>(kr, kb * 2 + ((lane >> 3) & 1)) * 8;
                ldsm_x4(bh, k_hi + at);
                if (KV_PLANES == 2) ldsm_x4(bl, k_lo + at);
                mma_kv(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                mma_kv(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
            }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                int q_pos_i = e < 2 ? pos0 : pos1;
                bool live = key <= q_pos_i && key < key_end;
                float score = ((float)softcap > 0.0f ? (float)softcap * tanhf(s[nb][e] * QUERY_SCALE / (float)softcap) : s[nb][e] * QUERY_SCALE);
                live = live && ((float)window <= 0.0f || (float)key > (float)q_pos_i - (float)window);
                float v = live ? score * LOG2E : NEG_INF;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG_INF ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == NEG_INF ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG_INF ? 0.0f : fast_exp2(s[nb][e] - nm);
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
        for (int i = 0; i < FM_NB; i++) {
            o[i][0] *= a0;
            o[i][1] *= a0;
            o[i][2] *= a1;
            o[i][3] *= a1;
        }
#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
            ph[0] = pack_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            if (KV_PLANES == 2) {
                pl[0] = pack_lo(s[2 * kb][0], s[2 * kb][1]);
                pl[1] = pack_lo(s[2 * kb][2], s[2 * kb][3]);
                pl[2] = pack_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
                pl[3] = pack_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            }
#pragma unroll
            for (int np = 0; np < FM_NB / 2; np++) {
                unsigned bh[4], bl[4] = {0, 0, 0, 0};
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = swz<CPR>(kr, np * 2 + (lane >> 4)) * 8;
                ldsm_x4_t(bh, v_hi + at);
                if (KV_PLANES == 2) ldsm_x4_t(bl, v_lo + at);
                mma_kv(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                mma_kv(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
            }
        }
        __syncthreads();
    }
    float i0 = 1.0f / l0;
    float i1 = 1.0f / l1;
#pragma unroll
    for (int nb = 0; nb < FM_NB; nb++) {
        int d = nb * 8 + 2 * t;
        int at = head * HEAD_DIM + d;
        if (row0 < tokens)
            split_store2(out_hi, out_lo, (size_t)row0 * Q_DIM + at, o[nb][0] * i0 * out_gate(qkv, row0, at),
                         o[nb][1] * i0 * out_gate(qkv, row0, at + 1));
        if (row1 < tokens)
            split_store2(out_hi, out_lo, (size_t)row1 * Q_DIM + at, o[nb][2] * i1 * out_gate(qkv, row1, at),
                         o[nb][3] * i1 * out_gate(qkv, row1, at + 1));
    }
}

