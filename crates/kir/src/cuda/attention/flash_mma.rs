use super::prim::{self, Mask};

pub fn body() -> String {
    [
        r##"__device__ __forceinline__ void fm_cp_async4(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
}

extern "C" __global__ void __launch_bounds__(32 * FM_WARPS) flash_mma(const float* qkv, const float* q_norm, const bf16_t* k_cache,
                                     const bf16_t* v_cache, const int* block_table, const int* pos_ptr, int layer,
                                     bf16_t* out_hi, bf16_t* out_lo, int tokens,
                                     const int* mm_meta, const int* mm_table, int mm_stride, int mm_slot, int window, const int* bidi) {
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = FM_BK * HEAD_DIM;
    static_assert(FM_BQ * HEAD_DIM * 2 <= FM_SMEM, "Q staging fits");
    extern __shared__ __align__(16) unsigned char fm_smem[];
    bf16_t* kvs = (bf16_t*)fm_smem;
    float* qs = (float*)fm_smem;
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = (gridDim.x - 1 - blockIdx.x) * FM_BQ;
    int base = pos_ptr[0];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int row0 = q0 + warp * 16 + g;
    int row1 = row0 + 8;
    int pos0 = base + row0;
    int pos1 = base + row1;
    float scale_log2 = QUERY_SCALE * 1.4426950408889634f;
    constexpr float LOG2E = 1.4426950408889634f;
    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
    constexpr int PAIRS = (ROT_HALF + 31) / 32;
    float* wq = qs + warp * 8 * HEAD_DIM;
    float freq[PAIRS];
#pragma unroll
    for (int i = 0; i < PAIRS; i++) freq[i] = rope_inv_freq(min(lane + 32 * i, ROT_HALF - 1), layer);
#pragma unroll
    for (int half = 0; half < 2; half++) {
        for (int r = 0; r < 8; r++) {
            int row = q0 + warp * 16 + half * 8 + r;
            const float* src = qkv + (size_t)row * QKV_ROW + head * HEAD_DIM;
            float* dst = wq + r * HEAD_DIM;
            float ss = 0.0f;
            for (int d = lane; d < HEAD_DIM; d += 32) {
                float v = row < tokens ? src[d] : 0.0f;
                dst[d] = v;
                ss += v * v;
            }
            float inv = QK_NORM ? rsqrtf(warp_sum(ss) / HEAD_DIM + EPS) : 1.0f;
            __syncwarp();
            if (QK_NORM)
                for (int d = lane; d < HEAD_DIM; d += 32) dst[d] = q_norm[d] * (dst[d] * inv);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < PAIRS; i++) {
                int d = lane + 32 * i;
                if (!NOPE[layer] && d < ROT_HALF) {
                    float x = dst[d], y = dst[d + ROT_HALF];
                    float sn, c;
                    sincosf(rope_position(mm_meta, mm_table, mm_stride, mm_slot, base + row, d) * freq[i], &sn, &c);
                    c *= ROPE_MSCALE;
                    sn *= ROPE_MSCALE;
                    dst[d] = x * c - y * sn;
                    dst[d + ROT_HALF] = y * c + x * sn;
                }
            }
            __syncwarp();
        }
#pragma unroll
        for (int kb = 0; kb < FM_KB; kb++) {
            int d = kb * 16 + 2 * t;
            float2 v0 = *(const float2*)&wq[g * HEAD_DIM + d];
            float2 v1 = *(const float2*)&wq[g * HEAD_DIM + d + 8];
            qh[kb][half] = pack_hi(v0.x, v0.y);
            qh[kb][2 + half] = pack_hi(v1.x, v1.y);
            if (KV_PLANES == 2) {
                ql[kb][half] = pack_lo(v0.x, v0.y);
                ql[kb][2 + half] = pack_lo(v1.x, v1.y);
            }
        }
        __syncwarp();
    }
    float o[FM_NB][4];
#pragma unroll
    for (int i = 0; i < FM_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    int last = min(q0 + FM_BQ, tokens) - 1;
    int key_end = base + last + 1;
    int warp_last = base + q0 + warp * 16 + 15;
    int span0 = span_end(mm_meta, bidi, mm_stride, mm_slot, pos0);
    int span1 = span_end(mm_meta, bidi, mm_stride, mm_slot, pos1);
    if (BIDI) {
        key_end = min(max(key_end, span_end(mm_meta, bidi, mm_stride, mm_slot, base + last) + 1), base + tokens);
        warp_last = max(warp_last, span_end(mm_meta, bidi, mm_stride, mm_slot, min(warp_last, base + tokens - 1)));
    }
    int tiles = (key_end + FM_BK - 1) / FM_BK;
    constexpr int PLANE_BYTES = PLANE * 2;
    float* kv_scales = (float*)(fm_smem + FM_STAGES * 2 * PLANE_BYTES);
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
        if (kv_q(layer)) {
            unsigned char* raw_st = (unsigned char*)st + PLANE_BYTES / 2;
            constexpr int RAW_CHUNKS = HEAD_DIM / 16;
            constexpr int TOTAL = 2 * FM_BK * RAW_CHUNKS;
#pragma unroll
            for (int i = tid; i < TOTAL; i += 32 * FM_WARPS) {
                int which = i / (FM_BK * RAW_CHUNKS);
                int rem = i % (FM_BK * RAW_CHUNKS);
                int r = rem / RAW_CHUNKS;
                int c = rem % RAW_CHUNKS;
                int tk = k0 + r;
                int at = min(tk, key_end - 1);
                const bf16_t* cache = which ? v_cache : k_cache;
                const unsigned char* src = (const unsigned char*)kv_row(cache, block_table, layer, kv_head, at);
                cp_async16_zfill(raw_st + which * PLANE_BYTES + r * HEAD_DIM + c * 16, src + c * 16, tk < key_end);
                if (c == 0) fm_cp_async4(kv_scales + buf * 2 * FM_BK + which * FM_BK + r, kv_scale_ptr(const_cast<bf16_t*>(cache), block_table, layer, kv_head, at));
            }
            return;
        }
"##.to_string(),
        prim::load_kv_planes(&prim::TileLoad { table: "block_table", bound: "key_end", stride: "32 * FM_WARPS", unroll: true }),
        r##"    };
    auto convert_kv = [&](int buf) {
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
        const unsigned char* raw_st = (const unsigned char*)st + PLANE_BYTES / 2;
        const float* sc = kv_scales + buf * 2 * FM_BK;
        constexpr int ITEMS = (2 * FM_BK * CPR + 32 * FM_WARPS - 1) / (32 * FM_WARPS);
        uint4 out[ITEMS];
#pragma unroll
        for (int j = 0; j < ITEMS; j++) {
            int i = tid + j * 32 * FM_WARPS;
            if (i < 2 * FM_BK * CPR) {
                int which = i / (FM_BK * CPR);
                int rem = i % (FM_BK * CPR);
                int r = rem / CPR;
                int c = rem % CPR;
                float s = sc[which * FM_BK + r];
                uint2 raw = *(const uint2*)(raw_st + which * PLANE_BYTES + r * HEAD_DIM + c * 8);
                unsigned w[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    unsigned two = ((e < 2 ? raw.x : raw.y) >> (16 * (e & 1))) & 0xffffu;
                    w[e] = attn_bits(e4m3_to_f32(two & 0xffu) * s) | (attn_bits(e4m3_to_f32(two >> 8) * s) << 16);
                }
                out[j] = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < ITEMS; j++) {
            int i = tid + j * 32 * FM_WARPS;
            if (i < 2 * FM_BK * CPR) {
                int which = i / (FM_BK * CPR);
                int rem = i % (FM_BK * CPR);
                *(uint4*)(st + which * PLANE + swz<CPR>(rem / CPR, rem % CPR) * 8) = out[j];
            }
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
        if (kv_q(layer)) {
            convert_kv(kt % FM_STAGES);
            __syncthreads();
        }
        const bf16_t* k_hi = kvs + (kt % FM_STAGES) * 2 * KV_PLANES * PLANE;
        const bf16_t* k_lo = k_hi + PLANE;
        const bf16_t* v_hi = k_hi + KV_PLANES * PLANE;
        const bf16_t* v_lo = v_hi + PLANE;
        if (k0 > warp_last || (window > 0 && k0 + FM_BK - 1 <= base + q0 + warp * 16 - window)) {
            __syncthreads();
            continue;
        }
        constexpr int SB = FM_BK / 8;
"##.to_string(),
        prim::zero_scores().to_string(),
        r##""##.to_string(),
        prim::qk_scores("FM_KB", "mma_kv", "                ldsm_x4(bh, k_hi + at);\n"),
        r##""##.to_string(),
        prim::online_softmax(&Mask { pos: "pos", live: "(key <= pos || key <= (e < 2 ? span0 : span1)) && key < key_end && (window <= 0 || key > pos - window)" }),
        r##""##.to_string(),
        prim::rescale_acc("FM_NB"),
        r##"#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
"##.to_string(),
        prim::pack_probs("KV_PLANES == 2"),
        r##""##.to_string(),
        prim::pv_accumulate("FM_NB", "mma_kv", "                ldsm_x4_t(bh, v_hi + at);\n"),
        r##"        }
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
"##.to_string(),
    ]
    .concat()
}
