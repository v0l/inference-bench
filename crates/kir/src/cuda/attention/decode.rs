use super::prim::{self, Mask};

pub fn body() -> String {
    [
        r##"extern "C" __global__ void __launch_bounds__(32 * DM_MAX_WARPS) batch_decode_attention(
    const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache, const int* block_table, const int* pos, const int* slots,
    int layer, int qlen, int pages, int splits, float* part_acc, float* part_ml, int window) {
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = DM_BK * HEAD_DIM;
    constexpr int KB = HEAD_DIM / 16;
    constexpr int NB = HEAD_DIM / 8;
    constexpr int SB = DM_BK / 8;
    extern __shared__ __align__(16) unsigned char dm_smem[];
    bf16_t* kvs = (bf16_t*)dm_smem;
    int kv_head = blockIdx.x;
    int group = blockIdx.y;
    int z = blockIdx.z;
    int len = group_len(pos, group, qlen);
    int chunk = decode_chunk(len, splits);
    int key_begin = z * chunk;
    if (key_begin >= len) return;
    int key_end = min(len, key_begin + chunk);
    int mine = (key_end - key_begin + DM_BK - 1) / DM_BK;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int rows = GROUP * qlen;
    int r0 = warp * 16 + g;
    int r1 = r0 + 8;
    int row0 = group * qlen + min(r0, rows - 1) / GROUP;
    int row1 = group * qlen + min(r1, rows - 1) / GROUP;
    int head0 = kv_head * GROUP + r0 % GROUP;
    int head1 = kv_head * GROUP + r1 % GROUP;
    int pos0 = r0 < rows ? pos[row0] : -1;
    int pos1 = r1 < rows ? pos[row1] : -1;
    const int* table = seq_table(block_table, slots, group * qlen, pages);
    float scale_log2 = QUERY_SCALE * 1.4426950408889634f;
    constexpr float LOG2E = 1.4426950408889634f;
    unsigned qh[KB][4];
    unsigned ql[KB][4];
    const float* q0 = qkv + (size_t)row0 * QKV_ROW + head0 * HEAD_DIM;
    const float* q1 = qkv + (size_t)row1 * QKV_ROW + head1 * HEAD_DIM;
#pragma unroll
    for (int kb = 0; kb < KB; kb++) {
        int d = kb * 16 + 2 * t;
        float2 a = pos0 >= 0 ? *(const float2*)&q0[d] : make_float2(0.0f, 0.0f);
        float2 b = pos1 >= 0 ? *(const float2*)&q1[d] : make_float2(0.0f, 0.0f);
        float2 c = pos0 >= 0 ? *(const float2*)&q0[d + 8] : make_float2(0.0f, 0.0f);
        float2 e = pos1 >= 0 ? *(const float2*)&q1[d + 8] : make_float2(0.0f, 0.0f);
        qh[kb][0] = pack_hi(a.x, a.y);
        qh[kb][1] = pack_hi(b.x, b.y);
        qh[kb][2] = pack_hi(c.x, c.y);
        qh[kb][3] = pack_hi(e.x, e.y);
        if (DM_SPLIT || KV_PLANES == 2) {
            ql[kb][0] = pack_lo(a.x, a.y);
            ql[kb][1] = pack_lo(b.x, b.y);
            ql[kb][2] = pack_lo(c.x, c.y);
            ql[kb][3] = pack_lo(e.x, e.y);
        }
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    constexpr int RAW_TILE = 2 * DM_BK * HEAD_DIM;
    constexpr int SCALE_TILE = 2 * DM_BK * 4;
    unsigned char* raw_base = dm_smem;
    auto raw_stage = [&](int buf) { return raw_base + buf * (RAW_TILE + SCALE_TILE); };
    auto load_kv = [&](int buf, int k0) {
        if (kv_q(layer)) {
            unsigned char* raw = raw_stage(buf);
            float* scales = (float*)(raw + RAW_TILE);
            constexpr int RAW_CHUNKS = HEAD_DIM / 16;
            constexpr int TOTAL = 2 * DM_BK * RAW_CHUNKS;
            for (int i = tid; i < TOTAL; i += blockDim.x) {
                int which = i / (DM_BK * RAW_CHUNKS);
                int rem = i % (DM_BK * RAW_CHUNKS);
                int r = rem / RAW_CHUNKS;
                int c = rem % RAW_CHUNKS;
                int tk = k0 + r;
                int at = min(tk, len - 1);
                const bf16_t* cache = which ? v_cache : k_cache;
                const unsigned char* src = (const unsigned char*)kv_row(cache, table, layer, kv_head, at);
                cp_async16_zfill(raw + (which * DM_BK + r) * HEAD_DIM + ((c ^ (r & 7)) << 4), src + c * 16, tk < len);
                if (c == 0) dm_cp_async4(scales + which * DM_BK + r, kv_scale_ptr(const_cast<bf16_t*>(cache), table, layer, kv_head, at));
            }
            return;
        }
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
"##.to_string(),
        prim::load_kv_planes(&prim::TileLoad { table: "table", bound: "len", stride: "blockDim.x", unroll: false }),
        r##"    };
#pragma unroll
    for (int s = 0; s < DM_STAGES - 1; s++) {
        if (s < mine) load_kv(s, key_begin + s * DM_BK);
        cp_async_commit();
    }
    for (int i = 0; i < mine; i++) {
        int ahead = i + DM_STAGES - 1;
        if (ahead < mine) load_kv(ahead % DM_STAGES, key_begin + ahead * DM_BK);
        cp_async_commit();
        cp_async_wait<DM_STAGES - 1>();
        __syncthreads();
        const unsigned char* k_raw = raw_stage(i % DM_STAGES);
        int k0 = key_begin + i * DM_BK;
        const bf16_t* k_hi = kvs + (kv_q(layer) ? 0 : i % DM_STAGES) * 2 * KV_PLANES * PLANE;
        const bf16_t* k_lo = k_hi + PLANE;
        const bf16_t* v_hi = k_hi + KV_PLANES * PLANE;
        const bf16_t* v_lo = v_hi + PLANE;
"##.to_string(),
        prim::zero_scores().to_string(),
        r##""##.to_string(),
        prim::qk_scores("KB", "mma_dec", DEC_K_FRAG),
        r##"        const float* tile_scales = (const float*)(raw_stage(i % DM_STAGES) + RAW_TILE);
        if (kv_q(layer)) {
#pragma unroll
            for (int nb = 0; nb < SB; nb++)
#pragma unroll
                for (int e = 0; e < 4; e++) s[nb][e] *= tile_scales[nb * 8 + 2 * t + (e & 1)];
        }
"##.to_string(),
        prim::online_softmax(&Mask { pos: "kpos", live: "key < key_end && key <= kpos && (window <= 0 || key > kpos - window)" }),
        r##"        if (kv_q(layer)) {
#pragma unroll
            for (int nb = 0; nb < SB; nb++)
#pragma unroll
                for (int e = 0; e < 4; e++) s[nb][e] *= tile_scales[DM_BK + nb * 8 + 2 * t + (e & 1)];
        }
"##.to_string(),
        prim::rescale_acc("NB"),
        r##"#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
"##.to_string(),
        prim::pack_probs("DM_SPLIT || KV_PLANES == 2"),
        r##""##.to_string(),
        prim::pv_accumulate("NB", "mma_dec", DEC_V_FRAG),
        r##"        }
        __syncthreads();
    }
    size_t base0 = ((size_t)row0 * HEADS + head0) * splits + z;
    size_t base1 = ((size_t)row1 * HEADS + head1) * splits + z;
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (pos0 >= 0) *(float2*)&part_acc[base0 * HEAD_DIM + d] = make_float2(o[nb][0], o[nb][1]);
        if (pos1 >= 0) *(float2*)&part_acc[base1 * HEAD_DIM + d] = make_float2(o[nb][2], o[nb][3]);
    }
    if (t == 0) {
        if (pos0 >= 0) *(float2*)&part_ml[base0 * 2] = make_float2(m0, l0);
        if (pos1 >= 0) *(float2*)&part_ml[base1 * 2] = make_float2(m1, l1);
    }
}
"##.to_string(),
    ]
    .concat()
}

const DEC_K_FRAG: &str = r##"                if (kv_q(layer)) {
                    const unsigned char* row0p = k_raw + (np * 16 + g) * HEAD_DIM + ((kb ^ g) << 4) + 2 * t;
                    const unsigned char* row1p = row0p + 8 * HEAD_DIM;
                    bh[0] = e4m3x2_to_f16x2(*(const unsigned short*)row0p);
                    bh[1] = e4m3x2_to_f16x2(*(const unsigned short*)(row0p + 8));
                    bh[2] = e4m3x2_to_f16x2(*(const unsigned short*)row1p);
                    bh[3] = e4m3x2_to_f16x2(*(const unsigned short*)(row1p + 8));
                } else {
                    ldsm_x4(bh, k_hi + at);
                }
"##;

const DEC_V_FRAG: &str = r##"                if (kv_q(layer)) {
                    const unsigned char* v_raw = k_raw + DM_BK * HEAD_DIM;
                    int key = kb * 16 + 2 * t;
                    int dim = np * 16 + g;
                    auto byte_at = [&](int k, int d) -> unsigned { return v_raw[k * HEAD_DIM + (((d >> 4) ^ (k & 7)) << 4) + (d & 15)]; };
                    bh[0] = e4m3x2_to_f16x2(byte_at(key, dim) | (byte_at(key + 1, dim) << 8));
                    bh[1] = e4m3x2_to_f16x2(byte_at(key + 8, dim) | (byte_at(key + 9, dim) << 8));
                    bh[2] = e4m3x2_to_f16x2(byte_at(key, dim + 8) | (byte_at(key + 1, dim + 8) << 8));
                    bh[3] = e4m3x2_to_f16x2(byte_at(key + 8, dim + 8) | (byte_at(key + 9, dim + 8) << 8));
                } else {
                    ldsm_x4_t(bh, v_hi + at);
                }
"##;
