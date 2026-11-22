use super::prim::{self, Scores};

pub struct TileScan<'a> {
    pub name: &'a str,
    pub sig: &'a str,
    pub lines: &'a [String],
    pub value: &'a str,
    pub sink: Option<&'a str>,
}

pub fn decode(s: &TileScan) -> String {
    let body: String = s.lines.iter().map(|l| format!("                {l}\n")).collect();
    let binds = format!(
        "                int q_pos_i = pos_row;
                int head = kv_head * GROUP + min(e & 2 ? r1 : r0, GROUP - 1);
                bool live = key <= q_pos_i && key < key_end && (e & 2 ? r1 : r0) < GROUP;
{body}"
    );
    let score = format!("{} * LOG2E", s.value);
    let (init_m, init_l) = match s.sink {
        Some(_) => ("sink_l2", "1.0f"),
        None => ("NEG_INF", "0.0f"),
    };
    let sink_decl = match s.sink {
        Some(n) => format!("    float sink_l2 = {n}[kv_head * GROUP + min(g, GROUP - 1)] * LOG2E;\n    float sink_l2b = {n}[kv_head * GROUP + min(g + 8, GROUP - 1)] * LOG2E;\n"),
        None => String::new(),
    };
    let (m1_init, l1_init) = match s.sink {
        Some(_) => ("sink_l2b", "1.0f"),
        None => ("NEG_INF", "0.0f"),
    };
    let _ = init_l;
    [
        format!(
            r##"extern "C" __global__ void __launch_bounds__(128) {name}_mma_dec({sig}) {{
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = FM_BK * HEAD_DIM;
    constexpr float LOG2E = 1.4426950408889634f;
    static_assert(!KV_FP8 && GROUP <= 16, "a tile decode scan reads the unquantised cache and one group of heads per tile");
    extern __shared__ __align__(16) unsigned char fm_smem[];
    bf16_t* kvs = (bf16_t*)fm_smem;
    const int kv_head = blockIdx.x;
    const int row = blockIdx.y;
    const int pos_row = pos[row * pos_step] + row * row_step;
    const int key_end = pos_row + 1;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int r0 = g;
    const int r1 = g + 8;
    const int head0 = kv_head * GROUP + min(r0, GROUP - 1);
    const int head1 = kv_head * GROUP + min(r1, GROUP - 1);
    const int* table = seq_table(block_table, slots, row, pages);
{sink_decl}    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
#pragma unroll
    for (int kb = 0; kb < FM_KB; kb++) {{
        int d = kb * 16 + 2 * t;
        float2 a = r0 < GROUP ? *(const float2*)&qkv[(size_t)row * QKV_ROW + head0 * HEAD_DIM + d] : make_float2(0.0f, 0.0f);
        float2 b = r1 < GROUP ? *(const float2*)&qkv[(size_t)row * QKV_ROW + head1 * HEAD_DIM + d] : make_float2(0.0f, 0.0f);
        float2 c = r0 < GROUP ? *(const float2*)&qkv[(size_t)row * QKV_ROW + head0 * HEAD_DIM + d + 8] : make_float2(0.0f, 0.0f);
        float2 e = r1 < GROUP ? *(const float2*)&qkv[(size_t)row * QKV_ROW + head1 * HEAD_DIM + d + 8] : make_float2(0.0f, 0.0f);
        qh[kb][0] = pack_hi(a.x, a.y);
        qh[kb][1] = pack_hi(b.x, b.y);
        qh[kb][2] = pack_hi(c.x, c.y);
        qh[kb][3] = pack_hi(e.x, e.y);
        if (KV_PLANES == 2) {{
            ql[kb][0] = pack_lo(a.x, a.y);
            ql[kb][1] = pack_lo(b.x, b.y);
            ql[kb][2] = pack_lo(c.x, c.y);
            ql[kb][3] = pack_lo(e.x, e.y);
        }}
    }}
    float o[FM_NB][4];
#pragma unroll
    for (int i = 0; i < FM_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = {init_m}, m1 = {m1_init}, l0 = {init_l}, l1 = {l1_init};
    const int tiles = (key_end + FM_BK - 1) / FM_BK;
    auto load_kv = [&](int buf, int k0) {{
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
"##,
            name = s.name,
            sig = s.sig,
            init_l = init_l,
        ),
        prim::load_kv_planes(&prim::TileLoad { table: "table", bound: "key_end", stride: "128", unroll: false }).replace("DM_BK", "FM_BK"),
        r##"    };
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
        if (warp != 0) {
            __syncthreads();
            continue;
        }
        constexpr int SB = FM_BK / 8;
"##
        .to_string(),
        prim::zero_scores().to_string(),
        prim::qk_scores("FM_KB", "mma_kv", "                ldsm_x4(bh, k_hi + at);\n"),
        prim::online_softmax_with(&Scores { neg: "NEG_INF", binds: &binds, live: "live", score: &score }),
        prim::rescale_acc("i", "FM_NB"),
        r##"#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
"##
        .to_string(),
        prim::pack_probs("KV_PLANES == 2"),
        prim::pv_accumulate("FM_NB", "mma_kv", "                ldsm_x4_t(bh, v_hi + at);\n"),
        r##"        }
        __syncthreads();
    }
    if (warp != 0) return;
    float i0 = 1.0f / l0;
    float i1 = 1.0f / l1;
#pragma unroll
    for (int nb = 0; nb < FM_NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (r0 < GROUP) {
            int at = head0 * HEAD_DIM + d;
            split_store2(out_hi, out_lo, (size_t)row * Q_DIM + at, o[nb][0] * i0 * out_gate(qkv, row, at), o[nb][1] * i0 * out_gate(qkv, row, at + 1));
        }
        if (r1 < GROUP) {
            int at = head1 * HEAD_DIM + d;
            split_store2(out_hi, out_lo, (size_t)row * Q_DIM + at, o[nb][2] * i1 * out_gate(qkv, row, at), o[nb][3] * i1 * out_gate(qkv, row, at + 1));
        }
    }
}

"##
        .to_string(),
    ]
    .concat()
}

pub fn prefill(s: &TileScan) -> String {
    let body: String = s.lines.iter().map(|l| format!("                {l}\n")).collect();
    let binds = format!(
        "                int q_pos_i = e < 2 ? pos0 : pos1;
                bool live = key <= q_pos_i && key < key_end;
{body}"
    );
    let score = format!("{} * LOG2E", s.value);
    let (init_m, init_l) = match s.sink {
        Some(_) => ("sink_l2", "1.0f"),
        None => ("NEG_INF", "0.0f"),
    };
    let sink_decl = match s.sink {
        Some(n) => format!("    float sink_l2 = {n}[head] * LOG2E;\n"),
        None => String::new(),
    };
    [
        format!(
            r##"extern "C" __global__ void __launch_bounds__(32 * FM_WARPS) {name}_mma({sig}, int tokens) {{
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
{sink_decl}    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
#pragma unroll
    for (int kb = 0; kb < FM_KB; kb++) {{
        int d = kb * 16 + 2 * t;
        float2 a = row0 < tokens ? *(const float2*)&qkv[(size_t)row0 * QKV_ROW + head * HEAD_DIM + d] : make_float2(0.0f, 0.0f);
        float2 b = row1 < tokens ? *(const float2*)&qkv[(size_t)row1 * QKV_ROW + head * HEAD_DIM + d] : make_float2(0.0f, 0.0f);
        float2 c = row0 < tokens ? *(const float2*)&qkv[(size_t)row0 * QKV_ROW + head * HEAD_DIM + d + 8] : make_float2(0.0f, 0.0f);
        float2 e = row1 < tokens ? *(const float2*)&qkv[(size_t)row1 * QKV_ROW + head * HEAD_DIM + d + 8] : make_float2(0.0f, 0.0f);
        qh[kb][0] = pack_hi(a.x, a.y);
        qh[kb][1] = pack_hi(b.x, b.y);
        qh[kb][2] = pack_hi(c.x, c.y);
        qh[kb][3] = pack_hi(e.x, e.y);
        if (KV_PLANES == 2) {{
            ql[kb][0] = pack_lo(a.x, a.y);
            ql[kb][1] = pack_lo(b.x, b.y);
            ql[kb][2] = pack_lo(c.x, c.y);
            ql[kb][3] = pack_lo(e.x, e.y);
        }}
    }}
    float o[FM_NB][4];
#pragma unroll
    for (int i = 0; i < FM_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = {init_m}, m1 = {init_m}, l0 = {init_l}, l1 = {init_l};
    const int last = min(q0 + FM_BQ, tokens) - 1;
    const int key_end = base + last + 1;
    const int warp_last = base + q0 + warp * 16 + 15;
    const int tiles = (key_end + FM_BK - 1) / FM_BK;
    auto load_kv = [&](int buf, int k0) {{
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
"##,
            name = s.name,
            sig = s.sig,
        ),
        prim::load_kv_planes(&prim::TileLoad { table: "table", bound: "key_end", stride: "32 * FM_WARPS", unroll: true }),
        r##"    };
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
"##
        .to_string(),
        prim::zero_scores().to_string(),
        prim::qk_scores("FM_KB", "mma_kv", "                ldsm_x4(bh, k_hi + at);\n"),
        prim::online_softmax_with(&Scores { neg: "NEG_INF", binds: &binds, live: "live", score: &score }),
        prim::rescale_acc("i", "FM_NB"),
        r##"#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
"##
        .to_string(),
        prim::pack_probs("KV_PLANES == 2"),
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

"##
        .to_string(),
    ]
    .concat()
}
