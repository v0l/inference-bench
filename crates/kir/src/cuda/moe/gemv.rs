use super::prim;

fn dot_fp4(k: &str) -> String {
    format!(
        r##"        const unsigned char* wb = (const unsigned char*)w + at * ({k} / 2);
        const unsigned char* sb = s + at * ({k} / 16);
        for (int grp = lane; grp < {k} / 16; grp += 32) {{
            uint2 q = *(const uint2*)(wb + (size_t)grp * 8);
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) {{
                unsigned byte = ((j < 4 ? q.x : q.y) >> (8 * (j & 3))) & 0xffu;
                part += moe_e2m1(byte & 0xfu) * xs[grp * 16 + 2 * j] + moe_e2m1(byte >> 4) * xs[grp * 16 + 2 * j + 1];
            }}
            acc += part * e4m3_to_f32(sb[grp]);
        }}
        acc *= g[expert];
"##
    )
}

fn dot_fp8() -> &'static str {
    r##"        const unsigned char* wb = (const unsigned char*)w + at * K;
        const float* sg = g + (size_t)expert * MOE_FP8_GRID<ROWS, K> + (row / FP8_BLOCK) * MOE_FP8_KB<K>;
        for (int k = lane * 16; k < K; k += 512) {
            uint4 q = *(const uint4*)(wb + k);
            float part = e4m3_dot4(q.x, *(const float4*)(xs + k)) + e4m3_dot4(q.y, *(const float4*)(xs + k + 4)) +
                         e4m3_dot4(q.z, *(const float4*)(xs + k + 8)) + e4m3_dot4(q.w, *(const float4*)(xs + k + 12));
            acc += part * sg[k / FP8_BLOCK];
        }
"##
}

fn dot_bf16() -> &'static str {
    "        const bf16_t* wb = (const bf16_t*)w + at * K;\n        for (int k = lane; k < K; k += 32) acc += bf16_to_f32(wb[k]) * xs[k];\n"
}

fn dot() -> String {
    format!(
        "template <int Q, int K, int ROWS>\n{}",
        prim::device(
            "float",
            "moe_dot",
            "const void* w, const unsigned char* s, const float* g, int expert, int row, const float* xs, int lane",
            &format!(
                "    size_t at = (size_t)expert * ROWS + row;\n    float acc = 0.0f;\n    if (Q == 1) {{\n{}    }} else if (Q == 2) {{\n{}    }} else {{\n{}    }}\n    return warp_sum(acc);\n",
                dot_fp4("K"),
                dot_fp8(),
                dot_bf16()
            )
        )
    )
}

fn dot2_fp4() -> &'static str {
    r##"        const unsigned char* wb = (const unsigned char*)w + at * (K / 2);
        const unsigned char* sb = s + at * (K / 16);
        for (int grp = lane; grp < K / 16; grp += 32) {
            uint2 q = *(const uint2*)(wb + (size_t)grp * 8);
            float v[16];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned byte = ((j < 4 ? q.x : q.y) >> (8 * (j & 3))) & 0xffu;
                v[2 * j] = moe_e2m1(byte & 0xfu);
                v[2 * j + 1] = moe_e2m1(byte >> 4);
            }
            float sc = e4m3_to_f32(sb[grp]);
#pragma unroll
            for (int j = 0; j < MOE_TJ; j++) {
                if (j < live) {
                    float part = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; i++) part += v[i] * xs[j * K + grp * 16 + i];
                    result[j] += part * sc;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < MOE_TJ; j++) result[j] *= g[expert];
"##
}

fn dot2_fp8() -> &'static str {
    r##"        const unsigned char* wb = (const unsigned char*)w + at * K;
        const float* sg = g + (size_t)expert * MOE_FP8_GRID<ROWS, K> + (row / FP8_BLOCK) * MOE_FP8_KB<K>;
        for (int k = lane * 16; k < K; k += 512) {
            uint4 q = *(const uint4*)(wb + k);
            float sc = sg[k / FP8_BLOCK];
#pragma unroll
            for (int j = 0; j < MOE_TJ; j++) {
                if (j < live) {
                    const float* x = xs + j * K + k;
                    float part = e4m3_dot4(q.x, *(const float4*)(x)) + e4m3_dot4(q.y, *(const float4*)(x + 4)) +
                                 e4m3_dot4(q.z, *(const float4*)(x + 8)) + e4m3_dot4(q.w, *(const float4*)(x + 12));
                    result[j] += part * sc;
                }
            }
        }
"##
}

fn dot2_bf16() -> &'static str {
    r##"        const bf16_t* wb = (const bf16_t*)w + at * K;
        for (int k = lane; k < K; k += 32) {
            float wv = bf16_to_f32(wb[k]);
#pragma unroll
            for (int j = 0; j < MOE_TJ; j++) if (j < live) result[j] += wv * xs[j * K + k];
        }
"##
}

fn dot2() -> String {
    format!(
        "template <int Q, int K, int ROWS>\n{}",
        prim::device(
            "void",
            "moe_dot2",
            "const void* w, const unsigned char* s, const float* g, int expert, int row, const float* xs, int lane, int live, float* result",
            &format!(
                "    size_t at = (size_t)expert * ROWS + row;\n#pragma unroll\n    for (int j = 0; j < MOE_TJ; j++) result[j] = 0.0f;\n    if (Q == 1) {{\n{}    }} else if (Q == 2) {{\n{}    }} else {{\n{}    }}\n#pragma unroll\n    for (int j = 0; j < MOE_TJ; j++) result[j] = warp_sum(result[j]);\n",
                dot2_fp4(),
                dot2_fp8(),
                dot2_bf16()
            )
        )
    )
}

const UP_ARGS: &str = "const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug";
const UP_CALL: &str = "gw, gs, gg, uw, us, ug";

fn up_impl() -> String {
    format!(
        "template <int Q>\n{}",
        prim::device(
            "void",
            "moe_up_impl",
            &format!("{UP_ARGS}, const float* xn, const int* idx, float* act"),
            &format!(
                r##"    __shared__ __align__(16) float xs[HIDDEN];
    int pair = blockIdx.x;
    int t = pair / MOE_PAIRS;
    int expert = idx[pair];
    if (expert < 0) return;
    for (int i = threadIdx.x; i < HIDDEN; i += 256) xs[i] = xn[(size_t)t * HIDDEN + i];
    __syncthreads();
{}    int row = blockIdx.y * MOE_WARPS + warp;
    if (row >= MOE_INTER) return;
    float g = moe_dot<Q, HIDDEN, MOE_INTER>(gw, gs, gg, expert, row, xs, lane);
    float u = moe_dot<Q, HIDDEN, MOE_INTER>(uw, us, ug, expert, row, xs, lane);
    if (lane == 0) act[(size_t)pair * MOE_INTER + row] = gated_act(g, u);
"##,
                prim::warp_ids()
            )
        )
    )
}

fn down_impl() -> String {
    format!(
        "template <int Q>\n{}",
        prim::device(
            "void",
            "moe_down_impl",
            "const void* w, const unsigned char* s, const float* g, const float* act, const int* idx, float* out",
            &format!(
                r##"    __shared__ __align__(16) float xs[MOE_INTER];
    int pair = blockIdx.x;
    int expert = idx[pair];
    if (expert < 0) return;
    for (int i = threadIdx.x; i < MOE_INTER; i += 256) xs[i] = act[(size_t)pair * MOE_INTER + i];
    __syncthreads();
{}    int row = blockIdx.y * MOE_WARPS + warp;
    if (row >= HIDDEN) return;
    float v = moe_dot<Q, MOE_INTER, HIDDEN>(w, s, g, expert, row, xs, lane);
    if (lane == 0) out[(size_t)pair * HIDDEN + row] = v;
"##,
                prim::warp_ids()
            )
        )
    )
}

fn group_up_impl() -> String {
    format!(
        "template <int Q>\n{}",
        prim::device(
            "void",
            "moe_group_up_impl",
            &format!("{UP_ARGS}, const float* xn, const int* offsets, const int* sorted, float* act"),
            &format!(
                r##"    __shared__ __align__(16) float xs[MOE_TJ * HIDDEN];
    int expert = blockIdx.x;
    int begin = offsets[expert];
    int end = offsets[expert + 1];
{}    int row = blockIdx.y * MOE_WARPS + warp;
    for (int base = begin; base < end; base += MOE_TJ) {{
        int live = min(MOE_TJ, end - base);
        __syncthreads();
        for (int j = 0; j < live; j++) {{
            int t = sorted[base + j] / MOE_PAIRS;
            for (int i = threadIdx.x; i < HIDDEN; i += 256) xs[j * HIDDEN + i] = xn[(size_t)t * HIDDEN + i];
        }}
        __syncthreads();
        if (row < MOE_INTER) {{
            float gv[MOE_TJ], uv[MOE_TJ];
            moe_dot2<Q, HIDDEN, MOE_INTER>(gw, gs, gg, expert, row, xs, lane, live, gv);
            moe_dot2<Q, HIDDEN, MOE_INTER>(uw, us, ug, expert, row, xs, lane, live, uv);
            if (lane == 0) {{
                for (int j = 0; j < live; j++) act[(size_t)sorted[base + j] * MOE_INTER + row] = gated_act(gv[j], uv[j]);
            }}
        }}
    }}
"##,
                prim::warp_ids()
            )
        )
    )
}

fn group_down_impl() -> String {
    format!(
        "template <int Q>\n{}",
        prim::device(
            "void",
            "moe_group_down_impl",
            "const void* w, const unsigned char* s, const float* g, const float* act, const int* offsets, const int* sorted, float* out",
            &format!(
                r##"    __shared__ __align__(16) float xs[MOE_TJ * MOE_INTER];
    int expert = blockIdx.x;
    int begin = offsets[expert];
    int end = offsets[expert + 1];
{}    int row = blockIdx.y * MOE_WARPS + warp;
    for (int base = begin; base < end; base += MOE_TJ) {{
        int live = min(MOE_TJ, end - base);
        __syncthreads();
        for (int j = 0; j < live; j++) {{
            size_t p = sorted[base + j];
            for (int i = threadIdx.x; i < MOE_INTER; i += 256) xs[j * MOE_INTER + i] = act[p * MOE_INTER + i];
        }}
        __syncthreads();
        if (row < HIDDEN) {{
            float v[MOE_TJ];
            moe_dot2<Q, MOE_INTER, HIDDEN>(w, s, g, expert, row, xs, lane, live, v);
            if (lane == 0) {{
                for (int j = 0; j < live; j++) out[(size_t)sorted[base + j] * HIDDEN + row] = v[j];
            }}
        }}
    }}
"##,
                prim::warp_ids()
            )
        )
    )
}

fn entries(out: &mut String, stem: &str, args: &str, impl_name: &str, call: &str) {
    for (q, name) in prim::QUANTS {
        out.push_str(&prim::delegate(256, &format!("{stem}_{name}"), args, &format!("{impl_name}<{q}>"), call));
    }
}

pub fn gemv() -> String {
    let mut out = String::new();
    out.push_str(prim::fp8_grid());
    out.push_str(&prim::e2m1_to_f32("moe_e2m1"));
    out.push_str(&dot());
    out.push_str(&up_impl());
    out.push_str(&down_impl());
    out
}

pub fn pair_entries() -> String {
    let mut out = String::new();
    entries(&mut out, "moe_up", &format!("{UP_ARGS}, const float* xn, const int* idx, float* act"), "moe_up_impl", &format!("{UP_CALL}, xn, idx, act"));
    entries(&mut out, "moe_down", "const void* w, const unsigned char* s, const float* g, const float* act, const int* idx, float* out", "moe_down_impl", "w, s, g, act, idx, out");
    out
}

pub fn grouped() -> String {
    let mut out = dot2();
    out.push_str(&group_up_impl());
    out.push_str(&group_down_impl());
    out
}

pub fn group_entries() -> String {
    let mut out = String::new();
    entries(&mut out, "moe_group_up", &format!("{UP_ARGS}, const float* xn, const int* offsets, const int* sorted, float* act"), "moe_group_up_impl", &format!("{UP_CALL}, xn, offsets, sorted, act"));
    entries(
        &mut out,
        "moe_group_down",
        "const void* w, const unsigned char* s, const float* g, const float* act, const int* offsets, const int* sorted, float* out",
        "moe_group_down_impl",
        "w, s, g, act, offsets, sorted, out",
    );
    out
}
