use super::prim;

fn dot_fp4() -> &'static str {
    "        const unsigned char* wb = (const unsigned char*)w + at * (K / 2);\n        const unsigned char* sb = s + at * (K / 16);\n        float col[1] = {0.0f};\n        moe_rows_fp4<K, 1>(wb, sb, xs, lane, 1, col);\n        acc = col[0] * g[expert];\n"
}

fn dot_mx() -> &'static str {
    "        constexpr int KP = (K + 127) / 128 * 128;\n        float col[1] = {0.0f};\n        moe_rows_mx<K, 1>((const unsigned char*)w + at * (KP / 2), s + at * (KP / MOE_BLOCK), xs, lane, 1, col);\n        acc = col[0];\n"
}

fn dot_fp8() -> &'static str {
    "        const float* sg = g + (size_t)expert * MOE_FP8_GRID<ROWS, K> + (row / FP8_BLOCK) * MOE_FP8_KB<K>;\n        acc = dot_fp8_scaled<K>((const unsigned char*)w + at * K, sg, xs, lane);\n"
}

fn dot_bf16() -> &'static str {
    "        acc = dot_bf16<K>(w, (int)at, xs, lane);\n"
}

fn dot() -> String {
    format!(
        "template <int Q, int K, int ROWS>\n{}",
        prim::device(
            "float",
            "moe_dot",
            "const void* w, const unsigned char* s, const float* g, int expert, int row, const float* xs, int lane",
            &format!(
                "    size_t at = (size_t)expert * ROWS + row;\n    float acc = 0.0f;\n    if (Q == 1) {{\n{}    }} else if (Q == 3) {{\n{}    }} else if (Q == 2) {{\n{}    }} else {{\n{}    }}\n    return warp_sum(acc);\n",
                dot_fp4(),
                dot_mx(),
                dot_fp8(),
                dot_bf16()
            )
        )
    )
}

fn dot2_fp4() -> &'static str {
    "        moe_rows_fp4<K, MOE_TJ>((const unsigned char*)w + at * (K / 2), s + at * (K / 16), xs, lane, live, result);\n#pragma unroll\n        for (int j = 0; j < MOE_TJ; j++) result[j] *= g[expert];\n"
}

fn dot2_mx() -> &'static str {
    "        constexpr int KP = (K + 127) / 128 * 128;\n        moe_rows_mx<K, MOE_TJ>((const unsigned char*)w + at * (KP / 2), s + at * (KP / MOE_BLOCK), xs, lane, live, result);\n"
}

fn dot2_fp8() -> &'static str {
    "        const float* sg = g + (size_t)expert * MOE_FP8_GRID<ROWS, K> + (row / FP8_BLOCK) * MOE_FP8_KB<K>;\n        moe_rows_fp8<K, MOE_TJ>((const unsigned char*)w + at * K, sg, xs, lane, live, result);\n"
}

fn dot2_bf16() -> &'static str {
    "        moe_rows_bf16<K, MOE_TJ>((const bf16_t*)w + at * K, xs, lane, live, result);\n"
}

fn dot2() -> String {
    format!(
        "template <int Q, int K, int ROWS>\n{}",
        prim::device(
            "void",
            "moe_dot2",
            "const void* w, const unsigned char* s, const float* g, int expert, int row, const float* xs, int lane, int live, float* result",
            &format!(
                "    size_t at = (size_t)expert * ROWS + row;\n#pragma unroll\n    for (int j = 0; j < MOE_TJ; j++) result[j] = 0.0f;\n    if (Q == 1) {{\n{}    }} else if (Q == 3) {{\n{}    }} else if (Q == 2) {{\n{}    }} else {{\n{}    }}\n#pragma unroll\n    for (int j = 0; j < MOE_TJ; j++) result[j] = warp_sum(result[j]);\n",
                dot2_fp4(),
                dot2_mx(),
                dot2_fp8(),
                dot2_bf16()
            )
        )
    )
}

const UP_ARGS: &str = "const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug, const float* eb";
const UP_CALL: &str = "gw, gs, gg, uw, us, ug, eb";

const TABLE_RESOLVE: &str = "    int eid = expert;\n    expert = tbl[expert];\n    if (expert == 0) return;\n    expert -= 1;\n";

fn by_id(body: String, table: bool) -> String {
    if table { body.replace("eb[(size_t)expert", "eb[(size_t)eid").replace("eb[((size_t)expert", "eb[((size_t)eid").replace("db[(size_t)expert", "db[(size_t)eid") } else { body }
}

fn up_impl(table: bool) -> String {
    let (name, extra, resolve) = if table { ("moe_up_tbl_impl", ", const int* tbl", TABLE_RESOLVE) } else { ("moe_up_impl", "", "") };
    format!(
        "template <int Q>\n{}",
        prim::device(
            "void",
            name,
            &format!("{UP_ARGS}, const float* xn, const int* idx, float* act{extra}"),
            &by_id(format!(
                r##"    __shared__ __align__(16) float xs[HIDDEN];
    int pair = blockIdx.x;
    int t = MOE_INPUT_SCALED ? pair : pair / MOE_PAIRS;
    int expert = idx[pair];
    if (expert < 0) return;
{}    for (int i = threadIdx.x; i < HIDDEN; i += 256) xs[i] = xn[(size_t)t * HIDDEN + i];
    __syncthreads();
{}    int row = blockIdx.y * MOE_WARPS + warp;
    if (row >= MOE_INTER) return;
    float g = MOE_MAPPED ? 0.0f : moe_dot<Q, HIDDEN, MOE_INTER>(gw, gs, gg, expert, row, xs, lane);
    float u = moe_dot<Q, HIDDEN, MOE_INTER>(uw, us, ug, expert, row, xs, lane);
    if (MOE_EBIAS) {{
        g += eb[(size_t)expert * 2 * MOE_INTER + row];
        u += eb[((size_t)expert * 2 + 1) * MOE_INTER + row];
    }}
    if (lane == 0) act[(size_t)pair * MOE_INTER + row] = MOE_MAPPED ? mapped_act(u) : gated_act(g, u);
"##,
                resolve,
                prim::warp_ids()
            ), table)
        )
    )
}

const DOWN_ARGS: &str = "const void* w, const unsigned char* s, const float* g, const float* db, const float* act, const int* idx, float* out";

fn down_impl(table: bool) -> String {
    let (name, args, resolve) = if table { ("moe_down_tbl_impl", format!("{DOWN_ARGS}, const int* tbl"), TABLE_RESOLVE) } else { ("moe_down_impl", DOWN_ARGS.to_string(), "") };
    format!(
        "template <int Q>\n{}",
        prim::device(
            "void",
            name,
            &args,
            &by_id(format!(
                r##"    __shared__ __align__(16) float xs[MOE_INTER];
    int pair = blockIdx.x;
    int expert = idx[pair];
    if (expert < 0) return;
{}    for (int i = threadIdx.x; i < MOE_INTER; i += 256) xs[i] = act[(size_t)pair * MOE_INTER + i];
    __syncthreads();
{}    int row = blockIdx.y * MOE_WARPS + warp;
    if (row >= HIDDEN) return;
    float v = moe_dot<Q, MOE_INTER, HIDDEN>(w, s, g, expert, row, xs, lane);
    if (MOE_EBIAS) v += db[(size_t)expert * HIDDEN + row];
    if (lane == 0) out[(size_t)pair * HIDDEN + row] = v;
"##,
                resolve,
                prim::warp_ids()
            ), table)
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
            int t = MOE_INPUT_SCALED ? sorted[base + j] : sorted[base + j] / MOE_PAIRS;
            for (int i = threadIdx.x; i < HIDDEN; i += 256) xs[j * HIDDEN + i] = xn[(size_t)t * HIDDEN + i];
        }}
        __syncthreads();
        if (row < MOE_INTER) {{
            float gv[MOE_TJ], uv[MOE_TJ];
            if (!MOE_MAPPED) moe_dot2<Q, HIDDEN, MOE_INTER>(gw, gs, gg, expert, row, xs, lane, live, gv);
            moe_dot2<Q, HIDDEN, MOE_INTER>(uw, us, ug, expert, row, xs, lane, live, uv);
            if (MOE_EBIAS) {{
                for (int j = 0; j < live; j++) {{
                    gv[j] += eb[(size_t)expert * 2 * MOE_INTER + row];
                    uv[j] += eb[((size_t)expert * 2 + 1) * MOE_INTER + row];
                }}
            }}
            if (lane == 0) {{
                for (int j = 0; j < live; j++) act[(size_t)sorted[base + j] * MOE_INTER + row] = MOE_MAPPED ? mapped_act(uv[j]) : gated_act(gv[j], uv[j]);
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
            "const void* w, const unsigned char* s, const float* g, const float* db, const float* act, const int* offsets, const int* sorted, float* out",
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
            if (MOE_EBIAS) {{
                for (int j = 0; j < live; j++) v[j] += db[(size_t)expert * HIDDEN + row];
            }}
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
        out.push_str(&prim::delegate(crate::moe::EXPERT_THREADS, &format!("{stem}_{name}"), args, &format!("{impl_name}<{q}>"), call));
    }
}

pub fn gemv() -> String {
    let mut out = String::new();
    out.push_str(&prim::fp8_grid());
    out.push_str(&prim::e2m1_to_f32("moe_e2m1"));
    out.push_str(&prim::multi_row_dot("moe_rows_fp4", prim::RowFormat::Fp4Rows { decode: "moe_e2m1" }, prim::RowLoad { unroll: 0, stream: false }));
    out.push_str(&prim::multi_row_dot("moe_rows_mx", prim::RowFormat::Mx { decode: "moe_e2m1", block: "MOE_BLOCK" }, prim::RowLoad { unroll: 0, stream: false }));
    out.push_str(&dot());
    out.push_str(&up_impl(false));
    out.push_str(&down_impl(false));
    out
}

pub fn table_gemv() -> String {
    let mut out = up_impl(true);
    out.push_str(&down_impl(true));
    out
}

pub fn table_entries() -> String {
    let mut out = String::new();
    entries(&mut out, "moe_up_t", &format!("{UP_ARGS}, const float* xn, const int* idx, float* act, const int* tbl"), "moe_up_tbl_impl", &format!("{UP_CALL}, xn, idx, act, tbl"));
    entries(&mut out, "moe_down_t", &format!("{DOWN_ARGS}, const int* tbl"), "moe_down_tbl_impl", "w, s, g, db, act, idx, out, tbl");
    out
}

pub fn pair_entries() -> String {
    let mut out = String::new();
    entries(&mut out, "moe_up", &format!("{UP_ARGS}, const float* xn, const int* idx, float* act"), "moe_up_impl", &format!("{UP_CALL}, xn, idx, act"));
    entries(&mut out, "moe_down", "const void* w, const unsigned char* s, const float* g, const float* db, const float* act, const int* idx, float* out", "moe_down_impl", "w, s, g, db, act, idx, out");
    out
}

pub fn grouped() -> String {
    let mut out = prim::multi_row_dot("moe_rows_fp8", prim::RowFormat::Fp8, prim::RowLoad { unroll: 0, stream: false });
    out.push_str(&prim::multi_row_dot("moe_rows_bf16", prim::RowFormat::Bf16, prim::RowLoad { unroll: 8, stream: true }));
    out.push_str(&dot2());
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
        "const void* w, const unsigned char* s, const float* g, const float* db, const float* act, const int* offsets, const int* sorted, float* out",
        "moe_group_down_impl",
        "w, s, g, db, act, offsets, sorted, out",
    );
    out
}
