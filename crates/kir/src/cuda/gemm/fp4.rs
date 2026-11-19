use super::prim;

fn fp4_rows_dot_body() -> String {
    [
        r##"    constexpr int KG = K / FP4_GROUP;
    const unsigned char* wb = (const unsigned char*)w;
    const unsigned char* sb = scales + FP4_HEADER;
    float acc[ROWS];
"##.to_string(),
        prim::unroll_nest(4, &[("r", "ROWS")], false, "acc[r] = 0.0f;"),
        r##"    for (int g0 = 0; g0 < KG; g0 += 32 * U) {
        uint2 q[ROWS][U];
        unsigned char sc[ROWS][U];
"##.to_string(),
        prim::unroll_nest(8, &[("r", "ROWS"), ("u", "U")], true, r##"int g = g0 + u * 32 + lane;
if (g < KG) {
    size_t row = (size_t)(row0 + r * stride);
    q[r][u] = load_stream8(wb + row * (K / 2) + (size_t)g * 8);
    sc[r][u] = __ldg(sb + row * KG + g);
} else {
    q[r][u] = make_uint2(0, 0);
    sc[r][u] = 0;
}
"##),
        r##"#pragma unroll
        for (int u = 0; u < U; u++) {
            int g = g0 + u * 32 + lane;
            if (g >= KG) break;
            float xv[16];
"##.to_string(),
        prim::unroll_nest(12, &[("i", "16")], false, "xv[i] = xt[i * KG + g];"),
        prim::unroll_nest(12, &[("r", "ROWS")], true, r##"float part = 0.0f;
#pragma unroll
for (int i = 0; i < 8; i++) {
    unsigned byte = ((i < 4 ? q[r][u].x : q[r][u].y) >> (8 * (i & 3))) & 0xffu;
    part += e2m1_to_f32(byte & 15u) * xv[2 * i];
    part += e2m1_to_f32(byte >> 4) * xv[2 * i + 1];
}
acc[r] += part * e4m3_to_f32(sc[r][u]);
"##),
        r##"        }
    }
    float inv = fp4_inv_global(scales);
"##.to_string(),
        prim::unroll_nest(4, &[("r", "ROWS")], false, "result[r] = warp_sum(acc[r]) * inv;"),
    ]
    .concat()
}

pub fn body() -> String {
    [
        [("FP4_GROUP", "16"), ("FP4_HEADER", "16"), ("FP4_GEMV_THREADS", "512")].iter().map(|(n, v)| prim::constexpr_int(n, v)).collect::<String>() + "\n",
        prim::device("float", "fp4_inv_global", "const unsigned char* scales", "    return __ldg((const float*)scales);\n"),
        r##""##.to_string(),
        prim::e2m1_to_f32("e2m1_to_f32"),
        prim::device("int", "fp4_t", "int k, int kg", "    return (k & (FP4_GROUP - 1)) * kg + (k >> 4);\n"),
        prim::device_t("int K, int ROWS, int U", "void", "fp4_rows_dot", "const void* w, const unsigned char* scales, int row0, int stride, const float* xt, int lane, float* result", &fp4_rows_dot_body()),
        prim::device("void", "cp_async4", "void* smem, const void* gmem", "    unsigned s = (unsigned)__cvta_generic_to_shared(smem);\n    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 4;\" :: \"r\"(s), \"l\"(gmem));\n"),
        prim::device("unsigned", "fp4_pair_bf16", "unsigned byte, float scale", &format!("{}{}    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);\n", "    float lo = e2m1_to_f32(byte & 15u) * scale;\n", "    float hi = e2m1_to_f32(byte >> 4) * scale;\n")),
        [("F4W_BK", "64"), ("F4W_ROW", "48")].iter().map(|(n, v)| prim::constexpr_int(n, v)).collect::<String>() + "\n",
        prim::dec_struct_open("DecFp4"),
        prim::static_const("int", "ACC", "ACC_DIRECT"),
        prim::static_const("bool", "LDSM", "false"),
        prim::static_const("int", "WARPS", "FP4_GEMV_THREADS / 32"),
        prim::static_const_t("int BK", "int", "wrow", "F4W_ROW"),
        prim::static_const_t("int BK", "int", "w_chunks", "3"),
        r##"    template <int W> static __device__ __forceinline__ int x_at(int i) { return fp4_t(i, W / FP4_GROUP); }
    template <int K, int ROWS>
    static __device__ __forceinline__ void rows(const void* w, const void* s, int row0, int stride, const float* x, int lane, float* out) {
"##.to_string(),
        format!("        {}\n", prim::cp_async_call("fp4_rows_dot<K, ROWS, 4 / ROWS>", &["w", "(const unsigned char*)s", "row0", "stride", "x", "lane", "out"])),
        r##"    }
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void* s, int r, int n, int kt, int ck) {
        static_assert(BK == F4W_BK, "nvfp4 tiles are 64 wide");
"##.to_string(),
        prim::split_copy(
            "ck < 2",
            &prim::cp_async_call("cp_async16", &["ws + r * F4W_ROW + ck * 16", "(const unsigned char*)w + (size_t)n * (K / 2) + kt * (BK / 2) + ck * 16"]),
            &prim::cp_async_call("cp_async4", &["ws + r * F4W_ROW + 32", "(const unsigned char*)s + FP4_HEADER + (size_t)n * (K / FP4_GROUP) + kt * (BK / FP4_GROUP)"]),
        )
        .lines()
        .map(|l| format!("        {l}\n"))
        .collect::<String>(),
        r##"    }
    static __device__ __forceinline__ void frag(unsigned* bw, const unsigned char* p, int ks, int t) {
        float sc = e4m3_to_f32(p[32 + ks]);
"##.to_string(),
        ["ks * 8 + t", "ks * 8 + 4 + t"].iter().enumerate().map(|(i, e)| format!("        bw[{i}] = fp4_pair_bf16(p[{e}], sc);\n")).collect::<String>(),
        r##"    }
    template <int MI, int NJ> static __device__ __forceinline__ void finish(float (&acc)[MI][NJ][4], const void* s) {
        float inv = fp4_inv_global((const unsigned char*)s);
"##.to_string(),
        prim::unroll_nest(8, &[("i", "MI"), ("j", "NJ"), ("e", "4")], false, "acc[i][j][e] *= inv;"),
        r##"    }
};

__device__ __forceinline__ void load_normed_t(const float* x, const float* nw, float* xt) {
    load_normed<DecFp4>(x, nw, xt);
}
"##.to_string(),
    ]
    .concat()
}
