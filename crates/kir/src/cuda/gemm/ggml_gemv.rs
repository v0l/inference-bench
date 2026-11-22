use super::prim;

pub fn body() -> String {
    [
        prim::device_t(
            "int K, bool SPLIT",
            "void",
            "ggv_load_x",
            "const bf16_t* x_hi, const bf16_t* x_lo, float* xs",
            &[
                "    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {\n".to_string(),
                "        unsigned h = ((const unsigned*)x_hi)[i];\n".to_string(),
                "        float a = __uint_as_float(h << 16), b = __uint_as_float(h & 0xffff0000u);\n".to_string(),
                "        if (SPLIT) {\n            unsigned l = ((const unsigned*)x_lo)[i];\n            a += __uint_as_float(l << 16);\n            b += __uint_as_float(l & 0xffff0000u);\n        }\n".to_string(),
                "        xs[2 * i] = a;\n        xs[2 * i + 1] = b;\n    }\n    __syncthreads();\n".to_string(),
            ]
            .concat(),
        ),
        prim::device_t(
            "int N, int K, int EPI, bool BIAS, bool SPLIT, int ROWS",
            "void",
            "gemv_ggv",
            "const void* w, const unsigned* table, const float* bias, const bf16_t* x_hi, const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo",
            &[
                "    __shared__ __align__(16) float xs[K];\n    ggv_load_x<K, SPLIT>(x_hi, x_lo, xs);\n".to_string(),
                prim::warp_ids().to_string(),
                "    constexpr int PER_BLOCK = DecGgml::WARPS * ROWS;\n    for (int row = blockIdx.x * PER_BLOCK + warp * ROWS; row < N; row += gridDim.x * PER_BLOCK) {\n        float acc[ROWS];\n        DecGgml::rows<K, ROWS>(w, (const void*)table, row, 1, xs, lane, acc);\n        if (lane != 0) continue;\n".to_string(),
                "        if (EPI == EPI_ACT) {\n".to_string(),
                prim::unroll_nest(12, &[("p", "ROWS / 2")], false, "ggr_store<N, EPI>(0, row + 2 * p, acc[2 * p], acc[2 * p + 1], out, out_hi, out_lo);"),
                "        } else {\n".to_string(),
                prim::unroll_nest(12, &[("r", "ROWS")], false, "ggr_store<N, EPI>(0, row + r, BIAS ? acc[r] + bias[row + r] : acc[r], 0.0f, out, out_hi, out_lo);"),
                "        }\n    }\n".to_string(),
            ]
            .concat(),
        ),
    ]
    .concat()
}
