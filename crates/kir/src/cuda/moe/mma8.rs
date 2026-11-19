use super::{mma, prim};

pub fn constants() -> &'static str {
    "constexpr int MMA8_ROW = 128;\n\n"
}

fn geometry() -> &'static str {
    r##"    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = FP8_BLOCK / 32;
    constexpr int ROWS = BM + NW * BN;
    constexpr int STAGE_BYTES = ROWS * MMA8_ROW;
    constexpr int KT = K / FP8_BLOCK;
    constexpr int CHUNKS = FP8_BLOCK / 16;
    constexpr int A_ITERS = (BM * CHUNKS + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % FP8_BLOCK == 0, "K must be a multiple of the FP8 block");
    static_assert(N % BN == 0 && FP8_BLOCK % BN == 0, "N must tile into the FP8 block");
    static_assert(NJ % 2 == 0, "B fragments come two n tiles per ldmatrix");
"##
}

fn load_stage() -> &'static str {
    r##"    auto load_stage = [&](int buf, int kt) {
        unsigned char* st = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int it = 0; it < A_ITERS; it++) {
            int idx = tid + it * THREADS;
            if (A_ITERS * THREADS != BM * CHUNKS && idx >= BM * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            int s = src[r];
            size_t row = (size_t)max(s, 0);
            cp_async16_zfill(st + r * MMA8_ROW + ((c ^ (r & 7)) << 4), aq + row * K + kt * FP8_BLOCK + c * 16, s >= 0);
        }
#pragma unroll
        for (int w = 0; w < NW; w++) {
            const unsigned char* wp = w == 0 ? w0 : w1;
#pragma unroll
            for (int it = 0; it < W_ITERS; it++) {
                int idx = tid + it * THREADS;
                if (W_ITERS * THREADS != BN * CHUNKS && idx >= BN * CHUNKS) break;
                int r = idx / CHUNKS;
                int c = idx % CHUNKS;
                size_t row = (size_t)e * N + n0 + r;
                cp_async16(st + (BM + w * BN + r) * MMA8_ROW + ((c ^ (r & 7)) << 4), wp + row * K + kt * FP8_BLOCK + c * 16);
            }
        }
    };
"##
}

fn mainloop() -> &'static str {
    r##"    const float* wg0 = g0 + (size_t)e * MOE_FP8_GRID<N, K> + (n0 / FP8_BLOCK) * MOE_FP8_KB<K>;
    const float* wg1 = g1 + (size_t)e * MOE_FP8_GRID<N, K> + (n0 / FP8_BLOCK) * MOE_FP8_KB<K>;
    for (int kt = 0; kt < KT; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT) load_stage(next % STAGES, next);
        cp_async_commit();
        const unsigned char* st = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        float sw[NW];
#pragma unroll
        for (int w = 0; w < NW; w++) sw[w] = (w == 0 ? wg0 : wg1)[kt];
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            float c[NW][MI][NJ][4];
#pragma unroll
            for (int w = 0; w < NW; w++)
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int q = 0; q < 4; q++) c[w][i][j][q] = 0.0f;
            unsigned a[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int ch = ks * 2 + (lane >> 4);
                ldsm_x4(a[i], st + r * MMA8_ROW + ((ch ^ (r & 7)) << 4));
            }
#pragma unroll
            for (int w = 0; w < NW; w++) {
                const unsigned char* ws = st + (BM + w * BN) * MMA8_ROW;
#pragma unroll
                for (int j = 0; j < NJ; j += 2) {
                    unsigned b[4];
                    int r = wn * TN + j * 8 + (lane & 7) + (lane >> 4) * 8;
                    int ch = ks * 2 + ((lane >> 3) & 1);
                    ldsm_x4(b, ws + r * MMA8_ROW + ((ch ^ (r & 7)) << 4));
#pragma unroll
                    for (int i = 0; i < MI; i++) {
                        moe_mma_e4m3(c[w][i][j], a[i], b[0], b[1]);
                        moe_mma_e4m3(c[w][i][j + 1], a[i], b[2], b[3]);
                    }
                }
            }
            float sa[MI][2];
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) sa[i][h] = xs[(size_t)max(src[wm * TM + i * 16 + g + 8 * h], 0) * (K / 32) + kt * KS + ks];
#pragma unroll
            for (int w = 0; w < NW; w++)
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int q = 0; q < 4; q++) acc[w][i][j][q] += c[w][i][j][q] * sa[i][q >> 1] * sw[w];
        }
    }
    cp_async_wait<0>();
"##
}

fn epilogue() -> &'static str {
    r##"#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = wm * TM + i * 16 + g + 8 * h;
                int d = dst[r];
                if (d < 0) continue;
                int col = n0 + wn * TN + j * 8 + 2 * t;
                float2 v;
                if (NW == 2) {
                    v.x = activation(acc[0][i][j][2 * h]) * acc[NW - 1][i][j][2 * h];
                    v.y = activation(acc[0][i][j][2 * h + 1]) * acc[NW - 1][i][j][2 * h + 1];
                } else {
                    v.x = acc[0][i][j][2 * h];
                    v.y = acc[0][i][j][2 * h + 1];
                }
                *(float2*)(out + (size_t)d * N + col) = v;
            }
"##
}

fn gemm() -> String {
    format!(
        "template <int N, int K, int NW, int BM, int BN, int WM, int WN, int STAGES, bool DOWN>\n{}",
        prim::device(
            "void",
            "moe_gemm_f8",
            "const unsigned char* w0, const float* g0, const unsigned char* w1, const float* g1, const unsigned char* aq, const float* xs,\n                                            const int* offsets, const int* sorted, const int* tile_map, float* out",
            &format!(
                "{}{}{}{}{}{}",
                geometry(),
                mma::tile_setup(),
                load_stage(),
                prim::pipeline_prologue("KT", "load_stage(st, st)"),
                mainloop(),
                epilogue()
            )
        )
    )
}

const ARGS: &str = "const unsigned char* gw, const float* gg, const unsigned char* uw, const float* ug, const unsigned char* aq, const float* xs,\n    const int* offsets, const int* sorted, const int* tile_map, float* out";

fn variant(name: &str, tile: &str) -> String {
    prim::delegate(128, name, ARGS, &format!("moe_gemm_f8<{tile}>"), "gw, gg, uw, ug, aq, xs, offsets, sorted, tile_map, out")
}

pub fn grouped() -> String {
    let mut out = String::new();
    out.push_str(constants());
    out.push_str(&prim::mma_e4m3("moe_mma_e4m3", true));
    out.push_str(&crate::cuda::quant::quant_fp8_f32_groups("moe_quant_f8", 128));
    out.push_str(&mma::tiles("moe_tiles_f8"));
    out.push_str(&gemm());
    out.push_str(&variant("moe_mma_up_f8", "MOE_INTER, HIDDEN, 2, 64, 64, 2, 2, 3, false"));
    out.push_str(&variant("moe_mma_up_f8_s", "MOE_INTER, HIDDEN, 2, 16, 64, 1, 4, 3, false"));
    out.push_str(&variant("moe_mma_down_f8", "HIDDEN, MOE_INTER, 1, 64, 64, 2, 2, 3, true"));
    out.push_str(&variant("moe_mma_down_f8_s", "HIDDEN, MOE_INTER, 1, 16, 64, 1, 4, 3, true"));
    out
}
