use super::{mma, prim};

pub fn constants() -> &'static str {
    "constexpr int MMB_ROW = 64;\nconstexpr int MMB_K = 32;\n\n"
}

pub fn split_f32() -> String {
    prim::global(
        Some(256),
        "moe_split_bf16",
        "const float* x, unsigned short* hi, unsigned short* lo, int width",
        r##"  size_t row = blockIdx.x;
  for (int i = threadIdx.x; i < width; i += 256) {
    unsigned h, l;
    split_hi_lo(x[row * width + i], h, l);
    hi[row * width + i] = (unsigned short)h;
    lo[row * width + i] = (unsigned short)l;
  }
"##,
    )
}

fn geometry() -> &'static str {
    r##"    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = MMB_K / 16;
    constexpr int ROWS = 2 * BM + NW * BN;
    constexpr int STAGE_BYTES = ROWS * MMB_ROW;
    constexpr int KT = K / MMB_K;
    constexpr int A_ITERS = (BM * 4 + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * 4 + THREADS - 1) / THREADS;
    static_assert(K % MMB_K == 0, "K must be a multiple of the k tile");
    static_assert(N % BN == 0, "N must be a multiple of the tile");
    static_assert(NJ % 2 == 0, "B fragments come two n tiles per ldmatrix");
"##
}

fn load_stage() -> &'static str {
    r##"    auto load_stage = [&](int buf, int kt) {
        unsigned char* st = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int it = 0; it < A_ITERS; it++) {
            int idx = tid + it * THREADS;
            if (A_ITERS * THREADS != BM * 4 && idx >= BM * 4) break;
            int r = idx >> 2;
            int c = idx & 3;
            int s = src[r];
            size_t row = (size_t)max(s, 0);
            size_t at = row * K + kt * MMB_K + c * 8;
            cp_async16_zfill(st + mmb_chunk(r, c), xh + at, s >= 0);
            cp_async16_zfill(st + BM * MMB_ROW + mmb_chunk(r, c), xl + at, s >= 0);
        }
#pragma unroll
        for (int w = 0; w < NW; w++) {
            const unsigned short* wp = w == 0 ? w0 : w1;
#pragma unroll
            for (int it = 0; it < W_ITERS; it++) {
                int idx = tid + it * THREADS;
                if (W_ITERS * THREADS != BN * 4 && idx >= BN * 4) break;
                int r = idx >> 2;
                int c = idx & 3;
                size_t row = (size_t)e * N + n0 + r;
                cp_async16(st + (2 * BM + w * BN) * MMB_ROW + mmb_chunk(r, c), wp + row * K + kt * MMB_K + c * 8);
            }
        }
    };
"##
}

fn mainloop() -> &'static str {
    r##"    for (int kt = 0; kt < KT; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT) load_stage(next % STAGES, next);
        cp_async_commit();
        const unsigned char* st = gemm_smem + (kt % STAGES) * STAGE_BYTES;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned fh[MI][4], fl[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 15);
                int ch = ks * 2 + (lane >> 4);
                ldsm_x4(fh[i], st + mmb_chunk(r, ch));
                ldsm_x4(fl[i], st + BM * MMB_ROW + mmb_chunk(r, ch));
            }
#pragma unroll
            for (int w = 0; w < NW; w++) {
                const unsigned char* ws = st + (2 * BM + w * BN) * MMB_ROW;
#pragma unroll
                for (int j = 0; j < NJ; j += 2) {
                    unsigned b[4];
                    int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                    ldsm_x4(b, ws + mmb_chunk(r, ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
                    for (int jj = 0; jj < 2; jj++) {
#pragma unroll
                        for (int i = 0; i < MI; i++) {
                            mma_bf16(acc[w][i][j + jj], fl[i][0], fl[i][1], fl[i][2], fl[i][3], b[2 * jj], b[2 * jj + 1]);
                            mma_bf16(acc[w][i][j + jj], fh[i][0], fh[i][1], fh[i][2], fh[i][3], b[2 * jj], b[2 * jj + 1]);
                        }
                    }
                }
            }
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
                float2 b0 = make_float2(0.0f, 0.0f);
                float2 b1 = b0;
                if (MOE_EBIAS && e < MOE_E) {
                    if (NW == 2) {
                        b0 = *(const float2*)(bias + (size_t)e * 2 * N + col);
                        b1 = *(const float2*)(bias + ((size_t)e * 2 + 1) * N + col);
                    } else {
                        b0 = *(const float2*)(bias + (size_t)e * N + col);
                    }
                }
                float2 v;
                if (NW == 2) {
                    v.x = gated_act(acc[0][i][j][2 * h] + b0.x, acc[NW - 1][i][j][2 * h] + b1.x);
                    v.y = gated_act(acc[0][i][j][2 * h + 1] + b0.y, acc[NW - 1][i][j][2 * h + 1] + b1.y);
                } else if (MOE_MAPPED && !DOWN) {
                    v.x = mapped_act(acc[0][i][j][2 * h] + b0.x);
                    v.y = mapped_act(acc[0][i][j][2 * h + 1] + b0.y);
                } else {
                    v.x = acc[0][i][j][2 * h] + b0.x;
                    v.y = acc[0][i][j][2 * h + 1] + b0.y;
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
            "moe_gemm_b16",
            "const unsigned short* w0, const unsigned short* w1, const float* bias, const unsigned short* xh, const unsigned short* xl,\n                                            const int* offsets, const int* sorted, const int* tile_map, float* out",
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

const ARGS: &str = "const unsigned short* gw, const unsigned short* uw, const float* bias, const unsigned short* xh, const unsigned short* xl,\n    const int* offsets, const int* sorted, const int* tile_map, float* out";

fn variant(name: &str, tile: &str) -> String {
    prim::delegate(128, name, ARGS, &format!("moe_gemm_b16<{tile}>"), "gw, uw, bias, xh, xl, offsets, sorted, tile_map, out")
}

pub fn grouped() -> String {
    let mut out = String::new();
    out.push_str(constants());
    out.push_str(&split_f32());
    out.push_str(&mma::tiles("moe_tiles_b16"));
    out.push_str(&prim::swizzle("mmb_chunk", "MMB_ROW"));
    out.push_str(&gemm());
    out.push_str(&variant("moe_mma_up_b16", "MOE_INTER, MOE_WIDTH, MOE_UP_WEIGHTS, 64, 64, 2, 2, 3, false"));
    out.push_str(&variant("moe_mma_up_b16_s", "MOE_INTER, MOE_WIDTH, MOE_UP_WEIGHTS, 16, 64, 1, 4, 3, false"));
    out.push_str(&variant("moe_mma_down_b16", "MOE_WIDTH, MOE_INTER, 1, 64, 64, 2, 2, 3, true"));
    out.push_str(&variant("moe_mma_down_b16_s", "MOE_WIDTH, MOE_INTER, 1, 16, 64, 1, 4, 3, true"));
    out
}
