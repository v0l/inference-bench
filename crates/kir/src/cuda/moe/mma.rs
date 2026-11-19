use super::prim;

pub fn constants() -> &'static str {
    "constexpr int MMA_DATA = 64;\nconstexpr int MMA_SCALES = 8;\nconstexpr int MMA_ROW = MMA_DATA + MMA_SCALES;\n\n"
}

pub fn quant_f32() -> String {
    prim::global(
        Some(256),
        "moe_quant_f32",
        "const float* x, unsigned char* q, unsigned char* qs, int width, float igs",
        r##"  int groups = width / 16;
  size_t row = blockIdx.x;
  for (int grp = threadIdx.x; grp < groups; grp += 256) {
    size_t at = row * width + (size_t)grp * 16;
    float v[16];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        float4 f = *(const float4*)(x + at + i);
        v[i] = f.x;
        v[i + 1] = f.y;
        v[i + 2] = f.z;
        v[i + 3] = f.w;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    unsigned sbits = f32x2_to_e4m3x2(fminf(igs * (amax * (1.0f / 6.0f)), 448.0f), 0.0f) & 0xffu;
    float sx = e4m3_to_f32(sbits);
    float mul = sx > 0.0f ? igs / sx : 0.0f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        unsigned b = moe_e2m1x2(v[i] * mul, v[i + 1] * mul);
        if (i < 8) w0 |= b << (4 * i); else w1 |= b << (4 * (i - 8));
    }
    *(uint2*)(q + at / 2) = make_uint2(w0, w1);
    qs[row * groups + grp] = (unsigned char)sbits;
  }
"##,
    )
}

pub fn tiles() -> String {
    prim::global(
        None,
        "moe_tiles",
        "const int* offsets, int* tile_map, int bm, int max_tiles",
        r##"    if (threadIdx.x != 0) return;
    int t = 0;
    for (int e = 0; e < MOE_SLOTS; e++) {
        int count = offsets[e + 1] - offsets[e];
        for (int m = 0; m * bm < count; m++) {
            if (t < max_tiles) {
                tile_map[2 * t] = e;
                tile_map[2 * t + 1] = m;
            }
            t++;
        }
    }
    for (; t < max_tiles; t++) tile_map[2 * t] = -1;
"##,
    )
}

fn geometry() -> &'static str {
    r##"    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = 2;
    constexpr int ROWS = BM + NW * BN;
    constexpr int SCALE_OFF = ROWS * MMA_DATA;
    constexpr int STAGE_BYTES = ROWS * MMA_ROW;
    constexpr int KT = K / 128;
    constexpr int KG = K / 16;
    constexpr int A_ITERS = (BM * 5 + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * 5 + THREADS - 1) / THREADS;
    static_assert(K % 128 == 0, "K must be a multiple of 128");
    static_assert(N % BN == 0, "N must be a multiple of the tile");
    static_assert(NJ % 2 == 0, "B fragments come two n tiles per ldmatrix");
"##
}

fn tile_setup_head() -> &'static str {
    r##"    extern __shared__ __align__(16) unsigned char gemm_smem[];
    __shared__ int src[BM];
    __shared__ int dst[BM];
    int tile = blockIdx.y;
    int e = tile_map[2 * tile];
    if (e < 0) return;
    int begin = offsets[e] + tile_map[2 * tile + 1] * BM;
    int end = offsets[e + 1];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    for (int i = tid; i < BM; i += THREADS) {
        int pos = begin + i;
        bool live = pos < end;
        int p = live ? sorted[pos] : -1;
        src[i] = live ? (DOWN ? pos : p / MOE_PAIRS) : -1;
        dst[i] = live ? (DOWN ? p : pos) : -1;
    }
    __syncthreads();
    int n0 = blockIdx.x * BN;
    float acc[NW][MI][NJ][4];
#pragma unroll
    for (int w = 0; w < NW; w++)
"##
}

fn zero_acc() -> String {
    prim::zero_frag("acc[w]", ("i", "MI"), ("j", "NJ"), 8)
}

fn tile_setup() -> String {
    format!("{}{}", tile_setup_head(), zero_acc())
}

fn prologue() -> String {
    prim::pipeline_prologue("KT", "load_stage(st, st)")
}

fn load_stage() -> &'static str {
    r##"    auto load_stage = [&](int buf, int kt) {
        unsigned char* st = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int it = 0; it < A_ITERS; it++) {
            int idx = tid + it * THREADS;
            if (A_ITERS * THREADS != BM * 5 && idx >= BM * 5) break;
            int r = idx / 5;
            int c = idx % 5;
            int s = src[r];
            size_t row = (size_t)max(s, 0);
            if (c < 4) cp_async16_zfill(st + mma_chunk(r, c), aq + row * (K / 2) + kt * 64 + c * 16, s >= 0);
            else moe_cp8(st + SCALE_OFF + r * MMA_SCALES, as + row * KG + kt * 8, s >= 0);
        }
#pragma unroll
        for (int w = 0; w < NW; w++) {
            const unsigned char* wp = w == 0 ? w0 : w1;
            const unsigned char* sp = w == 0 ? s0 : s1;
#pragma unroll
            for (int it = 0; it < W_ITERS; it++) {
                int idx = tid + it * THREADS;
                if (W_ITERS * THREADS != BN * 5 && idx >= BN * 5) break;
                int r = idx / 5;
                int c = idx % 5;
                size_t row = (size_t)e * N + n0 + r;
                if (c < 4) cp_async16(st + (BM + w * BN) * MMA_DATA + mma_chunk(r, c), wp + row * (K / 2) + kt * 64 + c * 16);
                else moe_cp8(st + SCALE_OFF + (BM + w * BN + r) * MMA_SCALES, sp + row * KG + kt * 8, true);
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
            unsigned a[MI][4], sfa[MI];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 15);
                ldsm_x4(a[i], st + mma_chunk(r, ks * 2 + (lane >> 4)));
                sfa[i] = *(const unsigned*)(st + SCALE_OFF + (wm * TM + i * 16 + g + 8 * (lane & 1)) * MMA_SCALES + ks * 4);
            }
#pragma unroll
            for (int w = 0; w < NW; w++) {
                const unsigned char* ws = st + (BM + w * BN) * MMA_DATA;
#pragma unroll
                for (int j = 0; j < NJ; j += 2) {
                    unsigned b[4];
                    int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                    ldsm_x4(b, ws + mma_chunk(r, ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
                    for (int jj = 0; jj < 2; jj++) {
                        unsigned sfb = *(const unsigned*)(st + SCALE_OFF + (BM + w * BN + wn * TN + (j + jj) * 8 + g) * MMA_SCALES + ks * 4);
#pragma unroll
                        for (int i = 0; i < MI; i++) moe_mma_fp4(acc[w][i][j + jj], a[i], b[2 * jj], b[2 * jj + 1], sfa[i], sfb);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
"##
}

fn epilogue() -> &'static str {
    r##"    float alpha0 = g0[e] * inv_in;
    float alpha1 = NW > 1 ? g1[e] * inv_in : 0.0f;
#pragma unroll
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
                    v.x = gated_act(acc[0][i][j][2 * h] * alpha0, acc[NW - 1][i][j][2 * h] * alpha1);
                    v.y = gated_act(acc[0][i][j][2 * h + 1] * alpha0, acc[NW - 1][i][j][2 * h + 1] * alpha1);
                } else {
                    v.x = acc[0][i][j][2 * h] * alpha0;
                    v.y = acc[0][i][j][2 * h + 1] * alpha0;
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
            "moe_gemm_f4",
            "const unsigned char* w0, const unsigned char* s0, const float* g0, const unsigned char* w1, const unsigned char* s1,\n                                            const float* g1, float inv_in, const unsigned char* aq, const unsigned char* as, const int* offsets, const int* sorted,\n                                            const int* tile_map, float* out",
            &format!("{}{}{}{}{}{}", geometry(), tile_setup(), load_stage(), prologue(), mainloop(), epilogue())
        )
    )
}

const ARGS: &str = "const unsigned char* gw, const unsigned char* gs, const float* gg, const unsigned char* uw, const unsigned char* us, const float* ug, float inv_in,\n    const unsigned char* aq, const unsigned char* as, const int* offsets, const int* sorted, const int* tile_map, float* out";

fn variant(name: &str, tile: &str) -> String {
    prim::delegate(
        128,
        name,
        ARGS,
        &format!("moe_gemm_f4<{tile}>"),
        "gw, gs, gg, uw, us, ug, inv_in, aq, as, offsets, sorted, tile_map, out",
    )
}

pub fn grouped() -> String {
    let mut out = String::new();
    out.push_str(constants());
    out.push_str(&prim::mma_fp4("moe_mma_fp4"));
    out.push_str(&prim::e2m1x2("moe_e2m1x2"));
    out.push_str(&prim::cp8_zfill("moe_cp8"));
    out.push_str(&quant_f32());
    out.push_str(&tiles());
    out.push_str(&prim::swizzle("mma_chunk", "MMA_DATA"));
    out.push_str(&gemm());
    out.push_str(&variant("moe_mma_up", "MOE_INTER, HIDDEN, 2, 64, 64, 2, 2, 3, false"));
    out.push_str(&variant("moe_mma_up_s", "MOE_INTER, HIDDEN, 2, 16, 64, 1, 4, 3, false"));
    out.push_str(&variant("moe_mma_down", "HIDDEN, MOE_INTER, 1, 64, 64, 2, 2, 3, true"));
    out.push_str(&variant("moe_mma_down_s", "HIDDEN, MOE_INTER, 1, 16, 64, 1, 4, 3, true"));
    out
}
