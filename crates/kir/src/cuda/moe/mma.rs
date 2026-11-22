use super::prim;

pub fn constants() -> &'static str {
    "constexpr int MMA_DATA = 64;\nconstexpr int MMA_SCALES = 8;\nconstexpr int MMA_SCALES_MX = 4;\nconstexpr int MMA_ROW = MMA_DATA + MMA_SCALES;\n\n"
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

pub fn quant_mx() -> String {
    prim::global(
        Some(256),
        "moe_quant_mx",
        "const float* x, unsigned char* q, unsigned char* qs, unsigned char* q2, unsigned char* qs2, int width",
        r##"  int groups = width / 32;
  int stride = (width + 127) / 128 * 128;
  int padded = stride / 32;
  size_t row = blockIdx.x;
  for (int grp = threadIdx.x; grp < padded; grp += 256) {
    size_t out_at = row * stride + (size_t)grp * 32;
    if (grp >= groups) {
        *(uint4*)(q + out_at / 2) = make_uint4(0u, 0u, 0u, 0u);
        qs[row * padded + grp] = 0;
        if (q2) {
            *(uint4*)(q2 + out_at / 2) = make_uint4(0u, 0u, 0u, 0u);
            qs2[row * padded + grp] = 0;
        }
        continue;
    }
    size_t at = row * width + (size_t)grp * 32;
    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        float4 f = *(const float4*)(x + at + i);
        v[i] = f.x;
        v[i + 1] = f.y;
        v[i + 2] = f.z;
        v[i + 3] = f.w;
    }
#pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    unsigned bits = __float_as_uint(amax * (1.0f / 6.0f));
    unsigned e = (bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u);
    e = min(max(e, 1u), 253u);
    float mul = __uint_as_float((254u - e) << 23);
    unsigned w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < 32; i += 2) w[i / 8] |= moe_e2m1x2(v[i] * mul, v[i + 1] * mul) << (4 * (i % 8));
    *(uint4*)(q + out_at / 2) = make_uint4(w[0], w[1], w[2], w[3]);
    qs[row * padded + grp] = (unsigned char)e;
    if (q2) {
        float back = __uint_as_float(e << 23);
        float r[32];
        float ramax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            r[i] = v[i] - moe_deq_e2m1((w[i / 8] >> (4 * (i % 8))) & 15u) * back;
            ramax = fmaxf(ramax, fabsf(r[i]));
        }
        unsigned rbits = __float_as_uint(ramax * (1.0f / 6.0f));
        unsigned re = (rbits >> 23) + ((rbits & 0x7fffffu) != 0u ? 1u : 0u);
        re = min(max(re, 1u), 253u);
        float rmul = __uint_as_float((254u - re) << 23);
        unsigned w2[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int i = 0; i < 32; i += 2) w2[i / 8] |= moe_e2m1x2(r[i] * rmul, r[i + 1] * rmul) << (4 * (i % 8));
        *(uint4*)(q2 + out_at / 2) = make_uint4(w2[0], w2[1], w2[2], w2[3]);
        qs2[row * padded + grp] = (unsigned char)re;
    }
  }
"##,
    )
}

pub fn tiles(name: &str) -> String {
    prim::global(
        None,
        name,
        "const int* offsets, int* tile_map, int bm, int max_tiles",
        r##"    __shared__ int first[MOE_SLOTS + 1];
    for (int e = threadIdx.x; e < MOE_SLOTS; e += blockDim.x) first[e] = (offsets[e + 1] - offsets[e] + bm - 1) / bm;
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int e = 0; e < MOE_SLOTS; e++) {
            int c = first[e];
            first[e] = run;
            run += c;
        }
        first[MOE_SLOTS] = run;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < MOE_SLOTS; e += blockDim.x) {
        int count = offsets[e + 1] - offsets[e];
        for (int m = 0; m * bm < count; m++) {
            int t = first[e] + m;
            if (t < max_tiles) {
                tile_map[2 * t] = e;
                tile_map[2 * t + 1] = m;
            }
        }
    }
    for (int t = first[MOE_SLOTS] + threadIdx.x; t < max_tiles; t += blockDim.x) tile_map[2 * t] = -1;
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
    constexpr int SB = MX ? MMA_SCALES_MX : MMA_SCALES;
    constexpr int ROWS = BM + NW * BN;
    constexpr int SCALE_OFF = ROWS * MMA_DATA;
    constexpr int STAGE_BYTES = ROWS * (MMA_DATA + SB);
    constexpr int KP = (K + 127) / 128 * 128;
    constexpr int KT = KP / 128;
    constexpr int KG = MX ? KP / 32 : KP / 16;
    constexpr int A_ITERS = (BM * 5 + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * 5 + THREADS - 1) / THREADS;
    static_assert(MX || K % 128 == 0, "K must be a multiple of 128 unless the weights are block scaled");
    static_assert(K % 32 == 0, "K must be a multiple of the block");
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

pub(super) fn tile_setup() -> String {
    format!("{}{}", tile_setup_head(), zero_acc())
}

fn prologue() -> String {
    prim::pipeline_prologue("KT * PL", "load_stage(st, st)")
}

fn load_stage() -> &'static str {
    r##"    auto load_stage = [&](int buf, int kt_all) {
        unsigned char* st = gemm_smem + buf * STAGE_BYTES;
        int plane = kt_all / KT;
        int kt = kt_all - plane * KT;
        const unsigned char* aq_p = plane ? aq2 : aq;
        const unsigned char* as_p = plane ? as2 : as;
#pragma unroll
        for (int it = 0; it < A_ITERS; it++) {
            int idx = tid + it * THREADS;
            if (A_ITERS * THREADS != BM * 5 && idx >= BM * 5) break;
            int r = idx / 5;
            int c = idx % 5;
            int s = src[r];
            size_t row = (size_t)max(s, 0);
            if (c < 4) cp_async16_zfill(st + mma_chunk(r, c), aq_p + row * (KP / 2) + kt * 64 + c * 16, s >= 0);
            else moe_cp_scale<MX>(st + SCALE_OFF + r * SB, as_p + row * KG + kt * SB, s >= 0);
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
                if (c < 4) cp_async16(st + (BM + w * BN) * MMA_DATA + mma_chunk(r, c), wp + row * (KP / 2) + kt * 64 + c * 16);
                else moe_cp_scale<MX>(st + SCALE_OFF + (BM + w * BN + r) * SB, sp + row * KG + kt * SB, true);
            }
        }
    };
"##
}

fn mainloop() -> &'static str {
    r##"    for (int kt = 0; kt < KT * PL; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT * PL) load_stage(next % STAGES, next);
        cp_async_commit();
        const unsigned char* st = gemm_smem + (kt % STAGES) * STAGE_BYTES;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned a[MI][4], sfa[MI];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 15);
                ldsm_x4(a[i], st + mma_chunk(r, ks * 2 + (lane >> 4)));
                sfa[i] = *(const unsigned*)(st + SCALE_OFF + (wm * TM + i * 16 + g + 8 * (lane & 1)) * SB + (MX ? 0 : ks * 4));
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
                        unsigned sfb = *(const unsigned*)(st + SCALE_OFF + (BM + w * BN + wn * TN + (j + jj) * 8 + g) * SB + (MX ? 0 : ks * 4));
#pragma unroll
                        for (int i = 0; i < MI; i++) {
                            moe_mma_block<MX>(acc[w][i][j + jj], a[i], b[2 * jj], b[2 * jj + 1], sfa[i], sfb, ks);
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
    r##"    float alpha0 = MX ? 1.0f : g0[e] * inv_in;
    float alpha1 = (NW > 1 && !MX) ? g1[e] * inv_in : 1.0f;
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
                    v.x = gated_act(acc[0][i][j][2 * h] * alpha0 + b0.x, acc[NW - 1][i][j][2 * h] * alpha1 + b1.x);
                    v.y = gated_act(acc[0][i][j][2 * h + 1] * alpha0 + b0.y, acc[NW - 1][i][j][2 * h + 1] * alpha1 + b1.y);
                } else {
                    v.x = acc[0][i][j][2 * h] * alpha0 + b0.x;
                    v.y = acc[0][i][j][2 * h + 1] * alpha0 + b0.y;
                }
                *(float2*)(out + (size_t)d * N + col) = v;
            }
"##
}

fn gemm() -> String {
    format!(
        "template <int N, int K, int NW, int BM, int BN, int WM, int WN, int STAGES, bool DOWN, bool MX, int PL>\n{}",
        prim::device(
            "void",
            "moe_gemm_f4",
            "const unsigned char* w0, const unsigned char* s0, const float* g0, const unsigned char* w1, const unsigned char* s1,\n                                            const float* g1, const float* bias, float inv_in, const unsigned char* aq, const unsigned char* as, const unsigned char* aq2, const unsigned char* as2, const int* offsets, const int* sorted,\n                                            const int* tile_map, float* out",
            &format!("{}{}{}{}{}{}", geometry(), tile_setup(), load_stage(), prologue(), mainloop(), epilogue())
        )
    )
}

const ARGS: &str = "const unsigned char* gw, const unsigned char* gs, const float* gg, const unsigned char* uw, const unsigned char* us, const float* ug, const float* bias, float inv_in,\n    const unsigned char* aq, const unsigned char* as, const int* offsets, const int* sorted, const int* tile_map, float* out";

const ARGS_MX: &str = "const unsigned char* gw, const unsigned char* gs, const float* gg, const unsigned char* uw, const unsigned char* us, const float* ug, const float* bias, float inv_in,\n    const unsigned char* aq, const unsigned char* as, const unsigned char* aq2, const unsigned char* as2, const int* offsets, const int* sorted, const int* tile_map, float* out";

fn variant(name: &str, tile: &str, mx: bool, planes: usize) -> String {
    let (args, call) = if mx {
        (ARGS_MX, "gw, gs, gg, uw, us, ug, bias, inv_in, aq, as, aq2, as2, offsets, sorted, tile_map, out")
    } else {
        (ARGS, "gw, gs, gg, uw, us, ug, bias, inv_in, aq, as, aq, as, offsets, sorted, tile_map, out")
    };
    prim::delegate(128, name, args, &format!("moe_gemm_f4<{tile}, {mx}, {planes}>"), call)
}

fn variants(out: &mut String, mx: bool) {
    let stem = if mx { "moe_mma_mx" } else { "moe_mma" };
    let planes: &[(usize, &str)] = if mx { &[(1, ""), (2, "2")] } else { &[(1, "")] };
    for &(p, tag) in planes {
        out.push_str(&variant(&format!("{stem}_up{tag}"), "MOE_INTER, HIDDEN, 2, 64, 64, 2, 2, 3, false", mx, p));
        out.push_str(&variant(&format!("{stem}_up{tag}_s"), "MOE_INTER, HIDDEN, 2, 16, 64, 1, 4, 3, false", mx, p));
        out.push_str(&variant(&format!("{stem}_down{tag}"), "HIDDEN, MOE_INTER, 1, 64, 64, 2, 2, 3, true", mx, p));
        out.push_str(&variant(&format!("{stem}_down{tag}_s"), "HIDDEN, MOE_INTER, 1, 16, 64, 1, 4, 3, true", mx, p));
    }
}

fn scale_helpers() -> String {
    let mut out = prim::mma_fp4("moe_mma_fp4");
    out.push_str(&prim::mma_mx("moe_mma_mx"));
    out.push_str(&prim::cp8_zfill("moe_cp8"));
    out.push_str(&prim::cp4_zfill("moe_cp4"));
    out.push_str(&format!(
        "template <bool MX>\n{}template <bool MX>\n{}",
        prim::device("void", "moe_cp_scale", "void* smem, const void* gmem, bool valid", "    if (MX) moe_cp4(smem, gmem, valid);\n    else moe_cp8(smem, gmem, valid);\n"),
        prim::device(
            "void",
            "moe_mma_block",
            "float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa, unsigned sfb, int ks",
            "    if (MX) moe_mma_mx(c, a, b0, b1, sfa, sfb, (unsigned short)(ks * 2));\n    else moe_mma_fp4(c, a, b0, b1, sfa, sfb);\n"
        )
    ));
    out
}

pub fn grouped(mx: bool) -> String {
    let mut out = String::new();
    out.push_str(constants());
    out.push_str(&prim::e2m1x2("moe_e2m1x2"));
    if mx {
        out.push_str(&prim::e2m1_to_f32("moe_deq_e2m1"));
    }
    out.push_str(&scale_helpers());
    out.push_str(&if mx { quant_mx() } else { quant_f32() });
    out.push_str(&tiles("moe_tiles"));
    out.push_str(&prim::swizzle("mma_chunk", "MMA_DATA"));
    out.push_str(&gemm());
    variants(&mut out, mx);
    out
}
