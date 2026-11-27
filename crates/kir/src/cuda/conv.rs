use crate::conv::{ConvError, ConvKernel, Family, Prologue, ROW_THREADS, THREADS, THREAD_COLS, THREAD_ROWS};

const LOAD_INPUT: &str = r#"template <int K, int KT, int CIN, bool UP, int GIN, bool SILU>
__device__ __forceinline__ float vae_conv_input(const float* __restrict__ x, const float* __restrict__ hist, int hist_frames,
                                                const float* __restrict__ norm, const float* __restrict__ gamma, const float* __restrict__ beta,
                                                int t, int oy, int ox, int dt, int dy, int dx, int c, int H, int W) {
    if (c >= CIN || oy >= H || ox >= W) return 0.0f;
    int iy = oy + dy - K / 2, ix = ox + dx - K / 2;
    if (iy < 0 || iy >= H || ix < 0 || ix >= W) return 0.0f;
    int IH = UP ? H / 2 : H, IW = UP ? W / 2 : W;
    int sy = UP ? iy / 2 : iy, sx = UP ? ix / 2 : ix;
    int f = t + dt - (KT - 1);
    const float* src;
    if (f >= 0) {
        src = x + (((size_t)f * IH + sy) * IW + sx) * CIN;
    } else if (hist_frames + f >= 0) {
        src = hist + (((size_t)(hist_frames + f) * IH + sy) * IW + sx) * CIN;
    } else {
        return 0.0f;
    }
    float v = src[c];
    if (GIN) {
        int g = c / (CIN / (GIN ? GIN : 1));
        v = (v - norm[2 * g]) * norm[2 * g + 1] * gamma[c] + beta[c];
        if (SILU) v = v / (1.0f + expf(-v));
    }
    return v;
}

__device__ __forceinline__ void vae_norm_table(const double* __restrict__ sin, float* norm, int groups, float eps, double cnt) {
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        double mean = sin[2 * g] / cnt;
        double var = sin[2 * g + 1] / cnt - mean * mean;
        norm[2 * g] = (float)mean;
        norm[2 * g + 1] = (float)(1.0 / sqrt((var > 0.0 ? var : 0.0) + (double)eps));
    }
}

template <int COUT, bool ILV, bool RES, bool CLAMP>
__device__ __forceinline__ float vae_conv_store(float v, const float* __restrict__ resid, float* __restrict__ y, float lo, float hi, int t, int oy, int ox, int n, int H, int W) {
    size_t at;
    if (ILV) {
        int half = COUT / 2;
        int s = n >= half ? 1 : 0;
        at = (((size_t)(2 * t + s) * H + oy) * W + ox) * half + (n - s * half);
    } else {
        at = (((size_t)t * H + oy) * W + ox) * COUT + n;
    }
    if (RES) v += resid[at];
    if (CLAMP) v = fminf(fmaxf(v, lo), hi);
    y[at] = v;
    return v;
}
"#;

const TILE_BODY: &str = r#"template <int BM, int BN, int BK, int PH, int PW, int TM, int TN, int STAGES, int K, int KT, int CIN, int COUT, bool UP, bool ILV, int GIN, bool SILU, int GOUT, bool RES, bool CLAMP>
__device__ __forceinline__ void vae_conv_tile(const float* __restrict__ x, const float* __restrict__ hist, int hist_frames, const float* __restrict__ w,
                                              const float* __restrict__ bias, const float* __restrict__ resid, float* __restrict__ y,
                                              const double* __restrict__ sin, const float* __restrict__ gamma, const float* __restrict__ beta, float eps, double cnt,
                                              double* __restrict__ sout, float lo, float hi, int T, int H, int W) {
    extern __shared__ __align__(16) unsigned char vae_smem[];
    double* gst = (double*)vae_smem;
    float* norm = (float*)(gst + 2 * GOUT);
    float* sa = norm + 2 * GIN;
    float* sb = sa + STAGES * BK * BM;
    constexpr int CHUNKS = (CIN + BK - 1) / BK;
    constexpr int STEPS = K * K * KT * CHUNKS;
    constexpr int NB = (COUT + BN - 1) / BN;
    int tid = threadIdx.x;
    int tr = tid / THREAD_COLS, tc = tid % THREAD_COLS;
    int t = blockIdx.z / NB, n0 = (blockIdx.z % NB) * BN;
    int py0 = blockIdx.y * PH, px0 = blockIdx.x * PW;
    for (int i = tid; i < 2 * GOUT; i += blockDim.x) gst[i] = 0.0;
    if (GIN) vae_norm_table(sin, norm, GIN, eps, cnt);
    __syncthreads();
    auto load = [&](int s, int buf) {
        int tap = s / CHUNKS, c0 = (s % CHUNKS) * BK;
        int dt = tap / (K * K), dy = (tap / K) % K, dx = tap % K;
        float* A = sa + buf * BK * BM;
        float* B = sb + buf * BK * BN;
        for (int e = tid; e < BM * BK; e += blockDim.x) {
            int p = e / BK, kk = e % BK;
            A[kk * BM + p] = vae_conv_input<K, KT, CIN, UP, GIN, SILU>(x, hist, hist_frames, norm, gamma, beta, t, py0 + p / PW, px0 + p % PW, dt, dy, dx, c0 + kk, H, W);
        }
        for (int e = tid; e < BN * BK; e += blockDim.x) {
            int j = e / BK, kk = e % BK;
            int n = n0 + j, c = c0 + kk;
            B[kk * BN + j] = (n < COUT && c < CIN) ? w[((size_t)n * (K * K * KT) + tap) * CIN + c] : 0.0f;
        }
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    load(0, 0);
    __syncthreads();
    for (int s = 0; s < STEPS; s++) {
        int buf = STAGES == 2 ? (s & 1) : 0;
        if (STAGES == 2 && s + 1 < STEPS) load(s + 1, buf ^ 1);
        const float* A = sa + buf * BK * BM;
        const float* B = sb + buf * BK * BN;
#pragma unroll 4
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = A[kk * BM + tr + THREAD_ROWS * i];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = B[kk * BN + tc + THREAD_COLS * j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
        if (STAGES == 1 && s + 1 < STEPS) {
            load(s + 1, 0);
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int p = tr + THREAD_ROWS * i;
        int oy = py0 + p / PW, ox = px0 + p % PW;
        if (oy >= H || ox >= W) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = n0 + tc + THREAD_COLS * j;
            if (n >= COUT) continue;
            float v = vae_conv_store<COUT, ILV, RES, CLAMP>(acc[i][j] + bias[n], resid, y, lo, hi, t, oy, ox, n, H, W);
            if (GOUT) {
                int g = n / (COUT / (GOUT ? GOUT : 1));
                atomicAdd(&gst[2 * g], (double)v);
                atomicAdd(&gst[2 * g + 1], (double)v * (double)v);
            }
        }
    }
    if (GOUT) {
        __syncthreads();
        for (int i = tid; i < 2 * GOUT; i += blockDim.x) atomicAdd(&sout[i], gst[i]);
    }
}
"#;

const DIRECT_BODY: &str = r#"template <int K, int KT, int CIN, int COUT, bool UP, bool ILV, int GIN, bool SILU, int GOUT, bool RES, bool CLAMP>
__device__ __forceinline__ void vae_conv_direct(const float* __restrict__ x, const float* __restrict__ hist, int hist_frames, const float* __restrict__ w,
                                                const float* __restrict__ bias, const float* __restrict__ resid, float* __restrict__ y,
                                                const double* __restrict__ sin, const float* __restrict__ gamma, const float* __restrict__ beta, float eps, double cnt,
                                                double* __restrict__ sout, float lo, float hi, int T, int H, int W) {
    extern __shared__ __align__(16) unsigned char vae_smem[];
    double* gst = (double*)vae_smem;
    float* norm = (float*)(gst + 2 * GOUT);
    if (GIN) vae_norm_table(sin, norm, GIN, eps, cnt);
    __syncthreads();
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * H * W * COUT) return;
    int n = idx % COUT;
    size_t pix = idx / COUT;
    int ox = pix % W, oy = (pix / W) % H, t = pix / ((size_t)W * H);
    float acc = bias[n];
    for (int dt = 0; dt < KT; dt++)
        for (int dy = 0; dy < K; dy++)
            for (int dx = 0; dx < K; dx++) {
                const float* wr = w + ((size_t)n * (K * K * KT) + (dt * K + dy) * K + dx) * CIN;
                for (int c = 0; c < CIN; c++) acc = fmaf(vae_conv_input<K, KT, CIN, UP, GIN, SILU>(x, hist, hist_frames, norm, gamma, beta, t, oy, ox, dt, dy, dx, c, H, W), wr[c], acc);
            }
    float v = vae_conv_store<COUT, ILV, RES, CLAMP>(acc, resid, y, lo, hi, t, oy, ox, n, H, W);
    if (GOUT) {
        int g = n / (COUT / (GOUT ? GOUT : 1));
        atomicAdd(&sout[2 * g], (double)v);
        atomicAdd(&sout[2 * g + 1], (double)v * (double)v);
    }
}
"#;

const ARGS: &str = "const float* x, const float* hist, int hist_frames, const float* w, const float* bias, const float* resid, float* y, const double* sin, const float* gamma, const float* beta, float eps, double cnt, double* sout, float lo, float hi, int T, int H, int W";
const PASS: &str = "x, hist, hist_frames, w, bias, resid, y, sin, gamma, beta, eps, cnt, sout, lo, hi, T, H, W";

const ROWS: &str = r#"__device__ __forceinline__ float vae_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int)(blockDim.x / 32); i++) t += red[i];
    return t;
}

__device__ __forceinline__ float vae_block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = red[0];
    for (int i = 1; i < (int)(blockDim.x / 32); i++) t = fmaxf(t, red[i]);
    return t;
}

extern "C" __global__ void vae_zero(double* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0;
}

extern "C" __global__ void vae_group_stats(const float* x, double* s, int P, int C, int G) {
    __shared__ double part[2 * MAX_GROUPS];
    for (int i = threadIdx.x; i < 2 * G; i += blockDim.x) part[i] = 0.0;
    __syncthreads();
    int per = C / G;
    size_t total = (size_t)P * C;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
        double v = x[e];
        int g = (e % C) / per;
        atomicAdd(&part[2 * g], v);
        atomicAdd(&part[2 * g + 1], v * v);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 2 * G; i += blockDim.x) atomicAdd(&s[i], part[i]);
}

extern "C" __global__ void vae_group_apply(const float* x, float* y, const double* s, const float* gamma, const float* beta, int P, int C, int G, float eps, double cnt, int silu) {
    __shared__ float norm[2 * MAX_GROUPS];
    vae_norm_table(s, norm, G, eps, cnt);
    __syncthreads();
    int per = C / G;
    size_t total = (size_t)P * C;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
        int c = e % C, g = c / per;
        float v = (x[e] - norm[2 * g]) * norm[2 * g + 1] * gamma[c] + beta[c];
        if (silu) v = v / (1.0f + expf(-v));
        y[e] = v;
    }
}

extern "C" __global__ void vae_rms(const float* x, float* y, const float* gamma, int P, int C, int silu) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    size_t p = (size_t)blockIdx.x * (blockDim.x / 32) + warp;
    if (p >= (size_t)P) return;
    const float* row = x + p * C;
    float ss = 0.0f;
    for (int c = lane; c < C; c += 32) ss += row[c] * row[c];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    float scale = sqrtf((float)C) / fmaxf(sqrtf(ss), 1e-12f);
    for (int c = lane; c < C; c += 32) {
        float v = row[c] * scale * gamma[c];
        if (silu) v = v / (1.0f + expf(-v));
        y[p * C + c] = v;
    }
}

extern "C" __global__ void vae_upsample(const float* x, float* y, int T, int H, int W, int C) {
    size_t total = (size_t)T * H * W * C;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
        int c = e % C;
        size_t pix = e / C;
        int ox = pix % W, oy = (pix / W) % H, t = pix / ((size_t)W * H);
        y[e] = x[(((size_t)t * (H / 2) + oy / 2) * (W / 2) + ox / 2) * C + c];
    }
}

extern "C" __global__ void vae_add(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void vae_clamp(const float* x, float* y, int n, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fminf(fmaxf(x[i], lo), hi);
}

extern "C" __global__ void vae_attention(const float* qkv, float* out, int S, int C, float scale) {
    extern __shared__ __align__(16) float att[];
    float* red = att;
    float* p = att + 64;
    int t = blockIdx.y, i = blockIdx.x;
    const float* base = qkv + (size_t)t * S * 3 * C;
    const float* q = base + (size_t)i * 3 * C;
    float m = -3.0e38f;
    for (int j = threadIdx.x; j < S; j += blockDim.x) {
        const float* k = base + (size_t)j * 3 * C + C;
        float d = 0.0f;
        for (int c = 0; c < C; c++) d = fmaf(q[c], k[c], d);
        d *= scale;
        p[j] = d;
        m = fmaxf(m, d);
    }
    m = vae_block_max(m, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < S; j += blockDim.x) {
        float e = expf(p[j] - m);
        p[j] = e;
        sum += e;
    }
    sum = vae_block_sum(sum, red + 32);
    float inv = 1.0f / sum;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) acc = fmaf(p[j], base[(size_t)j * 3 * C + 2 * C + c], acc);
        out[((size_t)t * S + i) * C + c] = acc * inv;
    }
}
"#;

fn flag(b: bool) -> &'static str {
    if b { "true" } else { "false" }
}

fn wrapper(k: &ConvKernel) -> String {
    let s = &k.shape;
    let (gin, silu) = match k.prologue {
        Prologue::None | Prologue::Act(_) => (0, false),
        Prologue::GroupApply { groups, silu } => (groups, silu),
    };
    let gout = k.epilogue.stats.unwrap_or(0);
    let call = match k.family {
        Family::Tile => {
            let t = &k.tile;
            format!(
                "vae_conv_tile<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {gin}, {}, {gout}, {}, CLAMP>",
                t.bm, t.bn, t.bk, t.ph, t.pw, t.tm, t.tn, t.stages, s.k, s.kt, s.cin, s.cout, flag(s.upsample), flag(s.interleave), flag(silu), flag(k.epilogue.residual)
            )
        }
        Family::Direct | Family::Rows => format!("vae_conv_direct<{}, {}, {}, {}, {}, {}, {gin}, {}, {gout}, {}, CLAMP>", s.k, s.kt, s.cin, s.cout, flag(s.upsample), flag(s.interleave), flag(silu), flag(k.epilogue.residual)),
    };
    let mut out = String::new();
    for (suffix, clamp) in [("", "false"), ("_clamp", "true")] {
        out += &format!(
            "extern \"C\" __global__ void __launch_bounds__({THREADS}) {}{suffix}({ARGS}) {{\n    {}({PASS});\n}}\n\n",
            k.name(),
            call.replace("CLAMP", clamp)
        );
    }
    out
}

const LINE: &str = r#"__device__ __forceinline__ float line_gelu(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }

template <int KT, int STRIDE, int CIN, int PAD, int ACT>
__device__ __forceinline__ float line_input(const float* __restrict__ x, const float* __restrict__ st, int T, int b, int o, int tap, int c) {
    if (c >= CIN) return 0.0f;
    constexpr int CTX = PAD >= 0 ? PAD : KT - STRIDE;
    int p = o * STRIDE + tap - CTX;
    if (p >= T) return 0.0f;
    if (p >= 0) return line_act<ACT>(x[((size_t)b * T + p) * CIN + c]);
    if (PAD >= 0) return 0.0f;
    return line_act<ACT>(st[((size_t)b * (CTX > 0 ? CTX : 1) + (CTX + p)) * CIN + c]);
}

template <int COUT, int PHASES, bool RES, bool GELU>
__device__ __forceinline__ void line_store(float v, const float* __restrict__ scale, float* __restrict__ y, int TO, int b, int o, int n) {
    if (GELU) v = line_gelu(v);
    constexpr int PER = COUT / PHASES;
    int r = n / PER, co = n % PER;
    size_t at = (((size_t)b * TO + o) * PHASES + r) * PER + co;
    if (RES) y[at] += scale[co] * v;
    else y[at] = v;
}

template <int BM, int BN, int BK, int TM, int TN, int STAGES, int KT, int STRIDE, int CIN, int COUT, int PHASES, bool RES, bool GELU, int PAD, int ACT>
__device__ __forceinline__ void line_conv_tile(const float* __restrict__ x, const float* __restrict__ st, const float* __restrict__ w, const float* __restrict__ bias,
                                               const float* __restrict__ scale, float* __restrict__ y, int B, int T, int TO) {
    extern __shared__ __align__(16) unsigned char line_smem[];
    float* sa = (float*)line_smem;
    float* sb = sa + STAGES * BK * BM;
    constexpr int CHUNKS = (CIN + BK - 1) / BK;
    constexpr int STEPS = KT * CHUNKS;
    int tid = threadIdx.x;
    int tr = tid / THREAD_COLS, tc = tid % THREAD_COLS;
    int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    int rows = B * TO;
    auto load = [&](int s, int buf) {
        int tap = s / CHUNKS, c0 = (s % CHUNKS) * BK;
        float* A = sa + buf * BK * BM;
        float* Bs = sb + buf * BK * BN;
        for (int e = tid; e < BM * BK; e += blockDim.x) {
            int p = e / BK, kk = e % BK;
            int m = m0 + p;
            A[kk * BM + p] = m < rows ? line_input<KT, STRIDE, CIN, PAD, ACT>(x, st, T, m / TO, m % TO, tap, c0 + kk) : 0.0f;
        }
        for (int e = tid; e < BN * BK; e += blockDim.x) {
            int j = e / BK, kk = e % BK;
            int n = n0 + j, c = c0 + kk;
            Bs[kk * BN + j] = (n < COUT && c < CIN) ? w[((size_t)n * KT + tap) * CIN + c] : 0.0f;
        }
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    load(0, 0);
    __syncthreads();
    for (int s = 0; s < STEPS; s++) {
        int buf = STAGES == 2 ? (s & 1) : 0;
        if (STAGES == 2 && s + 1 < STEPS) load(s + 1, buf ^ 1);
        const float* A = sa + buf * BK * BM;
        const float* Bs = sb + buf * BK * BN;
#pragma unroll 4
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], bb[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = A[kk * BM + tr + THREAD_ROWS * i];
#pragma unroll
            for (int j = 0; j < TN; j++) bb[j] = Bs[kk * BN + tc + THREAD_COLS * j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], bb[j], acc[i][j]);
        }
        __syncthreads();
        if (STAGES == 1 && s + 1 < STEPS) {
            load(s + 1, 0);
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int m = m0 + tr + THREAD_ROWS * i;
        if (m >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = n0 + tc + THREAD_COLS * j;
            if (n >= COUT) continue;
            line_store<COUT, PHASES, RES, GELU>(acc[i][j] + bias[n], scale, y, TO, m / TO, m % TO, n);
        }
    }
}

template <int KT, int STRIDE, int CIN, int COUT, int PHASES, bool RES, bool GELU, int PAD, int ACT>
__device__ __forceinline__ void line_conv_direct(const float* __restrict__ x, const float* __restrict__ st, const float* __restrict__ w, const float* __restrict__ bias,
                                                 const float* __restrict__ scale, float* __restrict__ y, int B, int T, int TO) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * TO * COUT) return;
    int n = idx % COUT;
    size_t m = idx / COUT;
    int b = m / TO, o = m % TO;
    float acc = bias[n];
    for (int tap = 0; tap < KT; tap++) {
        const float* wr = w + ((size_t)n * KT + tap) * CIN;
        for (int c = 0; c < CIN; c++) acc = fmaf(line_input<KT, STRIDE, CIN, PAD, ACT>(x, st, T, b, o, tap, c), wr[c], acc);
    }
    line_store<COUT, PHASES, RES, GELU>(acc, scale, y, TO, b, o, n);
}

template <int KT, int STRIDE, int CIN, int COUT, int PHASES, int CHUNK, bool RES, bool GELU, int PAD, int ACT>
__device__ __forceinline__ void line_conv_rows(const float* __restrict__ x, const float* __restrict__ st, const float* __restrict__ w, const float* __restrict__ bias,
                                               const float* __restrict__ scale, float* __restrict__ y, int B, int T, int TO) {
    __shared__ float part[CHUNK][32];
    constexpr int RED = KT * CIN;
    int n = blockIdx.x;
    int rows = B * TO;
    const float* wr = w + (size_t)n * RED;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int r0 = 0; r0 < rows; r0 += CHUNK) {
        float acc[CHUNK];
#pragma unroll
        for (int j = 0; j < CHUNK; j++) acc[j] = 0.0f;
        for (int e = threadIdx.x; e < RED; e += blockDim.x) {
            int tap = e / CIN, c = e % CIN;
            float wv = wr[e];
#pragma unroll
            for (int j = 0; j < CHUNK; j++) {
                int m = r0 + j;
                if (m < rows) acc[j] = fmaf(line_input<KT, STRIDE, CIN, PAD, ACT>(x, st, T, m / TO, m % TO, tap, c), wv, acc[j]);
            }
        }
#pragma unroll
        for (int j = 0; j < CHUNK; j++) {
            float v = acc[j];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == 0) part[j][warp] = v;
        }
        __syncthreads();
        if (threadIdx.x < CHUNK) {
            int m = r0 + threadIdx.x;
            if (m < rows) {
                float v = bias[n];
                for (int i = 0; i < (int)(blockDim.x >> 5); i++) v += part[threadIdx.x][i];
                line_store<COUT, PHASES, RES, GELU>(v, scale, y, TO, m / TO, m % TO, n);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void line_rms(const float* x, float* y, const float* gamma, int P, int C, float eps) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    size_t p = (size_t)blockIdx.x * (blockDim.x / 32) + warp;
    if (p >= (size_t)P) return;
    const float* row = x + p * C;
    float ss = 0.0f;
    for (int c = lane; c < C; c += 32) ss += row[c] * row[c];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    float r = rsqrtf(ss / (float)C + eps);
    for (int c = lane; c < C; c += 32) y[p * C + c] = row[c] * r * gamma[c];
}

extern "C" __global__ void line_dwconv(const float* x, const float* st, const float* w, const float* bias, const float* scale, float* y, int B, int T, int C, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * T * C) return;
    int c = idx % C;
    size_t m = idx / C;
    int b = m / T, t = m % T;
    int ctx = K - 1;
    float acc = bias[c];
    for (int tap = 0; tap < K; tap++) {
        int p = t + tap - ctx;
        float v = p >= 0 ? x[((size_t)b * T + p) * C + c] : st[((size_t)b * ctx + (ctx + p)) * C + c];
        acc = fmaf(w[c * K + tap], v, acc);
    }
    y[idx] += scale[c] * acc;
}

extern "C" __global__ void line_shift(const float* x, const float* st, float* next, int B, int T, int TP, int C, int CTX) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * CTX * C) return;
    int c = idx % C;
    size_t m = idx / C;
    int b = m / CTX, j = m % CTX;
    int p = TP - CTX + j;
    float v;
    if (p >= T) v = 0.0f;
    else if (p >= 0) v = x[((size_t)b * T + p) * C + c];
    else v = st[((size_t)b * CTX + (CTX + p)) * C + c];
    next[idx] = v;
}
"#;

const LINE_SEQ: &str = r#"extern "C" __global__ void line_layernorm(const float* x, float* y, const float* gamma, const float* beta, int P, int C, float eps) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    size_t p = (size_t)blockIdx.x * (blockDim.x / 32) + warp;
    if (p >= (size_t)P) return;
    const float* row = x + p * C;
    float s = 0.0f;
    for (int c = lane; c < C; c += 32) s += row[c];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    float mean = s / (float)C;
    float q = 0.0f;
    for (int c = lane; c < C; c += 32) {
        float d = row[c] - mean;
        q += d * d;
    }
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
    float r = rsqrtf(q / (float)C + eps);
    for (int c = lane; c < C; c += 32) y[p * C + c] = (row[c] - mean) * r * gamma[c] + beta[c];
}

extern "C" __global__ void line_dwup(const float* x, const float* st, const float* w, const float* bias, float* y, int B, int T, int C, int S) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * T * S * C) return;
    int c = idx % C;
    size_t m = idx / C;
    int r = m % S;
    size_t bt = m / S;
    int t = bt % T, b = bt / T;
    float cur = x[bt * C + c];
    float prev = t > 0 ? x[(bt - 1) * C + c] : st[(size_t)b * C + c];
    size_t n = (size_t)r * C + c;
    y[idx] = bias[n] + fmaf(prev, w[2 * n], cur * w[2 * n + 1]);
}

extern "C" __global__ void line_rope(float* qkv, const float* ring, int B, int T, int H, int D, int S, float theta) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    if (idx >= (size_t)B * T * 2 * H * half) return;
    int i = idx % half;
    size_t m = idx / half;
    int h = m % H;
    m /= H;
    int part = m % 2;
    size_t bt = m / 2;
    int t = bt % T, b = bt / T;
    float inv = 1.0f / powf(theta, (float)(2 * i) / (float)D);
    float f = ((float)((int)ring[(size_t)b * S] + t)) * inv;
    float c = (float)cos((double)f), s = (float)sin((double)f);
    float* v = qkv + bt * 3 * H * D + (size_t)part * H * D + (size_t)h * D + 2 * i;
    float a = v[0], e = v[1];
    v[0] = a * c - e * s;
    v[1] = e * c + a * s;
}

extern "C" __global__ void line_ring_attention(const float* qkv, const float* ring, float* out, int B, int T, int H, int D, int W, float scale) {
    extern __shared__ float ring_smem[];
    float* sc = ring_smem;
    float* qs = sc + W;
    float* red = qs + D;
    int h = blockIdx.x % H;
    size_t bt = blockIdx.x / H;
    int t = bt % T, b = bt / T;
    int DM = H * D;
    const float* rs = ring + (size_t)b * (1 + 2 * (size_t)W * DM);
    const float* rk = rs + 1;
    const float* rv = rk + (size_t)W * DM;
    int p0 = (int)rs[0];
    int a = p0 + t;
    int n = a + 1 < W ? a + 1 : W;
    for (int d = threadIdx.x; d < D; d += blockDim.x) qs[d] = qkv[bt * 3 * DM + (size_t)h * D + d];
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int k = a - n + 1 + j;
        const float* kr = k >= p0 ? qkv + ((size_t)b * T + (k - p0)) * 3 * DM + DM : rk + (size_t)(k % W) * DM;
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc = fmaf(qs[d], kr[(size_t)h * D + d], acc);
        acc *= scale;
        sc[j] = acc;
        mx = fmaxf(mx, acc);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = red[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, red[i]);
        red[32] = m;
    }
    __syncthreads();
    mx = red[32];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += red[i];
        red[33] = s;
    }
    __syncthreads();
    float inv = 1.0f / red[33];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            int k = a - n + 1 + j;
            const float* vr = k >= p0 ? qkv + ((size_t)b * T + (k - p0)) * 3 * DM + 2 * DM : rv + (size_t)(k % W) * DM;
            acc = fmaf(sc[j], vr[(size_t)h * D + d], acc);
        }
        out[bt * DM + (size_t)h * D + d] = acc * inv;
    }
}

extern "C" __global__ void line_ring_store(const float* qkv, float* ring, int B, int T, int DM, int W) {
    int n = T < W ? T : W;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * n * DM) return;
    int e = idx % DM;
    size_t m = idx / DM;
    int i = m % n, b = m / n;
    int t = T - n + i;
    float* rs = ring + (size_t)b * (1 + 2 * (size_t)W * DM);
    int slot = ((int)rs[0] + t) % W;
    const float* row = qkv + ((size_t)b * T + t) * 3 * DM;
    rs[1 + (size_t)slot * DM + e] = row[DM + e];
    rs[1 + (size_t)W * DM + (size_t)slot * DM + e] = row[2 * DM + e];
}

extern "C" __global__ void line_advance(float* ring, int B, int T, int S) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) ring[(size_t)b * S] += (float)T;
}
"#;

const LINE_ARGS: &str = "const float* x, const float* st, const float* w, const float* bias, const float* scale, float* y, int B, int T, int TO";
const LINE_PASS: &str = "x, st, w, bias, scale, y, B, T, TO";

fn line_wrapper(k: &ConvKernel) -> String {
    let s = &k.shape;
    let (res, gelu) = (flag(k.epilogue.residual), flag(k.epilogue.gelu));
    let pad = s.pad.map_or(-1, |p| p as i64);
    let act = crate::conv::line_act_id(k.prologue);
    let call = match k.family {
        Family::Tile => {
            let t = &k.tile;
            format!("line_conv_tile<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {res}, {gelu}, {pad}, {act}>", t.bm, t.bn, t.bk, t.tm, t.tn, t.stages, s.kt, s.stride, s.cin, s.cout, s.phases)
        }
        Family::Direct => format!("line_conv_direct<{}, {}, {}, {}, {}, {res}, {gelu}, {pad}, {act}>", s.kt, s.stride, s.cin, s.cout, s.phases),
        Family::Rows => format!("line_conv_rows<{}, {}, {}, {}, {}, {}, {res}, {gelu}, {pad}, {act}>", s.kt, s.stride, s.cin, s.cout, s.phases, crate::conv::ROWS_CHUNK),
    };
    format!("extern \"C\" __global__ void __launch_bounds__({THREADS}) {}({LINE_ARGS}) {{\n    {call}({LINE_PASS});\n}}\n\n", k.name())
}

fn line_acts() -> Result<String, ConvError> {
    let mut out = String::from("template <int ACT>\n__device__ __forceinline__ float line_act(float v) {\n");
    for (i, name) in crate::conv::LINE_ACTS.iter().enumerate() {
        let e = crate::expr::apply_macro(name, crate::expr::Expr::Raw("v".into())).and_then(|e| e.lower(crate::layout::Dialect::Cuda, &[])).map_err(ConvError)?;
        out += &format!("    if (ACT == {}) return {e};\n", i + 1);
    }
    out += "    return v;\n}\n\n";
    Ok(out)
}

pub fn emit_line(kernels: &[ConvKernel]) -> Result<crate::KirSource, ConvError> {
    let mut out = format!("constexpr int THREAD_ROWS = {THREAD_ROWS};\nconstexpr int THREAD_COLS = {THREAD_COLS};\n\n");
    out += &line_acts()?;
    out += LINE;
    out += LINE_SEQ;
    let mut seen = Vec::new();
    for k in kernels {
        k.validate()?;
        if !k.shape.line {
            return Err(ConvError(format!("{} is not a line conv", k.name())));
        }
        if k.family == Family::Tile && (THREAD_ROWS * k.tile.tm != k.tile.bm || THREAD_COLS * k.tile.tn != k.tile.bn) {
            return Err(ConvError(format!("{}: the tile {:?} does not map onto {THREADS} threads", k.name(), k.tile)));
        }
        if !seen.contains(&k.name()) {
            seen.push(k.name());
            out += &line_wrapper(k);
        }
    }
    Ok(crate::KirSource::emitted(out))
}

pub fn emit_vae(kernels: &[ConvKernel]) -> Result<crate::KirSource, ConvError> {
    let mut out = format!("constexpr int THREAD_ROWS = {THREAD_ROWS};\nconstexpr int THREAD_COLS = {THREAD_COLS};\nconstexpr int MAX_GROUPS = {};\n\n", crate::conv::MAX_GROUPS);
    out += LOAD_INPUT;
    out += TILE_BODY;
    out += DIRECT_BODY;
    out += ROWS;
    let mut seen = Vec::new();
    for k in kernels {
        k.validate()?;
        if THREAD_ROWS * k.tile.tm != k.tile.bm || THREAD_COLS * k.tile.tn != k.tile.bn || (THREAD_ROWS * THREAD_COLS) as u32 != THREADS || !ROW_THREADS.is_multiple_of(32) {
            return Err(ConvError(format!("{}: the tile {:?} does not map onto {THREADS} threads", k.name(), k.tile)));
        }
        if !seen.contains(&k.name()) {
            seen.push(k.name());
            out += &wrapper(k);
        }
    }
    Ok(crate::KirSource::emitted(out))
}
