use super::prim::{at, for_stride, kernel, lower_tiles, tri_leaf, tri_merge, upper_tiles};

const HEADER: &str = r##"extern "C" __global__ void __launch_bounds__(256, 2) gdn_chunk_prep(const float* qn, const float* kn, const float* mix,
                                                                 const float* g, const float* beta, float* w,
                                                                 float* u, float* aqk, float* gc, float* qe, float* kt, int tokens)"##;

const SETUP: &str = r##"constexpr int C = LA_CHUNK;
extern __shared__ __align__(16) unsigned char la_smem[];
float* ks = (float*)la_smem;
float* A = ks + C * LA_KS;
float* gs = A + C * LA_AS;
float* bs = gs + C;
int h = blockIdx.x;
int kh = la_kh(h);
int c = blockIdx.y;
int t0 = c * C;
int tid = threadIdx.x;
if (tid < C) {
    int t = t0 + tid;
    gs[tid] = t < tokens ? g[(size_t)t * LA_VH + h] : 0.0f;
    bs[tid] = t < tokens ? beta[(size_t)t * LA_VH + h] : 0.0f;
}
"##;

const LOAD_K: &str = r##"int r = i / LA_D;
int d = i % LA_D;
int t = t0 + r;
ks[r * LA_KS + d] = t < tokens ? kn[((size_t)t * LA_KH + kh) * LA_D + d] : 0.0f;
"##;

const CUMSUM: &str = r##"__syncthreads();
if (tid == 0)
    for (int i = 1; i < C; i++) gs[i] += gs[i - 1];
__syncthreads();
constexpr int WARPS = 8;
int lane = tid & 31;
int warp = tid >> 5;
int g8 = lane >> 2;
int t4 = lane & 3;
auto k_row = [&](int m, int k) { return ks[m * LA_KS + k]; };
auto k_col = [&](int k, int n) { return ks[n * LA_KS + k]; };
"##;

const KK: &str = r##"float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
mma_split_tile<LA_D / 16, LA_LO>(acc, m0, n0, lane, k_row, k_col);
#pragma unroll
for (int e = 0; e < 4; e++) {
    int i = m0 + g8 + (e >> 1) * 8;
    int j = n0 + 2 * t4 + (e & 1);
    A[i * LA_AS + j] = j < i ? -(bs[i] * acc[e]) * expf(gs[i] - gs[j]) : 0.0f;
}
"##;

const KK_UPPER: &str = r##"#pragma unroll
for (int e = 0; e < 4; e++) A[(m0 + g8 + (e >> 1) * 8) * LA_AS + n0 + 2 * t4 + (e & 1)] = 0.0f;
"##;

const QK_SETUP: &str = r##"const float* qc = qn + ((size_t)t0 * LA_KH + kh) * LA_D;
float* ac = aqk + (size_t)(c * LA_VH + h) * C * C;
int live = min(C, tokens - t0);
"##;

const QK: &str = r##"float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
mma_split_tile<LA_D / 16, LA_LO>(
    acc, m0, n0, lane, [&](int m, int k) { return m < live ? qc[(size_t)m * LA_KH * LA_D + k] : 0.0f; }, k_col);
#pragma unroll
for (int e = 0; e < 4; e += 2) {
    int i = m0 + g8 + (e >> 1) * 8;
    int j = n0 + 2 * t4;
    float v0 = j <= i && i < live ? acc[e] * expf(gs[i] - gs[j]) : 0.0f;
    float v1 = j + 1 <= i && i < live ? acc[e + 1] * expf(gs[i] - gs[j + 1]) : 0.0f;
    gdn_frag_put<C / 16, LA_LO>((unsigned*)ac, i, j, v0, v1);
}
"##;

const QK_UPPER: &str = r##"#pragma unroll
for (int e = 0; e < 4; e += 2) gdn_frag_put<C / 16, LA_LO>((unsigned*)ac, m0 + g8 + (e >> 1) * 8, n0 + 2 * t4, 0.0f, 0.0f);
"##;

const UNIT_DIAGONAL: &str = r##"if (tid < C) A[tid * LA_AS + tid] = 1.0f;
__syncthreads();
const float* vc = mix + (size_t)t0 * LA_CONV_DIM + LA_V_OFF + h * LA_D;
auto t_at = [&](int m, int k) { return k <= m ? A[m * LA_AS + k] : 0.0f; };
"##;

const WY: &str = r##"constexpr int MB = C / 16;
unsigned* wp = (unsigned*)w + (size_t)(c * LA_VH + h) * C * LA_D;
for (int nt = warp; nt < LA_D / 8; nt += WARPS) {
    int n = nt * 8 + g8;
    float vr[MB][4];
#pragma unroll
    for (int kk = 0; kk < MB; kk++) {
        int k = kk * 16 + 2 * t4;
        vr[kk][0] = k < live ? vc[(size_t)k * LA_CONV_DIM + n] : 0.0f;
        vr[kk][1] = k + 1 < live ? vc[(size_t)(k + 1) * LA_CONV_DIM + n] : 0.0f;
        vr[kk][2] = k + 8 < live ? vc[(size_t)(k + 8) * LA_CONV_DIM + n] : 0.0f;
        vr[kk][3] = k + 9 < live ? vc[(size_t)(k + 9) * LA_CONV_DIM + n] : 0.0f;
    }
    float uu[MB][4], ww[MB][4];
#pragma unroll
    for (int mb = 0; mb < MB; mb++)
#pragma unroll
        for (int e = 0; e < 4; e++) uu[mb][e] = ww[mb][e] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < MB; kk++) {
        int k = kk * 16 + 2 * t4;
        auto kb = [&](int r) { return ks[r * LA_KS + n] * bs[r] * expf(gs[r]); };
        unsigned uh0, ul0, uh1, ul1, wh0, wl0, wh1, wl1;
        gdn_pack2(vr[kk][0] * bs[k], vr[kk][1] * bs[k + 1], uh0, ul0);
        gdn_pack2(vr[kk][2] * bs[k + 8], vr[kk][3] * bs[k + 9], uh1, ul1);
        gdn_pack2(kb(k), kb(k + 1), wh0, wl0);
        gdn_pack2(kb(k + 8), kb(k + 9), wh1, wl1);
#pragma unroll
        for (int mb = kk; mb < MB; mb++) {
            int m0 = mb * 16;
            unsigned th[4], tl[4];
            gdn_pack2(t_at(m0 + g8, k), t_at(m0 + g8, k + 1), th[0], tl[0]);
            gdn_pack2(t_at(m0 + g8 + 8, k), t_at(m0 + g8 + 8, k + 1), th[1], tl[1]);
            gdn_pack2(t_at(m0 + g8, k + 8), t_at(m0 + g8, k + 9), th[2], tl[2]);
            gdn_pack2(t_at(m0 + g8 + 8, k + 8), t_at(m0 + g8 + 8, k + 9), th[3], tl[3]);
            if (LA_LO) {
                mma_bf16(uu[mb], tl[0], tl[1], tl[2], tl[3], uh0, uh1);
                mma_bf16(uu[mb], th[0], th[1], th[2], th[3], ul0, ul1);
            }
            mma_bf16(uu[mb], th[0], th[1], th[2], th[3], uh0, uh1);
            if (LA_LO) {
                mma_bf16(ww[mb], tl[0], tl[1], tl[2], tl[3], wh0, wh1);
                mma_bf16(ww[mb], th[0], th[1], th[2], th[3], wl0, wl1);
            }
            mma_bf16(ww[mb], th[0], th[1], th[2], th[3], wh0, wh1);
        }
    }
#pragma unroll
    for (int mb = 0; mb < MB; mb++)
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
            int i = mb * 16 + g8 + (e >> 1) * 8;
            int d = nt * 8 + 2 * t4;
            size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
            u[at] = uu[mb][e];
            u[at + 1] = uu[mb][e + 1];
            gdn_frag_put<LA_D / 16, LA_LO>(wp, i, d, ww[mb][e], ww[mb][e + 1]);
        }
}
"##;

const GATES_OUT: &str = r##"if (tid < C) gc[(size_t)(c * LA_VH + h) * C + tid] = gs[tid];
float glast = gs[C - 1];
unsigned* qp = (unsigned*)qe + (size_t)(c * LA_VH + h) * C * LA_D;
unsigned* kp = (unsigned*)kt + (size_t)(c * LA_VH + h) * C * LA_D;
"##;

const Q_EXP: &str = r##"int m = p / (LA_D / 2);
int k = (p % (LA_D / 2)) * 2;
float2 q = t0 + m < tokens ? *(const float2*)(qn + ((size_t)(t0 + m) * LA_KH + kh) * LA_D + k) : make_float2(0.0f, 0.0f);
float eg = t0 + m < tokens ? expf(gs[m]) : 0.0f;
gdn_frag_put<LA_D / 16, LA_LO>(qp, m, k, q.x * eg, q.y * eg);
"##;

const K_TAIL: &str = r##"int d = p % LA_D;
int k = (p / LA_D) * 2;
float v0 = ks[k * LA_KS + d] * expf(glast - gs[k]);
float v1 = ks[(k + 1) * LA_KS + d] * expf(glast - gs[k + 1]);
if (t0 + k >= tokens) v0 = 0.0f;
if (t0 + k + 1 >= tokens) v1 = 0.0f;
gdn_frag_put<C / 16, LA_LO>(kp, d, k, v0, v1);
"##;

pub fn body() -> String {
    let mut o = String::from(SETUP);
    o += &for_stride("i", "tid", "C * LA_D", "blockDim.x", LOAD_K);
    o += CUMSUM;
    o += &lower_tiles(KK);
    o += &upper_tiles(KK_UPPER);
    o += QK_SETUP;
    o += &lower_tiles(QK);
    o += &upper_tiles(QK_UPPER);
    o += "__syncthreads();\n";
    o += &tri_leaf(0);
    o += &tri_merge(0);
    o += UNIT_DIAGONAL;
    o += &format!("{{\n{}}}\n", at(4, WY));
    o += GATES_OUT;
    o += &for_stride("p", "tid", "C * LA_D / 2", "blockDim.x", Q_EXP);
    o += &for_stride("p", "tid", "C * LA_D / 2", "blockDim.x", K_TAIL);
    kernel(HEADER, &o)
}
