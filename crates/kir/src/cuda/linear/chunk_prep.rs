use super::prim::{for_stride, kernel, tri_leaf, tri_merge, warp_tiles};

const HEADER: &str = r##"extern "C" __global__ void __launch_bounds__(256) gdn_chunk_prep(const float* qn, const float* kn, const float* mix,
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
if (n0 <= m0 + 15) mma_split_tile<LA_D / 16>(acc, m0, n0, lane, k_row, k_col);
#pragma unroll
for (int e = 0; e < 4; e++) {
    int i = m0 + g8 + (e >> 1) * 8;
    int j = n0 + 2 * t4 + (e & 1);
    A[i * LA_AS + j] = j < i ? -(bs[i] * acc[e]) * expf(gs[i] - gs[j]) : 0.0f;
}
"##;

const QK_SETUP: &str = r##"const float* qc = qn + ((size_t)t0 * LA_KH + kh) * LA_D;
float* ac = aqk + (size_t)(c * LA_VH + h) * C * C;
int live = min(C, tokens - t0);
"##;

const QK: &str = r##"float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
if (n0 <= m0 + 15)
    mma_split_tile<LA_D / 16>(
        acc, m0, n0, lane, [&](int m, int k) { return m < live ? qc[(size_t)m * LA_KH * LA_D + k] : 0.0f; },
        k_col);
#pragma unroll
for (int e = 0; e < 4; e += 2) {
    int i = m0 + g8 + (e >> 1) * 8;
    int j = n0 + 2 * t4;
    float v0 = j <= i && i < live ? acc[e] * expf(gs[i] - gs[j]) : 0.0f;
    float v1 = j + 1 <= i && i < live ? acc[e + 1] * expf(gs[i] - gs[j + 1]) : 0.0f;
    gdn_frag_put<C / 16>((unsigned*)ac, i, j, v0, v1);
}
"##;

const UNIT_DIAGONAL: &str = r##"if (tid < C) A[tid * LA_AS + tid] = 1.0f;
__syncthreads();
const float* vc = mix + (size_t)t0 * LA_CONV_DIM + LA_V_OFF + h * LA_D;
auto t_at = [&](int m, int k) { return k <= m ? A[m * LA_AS + k] : 0.0f; };
"##;

const WY: &str = r##"float uu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float ww[4] = {0.0f, 0.0f, 0.0f, 0.0f};
mma_split_tile<C / 16>(uu, m0, n0, lane, t_at,
                       [&](int k, int n) { return k < live ? vc[(size_t)k * LA_CONV_DIM + n] * bs[k] : 0.0f; });
mma_split_tile<C / 16>(ww, m0, n0, lane, t_at,
                       [&](int k, int n) { return ks[k * LA_KS + n] * bs[k] * expf(gs[k]); });
unsigned* wp = (unsigned*)w + (size_t)(c * LA_VH + h) * C * LA_D;
#pragma unroll
for (int e = 0; e < 4; e += 2) {
    int i = m0 + g8 + (e >> 1) * 8;
    int d = n0 + 2 * t4;
    size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
    u[at] = uu[e];
    u[at + 1] = uu[e + 1];
    gdn_frag_put<LA_D / 16>(wp, i, d, ww[e], ww[e + 1]);
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
gdn_frag_put<LA_D / 16>(qp, m, k, q.x * eg, q.y * eg);
"##;

const K_TAIL: &str = r##"int d = p % LA_D;
int k = (p / LA_D) * 2;
float v0 = 0.0f, v1 = 0.0f;
if (t0 + k < tokens) v0 = kn[((size_t)(t0 + k) * LA_KH + kh) * LA_D + d] * expf(glast - gs[k]);
if (t0 + k + 1 < tokens) v1 = kn[((size_t)(t0 + k + 1) * LA_KH + kh) * LA_D + d] * expf(glast - gs[k + 1]);
gdn_frag_put<C / 16>(kp, d, k, v0, v1);
"##;

pub fn body() -> String {
    let mut o = String::from(SETUP);
    o += &for_stride("i", "tid", "C * LA_D", "blockDim.x", LOAD_K);
    o += CUMSUM;
    o += &warp_tiles("C", "C", KK);
    o += QK_SETUP;
    o += &warp_tiles("C", "C", QK);
    o += "__syncthreads();\n";
    o += &tri_leaf(0);
    o += &tri_merge(0);
    o += UNIT_DIAGONAL;
    o += &warp_tiles("C", "LA_D", WY);
    o += GATES_OUT;
    o += &for_stride("p", "tid", "C * LA_D / 2", "blockDim.x", Q_EXP);
    o += &for_stride("p", "tid", "C * LA_D / 2", "blockDim.x", K_TAIL);
    kernel(HEADER, &o)
}
