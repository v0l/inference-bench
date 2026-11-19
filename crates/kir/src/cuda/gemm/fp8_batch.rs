use super::prim;

pub fn body() -> String {
    [
        prim::batch_helpers("f8b"),
        r##"
__device__ __forceinline__ unsigned f8b_word(const uint4& a, const uint4& b, int i) {
    const uint4& q = i < 4 ? a : b;
    int j = i & 3;
    return j == 0 ? q.x : j == 1 ? q.y : j == 2 ? q.z : q.w;
}

"##.to_string(),
        prim::batch_store("f8b"),
        r##"
template <int N, int K, int EPI, bool BIAS, int NT, int WARPS, int STAGES, int MJ>
__device__ __forceinline__ void gemv_f8b(const void* w, const float* scales, const float* bias, const bf16_t* x_hi,
                                         const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, float* part,
                                         int tokens, int splits) {
    constexpr int M = NT * 8;
    constexpr int PLANE = M * F8B_KC;
    constexpr int STAGE = PLANE * 2;
    constexpr int CHUNKS = K / F8B_KC;
    constexpr int PIECES = F8B_KC / 4;
    constexpr int LOADS = 2 * M * PIECES;
    static_assert(F8B_KC == 128, "four lanes of 32 bytes span one chunk of a row");
    static_assert(F8B_KC == FP8_BLOCK, "one scale per row and chunk");
    static_assert(K % F8B_KC == 0, "K must be a multiple of the chunk");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    bf16_t* xs = (bf16_t*)gemm_smem;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int split = blockIdx.y;
    int c_begin = split * CHUNKS / splits;
    int nc = (split + 1) * CHUNKS / splits - c_begin;
    int row0 = blockIdx.x * (WARPS * MJ * 16) + warp * (MJ * 16);
    int last = max(tokens - 1, 0);
    const unsigned char* wb = (const unsigned char*)w;
    const unsigned char* wrow[MJ][2];
    const float* srow[MJ][2];
"##.to_string(),
        prim::batch_row_ptrs(4, &["wrow[i][h] = wb + (size_t)r * K + t * 32;".to_string(), "srow[i][h] = fp8_row_scales(scales, r);".to_string()]),
        prim::batch_stage("F8B_KC", "cp_async8", "f8b"),
        prim::batch_load_w(
            "uint4 (&q)[MJ][2][2], float (&s)[MJ][2]",
            &["q[i][h][0] = f8b_load16(wrow[i][h] + (size_t)c * F8B_KC);", "q[i][h][1] = f8b_load16(wrow[i][h] + (size_t)c * F8B_KC + 16);", "s[i][h] = __ldg(srow[i][h] + c);"],
        ),
        r##"    float acc[MJ][NT][4];
"##.to_string(),
        prim::zero_frag("acc", ("i", "MJ"), ("n", "NT"), 4),
        prim::pipeline_prologue("nc", "stage(st, c_begin + st)"),
        r##"    uint4 q[MJ][2][2];
    float s[MJ][2];
    if (nc > 0) load_w(c_begin, q, s);
"##.to_string(),
        prim::batch_loop_head(),
        r##"        uint4 nq[MJ][2][2];
        float ns[MJ][2];
        if (it + 1 < nc) load_w(c_begin + it + 1, nq, ns);
        const bf16_t* xh = xs + (it % STAGES) * STAGE;
        const bf16_t* xl = xh + PLANE;
        float c[MJ][NT][4];
"##.to_string(),
        prim::zero_frag("c", ("i", "MJ"), ("n", "NT"), 8),
        r##"#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint2 bh[NT], bl[NT];
#pragma unroll
            for (int n = 0; n < NT; n++) {
                bh[n] = *(const uint2*)(xh + ((k * M + n * 8 + g) * 4 + t) * 4);
                bl[n] = *(const uint2*)(xl + ((k * M + n * 8 + g) * 4 + t) * 4);
            }
#pragma unroll
            for (int i = 0; i < MJ; i++) {
                unsigned w0 = f8b_word(q[i][0][0], q[i][0][1], k);
                unsigned w1 = f8b_word(q[i][1][0], q[i][1][1], k);
                unsigned a0 = e4m3_pair_bf16(w0 & 0xffffu);
                unsigned a1 = e4m3_pair_bf16(w1 & 0xffffu);
                unsigned a2 = e4m3_pair_bf16(w0 >> 16);
                unsigned a3 = e4m3_pair_bf16(w1 >> 16);
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    mma_bf16(c[i][n], a0, a1, a2, a3, bl[n].x, bl[n].y);
                    mma_bf16(c[i][n], a0, a1, a2, a3, bh[n].x, bh[n].y);
                }
            }
        }
"##.to_string(),
        prim::unroll_nest(8, &[("i", "MJ"), ("n", "NT"), ("e", "4")], false, "acc[i][n][e] += c[i][n][e] * s[i][e >> 1];"),
        r##"        if (it + 1 < nc) {
"##.to_string(),
        prim::unroll_nest(12, &[("i", "MJ"), ("h", "2")], true, r##"q[i][h][0] = nq[i][h][0];
q[i][h][1] = nq[i][h][1];
s[i][h] = ns[i][h];
"##),
        r##"        }
    }
"##.to_string(),
        prim::batch_epilogue("f8b", "acc[i][n][e]", "BIAS ? v + bias[r] : v"),
        prim::batch_reduce("f8b", true),
    ]
    .concat()
}
