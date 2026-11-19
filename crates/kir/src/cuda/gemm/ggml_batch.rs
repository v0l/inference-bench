use super::prim;

pub fn body() -> String {
    [
        prim::batch_helpers("ggr"),
        r##"
__device__ __forceinline__ unsigned ggr_pack(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

"##.to_string(),
        prim::batch_store("ggr"),
        r##"
template <int N, int K, int EPI, bool BIAS, bool SPLIT, int NT, int WARPS, int STAGES, int MJ>
__device__ __forceinline__ void gemv_ggr(const void* w, const unsigned* table, const float* bias, const bf16_t* x_hi,
                                         const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, float* part,
                                         int tokens, int splits) {
    constexpr int M = NT * 8;
    constexpr int PLANE = M * GGR_KC;
    constexpr int STAGE = PLANE * 2;
    constexpr int CHUNKS = K / GGR_KC;
    constexpr int PIECES = GGR_KC / 4;
    constexpr int LOADS = (SPLIT ? 2 : 1) * M * PIECES;
    static_assert(GGR_KC * 2 == GG_SB, "one chunk is half a ggml super block");
    static_assert(K % GGR_KC == 0, "K must be a multiple of the chunk");
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
    const unsigned char* wrow[MJ][2];
    int wtype[MJ][2];
"##.to_string(),
        prim::batch_row_ptrs(4, &["wrow[i][h] = gg_row(w, table, r, wtype[i][h]);".to_string()]),
        prim::batch_stage("GGR_KC", "cp_async8", "ggr"),
        r##"    float acc[MJ][NT][4];
"##.to_string(),
        prim::zero_frag("acc", ("i", "MJ"), ("n", "NT"), 4),
        prim::pipeline_prologue("nc", "stage(st, c_begin + st)"),
        prim::batch_loop_head(),
        r##"        const bf16_t* xh = xs + (it % STAGES) * STAGE;
        const bf16_t* xl = xh + PLANE;
        int chunk = c_begin + it;
        unsigned a[MJ][2][16];
"##.to_string(),
        prim::unroll_nest(
            8,
            &[("i", "MJ"), ("h", "2")],
            true,
            r##"const unsigned char* sb = wrow[i][h] + (size_t)(chunk >> 1) * gg_bytes(wtype[i][h]);
#pragma unroll
for (int q = 0; q < 4; q++) {
    float v[8];
    gg_dequant8(sb, wtype[i][h], (chunk & 1) * 16 + t * 4 + q, v);
#pragma unroll
    for (int e = 0; e < 4; e++) a[i][h][4 * q + e] = ggr_pack(v[2 * e], v[2 * e + 1]);
}
"##,
        ),
        r##"#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint2 bh[NT], bl[NT];
#pragma unroll
            for (int n = 0; n < NT; n++) {
                bh[n] = *(const uint2*)(xh + ((k * M + n * 8 + g) * 4 + t) * 4);
                if (SPLIT) bl[n] = *(const uint2*)(xl + ((k * M + n * 8 + g) * 4 + t) * 4);
            }
#pragma unroll
            for (int i = 0; i < MJ; i++) {
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    if (SPLIT) mma_bf16(acc[i][n], a[i][0][2 * k], a[i][1][2 * k], a[i][0][2 * k + 1], a[i][1][2 * k + 1], bl[n].x, bl[n].y);
                    mma_bf16(acc[i][n], a[i][0][2 * k], a[i][1][2 * k], a[i][0][2 * k + 1], a[i][1][2 * k + 1], bh[n].x, bh[n].y);
                }
            }
        }
    }
"##.to_string(),
        prim::batch_epilogue("ggr", "acc[i][n][e]", "BIAS ? v + bias[r] : v"),
        prim::batch_reduce("ggr", true),
    ]
    .concat()
}
