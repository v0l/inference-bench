use super::prim;

pub fn body() -> String {
    [
        prim::batch_helpers("f4b", prim::L2_PREFETCH_128),
        prim::bf16x2_decode("f4b"),
        r##"
__device__ __forceinline__ unsigned f4b_byte(const uint4& q, int i) {
    unsigned word = (i >> 2) == 0 ? q.x : (i >> 2) == 1 ? q.y : (i >> 2) == 2 ? q.z : q.w;
    return (word >> (8 * (i & 3))) & 0xffu;
}

"##.to_string(),
        prim::batch_store("f4b"),
        r##"
template <int N, int K, int EPI, int NT, int WARPS, int STAGES, int MJ>
__device__ __forceinline__ void gemv_f4b(const void* w, const unsigned char* scales, const bf16_t* x_hi, const bf16_t* x_lo,
                                         float* out, bf16_t* out_hi, bf16_t* out_lo, float* part, int tokens, int splits) {
    constexpr int M = NT * 8;
    constexpr int PLANE = M * F4B_KC;
    constexpr int STAGE = PLANE * 2;
    constexpr int CHUNKS = K / F4B_KC;
    constexpr int KG = K / FP4_GROUP;
    constexpr int PIECES = F4B_KC / 4;
    constexpr int LOADS = 2 * M * PIECES;
    static_assert(F4B_KC == 128, "four lanes of 16 bytes span one chunk of a row");
    static_assert(K % F4B_KC == 0, "K must be a multiple of the chunk");
    static_assert(N % 16 == 0, "rows come in mma tiles of 16");
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
    const unsigned char* sb = scales + FP4_HEADER;
    const unsigned char* wrow[MJ][2];
    const unsigned char* srow[MJ][2];
"##.to_string(),
        prim::batch_row_ptrs(4, &["wrow[i][h] = wb + (size_t)r * (K / 2) + t * 16;".to_string(), "srow[i][h] = sb + (size_t)r * KG + 2 * t;".to_string()]),
        prim::batch_stage("F4B_KC", "cp_async8", "f4b"),
        prim::batch_load_w(
            "uint4 (&q)[MJ][2], unsigned (&s)[MJ][2]",
            &["q[i][h] = f4b_load16(wrow[i][h] + (size_t)c * (F4B_KC / 2));", "s[i][h] = __ldg((const unsigned short*)(srow[i][h] + c * (F4B_KC / FP4_GROUP)));"],
        ),
        r##"    float acc[MJ][NT][4];
"##.to_string(),
        prim::zero_frag("acc", ("i", "MJ"), ("n", "NT"), 4),
        prim::pipeline_prologue("nc", "stage(st, c_begin + st)"),
        r##"    uint4 q[MJ][2];
    unsigned s[MJ][2];
    if (nc > 0) load_w(c_begin, q, s);
"##.to_string(),
        prim::batch_loop_head(),
        r##"        uint4 nq[MJ][2];
        unsigned ns[MJ][2];
        if (it + 1 < nc) load_w(c_begin + it + 1, nq, ns);
        const bf16_t* xh = xs + (it % STAGES) * STAGE;
        const bf16_t* xl = xh + PLANE;
        unsigned sc[MJ][2][2];
"##.to_string(),
        prim::unroll_nest(8, &[("i", "MJ"), ("h", "2")], true, r##"sc[i][h][0] = f4b_scale_bf16x2(s[i][h] & 0xffu);
sc[i][h][1] = f4b_scale_bf16x2((s[i][h] >> 8) & 0xffu);
"##),
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
                unsigned s0 = sc[i][0][k >> 2];
                unsigned s1 = sc[i][1][k >> 2];
                unsigned a0 = f4b_e2m1x2_bf16(f4b_byte(q[i][0], 2 * k), s0);
                unsigned a1 = f4b_e2m1x2_bf16(f4b_byte(q[i][1], 2 * k), s1);
                unsigned a2 = f4b_e2m1x2_bf16(f4b_byte(q[i][0], 2 * k + 1), s0);
                unsigned a3 = f4b_e2m1x2_bf16(f4b_byte(q[i][1], 2 * k + 1), s1);
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    mma_bf16(acc[i][n], a0, a1, a2, a3, bl[n].x, bl[n].y);
                    mma_bf16(acc[i][n], a0, a1, a2, a3, bh[n].x, bh[n].y);
                }
            }
        }
        if (it + 1 < nc) {
"##.to_string(),
        prim::unroll_nest(12, &[("i", "MJ"), ("h", "2")], true, r##"q[i][h] = nq[i][h];
s[i][h] = ns[i][h];
"##),
        r##"        }
    }
"##.to_string(),
        prim::batch_epilogue("f4b", "acc[i][n][e] * inv", "v"),
        prim::batch_reduce("f4b", false),
    ]
    .concat()
}
