use super::prim::{acc_row, acc_zero, at, chunk_ptrs, mma_planes, nest, split_plane_store, state_tile_load, state_tile_store, store_state_planes, FRAG, FRAG_NE};

pub fn body() -> String {
    let mut o = String::new();
    o.push_str(
        "extern \"C\" __global__ void __launch_bounds__(LA_CHUNK * 2) gdn_chunk_scan(const float* w, const float* qe, const float* kt,
                                                                           const float* u, const float* aqk, const float* gc,
                                                                           float* state, float* o, int slot, int tokens) {
    constexpr int C = LA_CHUNK;
    constexpr int WARPS = C / 16;
    constexpr int NT = LA_COLS / 8;
    constexpr int KD = LA_D / 16;
    constexpr int KC = C / 16;
    constexpr int DPW = (KD + WARPS - 1) / WARPS;
    constexpr int SP = LA_D + 8;
    constexpr int VP = C + 8;
    __shared__ unsigned short sh[LA_COLS * SP];
    __shared__ unsigned short sl[LA_COLS * SP];
    __shared__ unsigned short vh[LA_COLS * VP];
    __shared__ unsigned short vl[LA_COLS * VP];
    int h = blockIdx.x;
    int c0 = blockIdx.y * LA_COLS;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int g = lane >> 2;
    int t4 = lane & 3;
    float* st = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    float sr[DPW][NT][4];
",
    );
    o.push_str(&state_tile_load(4));
    o.push_str(&store_state_planes(4));
    o.push_str(&at(
        4,
        "store_state_planes();
__syncthreads();
int chunks = (tokens + C - 1) / C;
for (int c = 0; c < chunks; c++) {
",
    ));
    o.push_str(&chunk_ptrs(8));
    o.push_str(&acc_zero(8));
    o.push_str(&mma_planes(8, "KD", "acc", "wp", "sh", "sl", "SP"));
    o.push_str(&at(8, "const float* uc = u + ((size_t)t0 * LA_VH + h) * LA_D + c0;\n"));
    let sub = format!("{}float v = uc[(size_t)m * LA_VH * LA_D + col] - acc[n][e];\n{}", acc_row(), split_plane_store(0, "vh", "vl", "VP", "m", "col", "v"));
    o.push_str(&nest(8, &FRAG_NE, &sub));
    o.push_str(&acc_zero(8));
    o.push_str(&mma_planes(8, "KD", "acc", "qp", "sh", "sl", "SP"));
    o.push_str(&at(8, "__syncthreads();\n"));
    o.push_str(&mma_planes(8, "KC", "acc", "ap", "vh", "vl", "VP"));
    let out = format!("{}if (t0 + m < tokens) o[((size_t)(t0 + m) * LA_VH + h) * LA_D + c0 + col] = acc[n][e];\n", acc_row());
    o.push_str(&nest(8, &FRAG_NE, &out));
    o.push_str(&at(
        8,
        "#pragma unroll
for (int j = 0; j < DPW; j++) {
    if (warp + j * WARPS >= KD) continue;
    const uint4* kj = kp + ch * (C * LA_D / 4) + LA_FS * ((warp + j * WARPS) * KC * 32 + lane);
    float a2[NT][4];
",
    ));
    o.push_str(&nest(12, &FRAG_NE, "a2[n][e] = sr[j][n][e] * decay;"));
    o.push_str(&mma_planes(12, "KC", "a2", "kj", "vh", "vl", "VP"));
    o.push_str(&nest(12, &FRAG_NE, "sr[j][n][e] = a2[n][e];"));
    o.push_str(&at(
        8,
        "}
__syncthreads();
store_state_planes();
__syncthreads();
",
    ));
    o.push_str("    }\n");
    o.push_str(&state_tile_store(4));
    o.push_str("}\n");
    let _ = FRAG;
    o
}
