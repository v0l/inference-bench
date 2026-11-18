pub fn body() -> String {
    r##"template <int N, int K, int ROWS>
__device__ __forceinline__ void head_norm_f4(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {
    static_assert(K == HIDDEN, "the head reads the normed hidden row");
    __shared__ __align__(16) float xt[K];
    load_normed_t(x, nw, xt);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (FP4_GEMV_THREADS / 32) + warp) * ROWS;
    if (row >= N) return;
    float acc[ROWS];
    DecFp4::rows<K, ROWS>(w, s, row, 1, xt, lane, acc);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) out[row + r] = acc[r];
    }
}
"##.to_string()
}
