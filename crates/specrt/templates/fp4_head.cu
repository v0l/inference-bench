constexpr int HEAD_F4_ROWS = {{HEAD_F4_ROWS}};

extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) lm_head_norm_f4(const void* w, const unsigned char* s, const float* b, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xt[HIDDEN];
    load_normed_t(x, nw, xt);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (FP4_GEMV_THREADS / 32) + warp) * HEAD_F4_ROWS;
    if (row >= VOCAB) return;
    float acc[HEAD_F4_ROWS];
    fp4_rows_dot<HIDDEN, HEAD_F4_ROWS, 4 / HEAD_F4_ROWS>(w, s, row, 1, xt, lane, acc);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < HEAD_F4_ROWS; r++) out[row + r] = acc[r];
    }
}

{{HEAD_F4W_KERNELS}}
