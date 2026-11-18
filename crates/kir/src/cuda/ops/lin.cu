extern "C" __global__ void lin_in_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<LIN_WF, LA_IN_ROWS, false>(w, s, b, x, nw, out);
}

extern "C" __global__ void lin_ba_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<DecBf16, LA_BA_ROWS, false>(w, s, b, x, nw, out);
}

constexpr int BA_ROWS_MAX_TOKENS = 32;
constexpr int BA_ROWS_THREADS = 256;

extern "C" __global__ void lin_ba_rows(const bf16_t* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl,
                                       float* out, const bf16_t* dummy_hi, const bf16_t* dummy_lo, int tokens) {
    constexpr int K = HIDDEN;
    static_assert(K % (8 * BA_ROWS_THREADS / 4) == 0 || K % 8 == 0, "rows are read 8 bf16 at a time");
    __shared__ float part[BA_ROWS_THREADS / 32][BA_ROWS_MAX_TOKENS];
    int row = blockIdx.x;
    int m0 = blockIdx.y * BA_ROWS_MAX_TOKENS;
    int chunk = min(tokens - m0, BA_ROWS_MAX_TOKENS);
    int tid = threadIdx.x;
    float acc[BA_ROWS_MAX_TOKENS];
#pragma unroll
    for (int m = 0; m < BA_ROWS_MAX_TOKENS; m++) acc[m] = 0.0f;
    const uint4* wr = (const uint4*)(w + (size_t)row * K);
    for (int i = tid; i < K / 8; i += BA_ROWS_THREADS) {
        uint4 q = wr[i];
        float wf[8];
        unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wf[2 * j] = __uint_as_float(words[j] << 16);
            wf[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u);
        }
#pragma unroll
        for (int m = 0; m < BA_ROWS_MAX_TOKENS; m++) {
            if (m < chunk) {
                uint4 h = ((const uint4*)(xh + (size_t)(m0 + m) * K))[i];
                uint4 l = ((const uint4*)(xl + (size_t)(m0 + m) * K))[i];
                unsigned hw[4] = {h.x, h.y, h.z, h.w};
                unsigned lw[4] = {l.x, l.y, l.z, l.w};
                float a = acc[m];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    a += wf[2 * j] * (__uint_as_float(hw[j] << 16) + __uint_as_float(lw[j] << 16));
                    a += wf[2 * j + 1] * (__uint_as_float(hw[j] & 0xffff0000u) + __uint_as_float(lw[j] & 0xffff0000u));
                }
                acc[m] = a;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < BA_ROWS_MAX_TOKENS; m++) {
        float v = acc[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if ((tid & 31) == 0) part[tid >> 5][m] = v;
    }
    __syncthreads();
    if (tid < chunk) {
        float v = 0.0f;
#pragma unroll
        for (int wp = 0; wp < BA_ROWS_THREADS / 32; wp++) v += part[wp][tid];
        out[(size_t)(m0 + tid) * LA_BA_ROWS + row] = v;
    }
}

extern "C" __global__ void lin_out_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    linear_w<LIN_WF, HIDDEN, LA_VALUE_DIM, false, true>(w, s, b, x, out);
}
