extern "C" __global__ void qkv_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    linear_w<DecBf16, 4096, 1024, false, false>(w, s, b, x, out);
}

extern "C" __global__ void qkv_gemm(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm<4096, 1024, true, false, DecQ8>(w, s, b, x, out, tokens);
}

extern "C" __global__ void lm_head_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<DecFp8, 151936, false>(w, s, b, x, nw, out);
}

extern "C" __global__ void gate_up_act(const void* w, const float* s, const float* x, const float* nw, float* out) {
    gate_up_act_norm<DecBf16>(w, s, x, nw, out);
}

extern "C" __global__ void lin_ba_rows(const bf16_t* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl,
                                       float* out, const bf16_t* dummy_hi, const bf16_t* dummy_lo, int tokens) {
    __shared__ float part[8][32];
    int row = blockIdx.x;
    int m0 = blockIdx.y * 32;
    int chunk = min(tokens - m0, 32);
    int tid = threadIdx.x;
    float acc[32];
#pragma unroll
    for (int m = 0; m < 32; m++) acc[m] = 0.0f;
    const uint4* wr = (const uint4*)(w + (size_t)row * 5120);
    for (int i = tid; i < 640; i += 256) {
        uint4 q = wr[i];
        float wf[8];
        unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wf[2 * j] = __uint_as_float(words[j] << 16);
            wf[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u);
        }
#pragma unroll
        for (int m = 0; m < 32; m++) {
            if (m < chunk) {
                uint4 h = ((const uint4*)(xh + (size_t)(m0 + m) * 5120))[i];
                uint4 l = ((const uint4*)(xl + (size_t)(m0 + m) * 5120))[i];
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
    for (int m = 0; m < 32; m++) {
        float v = acc[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if ((tid & 31) == 0) part[tid >> 5][m] = v;
    }
    __syncthreads();
    if (tid < chunk) {
        float v = 0.0f;
#pragma unroll
        for (int wp = 0; wp < 8; wp++) v += part[wp][tid];
        out[(size_t)(m0 + tid) * 96 + row] = v;
    }
}

