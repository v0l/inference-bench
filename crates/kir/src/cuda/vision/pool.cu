
__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < blockDim.x / 32; i++) t += red[i];
    return t;
}

extern "C" __global__ void vis_pool(const float* x, const bf16_t* w, const bf16_t* b, const float* soft, bf16_t* hi, bf16_t* lo,
                                    int hidden, int side, int kernel, int out_side, float eps, float soft_eps) {
    extern __shared__ float acc[];
    __shared__ float red[32];
    int o = blockIdx.x;
    int per_tile = out_side * out_side;
    int base = (o / per_tile) * side * side;
    int local = o % per_tile;
    int oy = local / out_side, ox = local % out_side;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();
    for (int dy = 0; dy < kernel; dy++) {
        for (int dx = 0; dx < kernel; dx++) {
            const float* row = x + (size_t)(base + (oy * kernel + dy) * side + ox * kernel + dx) * hidden;
            float s = 0.0f;
            for (int i = threadIdx.x; i < hidden; i += blockDim.x) s += row[i];
            float mean = block_sum(s, red) / hidden;
            float s2 = 0.0f;
            for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
                float d = row[i] - mean;
                s2 += d * d;
            }
            float inv = rsqrtf(block_sum(s2, red) / hidden + eps);
            for (int i = threadIdx.x; i < hidden; i += blockDim.x) acc[i] += (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
        }
    }
    float scale = 1.0f / (kernel * kernel);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        acc[i] *= scale;
        ss += acc[i] * acc[i];
    }
    float rinv = rsqrtf(block_sum(ss, red) / hidden + soft_eps);
    __syncthreads();
    size_t at = (size_t)o * hidden;
    for (int i = threadIdx.x * 2; i < hidden; i += blockDim.x * 2) store_act2(hi, lo, at + i, acc[i] * rinv * soft[i], acc[i + 1] * rinv * soft[i + 1]);
}
