
extern "C" __global__ void vis_split(const float* x, bf16_t* hi, bf16_t* lo, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;
    float2 v = *(const float2*)(x + i);
    store_act2(hi, lo, i, v.x, v.y);
}

extern "C" __global__ void vis_layernorm(const float* x, const bf16_t* w, const bf16_t* b, bf16_t* hi, bf16_t* lo, int width, float eps) {
    __shared__ float red[2][32];
    const float* row = x + (size_t)blockIdx.x * width;
    float s = 0.0f, s2 = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = row[i];
        s += v;
        s2 += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        red[0][warp] = s;
        red[1][warp] = s2;
    }
    __syncthreads();
    s = 0.0f;
    s2 = 0.0f;
    for (int i = 0; i < blockDim.x / 32; i++) {
        s += red[0][i];
        s2 += red[1][i];
    }
    float mean = s / width;
    float var = s2 / width - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    size_t base = (size_t)blockIdx.x * width;
    for (int i = threadIdx.x * 2; i < width; i += blockDim.x * 2) {
        float a = (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
        float c = (row[i + 1] - mean) * inv * bf(w[i + 1]) + bf(b[i + 1]);
        store_act2(hi, lo, base + i, a, c);
    }
}

