__device__ __forceinline__ float kir_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ float kir_block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    v = kir_warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < warps; i++) r += scratch[i];
    return r;
}

extern "C" __global__ void __launch_bounds__(1024) rmsnorm(const float* x, const float* weight, float* out) {
    constexpr int WIDTH = 1024;
    __shared__ float scratch[32];
    x += (size_t)blockIdx.x * WIDTH;
    out += (size_t)blockIdx.x * WIDTH;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) ss += x[i] * x[i];
    float inv = rsqrtf(kir_block_sum(ss, scratch) / WIDTH + 1e-6f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) out[i] = weight[i] * (x[i] * inv);
}

