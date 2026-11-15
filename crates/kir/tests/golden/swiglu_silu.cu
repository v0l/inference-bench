extern "C" __global__ void __launch_bounds__(256) swiglu(const float* gate_up, float* out) {
    constexpr int WIDTH = 3072;
    gate_up += (size_t)blockIdx.y * 2 * WIDTH;
    out += (size_t)blockIdx.y * WIDTH;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WIDTH; i += gridDim.x * blockDim.x) {
        float g = gate_up[i * 2];
        float a = g / (1.0f + expf(-g));
        out[i] = a * gate_up[i * 2 + 1];
    }
}

