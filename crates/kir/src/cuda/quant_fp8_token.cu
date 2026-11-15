extern "C" __global__ void quant_rows_fp8_token(const bf16_t* hi, const bf16_t* lo, unsigned char* q, float* scale, int width) {
    __shared__ float scratch[32];
    size_t row = (size_t)blockIdx.x * width;
    float amax = 0.0f;
    for (int c = threadIdx.x * 2; c < width; c += blockDim.x * 2) {
        float a = bf16_to_f32(hi[row + c]) + bf16_to_f32(lo[row + c]);
        float b = bf16_to_f32(hi[row + c + 1]) + bf16_to_f32(lo[row + c + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
    }
    amax = block_reduce<true>(amax, scratch);
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    for (int c = threadIdx.x * 2; c < width; c += blockDim.x * 2) {
        float a = bf16_to_f32(hi[row + c]) + bf16_to_f32(lo[row + c]);
        float b = bf16_to_f32(hi[row + c + 1]) + bf16_to_f32(lo[row + c + 1]);
        *(unsigned short*)(q + row + c) = (unsigned short)f32x2_to_e4m3x2(a / s, b / s);
    }
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
}
