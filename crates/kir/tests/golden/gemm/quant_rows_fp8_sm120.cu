// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm quant_rows_fp8 block=QuantRowsFp8 threads=64 smem=0
//   QuantFp8 { scale: Block }
extern "C" __global__ void quant_rows_fp8(const bf16_t* hi, const bf16_t* lo, unsigned char* q, float* scale, int width) {
    __shared__ float scratch[32];
    int row = blockIdx.y;
    int block = blockIdx.x;
    int c = block * FP8_BLOCK + threadIdx.x * 2;
    size_t at = (size_t)row * width + c;
    float a = bf16_to_f32(hi[at]) + bf16_to_f32(lo[at]);
    float b = bf16_to_f32(hi[at + 1]) + bf16_to_f32(lo[at + 1]);
    float amax = block_reduce<true>(fmaxf(fabsf(a), fabsf(b)), scratch);
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    *(unsigned short*)(q + at) = (unsigned short)f32x2_to_e4m3x2(a / s, b / s);
    if (threadIdx.x == 0) scale[(size_t)row * (width / FP8_BLOCK) + block] = s;
}

