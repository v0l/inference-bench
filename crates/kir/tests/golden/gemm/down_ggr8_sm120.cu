// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm down_ggr8 block=GgmlBatchRowsMma threads=128 smem=12288
//   GgmlBatch { shape: GemmShape { n: 5120, k: 17408, epi: Residual, bias: false, act: Silu }, batch: BatchShape { tokens: 8, warps: 4, stages: 3, mj: 1 }, blocks_per_sm: 4, split: false }
extern "C" __global__ void __launch_bounds__(128, 4) down_ggr8(const void* w, const unsigned* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {
    gemv_ggr<5120, 17408, EPI_RESIDUAL, false, false, 1, 4, 3, 1>(w, s, b, xh, xl, out, oh, ol, part, tokens, splits);
}

