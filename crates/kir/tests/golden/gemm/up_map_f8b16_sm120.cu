// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm up_map_f8b16 block=BatchRowsMma threads=128 smem=24576
//   Batch { shape: GemmShape { n: 15680, k: 4480, epi: Map, bias: false, act: Silu }, wf: Fp8, batch: BatchShape { tokens: 16, warps: 4, stages: 3, mj: 2 }, blocks_per_sm: 4 }
extern "C" __global__ void __launch_bounds__(128, 4) up_map_f8b16(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {
    gemv_f8b<15680, 4480, EPI_MAP, false, 2, 4, 3, 2>(w, s, b, xh, xl, out, oh, ol, part, tokens, splits);
}

