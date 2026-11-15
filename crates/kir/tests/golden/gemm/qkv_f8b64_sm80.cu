// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm qkv_f8b64 block=BatchRowsMma threads=256 smem=65536
//   Batch { shape: GemmShape { n: 14336, k: 5120, epi: Store, bias: true, act: Silu }, wf: Fp8, batch: BatchShape { tokens: 64, warps: 8, stages: 2, mj: 1 }, blocks_per_sm: 2 }
extern "C" __global__ void __launch_bounds__(256, 2) qkv_f8b64(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {
    gemv_f8b<14336, 5120, EPI_STORE, true, 8, 8, 2, 1>(w, s, b, xh, xl, out, oh, ol, part, tokens, splits);
}

