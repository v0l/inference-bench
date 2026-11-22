// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm down_ggv block=GgmlGemv threads=256 smem=0
//   GgmlGemv { shape: GemmShape { n: 1024, k: 3072, epi: Residual, bias: false, act: Silu }, split: false }
extern "C" __global__ void __launch_bounds__(256) down_ggv(const void* w, const unsigned* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {
    gemv_ggv<1024, 3072, EPI_RESIDUAL, false, false, 2>(w, s, b, xh, xl, out, oh, ol);
}

