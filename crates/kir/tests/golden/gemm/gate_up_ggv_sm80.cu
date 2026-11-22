// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm gate_up_ggv block=GgmlGemv threads=256 smem=0
//   GgmlGemv { shape: GemmShape { n: 6144, k: 1024, epi: Act, bias: false, act: Silu }, split: true }
extern "C" __global__ void __launch_bounds__(256) gate_up_ggv(const void* w, const unsigned* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {
    gemv_ggv<6144, 1024, EPI_ACT, false, true, 2>(w, s, b, xh, xl, out, oh, ol);
}

