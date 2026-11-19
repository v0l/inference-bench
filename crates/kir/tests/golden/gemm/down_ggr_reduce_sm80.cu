// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm down_ggr_reduce block=BatchReduce threads=256 smem=0
//   BatchReduce { shape: GemmShape { n: 5120, k: 17408, epi: Residual, bias: false, act: Silu }, wf: Ggml }
extern "C" __global__ void down_ggr_reduce(const float* part, const float* b, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {
    ggr_reduce<5120, EPI_RESIDUAL, false>(part, b, out, oh, ol, tokens, splits);
}

