// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm qkv_f8b_reduce block=BatchReduce threads=256 smem=0
//   BatchReduce { shape: GemmShape { n: 14336, k: 5120, epi: Store, bias: true, act: Silu }, wf: Fp8 }
extern "C" __global__ void qkv_f8b_reduce(const float* part, const float* b, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {
    f8b_reduce<14336, EPI_STORE, true>(part, b, out, oh, ol, tokens, splits);
}

