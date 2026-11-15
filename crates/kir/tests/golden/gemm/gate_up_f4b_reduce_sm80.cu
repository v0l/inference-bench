// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm gate_up_f4b_reduce block=BatchReduce threads=256 smem=0
//   BatchReduce { shape: GemmShape { n: 34816, k: 5120, epi: Act, bias: false, act: Silu }, wf: Fp4 }
extern "C" __global__ void gate_up_f4b_reduce(const float* part, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {
    f4b_reduce<34816, EPI_ACT>(part, out, oh, ol, tokens, splits);
}

