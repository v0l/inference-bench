// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm gate_up_f4b16 block=BatchRowsMma threads=128 smem=24576
//   Batch { shape: GemmShape { n: 34816, k: 5120, epi: Act, bias: false, act: Silu }, wf: Fp4, batch: BatchShape { tokens: 16, warps: 4, stages: 3, mj: 2 }, blocks_per_sm: 4 }
extern "C" __global__ void __launch_bounds__(128, 4) gate_up_f4b16(const void* w, const unsigned char* s, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {
    gemv_f4b<34816, 5120, EPI_ACT, 2, 4, 3, 2>(w, s, xh, xl, out, oh, ol, part, tokens, splits);
}

