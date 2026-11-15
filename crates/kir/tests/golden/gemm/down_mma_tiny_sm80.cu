// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm down_mma_tiny block=TileMma threads=64 smem=5120
//   Tile { shape: GemmShape { n: 5120, k: 17408, epi: Residual, bias: false, act: Silu }, wf: Q8, main: Mma(MmaShape { tile: Tile { bm: 16, bn: 32, wm: 1, wn: 2 }, bk: 32, stages: 2, split: false, wf: Q8 }), token_major: false }
extern "C" __global__ void __launch_bounds__(64) down_mma_tiny(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<5120, 17408, 16, 32, 1, 2, 32, 2, EPI_RESIDUAL, false, DecQ8, false, false>(w, s, b, xh, xl, out, oh, ol, tokens);
}

