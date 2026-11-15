// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm qkv_mma_big block=TileMma threads=256 smem=49152
//   Tile { shape: GemmShape { n: 14336, k: 5120, epi: Store, bias: true, act: Silu }, wf: Bf16, main: Mma(MmaShape { tile: Tile { bm: 128, bn: 128, wm: 2, wn: 4 }, bk: 32, stages: 2, split: true, wf: Bf16 }), token_major: false }
extern "C" __global__ void __launch_bounds__(256) qkv_mma_big(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<14336, 5120, 128, 128, 2, 4, 32, 2, EPI_STORE, true, DecBf16, true, false>(w, s, b, xh, xl, out, oh, ol, tokens);
}

