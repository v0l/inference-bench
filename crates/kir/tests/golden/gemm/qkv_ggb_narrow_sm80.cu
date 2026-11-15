// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm qkv_ggb_narrow block=TileMma threads=128 smem=20480
//   Tile { shape: GemmShape { n: 14336, k: 5120, epi: Store, bias: true, act: Silu }, wf: Bf16, main: Mma(MmaShape { tile: Tile { bm: 64, bn: 32, wm: 2, wn: 2 }, bk: 32, stages: 2, split: true, wf: Bf16 }), token_major: true }
extern "C" __global__ void __launch_bounds__(128) qkv_ggb_narrow(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<14336, 5120, 64, 32, 2, 2, 32, 2, EPI_STORE, true, DecBf16, true, true>(w, s, b, xh, xl, out, oh, ol, tokens);
}

