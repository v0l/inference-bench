// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm up_map_mma block=TileMma threads=128 smem=24576
//   Tile { shape: GemmShape { n: 15680, k: 4480, epi: Map, bias: false, act: Silu }, wf: Bf16, main: Mma(MmaShape { tile: Tile { bm: 64, bn: 64, wm: 2, wn: 2 }, bk: 32, stages: 2, split: true, wf: Bf16 }), token_major: false }
extern "C" __global__ void __launch_bounds__(128) up_map_mma(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<15680, 4480, 64, 64, 2, 2, 32, 2, EPI_MAP, false, DecBf16, true, false>(w, s, b, xh, xl, out, oh, ol, tokens);
}

