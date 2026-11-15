// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm gate_up_mma block=TileMma threads=128 smem=43008
//   Tile { shape: GemmShape { n: 34816, k: 5120, epi: Act, bias: false, act: Silu }, wf: Fp8, main: Mma(MmaShape { tile: Tile { bm: 64, bn: 64, wm: 2, wn: 2 }, bk: 64, stages: 2, split: true, wf: Fp8 }), token_major: false }
extern "C" __global__ void __launch_bounds__(128) gate_up_mma(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<34816, 5120, 64, 64, 2, 2, 64, 2, EPI_ACT, false, DecFp8, true, false>(w, s, b, xh, xl, out, oh, ol, tokens);
}

