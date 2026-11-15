// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm gate_up_f4w_big block=TileMma threads=256 smem=116736
//   Tile { shape: GemmShape { n: 34816, k: 5120, epi: Act, bias: false, act: Silu }, wf: Fp4, main: Mma(MmaShape { tile: Tile { bm: 128, bn: 128, wm: 2, wn: 4 }, bk: 64, stages: 3, split: true, wf: Fp4 }), token_major: false }
extern "C" __global__ void __launch_bounds__(256) gate_up_f4w_big(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<34816, 5120, 128, 128, 2, 4, 64, 3, EPI_ACT, false, DecFp4, true, false>(w, s, b, xh, xl, out, oh, ol, tokens);
}

