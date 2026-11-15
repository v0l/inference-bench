// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm gate_up_f4w_big block=TileMma threads=256 smem=77824
//   Tile { shape: GemmShape { n: 34816, k: 5120, epi: Act, bias: false, act: Silu }, wf: Fp4, main: Mma(MmaShape { tile: Tile { bm: 128, bn: 128, wm: 2, wn: 4 }, bk: 64, stages: 2, split: true, wf: Fp4 }), token_major: false }
extern "C" __global__ void __launch_bounds__(256) gate_up_f4w_big(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_tc<34816, 5120, 128, 128, 2, 4, 64, 2, EPI_ACT, false, DecFp4, true, false>(w, s, b, xh, xl, out, oh, ol, tokens);
}

