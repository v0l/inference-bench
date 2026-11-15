// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm gate_up_f8_huge block=Fp8ActMma threads=256 smem=65536
//   Fp8Act { shape: GemmShape { n: 34816, k: 5120, epi: Act, bias: false, act: Silu }, tile: Tile { bm: 128, bn: 128, wm: 2, wn: 4 }, stages: 2, scale: Channel }
extern "C" __global__ void __launch_bounds__(256) gate_up_f8_huge(const void* w, const float* s, const float* b, const unsigned char* xq, const float* xs, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_fp8<34816, 5120, 128, 128, 2, 4, 2, EPI_ACT, false, true>(w, s, b, xq, xs, out, oh, ol, tokens);
}

