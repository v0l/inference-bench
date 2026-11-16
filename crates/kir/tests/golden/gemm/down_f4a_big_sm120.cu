// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm down_f4a_big block=BlockScaledFp4Mma threads=512 smem=82944
//   Fp4Act { shape: GemmShape { n: 5120, k: 17408, epi: Residual, bias: false, act: Silu }, tile: Tile { bm: 128, bn: 256, wm: 4, wn: 4 }, stages: 3 }
extern "C" __global__ void __launch_bounds__(512) down_f4a_big(const void* w, const unsigned char* s, const unsigned char* xq, const unsigned char* xs, float* out, bf16_t* oh, bf16_t* ol, int tokens) {
    gemm_f4a<5120, 17408, 128, 256, 4, 4, 3, EPI_RESIDUAL>(w, s, xq, xs, out, oh, ol, tokens);
}

