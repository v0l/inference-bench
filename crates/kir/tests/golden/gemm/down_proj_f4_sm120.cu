// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm down_proj_f4 block=Fp4Gemv threads=512 smem=0
//   Gemv { op: Down, n: 5120, k: 17408 }
extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) down_proj_f4(const void* w, const unsigned char* s, const float* b, const float* xt, float* out) {
    linear_w<DecFp4, 5120, 17408, false, true>(w, s, b, xt, out);
}

