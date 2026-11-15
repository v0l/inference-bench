// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm lm_head_norm_f4 block=Fp4HeadGemv threads=512 smem=0
//   HeadNormFp4 { n: 248320, k: 5120, rows: 2 }
extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) lm_head_norm_f4(const void* w, const unsigned char* s, const float* b, const float* x, const float* nw, float* out) {
    head_norm_f4<248320, 5120, 2>(w, s, x, nw, out);
}

