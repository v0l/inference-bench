// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// gemm gate_up_act_f4 block=Fp4Gemv threads=512 smem=0
//   Gemv { op: GateUpAct, n: 34816, k: 5120 }
extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) gate_up_act_f4(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {
    gate_up_act_norm<DecFp4>(w, s, x, nw, out);
}

