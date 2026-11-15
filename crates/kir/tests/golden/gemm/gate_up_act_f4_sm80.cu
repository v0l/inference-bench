// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// gemm gate_up_act_f4 block=Fp4Gemv threads=512 smem=0
//   Gemv { op: GateUpAct, n: 34816, k: 5120 }
extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) gate_up_act_f4(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {
    gate_up_act_norm<DecFp4>(w, s, x, nw, out);
}

