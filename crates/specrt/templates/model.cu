extern "C" __global__ void qkv_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<{{QKV_N}}, {{HIDDEN}}, {{QKV_BIAS}}, false>(w, s, b, x, out);
}

extern "C" __global__ void o_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<{{HIDDEN}}, {{Q_DIM}}, false, true>(w, s, b, x, out);
}

extern "C" __global__ void gate_up_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<{{GATE_UP_N}}, {{HIDDEN}}, false, false>(w, s, b, x, out);
}

extern "C" __global__ void down_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<{{HIDDEN}}, {{INTER}}, false, true>(w, s, b, x, out);
}

extern "C" __global__ void lm_head(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<{{VOCAB}}, {{HIDDEN}}, false, false>(w, s, b, x, out);
}

extern "C" __global__ void qkv_gemm(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm<{{QKV_N}}, {{HIDDEN}}, {{QKV_BIAS}}, false, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void o_gemm(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm<{{HIDDEN}}, {{Q_DIM}}, false, true, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void gate_up_gemm(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm<{{GATE_UP_N}}, {{HIDDEN}}, false, false, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void down_gemm(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm<{{HIDDEN}}, {{INTER}}, false, true, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void lm_head_gemm(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm<{{VOCAB}}, {{HIDDEN}}, false, false, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void qkv_mma(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{QKV_N}}, {{HIDDEN}}, 64, 64, 2, 2, EPI_STORE, {{QKV_BIAS}}, {{Q8}}, true, {{SPLIT_QKV}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void qkv_mma_big(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{QKV_N}}, {{HIDDEN}}, {{BIG_BM}}, {{BIG_BN}}, {{BIG_WM}}, {{BIG_WN}}, EPI_STORE, {{QKV_BIAS}}, {{Q8}}, true, {{SPLIT_QKV}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void o_mma(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{Q_DIM}}, 64, 64, 2, 2, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_O}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void o_mma_big(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{Q_DIM}}, {{BIG_BM}}, {{BIG_BN}}, {{BIG_WM}}, {{BIG_WN}}, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_O}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void gate_up_mma(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{GATE_UP_N}}, {{HIDDEN}}, 64, 64, 2, 2, EPI_ACT, false, {{Q8}}, true, {{SPLIT_GATE_UP}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void gate_up_mma_big(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{GATE_UP_N}}, {{HIDDEN}}, {{BIG_BM}}, {{BIG_BN}}, {{BIG_WM}}, {{BIG_WN}}, EPI_ACT, false, {{Q8}}, true, {{SPLIT_GATE_UP}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void down_mma(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{INTER}}, 64, 64, 2, 2, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_DOWN}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void down_mma_big(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{INTER}}, {{BIG_BM}}, {{BIG_BN}}, {{BIG_WM}}, {{BIG_WN}}, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_DOWN}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void lm_head_mma(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{VOCAB}}, {{HIDDEN}}, 64, 64, 2, 2, EPI_STORE, false, {{Q8}}, true, {{SPLIT_LM_HEAD}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void lm_head_mma_big(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{VOCAB}}, {{HIDDEN}}, {{BIG_BM}}, {{BIG_BN}}, {{BIG_WM}}, {{BIG_WN}}, EPI_STORE, false, {{Q8}}, true, {{SPLIT_LM_HEAD}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void qkv_mma_narrow(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{QKV_N}}, {{HIDDEN}}, 64, 32, 2, 2, EPI_STORE, {{QKV_BIAS}}, {{Q8}}, true, {{SPLIT_QKV}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void o_mma_narrow(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{Q_DIM}}, 64, 32, 2, 2, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_O}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void gate_up_mma_narrow(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{GATE_UP_N}}, {{HIDDEN}}, 64, 32, 2, 2, EPI_ACT, false, {{Q8}}, true, {{SPLIT_GATE_UP}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void down_mma_narrow(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{INTER}}, 64, 32, 2, 2, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_DOWN}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void lm_head_mma_narrow(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{VOCAB}}, {{HIDDEN}}, 64, 32, 2, 2, EPI_STORE, false, {{Q8}}, true, {{SPLIT_LM_HEAD}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void qkv_mma_tiny(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{QKV_N}}, {{HIDDEN}}, 16, {{TINY_BN}}, 1, {{TINY_WN}}, EPI_STORE, {{QKV_BIAS}}, {{Q8}}, true, {{SPLIT_QKV}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void o_mma_tiny(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{Q_DIM}}, 16, {{TINY_BN}}, 1, {{TINY_WN}}, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_O}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void gate_up_mma_tiny(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{GATE_UP_N}}, {{HIDDEN}}, 16, {{TINY_BN}}, 1, {{TINY_WN}}, EPI_ACT, false, {{Q8}}, true, {{SPLIT_GATE_UP}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void down_mma_tiny(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{HIDDEN}}, {{INTER}}, 16, {{TINY_BN}}, 1, {{TINY_WN}}, EPI_RESIDUAL, false, {{Q8}}, false, {{SPLIT_DOWN}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void lm_head_mma_tiny(const void* w, const float* s, const float* b, const float* x, const float* ri, const float* nw, float* out, int tokens) {
    gemm_tc<{{VOCAB}}, {{HIDDEN}}, 16, {{TINY_BN}}, 1, {{TINY_WN}}, EPI_STORE, false, {{Q8}}, true, {{SPLIT_LM_HEAD}}>(w, s, b, x, ri, nw, out, tokens);
}

extern "C" __global__ void qkv_proj_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<{{Q8}}, {{QKV_N}}, {{QKV_BIAS}}>(w, s, b, x, nw, out);
}

extern "C" __global__ void gate_up_act(const void* w, const float* s, const float* x, const float* nw, float* out) {
    gate_up_act_norm<{{Q8}}>(w, s, x, nw, out);
}

extern "C" __global__ void lm_head_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<{{Q8}}, {{VOCAB}}, false>(w, s, b, x, nw, out);
}
