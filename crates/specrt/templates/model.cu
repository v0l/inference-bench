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

extern "C" __global__ void qkv_mma(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm_mma<{{QKV_N}}, {{HIDDEN}}, {{QKV_BIAS}}, false, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void o_mma(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm_mma<{{HIDDEN}}, {{Q_DIM}}, false, true, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void gate_up_mma(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm_mma<{{GATE_UP_N}}, {{HIDDEN}}, false, false, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void down_mma(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm_mma<{{HIDDEN}}, {{INTER}}, false, true, {{Q8}}>(w, s, b, x, out, tokens);
}

extern "C" __global__ void lm_head_mma(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {
    gemm_mma<{{VOCAB}}, {{HIDDEN}}, false, false, {{Q8}}>(w, s, b, x, out, tokens);
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
