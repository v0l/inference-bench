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
