template <int K>
__device__ __forceinline__ float fp4_row_dot_ref(const void* w, const unsigned char* scales, int row, const float* xt, int lane) {
    constexpr int KG = K / FP4_GROUP;
    const uint2* wr = (const uint2*)((const unsigned char*)w + (size_t)row * (K / 2));
    const unsigned char* sr = scales + FP4_HEADER + (size_t)row * KG;
    float acc = 0.0f;
#pragma unroll 4
    for (int g = lane; g < KG; g += 32) {
        uint2 q = load_stream8(wr + g);
        float s = e4m3_to_f32(__ldg(sr + g));
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned byte = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xffu;
            part += e2m1_to_f32(byte & 15u) * xt[(2 * i) * KG + g];
            part += e2m1_to_f32(byte >> 4) * xt[(2 * i + 1) * KG + g];
        }
        acc += part * s;
    }
    return warp_sum(acc) * fp4_inv_global(scales);
}

extern "C" __global__ void gate_up_act_f4_ref(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xt[HIDDEN];
    load_normed_t(x, nw, xt);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= INTER) return;
    float g = fp4_row_dot_ref<HIDDEN>(w, s, 2 * row, xt, lane);
    float u = fp4_row_dot_ref<HIDDEN>(w, s, 2 * row + 1, xt, lane);
    if (lane == 0) out[fp4_t(row, INTER / FP4_GROUP)] = activation(g) * u;
}

extern "C" __global__ void down_proj_f4_ref(const void* w, const unsigned char* s, const float* b, const float* xt, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= HIDDEN) return;
    float acc = fp4_row_dot_ref<INTER>(w, s, row, xt, lane);
    if (lane == 0) out[row] += acc;
}
