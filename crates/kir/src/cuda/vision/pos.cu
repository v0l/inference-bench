extern "C" __global__ void vis_pos(const float* table, const int* taps, const float* weights, float* out, int hidden) {
    int tok = blockIdx.x;
    const int* ti = taps + tok * 4;
    const float* tw = weights + tok * 4;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < 4; k++) v += tw[k] * table[(size_t)ti[k] * hidden + d];
        out[(size_t)tok * hidden + d] = v;
    }
}

