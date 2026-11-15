extern "C" __global__ void quant_rows_fp4(const bf16_t* hi, const bf16_t* lo, const unsigned char* wscales, unsigned char* q,
                                          unsigned char* qs, int width) {
    int groups = width / FP4_GROUP;
    int grp = blockIdx.x * blockDim.x + threadIdx.x;
    if (grp >= groups) return;
    size_t row = blockIdx.y;
    size_t at = row * width + (size_t)grp * FP4_GROUP;
    float igs = __ldg((const float*)wscales + 1);
    float v[FP4_GROUP];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < FP4_GROUP; i++) {
        v[i] = bf16_to_f32(hi[at + i]) + bf16_to_f32(lo[at + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    unsigned sbits = f32_to_e4m3_bits(fminf(igs * (amax * (1.0f / 6.0f)), 448.0f));
    float sx = e4m3_to_f32(sbits);
    float mul = sx > 0.0f ? igs / sx : 0.0f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int i = 0; i < FP4_GROUP; i += 2) {
        unsigned b = f32x2_to_e2m1x2(v[i] * mul, v[i + 1] * mul);
        if (i < 8) w0 |= b << (4 * i); else w1 |= b << (4 * (i - 8));
    }
    *(uint2*)(q + at / 2) = make_uint2(w0, w1);
    qs[row * groups + grp] = (unsigned char)sbits;
}
