__device__ __forceinline__ unsigned pack_hi(float a, float b) {
    return attn_bits(a) | (attn_bits(b) << 16);
}

__device__ __forceinline__ unsigned pack_lo(float a, float b) {
    return attn_bits(a - attn_float(attn_bits(a))) | (attn_bits(b - attn_float(attn_bits(b))) << 16);
}

__device__ __forceinline__ void mma_f16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mma_hi_lo(float* c, const unsigned* ah, const unsigned* al,
                                          unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma_f16(c, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

__device__ __forceinline__ void mma_kv(float* c, const unsigned* ah, const unsigned* al,
                                       unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    if (KV_PLANES == 2) {
        mma_hi_lo(c, ah, al, bh0, bh1, bl0, bl1);
        return;
    }
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}
