pub fn body() -> String {
    r##"__device__ __forceinline__ unsigned e4m3x2_to_f16x2(unsigned two) {
    unsigned short in = (unsigned short)two;
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(in));
    return r;
}

__device__ __forceinline__ void dm_cp_async4(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
}

__device__ __forceinline__ int group_len(const int* pos, int group, int qlen) {
    int len = 0;
    for (int j = 0; j < qlen; j++) len = max(len, pos[group * qlen + j] + 1);
    return len;
}

__device__ __forceinline__ int decode_chunk(int len, int splits) {
    int c = DM_CHUNK;
    while (c * splits < len) c <<= 1;
    return c;
}

__device__ __forceinline__ void mma_dec(float* c, const unsigned* ah, const unsigned* al,
                                        unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    if (KV_PLANES == 2) {
        mma_hi_lo(c, ah, al, bh0, bh1, bl0, bl1);
        return;
    }
    if (DM_SPLIT) mma_f16(c, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

"##
    .to_string()
}
