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

