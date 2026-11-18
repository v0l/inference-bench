pub const LIBRARY: &str = r##"template <int KSTEPS, typename FA, typename FB>
__device__ __forceinline__ void mma_split_tile(float* c, int m0, int n0, int lane, FA a_at, FB b_at) {
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll 2
    for (int kk = 0; kk < KSTEPS; kk++) {
        int k = kk * 16 + 2 * t;
        float av[8] = {a_at(m0 + g, k),     a_at(m0 + g, k + 1),     a_at(m0 + g + 8, k),     a_at(m0 + g + 8, k + 1),
                       a_at(m0 + g, k + 8), a_at(m0 + g, k + 9), a_at(m0 + g + 8, k + 8), a_at(m0 + g + 8, k + 9)};
        float bv[4] = {b_at(k, n0 + g), b_at(k + 1, n0 + g), b_at(k + 8, n0 + g), b_at(k + 9, n0 + g)};
        unsigned ah[4], al[4], bh[2], bl[2];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned h0, l0, h1, l1;
            split_hi_lo(av[2 * i], h0, l0);
            split_hi_lo(av[2 * i + 1], h1, l1);
            ah[i] = h0 | (h1 << 16);
            al[i] = l0 | (l1 << 16);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned h0, l0, h1, l1;
            split_hi_lo(bv[2 * i], h0, l0);
            split_hi_lo(bv[2 * i + 1], h1, l1);
            bh[i] = h0 | (h1 << 16);
            bl[i] = l0 | (l1 << 16);
        }
        mma_bf16(c, al[0], al[1], al[2], al[3], bh[0], bh[1]);
        mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
        mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
    }
}

__device__ __forceinline__ void gdn_pack2(float a, float b, unsigned& hi, unsigned& lo) {
    unsigned h0, l0, h1, l1;
    split_hi_lo(a, h0, l0);
    split_hi_lo(b, h1, l1);
    hi = h0 | (h1 << 16);
    lo = l0 | (l1 << 16);
}

template <int KSTEPS>
__device__ __forceinline__ void gdn_frag_put(unsigned* base, int m, int k, float v0, float v1) {
    unsigned hi, lo;
    gdn_pack2(v0, v1, hi, lo);
    int rec = ((m >> 4) * KSTEPS + (k >> 4)) * 32 + (m & 7) * 4 + ((k & 7) >> 1);
    int reg = ((m >> 3) & 1) + 2 * ((k >> 3) & 1);
    base[rec * 8 + reg] = hi;
    base[rec * 8 + 4 + reg] = lo;
}
"##;

pub const PLANES: &str = r##"template <int KS, int NT>
__device__ __forceinline__ void gdn_mma_planes(float (*acc)[4], const uint4* ap, const unsigned short* bh,
                                               const unsigned short* bl, int pitch, int g, int t) {
#pragma unroll
    for (int kk = 0; kk < KS; kk++) {
        uint4 ah = ap[kk * 64];
        uint4 al = ap[kk * 64 + 1];
#pragma unroll
        for (int n = 0; n < NT; n++) {
            const unsigned* rh = (const unsigned*)(bh + (n * 8 + g) * pitch);
            const unsigned* rl = (const unsigned*)(bl + (n * 8 + g) * pitch);
            unsigned bh0 = rh[kk * 8 + t], bh1 = rh[kk * 8 + t + 4];
            unsigned bl0 = rl[kk * 8 + t], bl1 = rl[kk * 8 + t + 4];
            mma_bf16(acc[n], al.x, al.y, al.z, al.w, bh0, bh1);
            mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bl0, bl1);
            mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bh0, bh1);
        }
    }
}
"##;
