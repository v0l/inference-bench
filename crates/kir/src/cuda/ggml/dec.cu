struct DecGgml : DecBase<DecGgml> {
    static constexpr int PAIR = 2;
    static constexpr bool STRIDE = true;
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void* s, int r, int n, int kt, int ck) {
        float v[8];
        gg_load8(w, (const unsigned*)s, n, kt * BK + ck * 8, v);
        uint4 packed = make_uint4(bf16_bits_rn(v[0]) | (bf16_bits_rn(v[1]) << 16), bf16_bits_rn(v[2]) | (bf16_bits_rn(v[3]) << 16),
                                  bf16_bits_rn(v[4]) | (bf16_bits_rn(v[5]) << 16), bf16_bits_rn(v[6]) | (bf16_bits_rn(v[7]) << 16));
        *(uint4*)(ws + swz<CPR>(r, ck) * 16) = packed;
    }
    template <int K> static __device__ __forceinline__ float dot(const void* w, const void* s, int row, const float* x, int lane) {
        return gg_row_dot<K>(w, (const unsigned*)s, row, x, lane);
    }
    template <int K, int ROWS>
    static __device__ __forceinline__ void rows(const void* w, const void* s, int row0, int stride, const float* x, int lane, float* out) {
        const unsigned char* base[ROWS];
        int type[ROWS];
        bool same = true;
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            base[r] = gg_row(w, (const unsigned*)s, row0 + r * stride, type[r]);            same = same && type[r] == type[0];
        }
        if (same && type[0] == GG_Q4K) {
            gg_q4k_rows<K, ROWS>(base, x, lane, out);
        } else if (same && type[0] == GG_Q6K) {
            gg_q6k_rows<K, ROWS>(base, x, lane, out);
        } else {
#pragma unroll
            for (int r = 0; r < ROWS; r++) out[r] = dot<K>(w, s, row0 + r * stride, x, lane);
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) out[r] = warp_sum(out[r]);
    }
    template <int K> static __device__ __forceinline__ float at(const void* w, const void* s, int n, int k) {
        float v[8];
        gg_load8(w, (const unsigned*)s, n, k & ~7, v);
        return v[k & 7];
    }
};
