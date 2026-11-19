use super::super::attention::prim::{Reduce, warp_reduce};
use super::super::gemm::prim;

pub fn body() -> String {
    [
        r##"struct DecGgml : DecBase<DecGgml> {
    static constexpr int PAIR = 2;
    static constexpr bool STRIDE = true;
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void* s, int r, int n, int kt, int ck) {
        float v[8];
        gg_load8(w, (const unsigned*)s, n, kt * BK + ck * 8, v);
        uint4 packed = "##.to_string(),
        prim::pack_bf16x8(34),
        r##";
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
"##.to_string(),
        super::dequant::rows_dispatch(),
        r##"#pragma unroll
            for (int r = 0; r < ROWS; r++) out[r] = dot<K>(w, s, row0 + r * stride, x, lane);
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) out[r] = "##.to_string(),
        warp_reduce(Reduce::Sum, "out[r]"),
        r##";
    }
    template <int K> static __device__ __forceinline__ float at(const void* w, const void* s, int n, int k) {
        float v[8];
        gg_load8(w, (const unsigned*)s, n, k & ~7, v);
        return v[k & 7];
    }
};
"##.to_string(),
    ]
    .concat()
}
