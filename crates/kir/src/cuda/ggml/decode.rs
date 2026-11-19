use super::dequant;
use super::rowmath;
use super::rows;

pub fn body() -> String {
    [
        rows::format_ids(),
        r##"__device__ __forceinline__ float gg_f16(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ unsigned gg_u16(const unsigned char* p) {
    return __ldg((const unsigned short*)p);
}

__device__ __forceinline__ unsigned gg_u32(const unsigned char* p) {
    return __ldg((const unsigned*)p);
}

"##.to_string(),
        rows::bytes_fn(),
        r##"__device__ __forceinline__ unsigned gg_byte(const unsigned* s, int i) {
    return (s[i >> 2] >> (8 * (i & 3))) & 0xffu;
}

"##.to_string(),
        dequant::scale_min(),
        dequant::dequant8(),
        r##"__device__ __forceinline__ const unsigned char* gg_row(const void* w, const unsigned* table, int row, int& type) {
    unsigned e = __ldg(table + row);
    type = (int)(e & 7u);
    return (const unsigned char*)w + (size_t)(e >> 3) * 16;
}

__device__ __forceinline__ void gg_load8(const void* w, const unsigned* table, int row, int k, float* out) {
    int type;
    const unsigned char* base = gg_row(w, table, row, type);
    gg_dequant8(base + (size_t)(k / GG_SB) * gg_bytes(type), type, (k % GG_SB) / 8, out);
}

"##.to_string(),
        rowmath::fused_rows(),
        rowmath::fused_dots(),
        dequant::row_dot(),
        dequant::dequant_rows_kernel(),
    ]
    .concat()
}
