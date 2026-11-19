use super::dequant;
use super::rowmath;
use super::rows;
use crate::cuda::gemm::prim;

pub fn body() -> String {
    [
        rows::format_ids(),
        prim::f16_to_f32("gg_f16"),
        prim::ldg_reader("gg_u16", "unsigned short"),
        prim::ldg_reader("gg_u32", "unsigned"),
        rows::bytes_fn(),
        prim::byte_of_words("gg_byte"),
        dequant::scale_min(),
        dequant::dequant8(),
        prim::table_row("gg_row"),
        r##"__device__ __forceinline__ void gg_load8(const void* w, const unsigned* table, int row, int k, float* out) {
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
