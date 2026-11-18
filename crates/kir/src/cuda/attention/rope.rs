pub fn body() -> String {
    r##"extern "C" __global__ void qk_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos, const int* slots, int layer, int per_row, int pages, int mm_stride,
                                        const int* mm_meta, const int* mm_table, int slot_base) {
    int row = blockIdx.y;
    const int* table = per_row ? seq_table(block_table, slots, row, pages) : block_table;
    int p = per_row ? pos[row] : pos[0] + row;
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, table, p, layer, blockIdx.x, row, mm_meta, mm_table, mm_stride,
                   per_row ? slots[row] : slot_base);
}

extern "C" __global__ void kv_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos_ptr, int layer,
                                        const int* mm_meta, const int* mm_table, int mm_stride, int slot) {
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, block_table, pos_ptr[0] + blockIdx.y, layer,
                   HEADS + blockIdx.x, blockIdx.y, mm_meta, mm_table, mm_stride, slot);
}
"##
    .to_string()
}
