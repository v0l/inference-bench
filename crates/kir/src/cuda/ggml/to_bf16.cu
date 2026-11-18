extern "C" __global__ void gg_dequant_bf16(const void* w, const unsigned* table, bf16_t* out, int rows, int cols) {
    int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    int per_row = cols / 8;
    if (chunk >= rows * per_row) return;
    int row = chunk / per_row;
    int k = (chunk % per_row) * 8;
    float v[8];
    gg_load8(w, table, row, k, v);
    *(uint4*)(out + (size_t)row * cols + k) = make_uint4(bf16_bits_rn(v[0]) | (bf16_bits_rn(v[1]) << 16), bf16_bits_rn(v[2]) | (bf16_bits_rn(v[3]) << 16),
                                                     bf16_bits_rn(v[4]) | (bf16_bits_rn(v[5]) << 16), bf16_bits_rn(v[6]) | (bf16_bits_rn(v[7]) << 16));
}
