pub fn body() -> String {
    r##"extern "C" __global__ void batch_decode_combine(const float* part_acc, const float* part_ml, const float* qkv, const int* pos, int qlen, int splits,
                                                bf16_t* out_hi, bf16_t* out_lo) {
    int head = blockIdx.x;
    int seq = blockIdx.y;
    int d = 2 * threadIdx.x;
    int len = group_len(pos, seq / qlen, qlen);
    int active = (len + decode_chunk(len, splits) - 1) / decode_chunk(len, splits);
    size_t base = ((size_t)seq * HEADS + head) * splits;
    float M = NEG_INF;
    for (int s = 0; s < active; s++) M = fmaxf(M, part_ml[(base + s) * 2]);
    float L = 0.0f, A0 = 0.0f, A1 = 0.0f;
    for (int s = 0; s < active; s++) {
        float ms = part_ml[(base + s) * 2];
        float c = ms == NEG_INF ? 0.0f : exp2f(ms - M);
        L += part_ml[(base + s) * 2 + 1] * c;
        A0 += part_acc[(base + s) * HEAD_DIM + d] * c;
        A1 += part_acc[(base + s) * HEAD_DIM + d + 1] * c;
    }
    int at = head * HEAD_DIM + d;
    split_store2(out_hi, out_lo, (size_t)seq * Q_DIM + at, A0 / L * out_gate(qkv, seq, at), A1 / L * out_gate(qkv, seq, at + 1));
}
"##
    .to_string()
}
