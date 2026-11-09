#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"

static uint32_t state = 12345;

static float next_float(void) {
    state = state * 1664525u + 1013904223u;
    float u = (float)(state >> 8) / 16777216.0f;
    state = state * 1664525u + 1013904223u;
    float v = (float)(state >> 8) / 16777216.0f;
    float g = sqrtf(-2.0f * logf(u + 1e-7f)) * cosf(6.2831853f * v);
    if ((state & 63u) == 0) g *= 8.0f;
    return g * 0.05f;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    FILE* out = fopen(argv[1], "wb");
    enum ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    const int rows = 4, cols = 512, n = rows * cols;
    for (int t = 0; t < 4; t++) {
        enum ggml_type ty = types[t];
        float* src = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) src[i] = next_float();
        size_t bytes = ggml_row_size(ty, cols) * rows;
        void* q = malloc(bytes);
        ggml_quantize_init(ty);
        ggml_quantize_chunk(ty, src, q, 0, rows, cols, NULL);
        float* back = malloc(n * sizeof(float));
        const struct ggml_type_traits* tr = ggml_get_type_traits(ty);
        tr->to_float(q, back, n);
        uint32_t head[3] = {(uint32_t)ty, (uint32_t)n, (uint32_t)bytes};
        fwrite(head, 4, 3, out);
        fwrite(q, 1, bytes, out);
        fwrite(back, 4, n, out);
        free(src); free(q); free(back);
    }
    fclose(out);
    return 0;
}
