// Writes random i-quant blocks and ggml's own dequantize_row_iq* output for them.
// Reference: llama.cpp ggml/src/ggml-quants.c (MIT).
// Build: gcc -O1 gen_iq_reference.c -o gen_iq_reference -L$LLAMA/build/bin -lggml-base -Wl,-rpath,$LLAMA/build/bin
// Run:   ./gen_iq_reference <out dir>
// Each file is NBLOCKS repetitions of: block bytes, then 256 little endian f32.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*deq_fn)(const void *, float *, int64_t);

extern void dequantize_row_iq1_m(const void *, float *, int64_t);
extern void dequantize_row_iq1_s(const void *, float *, int64_t);
extern void dequantize_row_iq2_s(const void *, float *, int64_t);
extern void dequantize_row_iq2_xs(const void *, float *, int64_t);
extern void dequantize_row_iq2_xxs(const void *, float *, int64_t);
extern void dequantize_row_iq3_s(const void *, float *, int64_t);
extern void dequantize_row_iq3_xxs(const void *, float *, int64_t);
extern void dequantize_row_iq4_nl(const void *, float *, int64_t);
extern void dequantize_row_iq4_xs(const void *, float *, int64_t);

#define NBLOCKS 64

static uint64_t state = 0x9e3779b97f4a7c15ull;
static uint8_t next(void) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint8_t)(state >> 56);
}

static void write_type(const char *dir, const char *name, deq_fn fn, int block_bytes, int half_at) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "wb");
    for (int b = 0; b < NBLOCKS; b++) {
        uint8_t block[256];
        for (int i = 0; i < block_bytes; i++) block[i] = next();
        if (half_at >= 0) {
            uint16_t h = 0x2c00 | (next() & 0x03ff);
            if (next() & 1) h |= 0x8000;
            memcpy(block + half_at, &h, 2);
        } else if (half_at == -1) {
            block[53] = (block[53] & 0x0f) | 0xc0;
            block[55] = (block[55] & 0x0f) | 0x20;
        } else if (half_at == -2) {
            for (int s = 0; s < 8; s++) {
                uint16_t h = 0x2c00 | (next() & 0x03ff);
                memcpy(block + 18 * s, &h, 2);
            }
        }
        float out[256];
        fn(block, out, 256);
        fwrite(block, 1, block_bytes, f);
        fwrite(out, 4, 256, f);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    const char *dir = argv[1];
    write_type(dir, "iq4_nl", dequantize_row_iq4_nl, 144, -2);
    write_type(dir, "iq4_xs", dequantize_row_iq4_xs, 136, 0);
    write_type(dir, "iq3_xxs", dequantize_row_iq3_xxs, 98, 0);
    write_type(dir, "iq3_s", dequantize_row_iq3_s, 110, 0);
    write_type(dir, "iq2_xxs", dequantize_row_iq2_xxs, 66, 0);
    write_type(dir, "iq2_xs", dequantize_row_iq2_xs, 74, 0);
    write_type(dir, "iq2_s", dequantize_row_iq2_s, 82, 0);
    write_type(dir, "iq1_s", dequantize_row_iq1_s, 50, 0);
    write_type(dir, "iq1_m", dequantize_row_iq1_m, 56, -1);
    return 0;
}
