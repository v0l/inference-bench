typedef unsigned short bf16_t;

__device__ __forceinline__ float bf(bf16_t v) { return __uint_as_float(((unsigned)v) << 16); }

__device__ __forceinline__ float activate(float x, int kind) {
    if (kind == 1) {
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + tanhf(u));
    }
    if (kind == 2) {
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
    return x;
}

__device__ __forceinline__ unsigned bf16_bits_rn(float v) {
    unsigned u = __float_as_uint(v);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ void split_bf16(float v, unsigned& hi, unsigned& lo) {
    hi = bf16_bits_rn(v);
    lo = bf16_bits_rn(v - __uint_as_float(hi << 16));
}

__device__ __forceinline__ void split_store2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    unsigned ha, la, hb, lb;
    split_bf16(a, ha, la);
    split_bf16(b, hb, lb);
    *(unsigned*)(hi + at) = ha | (hb << 16);
    *(unsigned*)(lo + at) = la | (lb << 16);
}

__device__ __forceinline__ unsigned f16_bits(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ float f16_float(unsigned bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)bits));
    return f;
}

__device__ __forceinline__ unsigned pack_f16_hi(float a, float b) { return f16_bits(a) | (f16_bits(b) << 16); }

__device__ __forceinline__ unsigned pack_f16_lo(float a, float b) {
    return f16_bits(a - f16_float(f16_bits(a))) | (f16_bits(b - f16_float(f16_bits(b))) << 16);
}

