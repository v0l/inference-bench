typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float bf16_f32(u16 v) { return __uint_as_float(((u32)v) << 16); }

__device__ __forceinline__ u8 f32_e4m3(float v) {
    float a = fabsf(v);
    u32 code;
    if (a < 0.015625f) {
        code = (u32)rintf(a * 512.0f);
    } else {
        u32 bits = min(__float_as_uint(a), 0x43f00000u);
        bits = bits + 0x7ffffu + ((bits >> 20) & 1u);
        code = (((u32)((int)(bits >> 23) - 120)) << 3) | ((bits >> 20) & 7u);
    }
    code = min(code, 0x7eu);
    return (u8)(code | ((__float_as_uint(v) >> 31) ? 0x80u : 0u));
}

__device__ __forceinline__ float e4m3_f32(u8 b) {
    u32 mag = b & 0x7f;
    int exp = (int)(mag >> 3);
    float man = (float)(mag & 7);
    float v = exp == 0 ? man * ldexpf(1.0f, -9) : (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
    return (b & 0x80) ? -v : v;
}

__device__ __forceinline__ u32 f32_e2m1(float v) {
    float a = fabsf(v);
    u32 index;
    if (a <= 0.25f) index = 0;
    else if (a < 0.75f) index = 1;
    else if (a <= 1.25f) index = 2;
    else if (a < 1.75f) index = 3;
    else if (a <= 2.5f) index = 4;
    else if (a < 3.5f) index = 5;
    else if (a <= 5.0f) index = 6;
    else index = 7;
    return index | (((__float_as_uint(v) >> 31) && index != 0) ? 8u : 0u);
}

__device__ __forceinline__ float global_scale(const u32* amax_bits) {
    float amax = __uint_as_float(*amax_bits);
    return amax == 0.0f ? 1.0f : 2688.0f / amax;
}

extern "C" __global__ void nvfp4_amax(const u16* w, u64 n, u32* amax_bits) {
    u32 local = 0;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
        local = max(local, ((u32)w[i] & 0x7fffu) << 16);
    }
    for (int o = 16; o > 0; o >>= 1) local = max(local, __shfl_xor_sync(0xffffffffu, local, o));
    if ((threadIdx.x & 31) == 0) atomicMax(amax_bits, local);
}

extern "C" __global__ void nvfp4_quantize(const u16* w, u64 groups, const u32* amax_bits, u8* packed, u8* scales) {
    float global = global_scale(amax_bits);
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g == 0) {
        float* header = (float*)scales;
        header[0] = 1.0f / global;
    }
    if (g >= groups) return;
    const u16* src = w + g * 16;
    float v[16];
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        v[i] = bf16_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    u8 code = f32_e4m3(fminf(amax / 6.0f * global, 448.0f));
    scales[16 + g] = code;
    float s = e4m3_f32(code) / global;
    u64 out = 0;
    for (int i = 0; i < 16; i++) {
        u32 q = s > 0.0f ? f32_e2m1(v[i] / s) : 0u;
        out |= (u64)q << (4 * i);
    }
    ((u64*)packed)[g] = out;
}
