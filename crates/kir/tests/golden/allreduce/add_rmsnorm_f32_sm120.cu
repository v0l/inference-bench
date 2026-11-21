// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// allreduce add_rmsnorm ranks=2 dtype=f32 threads=256 blocks=16 width=5120 eps=1e-6
constexpr int AR_RANKS = 2;
typedef unsigned short bf16_t;

constexpr int AR_MAX_BLOCKS = 64;
constexpr long long AR_SPIN_LIMIT = 20000000000LL;

__device__ __forceinline__ float ar_ld(const float* p) { return __ldcg(p); }
__device__ __forceinline__ float ar_ld(const bf16_t* p) { return __uint_as_float(((unsigned)__ldcg(p)) << 16); }
__device__ __forceinline__ bf16_t ar_to_bf16(float v) {
    unsigned u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}

__device__ __forceinline__ unsigned ar_bf16_rn(float v) {
    unsigned u = __float_as_uint(v);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ void ar_split2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    unsigned ha = ar_bf16_rn(a), hb = ar_bf16_rn(b);
    unsigned la = ar_bf16_rn(a - __uint_as_float(ha << 16)), lb = ar_bf16_rn(b - __uint_as_float(hb << 16));
    *(unsigned*)(hi + at) = ha | (hb << 16);
    *(unsigned*)(lo + at) = la | (lb << 16);
}

__device__ __forceinline__ unsigned ar_load_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void ar_store_release(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned* ar_flags(const unsigned long long* table, int rank) {
    return (unsigned*)table[AR_RANKS + rank];
}

__device__ __forceinline__ void ar_begin(const unsigned long long* table, int rank, unsigned* epoch) {
    if (threadIdx.x == 0) *epoch = ar_flags(table, rank)[AR_RANKS * AR_MAX_BLOCKS + blockIdx.x];
    __syncthreads();
}

__device__ __forceinline__ void ar_barrier(const unsigned long long* table, int rank, unsigned epoch) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < AR_RANKS) {
        int p = threadIdx.x;
        ar_store_release(ar_flags(table, p) + blockIdx.x * AR_RANKS + rank, epoch);
        const unsigned* mine = ar_flags(table, rank) + blockIdx.x * AR_RANKS + p;
        long long start = clock64();
        while ((int)(ar_load_acquire(mine) - epoch) < 0) {
            if (clock64() - start > AR_SPIN_LIMIT) asm volatile("trap;");
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void ar_end(const unsigned long long* table, int rank, unsigned epoch) {
    if (threadIdx.x == 0) ar_flags(table, rank)[AR_RANKS * AR_MAX_BLOCKS + blockIdx.x] = epoch;
}

__device__ float ar_block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < warps; i++) r += scratch[i];
    return r;
}

extern "C" __global__ void __launch_bounds__(256) ar_add_rmsnorm_f32(float* in, float* x, const float* weight, bf16_t* hi, bf16_t* lo, const unsigned long long* table, int rank, int rows) {
    constexpr int WIDTH = 5120;
    __shared__ unsigned epoch;
    __shared__ float scratch[32];
    ar_begin(table, rank, &epoch);
    float* mine = (float*)table[rank];
    for (int r = blockIdx.x; r < rows; r += gridDim.x)
        for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {
            mine[(size_t)r * WIDTH + i] = in[(size_t)r * WIDTH + i];
            in[(size_t)r * WIDTH + i] = 0.0f;
        }
    ar_barrier(table, rank, epoch + 1);
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        size_t base = (size_t)r * WIDTH;
        float ss = 0.0f;
        constexpr int QUADS = (WIDTH / 4 + 256 - 1) / 256;
        float4 pv[QUADS][AR_RANKS];
#pragma unroll
        for (int q = 0; q < QUADS; q++) {
            int i = threadIdx.x + q * 256;
            if (i < WIDTH / 4) {
#pragma unroll
                for (int p = 0; p < AR_RANKS; p++) pv[q][p] = __ldcg((const float4*)((const float*)table[p] + base) + i);
            }
        }
#pragma unroll
        for (int q = 0; q < QUADS; q++) {
            int i = threadIdx.x + q * 256;
            if (i < WIDTH / 4) {
                float4 v = ((const float4*)(x + base))[i];
                float s[4] = {v.x, v.y, v.z, v.w};
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int p = 0; p < AR_RANKS; p++) {
                    acc[0] += pv[q][p].x;
                    acc[1] += pv[q][p].y;
                    acc[2] += pv[q][p].z;
                    acc[3] += pv[q][p].w;
                }
#pragma unroll
                for (int j = 0; j < 4; j++) s[j] = s[j] + acc[j];
                ss += s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
                ((float4*)(x + base))[i] = make_float4(s[0], s[1], s[2], s[3]);
            }
        }
        float inv = rsqrtf(ar_block_sum(ss, scratch) / WIDTH + 1e-6f);
        for (int i = threadIdx.x; i < WIDTH / 4; i += blockDim.x) {
            float4 v = ((const float4*)(x + base))[i];
            float4 w = ((const float4*)weight)[i];
            ar_split2(hi, lo, base + 4 * i, w.x * (v.x * inv), w.y * (v.y * inv));
            ar_split2(hi, lo, base + 4 * i + 2, w.z * (v.z * inv), w.w * (v.w * inv));
        }
        __syncthreads();
    }
    ar_barrier(table, rank, epoch + 2);
    ar_end(table, rank, epoch + 2);
}

