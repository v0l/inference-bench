// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// allreduce two_shot ranks=2 dtype=f32 threads=512 blocks=16
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

extern "C" __global__ void __launch_bounds__(512) ar_two_shot_f32(float* in, float* out, const unsigned long long* table, int rank, int n, int add, int clear) {
    __shared__ unsigned epoch;
    ar_begin(table, rank, &epoch);
    float* mine = (float*)table[rank];
    size_t chunk = ((size_t)n + AR_RANKS - 1) / AR_RANKS;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int c = 0; c < AR_RANKS; c++) {
        size_t lo = (size_t)c * chunk;
        size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;
        for (size_t i = lo + first; i < hi; i += stride) {
            mine[i] = in[i];
            if (clear) in[i] = 0;
        }
    }
    ar_barrier(table, rank, epoch + 1);
    {
        size_t lo = (size_t)rank * chunk;
        size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;
        for (size_t i = lo + first; i < hi; i += stride) {
            float acc = 0.0f;
#pragma unroll
            for (int p = 0; p < AR_RANKS; p++) acc += ar_ld(&((float*)table[p])[i]);
            mine[i] = acc;
        }
    }
    ar_barrier(table, rank, epoch + 2);
    for (int c = 0; c < AR_RANKS; c++) {
        size_t lo = (size_t)c * chunk;
        size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;
        const float* src = (const float*)table[c];
        for (size_t i = lo + first; i < hi; i += stride) {
            float v = ar_ld(&src[i]);
            out[i] = add ? out[i] + v : v;
        }
    }
    ar_barrier(table, rank, epoch + 3);
    ar_end(table, rank, epoch + 3);
}

