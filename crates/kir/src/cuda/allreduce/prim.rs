pub fn device(ret: &str, name: &str, args: &str, body: &str) -> String {
    format!("__device__ __forceinline__ {ret} {name}({args}) {{\n{body}}}\n\n")
}

pub fn device_line(ret: &str, name: &str, args: &str, expr: &str) -> String {
    format!("__device__ __forceinline__ {ret} {name}({args}) {{ return {expr}; }}\n")
}

pub fn constants(max_blocks: u32, spin_limit: u64) -> String {
    format!("typedef unsigned short bf16_t;\n\nconstexpr int AR_MAX_BLOCKS = {max_blocks};\nconstexpr long long AR_SPIN_LIMIT = {spin_limit}LL;\n\n")
}

pub fn coherent_loads() -> String {
    device_line("float", "ar_ld", "const float* p", "__ldcg(p)") + &device_line("float", "ar_ld", "const bf16_t* p", "__uint_as_float(((unsigned)__ldcg(p)) << 16)")
}

const ROUND_BITS: &str = "    unsigned u = __float_as_uint(v);\n";

pub fn bf16_store() -> String {
    device(
        "bf16_t",
        "ar_to_bf16",
        "float v",
        &format!("{ROUND_BITS}    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_t)((u >> 16) | 0x40);\n    u += 0x7fffu + ((u >> 16) & 1u);\n    return (bf16_t)(u >> 16);\n"),
    )
}

pub fn bf16_round() -> String {
    device("unsigned", "ar_bf16_rn", "float v", &format!("{ROUND_BITS}    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;\n"))
}

pub fn split_planes() -> String {
    device(
        "void",
        "ar_split2",
        "bf16_t* hi, bf16_t* lo, size_t at, float a, float b",
        "    unsigned ha = ar_bf16_rn(a), hb = ar_bf16_rn(b);\n    unsigned la = ar_bf16_rn(a - __uint_as_float(ha << 16)), lb = ar_bf16_rn(b - __uint_as_float(hb << 16));\n    *(unsigned*)(hi + at) = ha | (hb << 16);\n    *(unsigned*)(lo + at) = la | (lb << 16);\n",
    )
}

pub fn system_flags() -> String {
    device("unsigned", "ar_load_acquire", "const unsigned* p", "    unsigned v;\n    asm volatile(\"ld.acquire.sys.global.u32 %0, [%1];\" : \"=r\"(v) : \"l\"(p) : \"memory\");\n    return v;\n")
        + &device("void", "ar_store_release", "unsigned* p, unsigned v", "    asm volatile(\"st.release.sys.global.u32 [%0], %1;\" :: \"l\"(p), \"r\"(v) : \"memory\");\n")
        + &device("unsigned*", "ar_flags", "const unsigned long long* table, int rank", "    return (unsigned*)table[AR_RANKS + rank];\n")
}

fn epoch_word() -> &'static str {
    "ar_flags(table, rank)[AR_RANKS * AR_MAX_BLOCKS + blockIdx.x]"
}

pub fn barrier() -> String {
    let wait = "        while ((int)(ar_load_acquire(mine) - epoch) < 0) {\n            if (clock64() - start > AR_SPIN_LIMIT) asm volatile(\"trap;\");\n        }\n";
    device("void", "ar_begin", "const unsigned long long* table, int rank, unsigned* epoch", &format!("    if (threadIdx.x == 0) *epoch = {};\n    __syncthreads();\n", epoch_word()))
        + &device(
            "void",
            "ar_barrier",
            "const unsigned long long* table, int rank, unsigned epoch",
            &format!(
                "    __threadfence_system();\n    __syncthreads();\n    if (threadIdx.x < AR_RANKS) {{\n        int p = threadIdx.x;\n        ar_store_release(ar_flags(table, p) + blockIdx.x * AR_RANKS + rank, epoch);\n        const unsigned* mine = ar_flags(table, rank) + blockIdx.x * AR_RANKS + p;\n        long long start = clock64();\n{wait}    }}\n    __syncthreads();\n"
            ),
        )
        + &device("void", "ar_end", "const unsigned long long* table, int rank, unsigned epoch", &format!("    if (threadIdx.x == 0) {} = epoch;\n", epoch_word()))
}

pub fn block_sum() -> String {
    "__device__ float ar_block_sum(float v, float* scratch) {\n    int lane = threadIdx.x & 31;\n    int warp = threadIdx.x >> 5;\n    int warps = (blockDim.x + 31) >> 5;\n    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);\n    __syncthreads();\n    if (lane == 0) scratch[warp] = v;\n    __syncthreads();\n    float r = 0.0f;\n    for (int i = 0; i < warps; i++) r += scratch[i];\n    return r;\n}\n\n".to_string()
}

pub fn library(max_blocks: u32, spin_limit: u64) -> String {
    [constants(max_blocks, spin_limit), coherent_loads(), bf16_store(), bf16_round(), split_planes(), system_flags(), barrier(), block_sum()].concat()
}

pub fn pad(n: usize) -> String {
    " ".repeat(n)
}

pub fn rank_sum(n: usize, t: &str, at: &str) -> String {
    let p = pad(n);
    format!("{p}float acc = 0.0f;\n#pragma unroll\n{p}for (int p = 0; p < AR_RANKS; p++) acc += ar_ld(&(({t}*)table[p])[{at}]);\n")
}

pub fn grid_stride(n: usize, from: &str, to: &str, body: &str) -> String {
    let p = pad(n);
    format!("{p}for (size_t i = {from}; i < {to}; i += stride) {{\n{body}{p}}}\n")
}

pub fn stage(n: usize) -> String {
    let p = pad(n);
    format!("{p}mine[i] = in[i];\n{p}if (clear) in[i] = 0;\n")
}

pub fn chunk_bounds(n: usize, c: &str) -> String {
    let p = pad(n);
    format!("{p}size_t lo = (size_t){c} * chunk;\n{p}size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;\n")
}

pub fn sync_point(epoch: u32) -> String {
    format!("    ar_barrier(table, rank, epoch + {epoch});\n")
}

pub fn finish(epoch: u32) -> String {
    format!("{}    ar_end(table, rank, epoch + {epoch});\n}}\n\n", sync_point(epoch))
}
