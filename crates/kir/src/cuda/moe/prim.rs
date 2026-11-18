pub const QUANTS: [(u8, &str); 3] = [(0, "bf16"), (1, "fp4"), (2, "fp8")];

pub fn device(ret: &str, name: &str, args: &str, body: &str) -> String {
    format!("__device__ __forceinline__ {ret} {name}({args}) {{\n{body}}}\n\n")
}

pub fn global(threads: Option<u32>, name: &str, args: &str, body: &str) -> String {
    let bounds = threads.map(|t| format!(" __launch_bounds__({t})")).unwrap_or_default();
    format!("extern \"C\" __global__ void{bounds} {name}({args}) {{\n{body}}}\n\n")
}

pub fn delegate(threads: u32, name: &str, args: &str, target: &str, call: &str) -> String {
    global(Some(threads), name, args, &format!("    {target}({call});\n"))
}

pub fn e2m1() -> String {
    device("float", "moe_e2m1", "unsigned n", "    return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22)) * __uint_as_float(0x7e800000u);\n")
}

pub fn mma_fp4() -> String {
    device(
        "void",
        "moe_mma_fp4",
        "float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa, unsigned sfb",
        r#"    unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
"#,
    )
}

pub fn e2m1x2() -> String {
    device(
        "unsigned",
        "moe_e2m1x2",
        "float lo, float hi",
        "    unsigned short r;\n    asm(\"{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }\" : \"=h\"(r) : \"f\"(hi), \"f\"(lo));\n    return r & 0xffu;\n",
    )
}

pub fn cp8() -> String {
    device(
        "void",
        "moe_cp8",
        "void* smem, const void* gmem, bool valid",
        "    unsigned s = (unsigned)__cvta_generic_to_shared(smem);\n    int n = valid ? 8 : 0;\n    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8, %2;\" :: \"r\"(s), \"l\"(gmem), \"r\"(n));\n",
    )
}

pub fn swizzle() -> String {
    device("int", "mma_chunk", "int r, int c", "    return r * MMA_DATA + ((c ^ (r >> 1)) & 3) * 16;\n")
}

pub fn warp_ids() -> &'static str {
    "    int warp = threadIdx.x >> 5;\n    int lane = threadIdx.x & 31;\n"
}

pub fn constants(m: &crate::ops::MoeShape) -> String {
    format!(
        "constexpr int MOE_E = {};\nconstexpr int MOE_LOCAL = {};\nconstexpr int MOE_K = {};\nconstexpr int MOE_PAIRS = {};\nconstexpr int MOE_INTER = {};\nconstexpr bool MOE_SHARED = {};\nconstexpr bool MOE_NORM_TOPK = {};\nconstexpr bool MOE_SHARED_GATE = {};\nconstexpr bool MOE_SIGMOID = {};\n",
        m.experts, m.local, m.top_k, m.pairs, m.inter, m.shared, m.norm_topk, m.shared_gate, m.sigmoid
    )
}

pub fn fp8_grid() -> &'static str {
    "constexpr int MOE_WARPS = 8;\ntemplate <int K> constexpr int MOE_FP8_KB = (K + FP8_BLOCK - 1) / FP8_BLOCK;\ntemplate <int ROWS, int K> constexpr int MOE_FP8_GRID = ((ROWS + FP8_BLOCK - 1) / FP8_BLOCK) * MOE_FP8_KB<K>;\n\n"
}
