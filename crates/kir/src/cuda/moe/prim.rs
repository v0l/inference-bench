pub use crate::cuda::gemm::prim::{cp8_zfill, device, e2m1_to_f32, e2m1x2, mma_fp4, swizzle, warp_ids, zero_frag, pipeline_prologue};

pub const QUANTS: [(u8, &str); 3] = [(0, "bf16"), (1, "fp4"), (2, "fp8")];

pub fn global(threads: Option<u32>, name: &str, args: &str, body: &str) -> String {
    let bounds = threads.map(|t| format!(" __launch_bounds__({t})")).unwrap_or_default();
    format!("extern \"C\" __global__ void{bounds} {name}({args}) {{\n{body}}}\n\n")
}

pub fn delegate(threads: u32, name: &str, args: &str, target: &str, call: &str) -> String {
    global(Some(threads), name, args, &format!("    {target}({call});\n"))
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
