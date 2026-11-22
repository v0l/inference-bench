pub use crate::cuda::gemm::prim::{cp4_zfill, cp8_zfill, device, e2m1_to_f32, e2m1x2, multi_row_dot, RowFormat, RowLoad, mma_e4m3, mma_fp4, mma_mx, swizzle, warp_ids, zero_frag, pipeline_prologue};

pub const QUANTS: [(u8, &str); 4] = [(0, "bf16"), (1, "fp4"), (2, "fp8"), (3, "mx")];

pub fn global(threads: Option<u32>, name: &str, args: &str, body: &str) -> String {
    let bounds = threads.map(|t| format!(" __launch_bounds__({t})")).unwrap_or_default();
    format!("extern \"C\" __global__ void{bounds} {name}({args}) {{\n{body}}}\n\n")
}

pub fn delegate(threads: u32, name: &str, args: &str, target: &str, call: &str) -> String {
    global(Some(threads), name, args, &format!("    {target}({call});\n"))
}

pub fn constants(m: &crate::ops::MoeShape) -> String {
    format!(
        "constexpr int MOE_E = {};\nconstexpr int MOE_LOCAL = {};\nconstexpr int MOE_K = {};\nconstexpr int MOE_PAIRS = {};\nconstexpr int MOE_INTER = {};\nconstexpr int MOE_SHARED = {};\nconstexpr bool MOE_NORM_TOPK = {};\nconstexpr float MOE_ROUTED_SCALE = {};\nconstexpr float MOE_OUT_SCALE = {};\nconstexpr bool MOE_SHARED_GATE = {};\nconstexpr bool MOE_SIGMOID = {};\nconstexpr int MOE_GROUPS = {};\nconstexpr int MOE_TOPK_GROUP = {};\nconstexpr bool MOE_BIAS = {};\nconstexpr bool MOE_ROUTER_BIAS = {};\nconstexpr bool MOE_EBIAS = {};\nconstexpr bool MOE_HASH = {};\nconstexpr int MOE_BLOCK = {};\nconstexpr bool MOE_MAPPED = {};\nconstexpr int MOE_UP_WEIGHTS = {};\nconstexpr bool MOE_INPUT_SCALED = {};\n",
        m.experts, m.local, m.top_k, m.pairs, m.inter, m.shared, m.norm_topk, crate::ops::f32_literal(m.routed_scale), crate::ops::f32_literal(m.out_scale), m.shared_gate, m.sigmoid, m.groups.max(1), m.topk_group.max(1), m.score_bias, m.router_bias, m.expert_bias, m.hash, if m.block_group == 0 { 32 } else { m.block_group }, m.mapped, if m.mapped { 1 } else { 2 }, m.input_scaled
    )
}

pub fn fp8_grid() -> &'static str {
    "constexpr int MOE_WARPS = 8;\ntemplate <int K> constexpr int MOE_FP8_KB = (K + FP8_BLOCK - 1) / FP8_BLOCK;\ntemplate <int ROWS, int K> constexpr int MOE_FP8_GRID = ((ROWS + FP8_BLOCK - 1) / FP8_BLOCK) * MOE_FP8_KB<K>;\n\n"
}
