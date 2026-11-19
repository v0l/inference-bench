use crate::vision::{VisionError, VisionKernel};

#[path = "vision/kernels.rs"]
mod kernels;
#[path = "vision/prim.rs"]
mod prim;

pub fn emit_vision(k: &VisionKernel) -> Result<String, VisionError> {
    k.validate()?;
    let split = k.precise.to_string();
    let mut out = prim::scalar_prelude();
    out += &format!("constexpr bool GEMM_SPLIT = {split};\nconstexpr bool ATT_SPLIT = {split};\n");
    out += &prim::store_act2("GEMM_SPLIT");
    out += &prim::to_f16_kernel();
    out += &prim::cp_async();
    out += &prim::ldsm_x4(false);
    out += &prim::ldsm_x4(true);
    out += &prim::mma("mma_bf16", "bf16");
    out += &prim::mma("mma_f16", "f16");
    out += &prim::mma_f16_hi_lo();
    out += &prim::fast_exp2();
    out += &prim::swz4();
    out += &kernels::gemm_kernel(k);
    out += &kernels::norm_kernels();
    out += &kernels::pos_kernel();
    out += &format!("constexpr int AT_HDP = {};\n", k.head_pad());
    out += &kernels::attention_kernels(k);
    out += &prim::block_sum();
    out += &kernels::pool_kernel();
    Ok(out)
}
