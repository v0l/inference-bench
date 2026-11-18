use crate::vision::{VisionError, VisionKernel};

const HEAD: &str = include_str!("vision/head.cu");
const GEMM: &str = include_str!("vision/gemm.cu");
const NORM: &str = include_str!("vision/norm.cu");
const POS: &str = include_str!("vision/pos.cu");
const ATTENTION: &str = include_str!("vision/attention.cu");
const POOL: &str = include_str!("vision/pool.cu");

pub fn emit_vision(k: &VisionKernel) -> Result<String, VisionError> {
    k.validate()?;
    Ok(format!(
        "{HEAD}constexpr bool GEMM_SPLIT = {p};\nconstexpr bool ATT_SPLIT = {p};\n{GEMM}{NORM}{POS}constexpr int AT_HDP = {hdp};\n{ATTENTION}{POOL}",
        p = k.precise,
        hdp = k.head_pad(),
    ))
}
