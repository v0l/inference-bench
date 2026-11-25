use crate::vision::{VisionError, VisionKernel};

#[path = "vision/kernels.rs"]
mod kernels;
#[path = "vision/prim.rs"]
mod prim;

pub fn emit_vision(k: &VisionKernel) -> Result<crate::KirSource, VisionError> {
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
    if k.tiled {
        out += &kernels::tiled_kernels();
    }
    if k.joint {
        out += &kernels::joint_kernels();
        out += &kernels::cross_attention_kernel(k).map_err(VisionError)?;
    }
    Ok(crate::KirSource::emitted(out))
}

pub fn step_gemm(k: &VisionKernel, rule: &crate::expr::Expr) -> Result<String, VisionError> {
    let bad = |s: String| VisionError(format!("vision step gemm: {s}"));
    let mut names = Vec::new();
    rule.names(&mut names);
    if let Some(n) = names.iter().find(|n| !["x", "v", "s0", "s1"].contains(&n.as_str())) {
        return Err(bad(format!("the epilogue rule names {n:?}; it reads x, v, s0 and s1")));
    }
    let raw = |n: &str| Some(crate::expr::Expr::Raw(format!("st_{n}")));
    let body = rule.subst(&raw).and_then(|e| e.lower(crate::layout::Dialect::Cuda, &[])).map_err(bad)?;
    let plain = kernels::gemm_kernel(k);
    let start = plain.find("extern \"C\"").ok_or_else(|| bad("no kernel entry".into()))?;
    let resid = "                    if (resid) {\n                        v0 += resid[at];\n                        v1 += resid[at + 1];\n                    }\n";
    let step = "                    v0 = gm_step(out[at], v0, resid[0], resid[1]);\n                    v1 = gm_step(out[at + 1], v1, resid[0], resid[1]);\n";
    let text = &plain[start..];
    if !text.contains(resid) || !text.contains(" vis_mma(") {
        return Err(bad("the gemm store has no residual slot".into()));
    }
    Ok(format!("__device__ __forceinline__ float gm_step(float st_x, float st_v, float st_s0, float st_s1) {{\n    return {body};\n}}\n\n{}", text.replace(" vis_mma(", " vis_mma_step(").replace(resid, step)))
}

pub fn emit_vision_step(k: &VisionKernel, rule: &crate::expr::Expr) -> Result<crate::KirSource, VisionError> {
    let mut out = emit_vision(k)?.into_string();
    out += &step_gemm(k, rule)?;
    Ok(crate::KirSource::emitted(out))
}
