use crate::moe::{Family, MoeKernel};

const MOE: &str = include_str!("moe.cu");
const MOE_MMA: &str = include_str!("moe_mma.cu");

pub fn emit_moe(kernels: &[MoeKernel]) -> Result<String, String> {
    let mut out = String::new();
    for k in kernels {
        k.validate()?;
        let m = &k.shape;
        match k.family {
            Family::Experts => out.push_str(&format!(
                "constexpr int MOE_E = {};
constexpr int MOE_K = {};
constexpr int MOE_PAIRS = {};
constexpr int MOE_INTER = {};
constexpr bool MOE_SHARED = {};
constexpr bool MOE_NORM_TOPK = {};
constexpr bool MOE_SHARED_GATE = {};
constexpr bool MOE_SIGMOID = {};
{MOE}",
                m.experts, m.top_k, m.pairs, m.inter, m.shared, m.norm_topk, m.shared_gate, m.sigmoid
            )),
            Family::GroupedMma => {
                out.push('\n');
                out.push_str(MOE_MMA);
            }
        }
    }
    Ok(out)
}
