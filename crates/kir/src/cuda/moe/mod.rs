use crate::moe::{Family, MoeKernel};

mod gemv;
mod mma;
mod prim;
mod route;

fn experts(k: &MoeKernel) -> String {
    let mut out = prim::constants(&k.shape);
    out.push_str(&gemv::gemv());
    out.push_str(&route::route_logits());
    out.push_str(&route::route());
    out.push_str(&gemv::pair_entries());
    out.push_str(&route::combine());
    out.push_str(&route::permute());
    out.push_str(&gemv::grouped());
    out.push_str(&gemv::group_entries());
    out
}

pub fn emit_moe(kernels: &[MoeKernel]) -> Result<String, String> {
    let mut out = String::new();
    for k in kernels {
        k.validate()?;
        match k.family {
            Family::Experts => out.push_str(&experts(k)),
            Family::GroupedMma => {
                out.push('\n');
                out.push_str(&mma::grouped());
            }
        }
    }
    Ok(out)
}
