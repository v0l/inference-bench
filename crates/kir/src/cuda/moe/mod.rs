use crate::moe::{Family, MoeKernel};

mod gemv;
mod mma;
mod mma8;
mod mma_bf16;
mod prim;
mod route;

fn experts(k: &MoeKernel) -> String {
    let mut out = prim::constants(&k.shape);
    out.push_str(&gemv::gemv());
    out.push_str(&route::route_logits());
    out.push_str(&route::route());
    out.push_str(&gemv::pair_entries());
    out.push_str(&route::combine());
    out.push_str(&route::scale_inputs());
    out.push_str(&route::permute());
    out.push_str(&gemv::grouped());
    out.push_str(&gemv::group_entries());
    out
}

pub fn emit_moe(kernels: &[MoeKernel]) -> Result<crate::KirSource, String> {
    let mut out = String::new();
    for k in kernels {
        k.validate()?;
        match k.family {
            Family::Experts => out.push_str(&experts(k)),
            Family::ExpertTable => {
                out.push_str(&gemv::table_gemv());
                out.push_str(&gemv::table_entries());
            }
            Family::ExpertsGgml => out.push_str(&gemv::ggml_experts()),
            Family::GroupedMma => {
                out.push('\n');
                out.push_str(&mma::grouped(false));
            }
            Family::GroupedMx => {
                out.push('\n');
                out.push_str(&mma::grouped(true));
            }
            Family::GroupedMmaBf16 => {
                out.push('\n');
                out.push_str(&mma_bf16::grouped());
            }
            Family::GroupedMmaFp8 => {
                out.push('\n');
                out.push_str(&mma8::grouped());
            }
        }
    }
    Ok(crate::KirSource::emitted(out))
}
