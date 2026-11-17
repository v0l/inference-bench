use crate::linear::{Family, LinearError, LinearKernel, constants};

const COMMON: &str = include_str!("linear/common.cu");
const CONV: &str = include_str!("linear/conv.cu");
const STEP: &str = include_str!("linear/step.cu");
const CHUNK: &str = include_str!("linear/chunk.cu");
const VERIFY: &str = include_str!("linear/verify.cu");
const DRAFT: &str = include_str!("linear/draft.cu");

pub fn emit_linear(kernels: &[LinearKernel]) -> Result<String, LinearError> {
    let Some(first) = kernels.iter().find(|k| k.family.uses_shape()).or(kernels.first()) else { return Ok(String::new()) };
    for k in kernels {
        k.validate()?;
        if k.family.uses_shape() && (k.shape != first.shape || k.tuning != first.tuning) {
            return Err(LinearError(format!("{}: kernels of one module must share a shape", k.family.block())));
        }
    }
    let has = |f: Family| kernels.iter().any(|k| k.family == f);
    let mut out = String::new();
    let mut push = |s: &str| {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(s);
    };
    if kernels.iter().any(|k| k.family.uses_shape()) {
        push(&constants(&first.shape, &first.tuning));
        push(COMMON);
    }
    if has(Family::Conv) {
        push(CONV);
    }
    if has(Family::Step) {
        push(STEP);
    }
    if has(Family::Chunk) {
        push(CHUNK);
    }
    if has(Family::Verify) {
        push(VERIFY);
    }
    if has(Family::Draft) {
        push(DRAFT);
    }
    Ok(out)
}
