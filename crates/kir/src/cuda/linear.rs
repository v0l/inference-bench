mod chunk;
mod common;
mod conv;
mod draft;
mod step;
mod verify;

use chunk::CHUNK;
use common::COMMON;
use conv::CONV;
use draft::DRAFT;
use step::STEP;
use verify::VERIFY;
use crate::linear::{Family, LinearError, LinearKernel, constants};


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
