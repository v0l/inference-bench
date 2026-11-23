mod chunk;
mod chunk_lib;
mod chunk_prep;
mod chunk_scan;
mod common;
mod conv;
mod prim;
mod draft;
mod ssd;
mod step;
mod verify;

use common::common;
use crate::linear::{Family, LinearError, LinearKernel, constants};

pub(crate) fn chunked_scans(kernels: &[crate::Kernel]) -> Result<String, String> {
    let mut out = String::new();
    for k in kernels {
        if let crate::Body::Recurrence(r) = &k.body
            && let Some(f) = r.chunk_form(k)
        {
            out.push_str(&ssd::emit(r, k, &f)?);
        }
    }
    Ok(if out.is_empty() { out } else { format!("{}\n{}\n{out}", chunk_lib::library(), chunk_lib::planes()) })
}


pub fn emit_linear(kernels: &[LinearKernel]) -> Result<crate::KirSource, LinearError> {
    let Some(first) = kernels.iter().find(|k| k.family.uses_shape()).or(kernels.first()) else { return Ok(crate::KirSource::default()) };
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
        push(&common());
    }
    if has(Family::Conv) {
        push(&conv::body());
    }
    if has(Family::Step) {
        push(&step::body());
    }
    if has(Family::Chunk) {
        push(&chunk::body());
    }
    if has(Family::Verify) {
        push(&verify::body());
    }
    if has(Family::Draft) {
        push(&draft::body());
    }
    Ok(crate::KirSource::emitted(out))
}
