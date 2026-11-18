use crate::attention::{AttentionError, AttentionKernel, Family, constants};

#[path = "attention/combine.rs"]
mod combine;
#[path = "attention/common.rs"]
mod common;
#[path = "attention/decode.rs"]
mod decode;
#[path = "attention/decode_common.rs"]
mod decode_common;
#[path = "attention/flash_legacy.rs"]
mod flash_legacy;
#[path = "attention/flash_mma.rs"]
mod flash_mma;
#[path = "attention/mma.rs"]
mod mma;
#[path = "attention/prim.rs"]
mod prim;
#[path = "attention/rope.rs"]
mod rope;
#[path = "attention/split.rs"]
mod split;


pub fn emit_attention(kernels: &[AttentionKernel]) -> Result<String, AttentionError> {
    let Some(first) = kernels.first() else { return Ok(String::new()) };
    for k in kernels {
        k.validate()?;
        if k.shape != first.shape {
            return Err(AttentionError(format!("{}: kernels of one module must share a shape", k.family.block())));
        }
    }
    let families: Vec<Family> = kernels.iter().map(|k| k.family).collect();
    let has = |f: Family| families.contains(&f);
    let mut out = constants(&first.shape, &families);
    let mut push = |s: &str| {
        out.push('\n');
        out.push_str(s);
    };
    push(&crate::rope::inv_freq_fn(crate::layout::Dialect::Cuda, &[1, 2, 3, 4]));
    push(&common::body());
    if has(Family::RopeKv) {
        push(&rope::body());
    }
    if has(Family::SplitDecode) {
        push(&split::body());
    }
    if has(Family::FlashLegacy) {
        push(&flash_legacy::body());
    }
    if has(Family::FlashMma) || has(Family::GroupedDecode) {
        push(&mma::body());
    }
    if has(Family::GroupedDecode) {
        push(&decode_common::body());
        push(&combine::body());
    }
    if has(Family::FlashMma) {
        push(&flash_mma::body());
    }
    if has(Family::GroupedDecode) {
        push(&decode::body());
    }
    Ok(out)
}
