use crate::attention::{AttentionError, AttentionKernel, Family, constants};

const COMMON: &str = include_str!("attention/common.cu");
const ROPE: &str = include_str!("attention/rope.cu");
const SPLIT: &str = include_str!("attention/split.cu");
const FLASH_LEGACY: &str = include_str!("attention/flash_legacy.cu");
const MMA: &str = include_str!("attention/mma.cu");
const DECODE_COMMON: &str = include_str!("attention/decode_common.cu");
const COMBINE: &str = include_str!("attention/combine.cu");
const FLASH_MMA: &str = include_str!("attention/flash_mma.cu");
const DECODE: &str = include_str!("attention/decode.cu");

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
    push(COMMON);
    if has(Family::RopeKv) {
        push(ROPE);
    }
    if has(Family::SplitDecode) {
        push(SPLIT);
    }
    if has(Family::FlashLegacy) {
        push(FLASH_LEGACY);
    }
    if has(Family::FlashMma) || has(Family::GroupedDecode) {
        push(MMA);
    }
    if has(Family::GroupedDecode) {
        push(DECODE_COMMON);
        push(COMBINE);
    }
    if has(Family::FlashMma) {
        push(FLASH_MMA);
    }
    if has(Family::GroupedDecode) {
        push(DECODE);
    }
    Ok(out)
}
