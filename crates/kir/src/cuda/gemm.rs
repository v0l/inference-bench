use crate::gemm::{BATCH_KC, Fp4Gemv, Fp8Scale, GemmError, Kind, KernelInstance, Mainloop, WeightFormat};
use crate::Language;

mod core;
mod fp4;
mod fp4_act;
mod fp4_batch;
mod fp4_head;
mod fp8_act;
mod fp8_batch;
mod ggml_batch;
pub mod prim;

use super::quant;

pub fn fp4_act_source() -> String {
    fp4_act::body()
}

fn uses_fp4(k: &KernelInstance) -> bool {
    match k.kind {
        Kind::Tile { wf, .. } | Kind::Batch { wf, .. } | Kind::BatchReduce { wf, .. } => wf == WeightFormat::Fp4,
        Kind::Fp4Act { .. } | Kind::QuantFp4 | Kind::Gemv { .. } | Kind::HeadNormFp4 { .. } => true,
        Kind::Fp8Act { .. } | Kind::QuantFp8 { .. } | Kind::GgmlBatch { .. } => false,
    }
}

fn batch_of(k: &KernelInstance) -> Option<WeightFormat> {
    match k.kind {
        Kind::Batch { wf, .. } | Kind::BatchReduce { wf, .. } => Some(wf),
        Kind::GgmlBatch { .. } => Some(WeightFormat::Ggml),
        _ => None,
    }
}

pub fn needs_arch_features(kernels: &[KernelInstance]) -> bool {
    kernels.iter().any(|k| matches!(k.kind, Kind::Fp4Act { .. } | Kind::QuantFp4))
}

pub fn gemm_library(kernels: &[KernelInstance]) -> String {
    let any = |f: &dyn Fn(&KernelInstance) -> bool| kernels.iter().any(f);
    let mut out = core::body();
    let mut push = |s: &str| {
        out.push('\n');
        out.push_str(s);
    };
    if any(&|k| matches!(k.kind, Kind::Fp8Act { .. })) {
        push(&fp8_act::body());
    }
    if any(&uses_fp4) {
        push(&fp4::body());
    }
    if any(&|k| matches!(k.kind, Kind::Fp4Act { .. } | Kind::QuantFp4)) {
        push(&fp4_act::body());
    }
    if any(&|k| batch_of(k) == Some(WeightFormat::Fp4)) {
        push(&format!("constexpr int F4B_KC = {BATCH_KC};\n{}", fp4_batch::body()));
    }
    if any(&|k| batch_of(k) == Some(WeightFormat::Ggml)) {
        push(&format!("constexpr int GGR_KC = {BATCH_KC};\n{}", ggml_batch::body()));
    }
    if any(&|k| batch_of(k) == Some(WeightFormat::Fp8)) {
        push(&format!("constexpr int F8B_KC = {BATCH_KC};\n{}", fp8_batch::body()));
    }
    if any(&|k| matches!(k.kind, Kind::HeadNormFp4 { .. })) {
        push(&fp4_head::body());
    }
    out
}

pub fn emit_gemm(k: &KernelInstance) -> Result<String, GemmError> {
    k.validate()?;
    if k.target.language != Language::Cuda {
        return Err(GemmError::Invalid(format!("{}: the CUDA emitter cannot target {:?}", k.name, k.target.language)));
    }
    let (name, threads) = (&k.name, k.threads);
    Ok(match k.kind {
        Kind::Tile { shape, wf, main: Mainloop::Mma(m), token_major } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const void* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_tc<{n}, {kk}, {bm}, {bn}, {wm}, {wn}, {bk}, {stages}, {epi}, {bias}, {dec}, {split}, {token_major}>(w, s, b, xh, xl, out, oh, ol, tokens);\n}}\n\n",
            n = shape.n,
            kk = shape.k,
            bm = m.tile.bm,
            bn = m.tile.bn,
            wm = m.tile.wm,
            wn = m.tile.wn,
            bk = m.bk,
            stages = m.stages,
            epi = shape.epi.cuda(),
            bias = shape.bias,
            dec = wf.decoder(),
            split = m.split,
        ),
        Kind::Tile { main: Mainloop::DequantF16 { .. }, .. } => unreachable!("validate refuses DequantF16 on CUDA"),
        Kind::Fp8Act { shape, tile, stages, scale } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const float* s, const float* b, const unsigned char* xq, const float* xs, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_fp8<{n}, {kk}, {bm}, {bn}, {wm}, {wn}, {stages}, {epi}, {bias}, {per_token}>(w, s, b, xq, xs, out, oh, ol, tokens);\n}}\n\n",
            n = shape.n,
            kk = shape.k,
            bm = tile.bm,
            bn = tile.bn,
            wm = tile.wm,
            wn = tile.wn,
            epi = shape.epi.cuda(),
            bias = shape.bias,
            per_token = scale == Fp8Scale::Channel,
        ),
        Kind::Fp4Act { shape, tile, stages } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const unsigned char* s, const unsigned char* xq, const unsigned char* xs, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_f4a<{n}, {kk}, {bm}, {bn}, {wm}, {wn}, {stages}, {epi}>(w, s, xq, xs, out, oh, ol, tokens);\n}}\n\n",
            n = shape.n,
            kk = shape.k,
            bm = tile.bm,
            bn = tile.bn,
            wm = tile.wm,
            wn = tile.wn,
            epi = shape.epi.cuda(),
        ),
        Kind::Batch { shape, wf: WeightFormat::Fp4, batch, blocks_per_sm } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}, {blocks_per_sm}) {name}(const void* w, const unsigned char* s, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {{\n    gemv_f4b<{n}, {kk}, {epi}, {nt}, {warps}, {stages}, {mj}>(w, s, xh, xl, out, oh, ol, part, tokens, splits);\n}}\n\n",
            n = shape.n,
            kk = shape.k,
            epi = shape.epi.cuda(),
            nt = batch.tokens / 8,
            warps = batch.warps,
            stages = batch.stages,
            mj = batch.mj,
        ),
        Kind::Batch { shape, batch, blocks_per_sm, .. } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}, {blocks_per_sm}) {name}(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {{\n    gemv_f8b<{n}, {kk}, {epi}, {bias}, {nt}, {warps}, {stages}, {mj}>(w, s, b, xh, xl, out, oh, ol, part, tokens, splits);\n}}\n\n",
            n = shape.n,
            kk = shape.k,
            epi = shape.epi.cuda(),
            bias = shape.bias,
            nt = batch.tokens / 8,
            warps = batch.warps,
            stages = batch.stages,
            mj = batch.mj,
        ),
        Kind::GgmlBatch { shape, batch, blocks_per_sm, split } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}, {blocks_per_sm}) {name}(const void* w, const unsigned* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, float* part, int tokens, int splits) {{\n    gemv_ggr<{n}, {kk}, {epi}, {bias}, {split}, {nt}, {warps}, {stages}>(w, s, b, xh, xl, out, oh, ol, part, tokens, splits);\n}}\n\n",
            n = shape.n,
            kk = shape.k,
            epi = shape.epi.cuda(),
            bias = shape.bias,
            nt = batch.tokens / 8,
            warps = batch.warps,
            stages = batch.stages,
        ),
        Kind::BatchReduce { shape, wf: WeightFormat::Ggml } => format!(
            "extern \"C\" __global__ void {name}(const float* part, const float* b, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {{\n    ggr_reduce<{n}, {epi}, {bias}>(part, b, out, oh, ol, tokens, splits);\n}}\n\n",
            n = shape.n,
            epi = shape.epi.cuda(),
            bias = shape.bias,
        ),
        Kind::BatchReduce { shape, wf: WeightFormat::Fp4 } => format!(
            "extern \"C\" __global__ void {name}(const float* part, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {{\n    f4b_reduce<{n}, {epi}>(part, out, oh, ol, tokens, splits);\n}}\n\n",
            n = shape.n,
            epi = shape.epi.cuda(),
        ),
        Kind::BatchReduce { shape, .. } => format!(
            "extern \"C\" __global__ void {name}(const float* part, const float* b, float* out, bf16_t* oh, bf16_t* ol, int tokens, int splits) {{\n    f8b_reduce<{n}, {epi}, {bias}>(part, b, out, oh, ol, tokens, splits);\n}}\n\n",
            n = shape.n,
            epi = shape.epi.cuda(),
            bias = shape.bias,
        ),
        Kind::QuantFp8 { scale: Fp8Scale::Block } => format!("{}\n", quant::quant_fp8(name, Fp8Scale::Block)),
        Kind::QuantFp8 { scale: Fp8Scale::Channel } => format!("{}\n", quant::quant_fp8(name, Fp8Scale::Channel)),
        Kind::QuantFp4 => format!("{}\n", quant::quant_fp4(name)),
        Kind::Gemv { op: Fp4Gemv::GateUpAct, .. } => format!(
            "extern \"C\" __global__ void __launch_bounds__(FP4_GEMV_THREADS) {name}(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {{\n    gate_up_act_norm<DecFp4>(w, s, x, nw, out);\n}}\n\n"
        ),
        Kind::Gemv { op: Fp4Gemv::Down, n, k } => format!(
            "extern \"C\" __global__ void __launch_bounds__(FP4_GEMV_THREADS) {name}(const void* w, const unsigned char* s, const float* b, const float* xt, float* out) {{\n    linear_w<DecFp4, {n}, {k}, false, true>(w, s, b, xt, out);\n}}\n\n"
        ),
        Kind::HeadNormFp4 { n, k: kk, rows } => format!(
            "extern \"C\" __global__ void __launch_bounds__(FP4_GEMV_THREADS) {name}(const void* w, const unsigned char* s, const float* b, const float* x, const float* nw, float* out) {{\n    head_norm_f4<{n}, {kk}, {rows}>(w, s, x, nw, out);\n}}\n\n"
        ),
    })
}

pub fn emit_gemms(kernels: &[KernelInstance]) -> Result<String, GemmError> {
    let mut out = gemm_library(kernels);
    out.push('\n');
    for k in kernels {
        out.push_str(&emit_gemm(k)?);
    }
    Ok(out)
}
