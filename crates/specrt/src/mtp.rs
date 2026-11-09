use crate::codegen::{FP4_STAGES, MMA_VARIANTS, MmaShape, Tile, WeightFormat, f4w_smem};
use crate::spec::{Quant, Spec};
use std::collections::BTreeMap;

pub const MTP_TEMPLATE: &str = include_str!("../templates/mtp.cu");

pub const MTP_KERNELS: &[&str] = &[
    "mtp_fc_in",
    "mtp_draft",
    "mtp_fill",
    "mtp_copy",
    "mtp_pick_rows",
    "mtp_round_in",
    "mtp_chain_rows",
    "spec_accept",
    "gdn_conv_verify",
    "gdn_conv_commit",
    "gdn_verify",
    "gdn_commit",
];

pub const MTP_LINEARS: [&str; 5] = ["mtp_fc", "mtp_qkv", "mtp_o", "mtp_gate_up", "mtp_down"];

pub const MTP_MMA_KERNELS: &[&str] = &[
    "mtp_fc_mma",
    "mtp_fc_mma_big",
    "mtp_fc_mma_narrow",
    "mtp_fc_mma_tiny",
    "mtp_qkv_mma",
    "mtp_qkv_mma_big",
    "mtp_qkv_mma_narrow",
    "mtp_qkv_mma_tiny",
    "mtp_o_mma",
    "mtp_o_mma_big",
    "mtp_o_mma_narrow",
    "mtp_o_mma_tiny",
    "mtp_gate_up_mma",
    "mtp_gate_up_mma_big",
    "mtp_gate_up_mma_narrow",
    "mtp_gate_up_mma_tiny",
    "mtp_down_mma",
    "mtp_down_mma_big",
    "mtp_down_mma_narrow",
    "mtp_down_mma_tiny",
];

pub fn dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str) {
    let d = &spec.dims;
    match linear {
        "mtp_fc" => (d.hidden, 2 * d.hidden, "EPI_STORE"),
        "mtp_qkv" => (spec.qkv_rows(), d.hidden, "EPI_STORE"),
        "mtp_o" => (d.hidden, spec.q_dim(), "EPI_RESIDUAL"),
        "mtp_gate_up" => (2 * d.intermediate, d.hidden, "EPI_ACT"),
        _ => (d.hidden, d.intermediate, "EPI_RESIDUAL"),
    }
}

pub const MTP_HEAD_MMA_KERNELS: &[&str] = &["mtp_head_mma", "mtp_head_mma_big", "mtp_head_mma_narrow", "mtp_head_mma_tiny"];
pub const MTP_HEAD_F4_KERNELS: &[&str] = &["mtp_head_f4w", "mtp_head_f4w_big", "mtp_head_f4w_narrow", "mtp_head_f4w_tiny"];

pub fn head_kernels(spec: &Spec) -> &'static [&'static str] {
    match spec.draft_head_quant() {
        None => &[],
        Some(Quant::Nvfp4G16) => MTP_HEAD_F4_KERNELS,
        Some(_) => MTP_HEAD_MMA_KERNELS,
    }
}

pub fn head_source(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> String {
    let Some(quant) = spec.draft_head_quant() else { return String::new() };
    let (n, k) = (spec.dims.vocab, spec.dims.hidden);
    let mut out = String::new();
    for (i, suffix) in MMA_VARIANTS.iter().enumerate() {
        let t = Tile::variant(suffix, &spec.runtime);
        if quant == Quant::Nvfp4G16 {
            let name = MTP_HEAD_F4_KERNELS[i];
            smem.insert(name, f4w_smem(t));
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const unsigned char* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_f4w<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {stages}, EPI_STORE>(w, s, xh, xl, out, oh, ol, tokens);\n}}\n\n",
                threads = t.threads(),
                bm = t.bm,
                bn = t.bn,
                wm = t.wm,
                wn = t.wn,
                stages = FP4_STAGES,
            ));
        } else {
            let wf = WeightFormat::of(quant);
            let shape = MmaShape::new(t, k, wf, true);
            let name = MTP_HEAD_MMA_KERNELS[i];
            smem.insert(name, shape.smem_bytes());
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_tc<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {bk}, {stages}, EPI_STORE, false, {wf}, true>(w, s, b, xh, xl, out, oh, ol, tokens);\n}}\n\n",
                threads = t.threads(),
                bm = t.bm,
                bn = t.bn,
                wm = t.wm,
                wn = t.wn,
                bk = shape.bk,
                stages = shape.stages,
                wf = wf.code(),
            ));
        }
    }
    out
}

pub fn mtp_source(spec: &Spec, smem: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::from(MTP_TEMPLATE);
    let wf = WeightFormat::Bf16;
    for linear in MTP_LINEARS {
        let (n, k, epi) = dims(spec, linear);
        for suffix in MMA_VARIANTS {
            let shape = MmaShape::new(Tile::variant(suffix, &spec.runtime), k, wf, true);
            let t = shape.tile;
            let name = format!("{linear}_mma{suffix}");
            let name = MTP_MMA_KERNELS.iter().copied().find(|k| *k == name).expect("listed in MTP_MMA_KERNELS");
            smem.insert(name, shape.smem_bytes());
            out.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const void* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl, float* out, bf16_t* oh, bf16_t* ol, int tokens) {{\n    gemm_tc<{n}, {k}, {bm}, {bn}, {wm}, {wn}, {bk}, {stages}, {epi}, false, {wf}, true>(w, s, b, xh, xl, out, oh, ol, tokens);\n}}\n\n",
                threads = t.threads(),
                bm = t.bm,
                bn = t.bn,
                wm = t.wm,
                wn = t.wn,
                bk = shape.bk,
                stages = shape.stages,
                wf = wf.code(),
            ));
        }
    }
    out
}
