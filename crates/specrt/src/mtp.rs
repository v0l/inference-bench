use crate::spec::Spec;

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

pub fn dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str) {
    let (n, k, epi, _) = crate::gemm::linear_dims(spec, linear);
    (n, k, epi)
}
