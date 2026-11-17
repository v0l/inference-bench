use crate::spec::Spec;

pub const MTP_LINEARS: [&str; 5] = ["mtp_fc", "mtp_qkv", "mtp_o", "mtp_gate_up", "mtp_down"];

pub fn dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str) {
    let (n, k, epi, _) = crate::gemm::linear_dims(spec, linear);
    (n, k, epi)
}
