use crate::spec::Spec;

pub const MTP_LINEARS: [&str; 5] = ["mtp_fc", "mtp_qkv", "mtp_o", "mtp_gate_up", "mtp_down"];

pub fn linears(spec: &Spec) -> Vec<&'static str> {
    let mut out = MTP_LINEARS.to_vec();
    if spec.streamed_mtp() {
        out.insert(1, "mtp_fc_hidden");
    }
    out
}

pub fn dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str) {
    let (n, k, epi, _) = crate::gemm::linear_dims(spec, linear);
    (n, k, epi)
}
