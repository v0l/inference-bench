use super::prim::{self, at, delta_update, gate, gated_output, load_state, memory_read, qk_norm};

pub fn body() -> String {
    let mut o = String::new();
    o.push_str(
        "extern \"C\" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_step(const float* mix, const float* lin,
                                                                        const float* ba,
                                                                        const float* neg_a, const float* dt_bias,
                                                                        const float* norm_w, float* state,
                                                                        float* out, int slot, int seq_slots,
                                                                        const int* slots) {\n",
    );
    o.push_str(prim::smem());
    o.push_str(&at(
        4,
        "int h = blockIdx.x;
int seq = blockIdx.y;
mix += (size_t)seq * LA_CONV_DIM;
lin += (size_t)seq * LA_IN_ROWS;
ba += (size_t)seq * LA_BA_ROWS;
out += (size_t)seq * LA_VALUE_DIM;
slot += slots[seq] * seq_slots;
int kh = la_kh(h);
int tid = threadIdx.x;
int v = tid % LA_D;
int ks = tid / LA_D;
",
    ));
    o.push_str(&qk_norm(4, "mix"));
    o.push_str(&gate(4, "ba"));
    o.push_str(&at(4, "float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;\nfloat s[KS];\n"));
    o.push_str(&memory_read(4, "S[k * LA_D + v]", "mix[LA_V_OFF + h * LA_D + v]"));
    o.push_str(&delta_update(4, true));
    o.push_str(&at(4, "__syncthreads();\n"));
    o.push_str(&gated_output(4, "out[h * LA_D + v]", "lin[LA_Z_OFF + h * LA_D + v]"));
    o.push_str("}\n\n");
    o.push_str(
        "extern \"C\" __global__ void gdn_qk_prep(const float* mix, float* qn, float* kn) {
    __shared__ float scratch[32];
    int kh = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* row = mix + (size_t)t * LA_CONV_DIM;
    float q = row[kh * LA_D + d];
    float k = row[LA_KEY_DIM + kh * LA_D + d];
    float qss = block_reduce<false>(q * q, scratch);
    float kss = block_reduce<false>(k * k, scratch);
    size_t at = ((size_t)t * LA_KH + kh) * LA_D + d;
    qn[at] = l2_q(q, qss);
    kn[at] = l2_k(k, kss);
}

extern \"C\" __global__ void gdn_gates(const float* ba, const float* neg_a, const float* dt_bias, float* g, float* beta) {
    int t = blockIdx.x;
    for (int h = threadIdx.x; h < LA_VH; h += blockDim.x) {
        size_t at = (size_t)t * LA_VH + h;
        gdn_gate(ba + (size_t)t * LA_BA_ROWS, neg_a, dt_bias, h, g[at], beta[at]);
    }
}
",
    );
    o
}
