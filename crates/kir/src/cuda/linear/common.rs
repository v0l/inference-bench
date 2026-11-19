use super::prim::{at, device_fn, unroll};

pub fn common() -> String {
    let parts = [
        device_fn("", "float silu(float x)", "return x / (1.0f + expf(-x));\n"),
        device_fn("", "float softplus(float x)", "return x > 20.0f ? x : log1pf(expf(x));\n"),
        device_fn(
            "",
            "void gdn_gate(const float* ba_row, const float* neg_a, const float* dt_bias, int h,\n                                         float& g, float& beta)",
            "g = neg_a[h] * softplus(ba_row[LA_A_OFF + h] + dt_bias[h]);\nbeta = 1.0f / (1.0f + expf(-ba_row[LA_B_OFF + h]));\n",
        ),
        device_fn(
            "",
            "float conv_tap(const float* w, const float* window)",
            &format!(
                "float acc = 0.0f;\n{}return silu(acc);\n",
                at(0, &unroll(0, "int j = 0; j < LA_K; j++", "acc += w[j] * window[j];"))
            ),
        ),
        device_fn("", "float l2_q(float v, float ss)", "return v * rsqrtf(ss + LA_L2_EPS) * rsqrtf((float)LA_D);\n"),
        device_fn("", "float l2_k(float v, float ss)", "return v * rsqrtf(ss + LA_L2_EPS);\n"),
        device_fn(
            "",
            "float* conv_state_row(float* state, int slot, int c)",
            "return state + ((size_t)slot * LA_CONV_DIM + c) * (LA_K - 1);\n",
        ),
    ];
    parts.join("\n")
}
