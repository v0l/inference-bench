use super::prim::{at, history_gather, window_load, window_shift, window_store};

pub fn body() -> String {
    let mut o = String::new();
    o.push_str(
        "extern \"C\" __global__ void gdn_conv_step(const float* lin, const float* w, float* state, float* mix, int slot, int seq_slots,
                                         const int* slots) {\n",
    );
    o.push_str(&at(
        4,
        "int c = blockIdx.x * blockDim.x + threadIdx.x;
int seq = blockIdx.y;
if (c >= LA_CONV_DIM) return;
lin += (size_t)seq * LA_IN_ROWS;
mix += (size_t)seq * LA_CONV_DIM;
float* s = conv_state_row(state, slot + slots[seq] * seq_slots, c);
float window[LA_K];
",
    ));
    o.push_str(&window_load(4));
    o.push_str(&at(4, "window[LA_K - 1] = lin[c];\n"));
    o.push_str(&window_store(4, "window[j + 1]"));
    o.push_str(&at(4, "mix[c] = conv_tap(w + (size_t)c * LA_K, window);\n"));
    o.push_str("}\n\n");

    o.push_str("extern \"C\" __global__ void gdn_conv_prefill(const float* lin, const float* w, const float* state, float* mix, int slot) {\n");
    o.push_str(&at(
        4,
        "int c = blockIdx.x * blockDim.x + threadIdx.x;
int t = blockIdx.y;
if (c >= LA_CONV_DIM) return;
const float* s = conv_state_row((float*)state, slot, c);
float window[LA_K];
",
    ));
    o.push_str(&history_gather(4, "window", "LA_K", "t"));
    o.push_str(&at(4, "mix[(size_t)t * LA_CONV_DIM + c] = conv_tap(w + (size_t)c * LA_K, window);\n"));
    o.push_str("}\n\n");

    o.push_str("extern \"C\" __global__ void gdn_conv_keep(const float* lin, float* state, int slot, int tokens) {\n");
    o.push_str(&at(
        4,
        "int c = blockIdx.x * blockDim.x + threadIdx.x;
if (c >= LA_CONV_DIM) return;
float* s = conv_state_row(state, slot, c);
float keep[LA_K - 1];
",
    ));
    o.push_str(&history_gather(4, "keep", "LA_K - 1", "tokens"));
    o.push_str(&window_store(4, "keep[j]"));
    o.push_str("}\n");
    let _ = window_shift;
    o
}
