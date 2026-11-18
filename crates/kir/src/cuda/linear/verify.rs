use super::prim::{self, at, delta_update, gate, gated_output, load_state, memory_read, qk_norm, store_state, unroll, window_load, window_shift, window_store};

pub fn body() -> String {
    let mut o = String::new();
    o.push_str(
        "extern \"C\" __global__ void gdn_conv_verify(const float* lin, const float* w, const float* state, float* mix, float* log,
                                           int slot, int seq_slots, const int* slots, int rows) {\n",
    );
    o.push_str(&at(
        4,
        "int c = blockIdx.x * blockDim.x + threadIdx.x;
int seq = blockIdx.y;
if (c >= LA_CONV_DIM) return;
const float* s = conv_state_row((float*)state, slot + slots[seq * rows] * seq_slots, c);
float window[LA_K];
",
    ));
    o.push_str(&window_load(4));
    o.push_str(&at(
        4,
        "for (int r = 0; r < rows; r++) {
    size_t row = (size_t)seq * rows + r;
    float x = lin[row * LA_IN_ROWS + c];
    window[LA_K - 1] = x;
    mix[row * LA_CONV_DIM + c] = conv_tap(w + (size_t)c * LA_K, window);
    log[row * LA_CONV_DIM + c] = x;
",
    ));
    o.push_str(&window_shift(8));
    o.push_str("    }\n}\n\n");

    o.push_str(
        "extern \"C\" __global__ void gdn_conv_commit(float* state, const float* log, int seq_slots, const int* slots, int rows,
                                           const int* accepted, int log_rows) {\n",
    );
    o.push_str(&at(
        4,
        "int c = blockIdx.x * blockDim.x + threadIdx.x;
int seq = blockIdx.y;
int layer = blockIdx.z;
if (c >= LA_CONV_DIM) return;
float* s = conv_state_row(state, layer + slots[seq * rows] * seq_slots, c);
log += (size_t)layer * log_rows * LA_CONV_DIM;
float window[LA_K];
",
    ));
    o.push_str(&window_load(4));
    o.push_str(&at(
        4,
        "for (int r = 0; r <= accepted[seq]; r++) {
    window[LA_K - 1] = log[((size_t)seq * rows + r) * LA_CONV_DIM + c];
",
    ));
    o.push_str(&window_shift(8));
    o.push_str("    }\n");
    o.push_str(&window_store(4, "window[j]"));
    o.push_str("}\n\n");

    o.push_str(
        "extern \"C\" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_verify(const float* mix, const float* lin,
                                                                          const float* ba, const float* neg_a,
                                                                          const float* dt_bias, const float* norm_w,
                                                                          const float* state, float* out, float* klog,
                                                                          float* dlog, float* glog, int slot,
                                                                          int seq_slots, const int* slots, int rows) {\n",
    );
    o.push_str(prim::smem());
    o.push_str(&at(
        4,
        "int h = blockIdx.x;
int seq = blockIdx.y;
slot += slots[seq * rows] * seq_slots;
int kh = la_kh(h);
int tid = threadIdx.x;
int v = tid % LA_D;
int ks = tid / LA_D;
const float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
",
    ));
    o.push_str(&load_state(4));
    o.push_str(&at(
        4,
        "for (int r = 0; r < rows; r++) {
    size_t row = (size_t)seq * rows + r;
    const float* m = mix + row * LA_CONV_DIM;
    const float* l = lin + row * LA_IN_ROWS;
    const float* b = ba + row * LA_BA_ROWS;
",
    ));
    o.push_str(&qk_norm(8, "m"));
    o.push_str(&gate(8, "b"));
    o.push_str(&memory_read(8, "s[i]", "m[LA_V_OFF + h * LA_D + v]"));
    o.push_str(&delta_update(8, false));
    o.push_str(&at(
        8,
        "if (tid < LA_D) {
    size_t at = (row * LA_VH + h) * LA_D + tid;
    klog[at] = k_s[tid];
    dlog[at] = delta_s[tid];
}
if (tid == 0) glog[row * LA_VH + h] = decay;
__syncthreads();
",
    ));
    o.push_str(&gated_output(8, "out[row * LA_VALUE_DIM + h * LA_D + v]", "l[LA_Z_OFF + h * LA_D + v]"));
    o.push_str("    }\n}\n\n");

    o.push_str(
        "extern \"C\" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_commit(float* state, const float* klog, const float* dlog,
                                                                          const float* glog, int seq_slots,
                                                                          const int* slots, int rows,
                                                                          const int* accepted, int log_rows) {
    constexpr int KS = LA_D / LA_KSPLIT;\n",
    );
    o.push_str(&at(
        4,
        "int h = blockIdx.x;
int seq = blockIdx.y;
int layer = blockIdx.z;
int slot = layer + slots[seq * rows] * seq_slots;
int v = threadIdx.x % LA_D;
int ks = threadIdx.x / LA_D;
size_t base = (size_t)layer * log_rows;
float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
",
    ));
    o.push_str(&load_state(4));
    o.push_str(&at(
        4,
        "for (int r = 0; r <= accepted[seq]; r++) {
    size_t row = base + (size_t)seq * rows + r;
    const float* k_s = klog + (row * LA_VH + h) * LA_D;
    float decay = glog[row * LA_VH + h];
    float d = dlog[(row * LA_VH + h) * LA_D + v];
",
    ));
    o.push_str(&prim::unroll_block(8, "int i = 0; i < KS; i++", "int k = ks * KS + i;\ns[i] = __fmul_rn(s[i], decay);\ns[i] = __fmaf_rn(k_s[k], d, s[i]);\n"));
    o.push_str("    }\n");
    o.push_str(&store_state(4));
    o.push_str("}\n");
    let _ = unroll;
    o
}
