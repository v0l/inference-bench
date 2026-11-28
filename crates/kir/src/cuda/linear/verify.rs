use super::prim::{self, at, delta_update, gated_output, load_state, memory_read, store_state, unroll, window_load, window_shift, window_store};

pub fn body(out_gate: &str) -> String {
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
                                                                          float* state, float* out, float* klog,
                                                                          float* dlog, float* glog, const int* pend,
                                                                          int slot, int seq_slots, const int* slots,
                                                                          int rows, int log_rows) {\n",
    );
    o.push_str(&verify_smem());
    o.push_str(&at(
        4,
        "int h = blockIdx.x;
int seq = blockIdx.y;
int owner = slots[seq * rows];
slot += owner * seq_slots;
int kh = la_kh(h);
int tid = threadIdx.x;
int v = tid % LA_D;
int ks = tid / LA_D;
float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
",
    ));
    o.push_str(&load_state(4));
    o.push_str(&hoist(4, "0"));
    o.push_str(&at(4, "int pending = pend[owner];\n"));
    o.push_str(&replay(4, "pending"));
    o.push_str(&at(4, "if (pending > 0) {\n"));
    o.push_str(&store_state(8));
    o.push_str(&at(4, "}\n"));
    o.push_str(&at(4, "for (int r0 = 0; r0 < rows; r0 += VR) {\n"));
    o.push_str(&at(4, "if (r0 > 0) {\n"));
    o.push_str(&hoist(8, "r0"));
    o.push_str(&at(4, "}\n"));
    o.push_str(&at(
        4,
        "for (int r = r0; r < rows && r < r0 + VR; r++) {
    size_t row = (size_t)seq * rows + r;
    size_t logged = (size_t)slot * log_rows + r;
    const float* l = lin + row * LA_IN_ROWS;
    const float* m = mix + row * LA_CONV_DIM;
    const float* q_s = q_rows[r - r0];
    const float* k_s = k_rows[r - r0];
    float decay = decay_rows[r - r0];
    float beta = beta_rows[r - r0];
",
    ));
    o.push_str(&memory_read(8, "s[i]", "m[LA_V_OFF + h * LA_D + v]"));
    o.push_str(&delta_update(8, false));
    o.push_str(&at(
        8,
        "if (tid < LA_D) {
    size_t at = (logged * LA_VH + h) * LA_D + tid;
    klog[at] = k_s[tid];
    dlog[at] = delta_s[tid];
}
if (tid == 0) glog[logged * LA_VH + h] = decay;
__syncthreads();
",
    ));
    o.push_str(&gated_output(8, "out[row * LA_VALUE_DIM + h * LA_D + v]", "l[LA_Z_OFF + h * LA_D + v]", out_gate));
    o.push_str("    }\n    }\n}\n\n");

    o.push_str(
        "extern \"C\" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_commit(float* state, const float* klog, const float* dlog,
                                                                          const float* glog, const int* pend,
                                                                          int seq_slots, const int* slots, int rows,
                                                                          int log_rows) {
    constexpr int KS = LA_D / LA_KSPLIT;\n",
    );
    o.push_str(&at(
        4,
        "int h = blockIdx.x;
int seq = blockIdx.y;
int layer = blockIdx.z;
int owner = slots[seq * rows];
int pending = pend[owner];
if (pending == 0) return;
int slot = layer + owner * seq_slots;
int v = threadIdx.x % LA_D;
int ks = threadIdx.x / LA_D;
float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
",
    ));
    o.push_str(&load_state(4));
    o.push_str(&replay(4, "pending"));
    o.push_str(&store_state(4));
    o.push_str("}\n\n");
    o.push_str(
        "extern \"C\" __global__ void gdn_defer(int* pend, const int* slots, int rows, const int* accepted, int n, int clear) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pend[slots[i * rows]] = clear ? 0 : accepted[i] + 1;
}
",
    );
    let _ = unroll;
    o
}

fn verify_smem() -> String {
    "    constexpr int KS = LA_D / LA_KSPLIT;
    constexpr int VR = 4;
    __shared__ float scratch[32];
    __shared__ float q_rows[VR][LA_D];
    __shared__ float k_rows[VR][LA_D];
    __shared__ float decay_rows[VR];
    __shared__ float beta_rows[VR];
    __shared__ float part[LA_KSPLIT][LA_D];
    __shared__ float delta_s[LA_D];
"
    .to_string()
}

fn hoist(n: usize, from: &str) -> String {
    at(
        n,
        &format!(
            "for (int r = {from}; r < rows && r < {from} + VR; r++) {{
    size_t row = (size_t)seq * rows + r;
    const float* m = mix + row * LA_CONV_DIM;
    float qv = tid < LA_D ? m[kh * LA_D + tid] : 0.0f;
    float kv = tid < LA_D ? m[LA_KEY_DIM + kh * LA_D + tid] : 0.0f;
    float qss = block_reduce<false>(qv * qv, scratch);
    float kss = block_reduce<false>(kv * kv, scratch);
    if (tid < LA_D) {{
        q_rows[r - {from}][tid] = l2_q(qv, qss);
        k_rows[r - {from}][tid] = l2_k(kv, kss);
    }}
    if (tid == 0) {{
        float g, beta;
        gdn_gate(ba + row * LA_BA_ROWS, neg_a, dt_bias, h, g, beta);
        decay_rows[r - {from}] = expf(g);
        beta_rows[r - {from}] = beta;
    }}
}}
__syncthreads();
"
        ),
    )
}

fn replay(n: usize, count: &str) -> String {
    let mut o = at(
        n,
        &format!(
            "#pragma unroll 1
for (int r = 0; r < {count}; r++) {{
    size_t row = (size_t)slot * log_rows + r;
    const float* kr = klog + (row * LA_VH + h) * LA_D;
    float decay = glog[row * LA_VH + h];
    float d = dlog[(row * LA_VH + h) * LA_D + v];
"
        ),
    );
    o.push_str(&prim::unroll_block(n + 4, "int i = 0; i < KS; i++", "int k = ks * KS + i;\ns[i] = __fmul_rn(s[i], decay);\ns[i] = __fmaf_rn(kr[k], d, s[i]);\n"));
    o.push_str(&at(n, "}\n"));
    o
}
