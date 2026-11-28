pub fn at(n: usize, s: &str) -> String {
    let pad = " ".repeat(n);
    let mut o = String::new();
    for line in s.split_inclusive('\n') {
        if line.trim().is_empty() || line.starts_with('#') {
            o.push_str(line);
        } else {
            o.push_str(&pad);
            o.push_str(line);
        }
    }
    o
}

pub fn unroll(n: usize, header: &str, body: &str) -> String {
    at(n, &format!("#pragma unroll\nfor ({header}) {body}\n"))
}

pub fn unroll_block(n: usize, header: &str, body: &str) -> String {
    at(n, &format!("#pragma unroll\nfor ({header}) {{\n{}}}\n", at(4, body)))
}

pub fn nest(n: usize, headers: &[&str], body: &str) -> String {
    let mut o = String::new();
    for (depth, h) in headers.iter().enumerate() {
        o.push_str("#pragma unroll\n");
        o.push_str(&at(n + 4 * depth, &format!("for ({h})")));
        if depth + 1 < headers.len() {
            o.push('\n');
        }
    }
    let inner = n + 4 * headers.len();
    if body.trim_end().contains('\n') {
        o.push_str(" {\n");
        o.push_str(&at(inner, body));
        o.push_str(&at(inner - 4, "}\n"));
    } else {
        o.push(' ');
        o.push_str(body.trim());
        o.push('\n');
    }
    o
}

pub const WINDOW_TAPS: &str = "int j = 0; j < LA_K - 1; j++";

pub fn window_load(n: usize) -> String {
    unroll(n, WINDOW_TAPS, "window[j] = s[j];")
}

pub fn window_shift(n: usize) -> String {
    unroll(n, WINDOW_TAPS, "window[j] = window[j + 1];")
}

pub fn window_store(n: usize, from: &str) -> String {
    unroll(n, WINDOW_TAPS, &format!("s[j] = {from};"))
}

pub fn history_gather(n: usize, dst: &str, count: &str, upto: &str) -> String {
    unroll_block(
        n,
        &format!("int j = 0; j < {count}; j++"),
        &format!("int src = {upto} - (LA_K - 1) + j;\n{dst}[j] = src >= 0 ? lin[(size_t)src * LA_IN_ROWS + c] : s[LA_K - 1 + src];\n"),
    )
}

pub fn smem() -> &'static str {
    "    constexpr int KS = LA_D / LA_KSPLIT;
    __shared__ float scratch[32];
    __shared__ float q_s[LA_D];
    __shared__ float k_s[LA_D];
    __shared__ float part[LA_KSPLIT][LA_D];
    __shared__ float delta_s[LA_D];
"
}

pub fn qk_norm(n: usize, m: &str) -> String {
    at(
        n,
        &format!(
            "float qv = tid < LA_D ? {m}[kh * LA_D + tid] : 0.0f;
float kv = tid < LA_D ? {m}[LA_KEY_DIM + kh * LA_D + tid] : 0.0f;
float qss = block_reduce<false>(qv * qv, scratch);
float kss = block_reduce<false>(kv * kv, scratch);
if (tid < LA_D) {{
    q_s[tid] = l2_q(qv, qss);
    k_s[tid] = l2_k(kv, kss);
}}
"
        ),
    )
}

pub fn gate(n: usize, ba: &str) -> String {
    at(n, &format!("float g, beta;\ngdn_gate({ba}, neg_a, dt_bias, h, g, beta);\nfloat decay = expf(g);\n__syncthreads();\n"))
}

pub fn load_state(n: usize) -> String {
    at(n, "float s[KS];\n") + &unroll(n, "int i = 0; i < KS; i++", "s[i] = S[(ks * KS + i) * LA_D + v];")
}

pub fn store_state(n: usize) -> String {
    unroll(n, "int i = 0; i < KS; i++", "S[(ks * KS + i) * LA_D + v] = s[i];")
}

pub fn memory_read(n: usize, decayed: &str, value: &str) -> String {
    let mut o = at(n, "float acc = 0.0f;\n");
    o += &unroll_block(
        n,
        "int i = 0; i < KS; i++",
        &format!("int k = ks * KS + i;\ns[i] = __fmul_rn({decayed}, decay);\nacc = __fmaf_rn(s[i], k_s[k], acc);\n"),
    );
    o += &at(
        n,
        "part[ks][v] = acc;
__syncthreads();
if (ks == 0) {
    float mem = 0.0f;
",
    );
    o += &unroll(n + 4, "int j = 0; j < LA_KSPLIT; j++", "mem += part[j][v];");
    o += &at(n, &format!("    delta_s[v] = ({value} - mem) * beta;\n}}\n__syncthreads();\n"));
    o
}

pub fn delta_update(n: usize, store: bool) -> String {
    let mut o = at(n, "float d = delta_s[v];\nacc = 0.0f;\n");
    let st = if store { "s[i] = __fmaf_rn(k_s[k], d, s[i]);\nS[k * LA_D + v] = s[i];\n" } else { "s[i] = __fmaf_rn(k_s[k], d, s[i]);\n" };
    o += &unroll_block(n, "int i = 0; i < KS; i++", &format!("int k = ks * KS + i;\n{st}acc = __fmaf_rn(s[i], q_s[k], acc);\n"));
    o
}

pub fn gated_output(n: usize, dst: &str, z: &str, gate: &str) -> String {
    let mut o = at(n, "part[ks][v] = acc;\n__syncthreads();\nfloat o = 0.0f;\nif (ks == 0) {\n");
    o += &unroll(n + 4, "int j = 0; j < LA_KSPLIT; j++", "o += part[j][v];");
    o += &at(
        n,
        &format!(
            "}}
float inv = rsqrtf(block_reduce<false>(o * o, scratch) / LA_D + EPS);
if (ks == 0) {dst} = norm_w[v] * (o * inv) * {gate}({z});
"
        ),
    );
    o
}

pub fn block_argmax_tree(n: usize) -> String {
    at(
        n,
        "best_v[threadIdx.x] = bv;
best_i[threadIdx.x] = bi;
__syncthreads();
for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) {
        float ov = best_v[threadIdx.x + w];
        int oi = best_i[threadIdx.x + w];
        if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
            best_v[threadIdx.x] = ov;
            best_i[threadIdx.x] = oi;
        }
    }
    __syncthreads();
}
",
    )
}

pub const FRAG: [&str; 3] = ["int j = 0; j < DPW; j++", "int n = 0; n < NT; n++", "int e = 0; e < 4; e++"];
pub const FRAG_NE: [&str; 2] = ["int n = 0; n < NT; n++", "int e = 0; e < 4; e++"];

pub const FRAG_ROW: &str = "int d = (warp + j * WARPS) * 16 + g + (e >> 1) * 8;";
pub const FRAG_COL: &str = "int col = n * 8 + 2 * t4 + (e & 1);";

pub fn state_tile_load(n: usize) -> String {
    nest(n, &FRAG, &format!("{FRAG_ROW}\nif (warp + j * WARPS >= KD) continue;\nsr[j][n][e] = st[(size_t)d * LA_D + c0 + n * 8 + 2 * t4 + (e & 1)];\n"))
}

pub fn state_tile_store(n: usize) -> String {
    nest(n, &FRAG, &format!("{FRAG_ROW}\nif (warp + j * WARPS >= KD) continue;\nst[(size_t)d * LA_D + c0 + n * 8 + 2 * t4 + (e & 1)] = sr[j][n][e];\n"))
}

pub fn split_plane_store(n: usize, hi: &str, lo: &str, pitch: &str, row: &str, col: &str, value: &str) -> String {
    at(
        n,
        &format!(
            "unsigned hi, lo;
split_hi_lo({value}, hi, lo);
{hi}[{col} * {pitch} + {row}] = (unsigned short)hi;
if (LA_LO) {lo}[{col} * {pitch} + {row}] = (unsigned short)lo;
"
        ),
    )
}

pub fn store_state_planes(n: usize) -> String {
    let mut o = at(n, "auto store_state_planes = [&]() {\n");
    let body = format!("{FRAG_ROW}\nif (warp + j * WARPS >= KD) continue;\n{FRAG_COL}\n{}", split_plane_store(0, "sh", "sl", "SP", "d", "col", "sr[j][n][e]"));
    o += &nest(n + 4, &FRAG, &body);
    o += &at(n, "};\n");
    o
}

pub fn acc_row() -> &'static str {
    "int m = warp * 16 + g + (e >> 1) * 8;\nint col = n * 8 + 2 * t4 + (e & 1);\n"
}

pub fn acc_zero(n: usize) -> String {
    nest(n, &FRAG_NE, "acc[n][e] = 0.0f;")
}

pub fn chunk_ptrs(n: usize) -> String {
    at(
        n,
        "int t0 = c * C;
size_t ch = (size_t)(c * LA_VH + h);
const uint4* wp = (const uint4*)w + ch * (C * LA_D / 4) + LA_FS * (warp * KD * 32 + lane);
const uint4* qp = (const uint4*)qe + ch * (C * LA_D / 4) + LA_FS * (warp * KD * 32 + lane);
const uint4* ap = (const uint4*)aqk + ch * (C * C / 4) + LA_FS * (warp * KC * 32 + lane);
const uint4* kp = (const uint4*)kt;
float decay = expf(gc[ch * C + C - 1]);
float acc[NT][4];
",
    )
}

pub fn mma_planes(n: usize, k: &str, acc: &str, frag: &str, hi: &str, lo: &str, pitch: &str) -> String {
    at(n, &format!("gdn_mma_planes<{k}, NT, LA_LO>({acc}, {frag}, {hi}, {lo}, {pitch}, g, t4);\n"))
}

pub fn kernel(header: &str, body: &str) -> String {
    format!("{header} {{\n{}}}\n", at(4, body))
}

pub fn for_stride(var: &str, init: &str, bound: &str, step: &str, body: &str) -> String {
    let head = format!("for (int {var} = {init}; {var} < {bound}; {var} += {step})");
    if body.trim_end().contains('\n') {
        format!("{head} {{\n{}}}\n", at(4, body))
    } else {
        format!("{head} {}\n", body.trim())
    }
}

pub fn warp_tiles(rows: &str, cols: &str, body: &str) -> String {
    let head = format!(
        "for (int tile = warp; tile < ({rows} / 16) * ({cols} / 8); tile += WARPS) {{
    int m0 = (tile / ({cols} / 8)) * 16;
    int n0 = (tile % ({cols} / 8)) * 8;
"
    );
    format!("{head}{}}}\n", at(4, body))
}

pub fn tri_leaf(n: usize) -> String {
    at(
        n,
        "constexpr int LEAF = 16;
for (int i = 1; i < LEAF; i++) {
    int b = (tid / LEAF) * LEAF;
    int j = tid % LEAF;
    float next = 0.0f;
    if (tid < C && j < i) {
        next = A[(b + i) * LA_AS + b + j];
        for (int m = j + 1; m < i; m++) next += A[(b + i) * LA_AS + b + m] * A[(b + m) * LA_AS + b + j];
    }
    __syncthreads();
    if (tid < C && j < i) A[(b + i) * LA_AS + b + j] = next;
    __syncthreads();
}
",
    )
}

fn tri_pass(zero: bool, body: &str) -> String {
    let init = if zero { "acc[p] = 0.0f;\n" } else { "" };
    unroll_block(
        0,
        "int p = 0; p < PER; p++",
        &format!(
            "int e = tid + p * 256;
{init}if (e < per) {{
    int blk = e / (sz * sz);
    int r = (e / sz) % sz;
    int c2 = e % sz;
    int base = blk * 2 * sz;
{}}}
",
            at(4, body)
        ),
    )
}

pub fn tri_merge(n: usize) -> String {
    let corner = "A[(base + sz + r) * LA_AS + base + c2]";
    let mut o = String::from("constexpr int PER = C * C / 4 / 256;\nint per = sz * sz * (C / (2 * sz));\nfloat acc[PER];\n");
    o += &tri_pass(
        true,
        &format!(
            "float y = {corner};
for (int m = c2 + 1; m < sz; m++)
    y += A[(base + sz + r) * LA_AS + base + m] * A[(base + m) * LA_AS + base + c2];
acc[p] = y;
"
        ),
    );
    o += "__syncthreads();\n";
    o += &tri_pass(false, &format!("{corner} = acc[p];\n"));
    o += "__syncthreads();\n";
    o += &tri_pass(
        false,
        &format!(
            "float x = {corner};
for (int m = 0; m < r; m++)
    x += A[(base + sz + r) * LA_AS + base + sz + m] * A[(base + sz + m) * LA_AS + base + c2];
acc[p] = x;
"
        ),
    );
    o += "__syncthreads();\n";
    o += &tri_pass(false, &format!("{corner} = acc[p];\n"));
    o += "__syncthreads();\n";
    at(n, &format!("for (int sz = LEAF; sz < C; sz *= 2) {{\n{}}}\n", at(4, &o)))
}

pub fn device_fn(template: &str, sig: &str, body: &str) -> String {
    let head = if template.is_empty() { String::new() } else { format!("template <{template}>\n") };
    format!("{head}__device__ __forceinline__ {sig} {{\n{}}}\n", at(4, body))
}
