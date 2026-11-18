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

pub fn conv_thread(n: usize, z: Option<&str>) -> String {
    let mut s = String::from("int c = blockIdx.x * blockDim.x + threadIdx.x;\n");
    match z {
        Some(axis) => s.push_str(&format!("int {axis};\n")),
        None => {}
    }
    at(n, &s)
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
        &format!(
            "part[ks][v] = acc;
__syncthreads();
if (ks == 0) {{
    float mem = 0.0f;
"
        ),
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

pub fn gated_output(n: usize, dst: &str, z: &str) -> String {
    let mut o = at(n, "part[ks][v] = acc;\n__syncthreads();\nfloat o = 0.0f;\nif (ks == 0) {\n");
    o += &unroll(n + 4, "int j = 0; j < LA_KSPLIT; j++", "o += part[j][v];");
    o += &at(
        n,
        &format!(
            "}}
float inv = rsqrtf(block_reduce<false>(o * o, scratch) / LA_D + EPS);
if (ks == 0) {dst} = norm_w[v] * (o * inv) * silu({z});
"
        ),
    );
    o
}

pub fn lane_ids(n: usize, g: &str, t: &str) -> String {
    at(n, &format!("int {g} = lane >> 2;\nint {t} = lane & 3;\n"))
}

pub fn zero_acc(n: usize, headers: &[&str], target: &str) -> String {
    nest(n, headers, &format!("{target} = 0.0f;"))
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
