use super::prim::{at, kernel, unroll_block};
use crate::recur::{ChunkForm, Recurrence, signature};
use crate::Kernel;

fn header(name: &str, k: &Kernel, f: &ChunkForm) -> String {
    format!("extern \"C\" __global__ void __launch_bounds__({}) {name}({}, float* ck_state, float* ck_decay)", f.threads, signature(&k.params))
}

fn constants(r: &Recurrence, f: &ChunkForm) -> String {
    format!(
        "constexpr int C = {};
constexpr int M = {};
constexpr int N = {};
constexpr int RK = {};
constexpr int RV = {};
constexpr int WARPS = {};
constexpr int THREADS = WARPS * 32;
constexpr int SUBS = C / M;
constexpr int BP = RK + 8;
constexpr int UP = N + 8;
constexpr int KS = RK % N == 0 ? N : 16;
constexpr int KSP = KS + 8;
constexpr int COLG = WARPS / (M / 16);
constexpr int NTO = RV / 8 / COLG;
constexpr int NTS = N / 8 / COLG;
constexpr int MTS = (RK / 16 + WARPS - 1) / WARPS;
static_assert(C % 32 == 0 && C % M == 0 && M % 16 == 0 && N % 16 == 0 && RK % 16 == 0, \"chunk tiling\");
static_assert(WARPS % (M / 16) == 0 && RV % (8 * COLG) == 0 && N % (8 * COLG) == 0, \"warp tiling\");
",
        f.rows,
        f.rows / f.subs,
        f.slice,
        r.keys,
        r.values,
        f.threads / 32
    )
}

const IDS: &str = "const int head = blockIdx.x;
const int seq = blockIdx.z;
const int ck_base = seq * rows_per_seq + ck_c * C;
const int ck_live = min(C, rows_per_seq - ck_c * C);
const size_t ck_at = (size_t)(seq * ck_nc + ck_c) * gridDim.x + head;
const int lane = threadIdx.x & 31;
const int warp = threadIdx.x >> 5;
const int ck_g = lane >> 2;
const int ck_t4 = lane & 3;
";

const CUMSUM: &str = "__syncthreads();
if (warp == 0) {
    constexpr int PL = C / 32;
    float ck_v[PL];
    float ck_run = 0.0f;
#pragma unroll
    for (int j = 0; j < PL; j++) {
        ck_run += ck_l[lane * PL + j];
        ck_v[j] = ck_run;
    }
    float ck_inc = ck_run;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float ck_up = __shfl_up_sync(0xffffffffu, ck_inc, o);
        if (lane >= o) ck_inc += ck_up;
    }
    const float ck_off = ck_inc - ck_run;
#pragma unroll
    for (int j = 0; j < PL; j++) ck_l[lane * PL + j] = ck_v[j] + ck_off;
}
__syncthreads();
";

fn strided(total: &str, body: &str) -> String {
    unroll_block(0, &format!("int ck_p = 0; ck_p < ({total} + THREADS - 1) / THREADS; ck_p++"), &format!("const int i = ck_p * THREADS + threadIdx.x;\nif (i < {total}) {{\n{}}}\n", at(4, body)))
}

fn row_fn(name: &str, args: &str, before: &str, value: &str) -> String {
    format!("auto {name} = [&]({args}) -> float {{\n    if (ck_t >= ck_live) return 0.0f;\n    const int row = ck_base + ck_t;\n{}    return {value};\n}};\n", at(4, before))
}

fn lambdas(r: &Recurrence, k: &Kernel) -> Result<String, String> {
    let t = r.chunk_text(&k.params)?;
    let (b, c) = (&t.update_key, &t.read_key);
    let mut o = row_fn("ck_decay_at", "int ck_t", &format!("const int col = 0;\n{}", t.before), &format!("logf(fmaxf({}, 1e-30f))", t.decay));
    o += &row_fn("ck_upd", "int ck_t, int col", &t.before, &format!("({})", t.update));
    o += &row_fn("ck_kb", "int ck_t, int d", "", &format!("{}[{}]", b.src, b.addr));
    o += &row_fn("ck_kc", "int ck_t, int d", "", &format!("{}[{}]", c.src, c.addr));
    Ok(o)
}

const LOG_DECAY: &str = "for (int i = threadIdx.x; i < C; i += THREADS) ck_l[i] = ck_decay_at(i);\n";

const PLANE: &str = "unsigned ck_hi, ck_lo;\nsplit_hi_lo(ck_x, ck_hi, ck_lo);\n";

fn plane_put(hi: &str, lo: &str, at: &str) -> String {
    format!("{PLANE}{hi}[{at}] = (unsigned short)ck_hi;\n{lo}[{at}] = (unsigned short)ck_lo;\n")
}

fn state_kernel(r: &Recurrence, k: &Kernel, f: &ChunkForm) -> Result<String, String> {
    let mut o = constants(r, f);
    o += "extern __shared__ __align__(16) unsigned char ck_smem[];
float* ck_l = (float*)ck_smem;
unsigned* ck_a = (unsigned*)(ck_l + C);
unsigned short* ck_uh = (unsigned short*)(ck_a + RK * N);
unsigned short* ck_ul = ck_uh + RV * UP;
const int ck_c = blockIdx.y;
const int ck_nc = gridDim.y;
";
    o += IDS;
    o += &lambdas(r, k)?;
    o += LOG_DECAY;
    o += CUMSUM;
    o += "const float ck_last = ck_l[C - 1];\nfloat ck_acc[MTS][RV / 8][4] = {};\n";
    let mut slice = strided("RK * N / 2", "const int d = i / (N / 2);\nconst int s = 2 * (i % (N / 2));\ngdn_frag_put<N / 16>(ck_a, d, s, ck_kb(ck_s0 + s, d), ck_kb(ck_s0 + s + 1, d));\n");
    slice += &strided("N * RV", &format!("const int s = i / RV;\nconst int v = i % RV;\nconst float ck_x = ck_upd(ck_s0 + s, v) * expf(ck_last - ck_l[min(ck_s0 + s, C - 1)]);\n{}", plane_put("ck_uh", "ck_ul", "v * UP + s")));
    slice += "__syncthreads();\n";
    slice += &unroll_block(0, "int j = 0; j < MTS; j++", "const int mt = warp + j * WARPS;\nif (mt < RK / 16) gdn_mma_planes<N / 16, RV / 8>(ck_acc[j], (const uint4*)ck_a + 2 * (mt * (N / 16) * 32 + lane), ck_uh, ck_ul, UP, ck_g, ck_t4);\n");
    slice += "__syncthreads();\n";
    o += &format!("for (int ck_s0 = 0; ck_s0 < ck_live; ck_s0 += N) {{\n{}}}\n", at(4, &slice));
    o += "if (threadIdx.x == 0) ck_decay[ck_at] = expf(ck_last);\nfloat* ck_out = ck_state + ck_at * (RK * RV);\n";
    o += &unroll_block(
        0,
        "int j = 0; j < MTS; j++",
        "const int mt = warp + j * WARPS;\nif (mt < RK / 16) {\n#pragma unroll\n    for (int n = 0; n < RV / 8; n++)\n#pragma unroll\n        for (int e = 0; e < 4; e++) ck_out[(mt * 16 + ck_g + (e >> 1) * 8) * RV + n * 8 + 2 * ck_t4 + (e & 1)] = ck_acc[j][n][e];\n}\n",
    );
    Ok(kernel(&header(f.state, k, f), &o))
}

fn carry_kernel(r: &Recurrence, k: &Kernel, f: &ChunkForm) -> String {
    let mut o = constants(r, f);
    o += &format!(
        "const int head = blockIdx.x;
const int seq = blockIdx.y;
const int i = blockIdx.z * blockDim.x + threadIdx.x;
const int ck_nc = (rows_per_seq + C - 1) / C;
const int slot = slot_step ? slots[seq] : slot_base;
float* st = {} + ((size_t)slot * {} + layer) * ((size_t)gridDim.x * RK * RV) + (size_t)head * RK * RV;
if (i < RK * RV) {{
    float ck_s = st[i];
    for (int ck_c0 = 0; ck_c0 < ck_nc; ck_c0 += 8) {{
        float ck_d[8], ck_a[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {{
            const size_t ck_at = (size_t)(seq * ck_nc + ck_c0 + j) * gridDim.x + head;
            ck_d[j] = ck_c0 + j < ck_nc ? ck_state[ck_at * (RK * RV) + i] : 0.0f;
            ck_a[j] = ck_c0 + j < ck_nc ? ck_decay[ck_at] : 1.0f;
        }}
#pragma unroll
        for (int j = 0; j < 8; j++) {{
            if (ck_c0 + j < ck_nc) ck_state[((size_t)(seq * ck_nc + ck_c0 + j) * gridDim.x + head) * (RK * RV) + i] = ck_s;
            ck_s = fmaf(ck_a[j], ck_s, ck_d[j]);
        }}
    }}
    st[i] = ck_s;
}}
",
        r.state, r.layers
    );
    kernel(&header(f.carry, k, f), &o)
}

fn out_kernel(r: &Recurrence, k: &Kernel, f: &ChunkForm) -> Result<String, String> {
    let t = r.chunk_text(&k.params)?;
    let mut o = constants(r, f);
    o += "extern __shared__ __align__(16) unsigned char ck_smem[];
float* ck_l = (float*)ck_smem;
unsigned* ck_q = (unsigned*)(ck_l + C);
unsigned* ck_sc = ck_q + M * RK;
unsigned short* ck_bh = (unsigned short*)(ck_sc + M * N);
unsigned short* ck_bl = ck_bh + N * BP;
unsigned short* ck_uh = ck_bl + N * BP;
unsigned short* ck_ul = ck_uh + RV * UP;
unsigned short* ck_sh = ck_bh;
unsigned short* ck_sl = ck_sh + RV * KSP;
const int ck_c = blockIdx.y / SUBS;
const int ck_m0 = (blockIdx.y % SUBS) * M;
const int ck_nc = gridDim.y / SUBS;
";
    o += IDS;
    o += "if (ck_m0 >= ck_live) return;\n";
    o += &lambdas(r, k)?;
    o += LOG_DECAY;
    o += CUMSUM;
    o += &strided("M * RK / 2", "const int m = i / (RK / 2);\nconst int d = 2 * (i % (RK / 2));\ngdn_frag_put<RK / 16>(ck_q, m, d, ck_kc(ck_m0 + m, d), ck_kc(ck_m0 + m, d + 1));\n");
    o += "const int mt = warp % (M / 16);
const int cg = warp / (M / 16);
const uint4* ck_qa = (const uint4*)ck_q + 2 * (mt * (RK / 16) * 32 + lane);
const uint4* ck_sa = (const uint4*)ck_sc + 2 * (mt * (N / 16) * 32 + lane);
float ck_acc[NTO][4] = {};
float ck_car[NTO][4] = {};
const int ck_end = min(ck_m0 + M, ck_live);
";
    let mut slice = strided("N * RK", &format!("const int s = i / RK;\nconst int d = i % RK;\nconst float ck_x = ck_kb(ck_s0 + s, d);\n{}", plane_put("ck_bh", "ck_bl", "s * BP + d")));
    slice += &strided("N * RV", &format!("const int s = i / RV;\nconst int v = i % RV;\nconst float ck_x = ck_upd(ck_s0 + s, v);\n{}", plane_put("ck_uh", "ck_ul", "v * UP + s")));
    slice += "__syncthreads();
float ck_x[NTS][4] = {};
gdn_mma_planes<RK / 16, NTS>(ck_x, ck_qa, ck_bh + cg * NTS * 8 * BP, ck_bl + cg * NTS * 8 * BP, BP, ck_g, ck_t4);
";
    slice += &unroll_block(
        0,
        "int n = 0; n < NTS; n++",
        "#pragma unroll\nfor (int e = 0; e < 4; e += 2) {\n    const int m = mt * 16 + ck_g + (e >> 1) * 8;\n    const int s = (cg * NTS + n) * 8 + 2 * ck_t4;\n    const int tm = ck_m0 + m;\n    const int ts = ck_s0 + s;\n    const float v0 = ts <= tm ? ck_x[n][e] * expf(ck_l[tm] - ck_l[min(ts, C - 1)]) : 0.0f;\n    const float v1 = ts + 1 <= tm ? ck_x[n][e + 1] * expf(ck_l[tm] - ck_l[min(ts + 1, C - 1)]) : 0.0f;\n    gdn_frag_put<N / 16>(ck_sc, m, s, v0, v1);\n}\n",
    );
    slice += "__syncthreads();
gdn_mma_planes<N / 16, NTO>(ck_acc, ck_sa, ck_uh + cg * NTO * 8 * UP, ck_ul + cg * NTO * 8 * UP, UP, ck_g, ck_t4);
__syncthreads();
";
    o += &format!("for (int ck_s0 = 0; ck_s0 < ck_end; ck_s0 += N) {{\n{}}}\n", at(4, &slice));
    o += "const float* ck_in = ck_state + ck_at * (RK * RV);\n";
    let mut carry = strided("KS * RV", &format!("const int d = i / RV;\nconst int v = i % RV;\nconst float ck_x = ck_in[(size_t)(ck_k0 + d) * RV + v];\n{}", plane_put("ck_sh", "ck_sl", "v * KSP + d")));
    carry += "__syncthreads();\ngdn_mma_planes<KS / 16, NTO>(ck_car, ck_qa + 4 * ck_k0, ck_sh + cg * NTO * 8 * KSP, ck_sl + cg * NTO * 8 * KSP, KSP, ck_g, ck_t4);\n__syncthreads();\n";
    o += &format!("for (int ck_k0 = 0; ck_k0 < RK; ck_k0 += KS) {{\n{}}}\n", at(4, &carry));
    let finish = format!(
        "const int tm = ck_m0 + mt * 16 + ck_g + (e >> 1) * 8;\nif (tm < ck_live) {{\n{}}}\n",
        at(4, &format!("const int row = ck_base + tm;\nconst int col = (cg * NTO + n) * 8 + 2 * ck_t4 + (e & 1);\nconst float ck_o = ck_acc[n][e] + expf(ck_l[tm]) * ck_car[n][e];\n{}", t.finish))
    );
    o += &unroll_block(0, "int n = 0; n < NTO; n++", &unroll_block(0, "int e = 0; e < 4; e++", &finish));
    Ok(kernel(&header(f.out, k, f), &o))
}

pub fn emit(r: &Recurrence, k: &Kernel, f: &ChunkForm) -> Result<String, String> {
    Ok(format!("{}\n{}\n{}\n", state_kernel(r, k, f)?, carry_kernel(r, k, f), out_kernel(r, k, f)?))
}
