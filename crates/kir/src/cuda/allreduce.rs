use crate::allreduce::{AllReduceError, AllReduceKernel, Dtype, Mode};
use crate::lit;

#[path = "allreduce/prim.rs"]
mod prim;

const SPIN_LIMIT: u64 = 20_000_000_000;

fn store(d: Dtype, v: &str) -> String {
    match d {
        Dtype::F32 => v.to_string(),
        Dtype::Bf16 => format!("ar_to_bf16({v})"),
    }
}

fn accumulate(d: Dtype, at: &str, v: &str) -> String {
    match d {
        Dtype::F32 => format!("out[{at}] = add ? out[{at}] + {v} : {v};"),
        Dtype::Bf16 => format!("out[{at}] = {};", store(d, &format!("(add ? ar_ld(&out[{at}]) + {v} : {v})"))),
    }
}

fn header(threads: u32, name: &str, args: &str) -> String {
    format!("extern \"C\" __global__ void __launch_bounds__({threads}) {name}({args}) {{\n")
}

const PLAIN_ARGS: &str = "int rank, int n, int add, int clear";

fn plain_header(k: &AllReduceKernel) -> String {
    let t = k.dtype.ctype();
    let mut s = header(k.threads, &k.name(), &format!("{t}* in, {t}* out, const unsigned long long* table, {PLAIN_ARGS}"));
    s += &format!("    __shared__ unsigned epoch;\n    ar_begin(table, rank, &epoch);\n    {t}* mine = ({t}*)table[rank];\n");
    s
}

const STRIDE: &str = "    size_t stride = (size_t)gridDim.x * blockDim.x;\n    size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n";

fn one_shot(k: &AllReduceKernel) -> String {
    let t = k.dtype.ctype();
    let mut s = plain_header(k);
    s += STRIDE;
    s += &prim::grid_stride(4, "first", "(size_t)n", &prim::stage(8));
    s += &prim::sync_point(1);
    s += &prim::grid_stride(4, "first", "(size_t)n", &format!("{}        {}\n", prim::rank_sum(8, t, "i"), accumulate(k.dtype, "i", "acc")));
    s += &prim::finish(2);
    s
}

fn per_chunk(n: usize, c: &str, body: &str) -> String {
    let p = prim::pad(n);
    format!("{p}for (int c = 0; c < AR_RANKS; c++) {{\n{}{body}{p}}}\n", prim::chunk_bounds(n + 4, c))
}

fn two_shot(k: &AllReduceKernel) -> String {
    let t = k.dtype.ctype();
    let mut s = plain_header(k);
    s += "    size_t chunk = ((size_t)n + AR_RANKS - 1) / AR_RANKS;\n";
    s += STRIDE;
    s += &per_chunk(4, "c", &prim::grid_stride(8, "lo + first", "hi", &prim::stage(12)));
    s += &prim::sync_point(1);
    s += &format!(
        "    {{\n{}{}    }}\n",
        prim::chunk_bounds(8, "rank"),
        prim::grid_stride(8, "lo + first", "hi", &format!("{}            mine[i] = {};\n", prim::rank_sum(12, t, "i"), store(k.dtype, "acc")))
    );
    s += &prim::sync_point(2);
    s += &per_chunk(
        4,
        "c",
        &format!("        const {t}* src = (const {t}*)table[c];\n{}", prim::grid_stride(8, "lo + first", "hi", &format!("            float v = ar_ld(&src[i]);\n            {}\n", accumulate(k.dtype, "i", "v")))),
    );
    s += &prim::finish(3);
    s
}

fn rows_loop(n: usize, body: &str) -> String {
    let p = prim::pad(n);
    format!("{p}for (int r = blockIdx.x; r < rows; r += gridDim.x) {{\n{body}{p}}}\n")
}

fn quads(n: usize, body: &str) -> String {
    let p = prim::pad(n);
    format!("{p}for (int i = threadIdx.x; i < WIDTH / 4; i += blockDim.x) {{\n{body}{p}}}\n")
}

fn peer_loads(threads: u32) -> String {
    format!(
        "        constexpr int QUADS = (WIDTH / 4 + {threads} - 1) / {threads};\n        float4 pv[QUADS][AR_RANKS];\n#pragma unroll\n        for (int q = 0; q < QUADS; q++) {{\n            int i = threadIdx.x + q * {threads};\n            if (i < WIDTH / 4) {{\n#pragma unroll\n                for (int p = 0; p < AR_RANKS; p++) pv[q][p] = __ldcg((const float4*)((const float*)table[p] + base) + i);\n            }}\n        }}\n"
    )
}

fn residual_sum(threads: u32) -> String {
    format!(
        "#pragma unroll\n        for (int q = 0; q < QUADS; q++) {{\n            int i = threadIdx.x + q * {threads};\n            if (i < WIDTH / 4) {{\n                float4 v = ((const float4*)(x + base))[i];\n                float s[4] = {{v.x, v.y, v.z, v.w}};\n                float acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};\n#pragma unroll\n                for (int p = 0; p < AR_RANKS; p++) {{\n                    acc[0] += pv[q][p].x;\n                    acc[1] += pv[q][p].y;\n                    acc[2] += pv[q][p].z;\n                    acc[3] += pv[q][p].w;\n                }}\n#pragma unroll\n                for (int j = 0; j < 4; j++) s[j] = s[j] + acc[j];\n                ss += s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];\n                ((float4*)(x + base))[i] = make_float4(s[0], s[1], s[2], s[3]);\n            }}\n        }}\n"
    )
}

fn norm_planes() -> &'static str {
    "            float4 v = ((const float4*)(x + base))[i];\n            float4 w = ((const float4*)weight)[i];\n            ar_split2(hi, lo, base + 4 * i, w.x * (v.x * inv), w.y * (v.y * inv));\n            ar_split2(hi, lo, base + 4 * i + 2, w.z * (v.z * inv), w.w * (v.w * inv));\n"
}

fn add_rmsnorm(k: &AllReduceKernel, width: u32, eps: f32) -> String {
    let mut s = header(k.threads, &k.name(), "float* in, float* x, const float* weight, bf16_t* hi, bf16_t* lo, const unsigned long long* table, int rank, int rows");
    s += &format!("    constexpr int WIDTH = {width};\n    __shared__ unsigned epoch;\n    __shared__ float scratch[32];\n    ar_begin(table, rank, &epoch);\n    float* mine = (float*)table[rank];\n");
    s += "    for (int r = blockIdx.x; r < rows; r += gridDim.x)\n        for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {\n            mine[(size_t)r * WIDTH + i] = in[(size_t)r * WIDTH + i];\n            in[(size_t)r * WIDTH + i] = 0.0f;\n        }\n";
    s += &prim::sync_point(1);
    let body = format!(
        "        size_t base = (size_t)r * WIDTH;\n        float ss = 0.0f;\n{}{}        float inv = rsqrtf(ar_block_sum(ss, scratch) / WIDTH + {}f);\n{}        __syncthreads();\n",
        peer_loads(k.threads),
        residual_sum(k.threads),
        lit(eps),
        quads(8, norm_planes())
    );
    s += &rows_loop(4, &body);
    s += &prim::finish(2);
    s
}

pub const PROBE_WAIT: &str = "ar_probe_wait";
pub const PROBE_STORE: &str = "ar_probe_store";

pub fn emit_peer_probe() -> String {
    let mut s = String::from("typedef unsigned short bf16_t;\n\n");
    s += &prim::flag_ops();
    s += &prim::global_timer();
    s += &header(1, PROBE_WAIT, "const unsigned* flag, unsigned magic, unsigned long long timeout_ns, unsigned* result");
    s += "    unsigned long long start = ar_now_ns();\n    while (ar_load_acquire(flag) != magic) {\n        if (ar_now_ns() - start > timeout_ns) {\n            *result = 2;\n            return;\n        }\n    }\n    *result = 1;\n}\n\n";
    s += &header(1, PROBE_STORE, "unsigned* flag, unsigned magic");
    s += "    __threadfence_system();\n    ar_store_release(flag, magic);\n}\n\n";
    s
}

pub fn emit_allreduce(kernels: &[AllReduceKernel]) -> Result<String, AllReduceError> {
    let Some(first) = kernels.first() else { return Ok(String::new()) };
    for k in kernels {
        k.validate()?;
        if k.ranks != first.ranks {
            return Err(AllReduceError(format!("{}: kernels of one module must share a rank count", k.name())));
        }
    }
    let mut out = format!("constexpr int AR_RANKS = {};\n", first.ranks);
    out.push_str(&prim::library(crate::allreduce::MAX_BLOCKS, SPIN_LIMIT));
    for k in kernels {
        out.push_str(&match k.mode {
            Mode::OneShot => one_shot(k),
            Mode::TwoShot => two_shot(k),
            Mode::AddRmsNorm { width, eps } => add_rmsnorm(k, width, eps),
        });
    }
    Ok(out)
}
