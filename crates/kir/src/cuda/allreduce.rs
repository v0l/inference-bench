use crate::allreduce::{AllReduceError, AllReduceKernel, Dtype, Mode};
use crate::lit;

const COMMON: &str = include_str!("allreduce/common.cu");

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

fn one_shot(k: &AllReduceKernel) -> String {
    let t = k.dtype.ctype();
    format!(
        "extern \"C\" __global__ void __launch_bounds__({threads}) {name}({t}* in, {t}* out, const unsigned long long* table, int rank, int n, int add, int clear) {{
    __shared__ unsigned epoch;
    ar_begin(table, rank, &epoch);
    {t}* mine = ({t}*)table[rank];
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = first; i < (size_t)n; i += stride) {{
        mine[i] = in[i];
        if (clear) in[i] = 0;
    }}
    ar_barrier(table, rank, epoch + 1);
    for (size_t i = first; i < (size_t)n; i += stride) {{
        float acc = 0.0f;
#pragma unroll
        for (int p = 0; p < AR_RANKS; p++) acc += ar_ld(&(({t}*)table[p])[i]);
        {acc}
    }}
    ar_barrier(table, rank, epoch + 2);
    ar_end(table, rank, epoch + 2);
}}

",
        threads = k.threads,
        name = k.name(),
        acc = accumulate(k.dtype, "i", "acc"),
    )
}

fn two_shot(k: &AllReduceKernel) -> String {
    let t = k.dtype.ctype();
    format!(
        "extern \"C\" __global__ void __launch_bounds__({threads}) {name}({t}* in, {t}* out, const unsigned long long* table, int rank, int n, int add, int clear) {{
    __shared__ unsigned epoch;
    ar_begin(table, rank, &epoch);
    {t}* mine = ({t}*)table[rank];
    size_t chunk = ((size_t)n + AR_RANKS - 1) / AR_RANKS;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int c = 0; c < AR_RANKS; c++) {{
        size_t lo = (size_t)c * chunk;
        size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;
        for (size_t i = lo + first; i < hi; i += stride) {{
            mine[i] = in[i];
            if (clear) in[i] = 0;
        }}
    }}
    ar_barrier(table, rank, epoch + 1);
    {{
        size_t lo = (size_t)rank * chunk;
        size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;
        for (size_t i = lo + first; i < hi; i += stride) {{
            float acc = 0.0f;
#pragma unroll
            for (int p = 0; p < AR_RANKS; p++) acc += ar_ld(&(({t}*)table[p])[i]);
            mine[i] = {st};
        }}
    }}
    ar_barrier(table, rank, epoch + 2);
    for (int c = 0; c < AR_RANKS; c++) {{
        size_t lo = (size_t)c * chunk;
        size_t hi = lo + chunk < (size_t)n ? lo + chunk : (size_t)n;
        const {t}* src = (const {t}*)table[c];
        for (size_t i = lo + first; i < hi; i += stride) {{
            float v = ar_ld(&src[i]);
            {acc}
        }}
    }}
    ar_barrier(table, rank, epoch + 3);
    ar_end(table, rank, epoch + 3);
}}

",
        threads = k.threads,
        name = k.name(),
        st = store(k.dtype, "acc"),
        acc = accumulate(k.dtype, "i", "v"),
    )
}

fn add_rmsnorm(k: &AllReduceKernel, width: u32, eps: f32) -> String {
    format!(
        "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(float* in, float* x, const float* weight, bf16_t* hi, bf16_t* lo, const unsigned long long* table, int rank, int rows) {{
    constexpr int WIDTH = {width};
    __shared__ unsigned epoch;
    __shared__ float scratch[32];
    ar_begin(table, rank, &epoch);
    float* mine = (float*)table[rank];
    for (int r = blockIdx.x; r < rows; r += gridDim.x)
        for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {{
            mine[(size_t)r * WIDTH + i] = in[(size_t)r * WIDTH + i];
            in[(size_t)r * WIDTH + i] = 0.0f;
        }}
    ar_barrier(table, rank, epoch + 1);
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {{
        size_t base = (size_t)r * WIDTH;
        float ss = 0.0f;
        for (int i = threadIdx.x; i < WIDTH / 4; i += blockDim.x) {{
            float4 v = ((const float4*)(x + base))[i];
            float s[4] = {{v.x, v.y, v.z, v.w}};
#pragma unroll
            for (int j = 0; j < 4; j++) {{
                float acc = 0.0f;
#pragma unroll
                for (int p = 0; p < AR_RANKS; p++) acc += ar_ld(&((float*)table[p])[base + 4 * i + j]);
                s[j] = s[j] + acc;
            }}
            ss += s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
            ((float4*)(x + base))[i] = make_float4(s[0], s[1], s[2], s[3]);
        }}
        float inv = rsqrtf(ar_block_sum(ss, scratch) / WIDTH + {eps}f);
        for (int i = threadIdx.x; i < WIDTH / 4; i += blockDim.x) {{
            float4 v = ((const float4*)(x + base))[i];
            float4 w = ((const float4*)weight)[i];
            ar_split2(hi, lo, base + 4 * i, w.x * (v.x * inv), w.y * (v.y * inv));
            ar_split2(hi, lo, base + 4 * i + 2, w.z * (v.z * inv), w.w * (v.w * inv));
        }}
        __syncthreads();
    }}
    ar_barrier(table, rank, epoch + 2);
    ar_end(table, rank, epoch + 2);
}}

",
        threads = k.threads,
        name = k.name(),
        eps = lit(eps),
    )
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
    out.push_str(COMMON);
    for k in kernels {
        out.push_str(&match k.mode {
            Mode::OneShot => one_shot(k),
            Mode::TwoShot => two_shot(k),
            Mode::AddRmsNorm { width, eps } => add_rmsnorm(k, width, eps),
        });
    }
    Ok(out)
}
