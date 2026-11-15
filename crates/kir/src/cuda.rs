use crate::layout::{Dialect, Layout};
use crate::{Activation, BlockSum, Body, Kernel, Language, lit};

const WARP_SUM: &str = "__device__ __forceinline__ float kir_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ float kir_block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    v = kir_warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < warps; i++) r += scratch[i];
    return r;
}

";

fn check(k: &Kernel) -> Result<(), String> {
    k.validate()?;
    if k.target.language != Language::Cuda {
        return Err(format!("{}: the CUDA emitter cannot target {:?}", k.name, k.target.language));
    }
    Ok(())
}

pub fn gate_up_tile(width: u32, i: &str, k: &str) -> Result<String, String> {
    gate_up_tile_in(width, i, k, Dialect::Cuda)
}

pub fn gate_up_tile_in(width: u32, i: &str, k: &str, dialect: Dialect) -> Result<String, String> {
    Layout::flat(&[width as u64, 2], &[2, 1])?.expr(&[i, k], dialect)
}

pub fn emit_kernel(k: &Kernel) -> Result<String, String> {
    check(k)?;
    Ok(match &k.body {
        Body::RmsNorm { width, eps, sum: BlockSum::WarpShuffle } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {x}, const float* {w}, float* {out}) {{
    constexpr int WIDTH = {width};
    __shared__ float scratch[32];
    {x} += (size_t)blockIdx.x * WIDTH;
    {out} += (size_t)blockIdx.x * WIDTH;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) ss += {x}[i] * {x}[i];
    float inv = rsqrtf(kir_block_sum(ss, scratch) / WIDTH + {eps}f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {out}[i] = {w}[i] * ({x}[i] * inv);
}}

",
            threads = k.threads,
            name = k.name,
            x = k.params[0].name,
            w = k.params[1].name,
            out = k.params[2].name,
            eps = lit(*eps),
        ),
        Body::RmsNorm { sum, .. } => return Err(format!("{}: block sum {sum:?} has no CUDA snippet", k.name)),
        Body::GatedActivation { width, act } => {
            let a = match act {
                Activation::Silu => "g / (1.0f + expf(-g))",
                Activation::GeluTanh => "0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))",
            };
            format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {gu}, float* {out}) {{
    constexpr int WIDTH = {width};
    {gu} += (size_t)blockIdx.y * 2 * WIDTH;
    {out} += (size_t)blockIdx.y * WIDTH;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WIDTH; i += gridDim.x * blockDim.x) {{
        float g = {gu}[{g}];
        float a = {a};
        {out}[i] = a * {gu}[{u}];
    }}
}}

",
                g = gate_up_tile(*width, "i", "0")?,
                u = gate_up_tile(*width, "i", "1")?,
                threads = k.threads,
                name = k.name,
                gu = k.params[0].name,
                out = k.params[1].name,
            )
        }
    })
}

pub fn emit_module(kernels: &[Kernel]) -> Result<String, String> {
    let mut out = String::new();
    if kernels.iter().any(|k| matches!(k.body, Body::RmsNorm { .. })) {
        out.push_str(WARP_SUM);
    }
    for k in kernels {
        out.push_str(&emit_kernel(k)?);
    }
    Ok(out)
}
