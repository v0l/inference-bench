mod allreduce;
mod attention;
mod gemm;
mod quant;
mod linear;
mod ggml;
mod vision;
mod ops;
mod moe;
pub(crate) mod topk;

pub use allreduce::{PROBE_STORE, PROBE_WAIT, emit_allreduce, emit_peer_probe};
pub use attention::emit_attention;
pub use linear::emit_linear;
pub use ggml::emit_ggml;
pub use vision::emit_vision;
pub use ops::emit_ops;
pub use moe::emit_moe;
pub use gemm::{fp4_act_source, emit_gemm, emit_gemms, gemm_library, needs_arch_features};

use crate::layout::{Dialect, Layout};
use crate::{Activation, BlockSum, Body, Kernel, Language, Pointwise, lit};

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

fn weight_expr(w: &str, offset: f32) -> String {
    if offset == 0.0 { format!("{w}[i]") } else { format!("({w}[i] + {}f)", lit(offset)) }
}

pub fn emit_kernel(k: &Kernel) -> Result<String, String> {
    check(k)?;
    Ok(match &k.body {
        Body::RmsNorm { width, eps, sum: BlockSum::WarpShuffle, offset } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {x}, const float* {w}, float* {out}) {{
    constexpr int WIDTH = {width};
    __shared__ float scratch[32];
    {x} += (size_t)blockIdx.x * WIDTH;
    {out} += (size_t)blockIdx.x * WIDTH;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) ss += {x}[i] * {x}[i];
    float inv = rsqrtf(kir_block_sum(ss, scratch) / WIDTH + {eps}f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {out}[i] = {wi} * ({x}[i] * inv);
}}

",
            threads = k.threads,
            name = k.name,
            x = k.params[0].name,
            w = k.params[1].name,
            out = k.params[2].name,
            eps = lit(*eps),
            wi = weight_expr(k.params[1].name, *offset),
        ),
        Body::RmsNormSegment { width, stride, start, eps, sum: BlockSum::WarpShuffle } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {x}, const float* {w}, float* {out}) {{
    constexpr int WIDTH = {width};
    __shared__ float scratch[32];
    size_t base = (size_t)blockIdx.x * {stride} + {start};
    float ss = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) ss += {x}[base + i] * {x}[base + i];
    float inv = rsqrtf(kir_block_sum(ss, scratch) / WIDTH + {eps}f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {out}[base + i] = {w}[i] * ({x}[base + i] * inv);
}}

",
            threads = k.threads,
            name = k.name,
            x = k.params[0].name,
            w = k.params[1].name,
            out = k.params[2].name,
            eps = lit(*eps),
        ),
        Body::RmsNormSegment { sum, .. } => return Err(format!("{}: block sum {sum:?} has no CUDA snippet", k.name)),
        Body::AddRmsNorm { width, eps, sum: BlockSum::WarpShuffle, offset } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {x}, const float* {d}, const float* {w}, float* {sm}, float* {out}) {{
    constexpr int WIDTH = {width};
    __shared__ float scratch[32];
    size_t base = (size_t)blockIdx.x * WIDTH;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {{
        float v = {x}[base + i] + {d}[base + i];
        {sm}[base + i] = v;
        ss += v * v;
    }}
    float inv = rsqrtf(kir_block_sum(ss, scratch) / WIDTH + {eps}f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {out}[base + i] = {wi} * ({sm}[base + i] * inv);
}}

",
            threads = k.threads,
            name = k.name,
            x = k.params[0].name,
            d = k.params[1].name,
            w = k.params[2].name,
            sm = k.params[3].name,
            out = k.params[4].name,
            eps = lit(*eps),
            wi = weight_expr(k.params[2].name, *offset),
        ),
        Body::Pointwise { width, op } => {
            let e = match op {
                Pointwise::Scale(f) => format!("v * {}f", lit(*f)),
                Pointwise::SoftCap(c) => format!("{c}f * tanhf(v / {c}f)", c = lit(*c)),
                Pointwise::Fill(f) => format!("{}f", lit(*f)),
            };
            format!(
                "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {x}, float* {out}) {{\n    constexpr int WIDTH = {width};\n    {x} += (size_t)blockIdx.y * WIDTH;\n    {out} += (size_t)blockIdx.y * WIDTH;\n    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WIDTH; i += gridDim.x * blockDim.x) {{\n        float v = {x}[i];\n        {out}[i] = {e};\n    }}\n}}\n\n",
                threads = k.threads,
                name = k.name,
                x = k.params[0].name,
                out = k.params[1].name,
            )
        }
        Body::NormAdd { width, eps, sum: BlockSum::WarpShuffle, offset } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {d}, const float* {w}, const float* {r}, float* {out}) {{
    constexpr int WIDTH = {width};
    __shared__ float scratch[32];
    size_t base = (size_t)blockIdx.x * WIDTH;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) ss += {d}[base + i] * {d}[base + i];
    float inv = rsqrtf(kir_block_sum(ss, scratch) / WIDTH + {eps}f);
    for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {out}[base + i] = {r}[base + i] + {wi} * ({d}[base + i] * inv);
}}

",
            threads = k.threads,
            name = k.name,
            d = k.params[0].name,
            w = k.params[1].name,
            r = k.params[2].name,
            out = k.params[3].name,
            eps = lit(*eps),
            wi = weight_expr(k.params[1].name, *offset),
        ),
        Body::NormAdd { sum, .. } => return Err(format!("{}: block sum {sum:?} has no CUDA snippet", k.name)),
        Body::Add { width, scale } => format!(
            "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const float* {x}, const float* {d}, float* {out}) {{
    constexpr int WIDTH = {width};
    size_t base = (size_t)blockIdx.y * WIDTH;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WIDTH; i += gridDim.x * blockDim.x) {out}[base + i] = {x}[base + i] + {scaled}{d}[base + i];
}}

",
            threads = k.threads,
            name = k.name,
            x = k.params[0].name,
            d = k.params[1].name,
            out = k.params[2].name,
            scaled = if *scale == 1.0 { String::new() } else { format!("{}f * ", lit(*scale)) },
        ),
        Body::AddRmsNorm { sum, .. } => return Err(format!("{}: block sum {sum:?} has no CUDA snippet", k.name)),
        Body::RmsNorm { sum, .. } => return Err(format!("{}: block sum {sum:?} has no CUDA snippet", k.name)),
        Body::Compose(c) => c.emit_cuda(k)?,
        Body::Scan(c) => c.emit_cuda(k)?,
        Body::Rows(c) => c.emit_cuda(k)?,
        Body::Recurrence(c) => c.emit_cuda(k)?,
        Body::Seq(c) => c.emit_cuda(k)?,
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
    if kernels.iter().any(|k| matches!(k.body, Body::RmsNorm { .. } | Body::RmsNormSegment { .. } | Body::AddRmsNorm { .. } | Body::NormAdd { .. }) || matches!(&k.body, Body::Compose(c) if c.has_sum()) || matches!(&k.body, Body::Rows(crate::rows::RowBody::LayerNorm { .. }))) {
        out.push_str(WARP_SUM);
    }
    for k in kernels {
        out.push_str(&emit_kernel(k)?);
    }
    out.push_str(&linear::chunked_scans(kernels)?);
    Ok(out)
}

fn pdl_macro(single_wave_blocks: u32) -> String {
    format!(
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define PDL_WAIT() do {{ asm volatile(\"griddepcontrol.wait;\" ::: \"memory\"); if (gridDim.x * gridDim.y * gridDim.z <= {single_wave_blocks}u) asm volatile(\"griddepcontrol.launch_dependents;\"); }} while (0)\n#else\n#define PDL_WAIT()\n#endif\n"
    )
}

pub fn with_pdl_wait(source: &str, single_wave_blocks: u32) -> String {
    let mut out = String::with_capacity(source.len() + 1024);
    out.push_str(&pdl_macro(single_wave_blocks));
    let mut rest = source;
    while let Some(at) = rest.find("__global__") {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let open = tail.find('{').map(|i| i + 1).unwrap_or(tail.len());
        out.push_str(&tail[..open]);
        if open < tail.len() {
            out.push_str("\n    PDL_WAIT();");
        }
        rest = &tail[open..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod pdl_tests {
    use super::with_pdl_wait;

    #[test]
    fn wait_is_first_statement_of_every_kernel() {
        let src = "extern \"C\" __global__ void a(int* x) {\n x[0] = 1;\n}\nextern \"C\" __global__ void __launch_bounds__(32) b(float* y) { y[0] = 0; }\n";
        let out = with_pdl_wait(src, 188);
        assert_eq!(out.matches("PDL_WAIT();").count(), 2);
        assert!(out.contains("void a(int* x) {\n    PDL_WAIT();\n x[0]"));
        assert!(out.contains("b(float* y) {\n    PDL_WAIT(); y[0]"));
    }
}
