use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use kir::{Activation, Kernel};
use specrt::codegen::Module;
use specrt::jit::Jit;
use std::collections::BTreeMap;

const HIDDEN: usize = 5120;
const INTER: usize = 17408;
const EPS: f32 = 1e-6;

const OLD: &str = "
#define NEG_INF __int_as_float(0xff800000)
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
template <bool IS_MAX>
__device__ float block_reduce(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = IS_MAX ? NEG_INF : 0.0f;
    for (int i = 0; i < warps; i++) r = IS_MAX ? fmaxf(r, scratch[i]) : r + scratch[i];
    return r;
}
constexpr int HIDDEN = 5120;
constexpr int INTER = 17408;
constexpr int ACT_KIND = 0;
constexpr float EPS = 1e-6f;
extern \"C\" __global__ void rmsnorm(const float* x, const float* weight, float* out) {
    __shared__ float scratch[32];
    x += (size_t)blockIdx.x * HIDDEN;
    out += (size_t)blockIdx.x * HIDDEN;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) ss += x[i] * x[i];
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) out[i] = weight[i] * (x[i] * inv);
}
extern \"C\" __global__ void swiglu(const float* gate_up, float* out) {
    gate_up += (size_t)blockIdx.y * 2 * INTER;
    out += (size_t)blockIdx.y * INTER;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < INTER; i += gridDim.x * blockDim.x) {
        float g = gate_up[2 * i];
        float a = ACT_KIND == 1
            ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))
            : g / (1.0f + expf(-g));
        out[i] = a * gate_up[2 * i + 1];
    }
}
";

fn module(source: kir::KirSource) -> Module {
    Module { source, kernels: vec![], smem: BTreeMap::new(), arch_features: false, kir_hash: String::new() }
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn kir_rmsnorm_and_swiglu_are_not_slower_than_the_templates() {
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-kir-speed-{}", std::process::id()))).unwrap();
    let stream = ctx.default_stream();
    let new_src = kir::cuda::emit_module(&[
        Kernel::rmsnorm(&target, HIDDEN as u32, EPS, 1024).unwrap(),
        Kernel::gated_activation(&target, "swiglu", INTER as u32, Activation::Silu, 256).unwrap(),
    ])
    .unwrap();
    let old = jit.load(&module(kir::KirSource::test_text(OLD))).unwrap();
    let new = jit.load(&module(new_src)).unwrap();
    let max_rows = 4096;
    let x: CudaSlice<f32> = stream.clone_htod(&(0..max_rows * HIDDEN).map(|i| ((i % 97) as f32 - 48.0) * 0.03).collect::<Vec<_>>()).unwrap();
    let w: CudaSlice<f32> = stream.clone_htod(&vec![1.1f32; HIDDEN]).unwrap();
    let out: CudaSlice<f32> = stream.alloc_zeros(max_rows * HIDDEN).unwrap();
    let gu: CudaSlice<f32> = stream.clone_htod(&(0..max_rows * 2 * INTER).map(|i| ((i % 89) as f32 - 44.0) * 0.05).collect::<Vec<_>>()).unwrap();
    let act: CudaSlice<f32> = stream.alloc_zeros(max_rows * INTER).unwrap();
    let mut worst: f32 = 0.0;
    for rows in [1usize, 8, 4096] {
        for name in ["rmsnorm", "swiglu"] {
            let mut samples: [Vec<f32>; 2] = [vec![], vec![]];
            let reps = if rows == 4096 { 10 } else { 200 };
            for round in 0..21 {
                for (which, loaded) in [&old, &new].into_iter().enumerate() {
                    let f = loaded.function(name).unwrap();
                    let launch = || {
                        let mut b = stream.launch_builder(&f);
                        let config = if name == "rmsnorm" {
                            b.arg(&x).arg(&w).arg(&out);
                            LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }
                        } else {
                            b.arg(&gu).arg(&act);
                            LaunchConfig { grid_dim: (INTER.div_ceil(256) as u32, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
                        };
                        unsafe { b.launch(config) }.unwrap();
                    };
                    launch();
                    let e0 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    let e1 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    e0.record(&stream).unwrap();
                    for _ in 0..reps {
                        launch();
                    }
                    e1.record(&stream).unwrap();
                    e1.synchronize().unwrap();
                    if round > 0 {
                        samples[which].push(e0.elapsed_ms(&e1).unwrap() * 1000.0 / reps as f32);
                    }
                }
            }
            let [o, n] = samples;
            let (o, n) = (median(o), median(n));
            let pct = (n / o - 1.0) * 100.0;
            worst = worst.max(pct);
            eprintln!("{name} rows {rows}: templates {o:.2} us, kir {n:.2} us, {pct:+.2}%");
        }
    }
    assert!(worst <= 2.0, "kir is {worst:.2}% slower");
}
