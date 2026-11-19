use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::{Module, WeightKind, cuda_target, select};
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

fn bf16(v: f32) -> u16 {
    let u = v.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn run(linear: &str, tokens: usize) -> f64 {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.5-122b-a10b-nvfp4.config.json");
    let spec = Spec::from_hf_config_serving(&std::fs::read_to_string(path).unwrap()).unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let target = cuda_target(&ctx).unwrap();
    let module = Module::generate_kv(&spec, 256, 2, &target).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-gemm-shapes-{}", std::process::id()))).unwrap();
    let loaded = jit.load(&module).unwrap();
    let stream = ctx.default_stream();
    let choice = select(&spec, &target, linear, tokens, WeightKind::Bf16).unwrap();
    let shape = specrt::codegen::linear_shape(&spec, linear);
    let (n, k) = (shape.n, shape.k);
    let kernel = loaded.function(&choice.name).unwrap();
    use cudarc::driver::sys::CUfunction_attribute as A;
    let smem = module.smem.get(choice.name.as_str()).copied().unwrap_or(0);
    kernel.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
    let mut seed = 99u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rnd() * 0.1)).collect();
    let x: Vec<f32> = (0..tokens * k).map(|_| rnd()).collect();
    let hi: Vec<u16> = x.iter().map(|&v| bf16(v)).collect();
    let lo: Vec<u16> = x.iter().zip(&hi).map(|(&v, &h)| bf16(v - f(h))).collect();
    let (dw, dhi, dlo) = (stream.clone_htod(&w).unwrap(), stream.clone_htod(&hi).unwrap(), stream.clone_htod(&lo).unwrap());
    let dummy = stream.alloc_zeros::<f32>(1).unwrap();
    let dummy16 = stream.alloc_zeros::<u16>(8).unwrap();
    let mut out = stream.alloc_zeros::<f32>(tokens * n).unwrap();
    let nt = tokens as i32;
    let tile = choice.tile;
    let mut b = stream.launch_builder(&kernel);
    b.arg(&dw).arg(&dummy).arg(&dummy).arg(&dhi).arg(&dlo).arg(&mut out).arg(&dummy16).arg(&dummy16).arg(&nt);
    let config = LaunchConfig { grid_dim: (n.div_ceil(tile.bn) as u32, tokens.div_ceil(tile.bm) as u32, 1), block_dim: (tile.threads(), 1, 1), shared_mem_bytes: smem };
    unsafe { b.launch(config) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    let mut worst = 0f64;
    let mut scale = 0f64;
    for t in 0..tokens {
        for r in (0..n).step_by(7) {
            let want: f64 = (0..k).map(|c| f(w[r * k + c]) as f64 * (f(hi[t * k + c]) + f(lo[t * k + c])) as f64).sum();
            worst = worst.max((got[t * n + r] as f64 - want).abs());
            scale = scale.max(want.abs());
        }
    }
    eprintln!("{linear} n={n} k={k} tokens {tokens} kernel {}: max err {worst:.2e} of {scale:.2}", choice.name);
    worst / scale
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn attention_linears_match_a_cpu_reference_at_122b_shapes() {
    for linear in ["qkv", "o"] {
        for tokens in [1, 16, 17, 32, 96] {
            let e = run(linear, tokens);
            assert!(e < 1e-4, "{linear} tokens {tokens}: {e}");
        }
    }
}
