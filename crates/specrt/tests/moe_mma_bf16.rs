use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use specrt::codegen::{MOE_MMA_BM, MOE_MMA_BM_SMALL, MOE_MMA_BN, Module, cuda_target, moe_mma_bf16_smem};
use specrt::jit::Jit;
use specrt::spec::{Quant, Spec};
use specrt::weights::{bf16_to_f32, f32_to_bf16};
use std::path::Path;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn float(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    }
}

const HIDDEN: usize = 192;
const INTER: usize = 128;
const EXPERTS: usize = 16;
const TOP_K: usize = 4;
const PAIRS: usize = TOP_K + 1;
const SLOTS: usize = EXPERTS + 1;

fn spec() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-moe");
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut s = Spec::from_hf_config(&json).unwrap();
    s.dims.hidden = HIDDEN;
    s.dims.intermediate = INTER;
    let m = s.moe.as_mut().unwrap();
    m.experts = EXPERTS;
    m.active = Some(TOP_K);
    m.expert_intermediate = INTER;
    m.shared_intermediate = INTER;
    s.quant = Quant::Bf16;
    s.mlp.quant = None;
    s
}

struct Stack {
    bits: Vec<u16>,
    values: Vec<f32>,
}

fn stack(rows: usize, cols: usize, rng: &mut Lcg) -> Stack {
    let bits: Vec<u16> = (0..SLOTS * rows * cols).map(|_| f32_to_bf16(rng.float() * 0.5)).collect();
    let values = bits.iter().map(|&b| bf16_to_f32(b)).collect();
    Stack { bits, values }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn run(tokens: usize, expected_bm: usize) {
    let spec = spec();
    let ctx = CudaContext::new(0).unwrap();
    let target = cuda_target(&ctx).unwrap();
    assert!(specrt::codegen::moe_mma_bf16_enabled(&spec, &target));
    let module = Module::generate_kv(&spec, 256, 2, &target).unwrap();
    let jit = Jit::new(ctx.clone(), specrt::jit::default_cache_dir()).unwrap();
    let loaded = jit.load(&module).unwrap();
    use cudarc::driver::sys::CUfunction_attribute as A;
    let f = |name: &str| {
        let k = loaded.function(name).unwrap();
        if let Some(&bytes) = module.smem.get(name) {
            k.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32).unwrap();
        }
        k
    };
    let stream = ctx.default_stream();
    let mut rng = Lcg(907);
    let gate = stack(INTER, HIDDEN, &mut rng);
    let up = stack(INTER, HIDDEN, &mut rng);
    let down = stack(HIDDEN, INTER, &mut rng);
    let x: Vec<f32> = (0..tokens * HIDDEN).map(|_| rng.float() * 2.0).collect();
    let mut idx = vec![0i32; tokens * PAIRS];
    for t in 0..tokens {
        let mut picked: Vec<i32> = vec![];
        while picked.len() < TOP_K {
            let e = (rng.next() as usize % EXPERTS) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        idx[t * PAIRS..t * PAIRS + TOP_K].copy_from_slice(&picked);
        idx[t * PAIRS + TOP_K] = EXPERTS as i32;
    }
    let pairs = tokens * PAIRS;
    let mut expected = vec![0f32; pairs * HIDDEN];
    for p in 0..pairs {
        let e = idx[p] as usize;
        let xin = &x[(p / PAIRS) * HIDDEN..(p / PAIRS + 1) * HIDDEN];
        let act: Vec<f32> = (0..INTER)
            .map(|r| {
                let dot = |w: &Stack| -> f32 { (0..HIDDEN).map(|c| w.values[(e * INTER + r) * HIDDEN + c] * xin[c]).sum() };
                silu(dot(&gate)) * dot(&up)
            })
            .collect();
        for r in 0..HIDDEN {
            expected[p * HIDDEN + r] = (0..INTER).map(|c| down.values[(e * HIDDEN + r) * INTER + c] * act[c]).sum();
        }
    }
    let d = |v: &[u16]| stream.clone_htod(v).unwrap();
    let (gw, uw, dw) = (d(&gate.bits), d(&up.bits), d(&down.bits));
    let xd = stream.clone_htod(&x).unwrap();
    let bias = stream.clone_htod(&[0.0f32]).unwrap();
    let idx_d = stream.clone_htod(&idx).unwrap();
    let mut offsets: CudaSlice<i32> = stream.alloc_zeros(SLOTS + 2).unwrap();
    let mut sorted: CudaSlice<i32> = stream.alloc_zeros(pairs).unwrap();
    let max_tiles = SLOTS + pairs / expected_bm;
    let mut tiles: CudaSlice<i32> = stream.alloc_zeros(2 * max_tiles).unwrap();
    let mut xh: CudaSlice<u16> = stream.alloc_zeros(tokens * HIDDEN).unwrap();
    let mut xl: CudaSlice<u16> = stream.alloc_zeros(tokens * HIDDEN).unwrap();
    let mut act: CudaSlice<f32> = stream.alloc_zeros(pairs * INTER).unwrap();
    let mut ah: CudaSlice<u16> = stream.alloc_zeros(pairs * INTER).unwrap();
    let mut al: CudaSlice<u16> = stream.alloc_zeros(pairs * INTER).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(pairs * HIDDEN).unwrap();
    let cfg = |g: u32, b: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    let count = pairs as i32;
    let kernel = f("moe_permute");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&idx_d).arg(&mut offsets).arg(&mut sorted).arg(&count);
    unsafe { b.launch(cfg(1, 1024)) }.unwrap();
    let (bm_i, max_i) = (expected_bm as i32, max_tiles as i32);
    let kernel = f("moe_tiles_b16");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&offsets).arg(&mut tiles).arg(&bm_i).arg(&max_i);
    unsafe { b.launch(cfg(1, 32)) }.unwrap();
    let w = HIDDEN as i32;
    let kernel = f("moe_split_bf16");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&xd).arg(&mut xh).arg(&mut xl).arg(&w);
    unsafe { b.launch(cfg(tokens as u32, 256)) }.unwrap();
    let small = expected_bm == MOE_MMA_BM_SMALL;
    let kernel = f(if small { "moe_mma_up_b16_s" } else { "moe_mma_up_b16" });
    let mut b = stream.launch_builder(&kernel);
    b.arg(&gw).arg(&uw).arg(&bias).arg(&xh).arg(&xl).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut act);
    let config = LaunchConfig { grid_dim: ((INTER / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_bf16_smem(expected_bm, 2) };
    unsafe { b.launch(config) }.unwrap();
    let w = INTER as i32;
    let kernel = f("moe_split_bf16");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&act).arg(&mut ah).arg(&mut al).arg(&w);
    unsafe { b.launch(cfg(pairs as u32, 256)) }.unwrap();
    let kernel = f(if small { "moe_mma_down_b16_s" } else { "moe_mma_down_b16" });
    let mut b = stream.launch_builder(&kernel);
    b.arg(&dw).arg(&dw).arg(&bias).arg(&ah).arg(&al).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut out);
    let config = LaunchConfig { grid_dim: ((HIDDEN / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_bf16_smem(expected_bm, 1) };
    unsafe { b.launch(config) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(&expected).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("tokens {tokens} bm {expected_bm}: max err {err:.3e} of {scale:.3}");
    assert!(err <= 1e-3 * scale, "{err} of {scale}");
}

#[test]
#[ignore = "needs a CUDA GPU with tensor cores"]
fn grouped_mma_bf16_matches_cpu_with_big_tiles() {
    run(48, MOE_MMA_BM);
}

#[test]
#[ignore = "needs a CUDA GPU with tensor cores"]
fn grouped_mma_bf16_matches_cpu_with_small_tiles() {
    run(8, MOE_MMA_BM_SMALL);
}

#[test]
fn bf16_mma_path_is_selected_for_bf16_moe_with_tensor_cores() {
    let s = spec();
    assert!(specrt::codegen::moe_mma_bf16_enabled(&s, &kir::Target::cuda(8, 0, 48, 99 * 1024)));
    assert!(specrt::codegen::moe_mma_bf16_enabled(&s, &kir::Target::cuda(12, 1, 48, 99 * 1024)));
    let mut fp8 = s.clone();
    fp8.quant = Quant::Fp8Block128;
    assert!(!specrt::codegen::moe_mma_bf16_enabled(&fp8, &kir::Target::cuda(12, 0, 48, 99 * 1024)));
}
