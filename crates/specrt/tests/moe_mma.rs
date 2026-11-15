use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use specrt::codegen::{MOE_MMA_BM, MOE_MMA_BM_SMALL, MOE_MMA_BN, Module, cuda_target, moe_mma_smem};
use specrt::jit::Jit;
use specrt::nvfp4::{e2m1_to_f32, quantize};
use specrt::spec::{ActQuant, Quant, Spec};
use specrt::weights::{bf16_to_f32, e4m3_to_f32};
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

const HIDDEN: usize = 512;
const INTER: usize = 512;
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
    s.mlp.quant = Some(Quant::Nvfp4G16);
    s.mlp.act_quant = ActQuant::Fp4;
    s
}

struct Stack {
    packed: Vec<u8>,
    scales: Vec<u8>,
    inv_global: Vec<f32>,
    dequant: Vec<f32>,
}

fn stack(rows: usize, cols: usize, rng: &mut Lcg) -> Stack {
    let global = 400.0f32;
    let mut s = Stack { packed: vec![], scales: vec![], inv_global: vec![], dequant: vec![] };
    for _ in 0..SLOTS {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.float() * 0.5).collect();
        let (packed, scales) = quantize(&w, cols, global);
        for (i, &b) in packed.iter().enumerate() {
            let _ = i;
            let _ = b;
        }
        for r in 0..rows {
            for c in 0..cols {
                let b = packed[(r * cols + c) / 2];
                let nib = if c % 2 == 0 { b & 15 } else { b >> 4 };
                s.dequant.push(e2m1_to_f32(nib) * e4m3_to_f32(scales[(r * cols + c) / 16]) / global);
            }
        }
        s.packed.extend(packed);
        s.scales.extend(scales);
        s.inv_global.push(1.0 / global);
    }
    s
}

fn quant_dequant(v: &[f32], igs: f32) -> Vec<f32> {
    let (packed, scales) = quantize(v, v.len(), igs);
    (0..v.len())
        .map(|i| {
            let b = packed[i / 2];
            let nib = if i % 2 == 0 { b & 15 } else { b >> 4 };
            e2m1_to_f32(nib) * e4m3_to_f32(scales[i / 16]) / igs
        })
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn run(tokens: usize, expected_bm: usize) {
    let spec = spec();
    let ctx = CudaContext::new(0).unwrap();
    let target = cuda_target(&ctx).unwrap();
    if target.compute_capability.0 < 12 {
        eprintln!("needs sm_120, skipped");
        return;
    }
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
    let mut rng = Lcg(644);
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
    let igs_up = 300.0f32;
    let igs_down = 120.0f32;
    let up_in: Vec<Vec<f32>> = (0..tokens).map(|t| quant_dequant(&x[t * HIDDEN..(t + 1) * HIDDEN], igs_up)).collect();
    let mut expected = vec![0f32; pairs * HIDDEN];
    for p in 0..pairs {
        let e = idx[p] as usize;
        let xin = &up_in[p / PAIRS];
        let act: Vec<f32> = (0..INTER)
            .map(|r| {
                let dot = |w: &Stack| -> f32 { (0..HIDDEN).map(|c| w.dequant[(e * INTER + r) * HIDDEN + c] * xin[c]).sum() };
                silu(dot(&gate)) * dot(&up)
            })
            .collect();
        let mid = quant_dequant_rows(&act, igs_down);
        for r in 0..HIDDEN {
            expected[p * HIDDEN + r] = (0..INTER).map(|c| down.dequant[(e * HIDDEN + r) * INTER + c] * mid[c]).sum();
        }
    }
    let d = |v: &[u8]| stream.clone_htod(v).unwrap();
    let df = |v: &[f32]| stream.clone_htod(v).unwrap();
    let (gw, gs, gg) = (d(&gate.packed), d(&gate.scales), df(&gate.inv_global));
    let (uw, us, ug) = (d(&up.packed), d(&up.scales), df(&up.inv_global));
    let (dw, ds, dg) = (d(&down.packed), d(&down.scales), df(&down.inv_global));
    let xd = df(&x);
    let idx_d = stream.clone_htod(&idx).unwrap();
    let mut offsets: CudaSlice<i32> = stream.alloc_zeros(SLOTS + 2).unwrap();
    let mut sorted: CudaSlice<i32> = stream.alloc_zeros(pairs).unwrap();
    let max_tiles = SLOTS + pairs / expected_bm;
    let mut tiles: CudaSlice<i32> = stream.alloc_zeros(2 * max_tiles).unwrap();
    let mut xq: CudaSlice<u8> = stream.alloc_zeros(tokens * HIDDEN / 2).unwrap();
    let mut xs: CudaSlice<u8> = stream.alloc_zeros(tokens * HIDDEN / 16).unwrap();
    let mut act: CudaSlice<f32> = stream.alloc_zeros(pairs * INTER).unwrap();
    let mut aq: CudaSlice<u8> = stream.alloc_zeros(pairs * INTER / 2).unwrap();
    let mut asc: CudaSlice<u8> = stream.alloc_zeros(pairs * INTER / 16).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(pairs * HIDDEN).unwrap();
    let cfg = |g: u32, b: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    let count = pairs as i32;
    let kernel = f("moe_permute");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&idx_d).arg(&mut offsets).arg(&mut sorted).arg(&count);
    unsafe { b.launch(cfg(1, 1024)) }.unwrap();
    let (bm_i, max_i) = (expected_bm as i32, max_tiles as i32);
    let kernel = f("moe_tiles");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&offsets).arg(&mut tiles).arg(&bm_i).arg(&max_i);
    unsafe { b.launch(cfg(1, 32)) }.unwrap();
    let w = HIDDEN as i32;
    let kernel = f("moe_quant_f32");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&xd).arg(&mut xq).arg(&mut xs).arg(&w).arg(&igs_up);
    unsafe { b.launch(cfg(tokens as u32, 256)) }.unwrap();
    let inv_up = 1.0 / igs_up;
    let small = expected_bm == MOE_MMA_BM_SMALL;
    let kernel = f(if small { "moe_mma_up_s" } else { "moe_mma_up" });
    let mut b = stream.launch_builder(&kernel);
    b.arg(&gw).arg(&gs).arg(&gg).arg(&uw).arg(&us).arg(&ug).arg(&inv_up).arg(&xq).arg(&xs).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut act);
    let config = LaunchConfig { grid_dim: ((INTER / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(expected_bm, 2) };
    unsafe { b.launch(config) }.unwrap();
    let w = INTER as i32;
    let kernel = f("moe_quant_f32");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&act).arg(&mut aq).arg(&mut asc).arg(&w).arg(&igs_down);
    unsafe { b.launch(cfg(pairs as u32, 256)) }.unwrap();
    let inv_down = 1.0 / igs_down;
    let kernel = f(if small { "moe_mma_down_s" } else { "moe_mma_down" });
    let mut b = stream.launch_builder(&kernel);
    b.arg(&dw).arg(&ds).arg(&dg).arg(&dw).arg(&ds).arg(&dg).arg(&inv_down).arg(&aq).arg(&asc).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut out);
    let config = LaunchConfig { grid_dim: ((HIDDEN / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(expected_bm, 1) };
    unsafe { b.launch(config) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(&expected).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("tokens {tokens} bm {expected_bm}: max err {err:.3e} of {scale:.3}");
    assert!(err <= 0.02 * scale, "{err} of {scale}");
    let _ = bf16_to_f32(0);
}

fn quant_dequant_rows(v: &[f32], igs: f32) -> Vec<f32> {
    quant_dequant(v, igs)
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn grouped_mma_matches_cpu_with_big_tiles() {
    run(48, MOE_MMA_BM);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn grouped_mma_matches_cpu_with_small_tiles() {
    run(8, MOE_MMA_BM_SMALL);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn grouped_mma_matches_cpu_with_one_token() {
    run(1, MOE_MMA_BM_SMALL);
}
