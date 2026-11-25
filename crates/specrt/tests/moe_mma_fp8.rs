use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use specrt::codegen::{MOE_MMA_BM, MOE_MMA_BM_SMALL, MOE_MMA_BN, Module, cuda_target, moe_mma8_smem};
use specrt::jit::Jit;
use specrt::spec::{ActQuant, Quant, Spec};
use specrt::weights::{e4m3_to_f32, f32_to_e4m3};
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
    spec_of(HIDDEN, INTER)
}

fn spec_of(hidden: usize, inter: usize) -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-moe");
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut s = Spec::from_hf_config(&json).unwrap();
    s.dims.hidden = hidden;
    s.dims.intermediate = inter;
    let m = s.moe.as_mut().unwrap();
    m.experts = EXPERTS;
    m.active = Some(TOP_K);
    m.expert_intermediate = inter;
    m.shared_intermediate = inter;
    s.quant = Quant::Fp8Block128;
    s.mlp.act_quant = ActQuant::Fp8;
    s
}

const BLOCK: usize = 128;

struct Stack {
    values: Vec<u8>,
    grid: Vec<f32>,
    dequant: Vec<f32>,
}

fn stack(rows: usize, cols: usize, rng: &mut Lcg) -> Stack {
    let mut s = Stack { values: vec![], grid: vec![], dequant: vec![] };
    let (rb, cb) = (rows / BLOCK, cols / BLOCK);
    for _ in 0..SLOTS {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.float() * 0.5).collect();
        let grid: Vec<f32> = (0..rb * cb).map(|_| 0.002 + 0.001 * (rng.next() % 5) as f32).collect();
        for r in 0..rows {
            for c in 0..cols {
                let sc = grid[(r / BLOCK) * cb + c / BLOCK];
                let q = f32_to_e4m3(w[r * cols + c] / sc);
                s.values.push(q);
                s.dequant.push(e4m3_to_f32(q) * sc);
            }
        }
        s.grid.extend(grid);
    }
    s
}

fn quant_dequant(v: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![];
    let mut scales = vec![];
    for blk in v.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
        let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
        scales.push(s);
        out.extend(blk.iter().map(|x| e4m3_to_f32(f32_to_e4m3(x / s)) * s));
    }
    (out, scales)
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn run(tokens: usize, expected_bm: usize) {
    run_shape(tokens, expected_bm, HIDDEN, INTER);
}

fn run_shape(tokens: usize, expected_bm: usize, hidden: usize, inter: usize) {
    let spec = spec_of(hidden, inter);
    let ctx = CudaContext::new(0).unwrap();
    let target = cuda_target(&ctx).unwrap();
    if target.compute_capability < (8, 9) {
        eprintln!("needs sm_89 or newer, skipped");
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
    let mut rng = Lcg(885);
    let gate = stack(inter, hidden, &mut rng);
    let up = stack(inter, hidden, &mut rng);
    let down = stack(hidden, inter, &mut rng);
    let x: Vec<f32> = (0..tokens * hidden).map(|_| rng.float() * 2.0).collect();
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
    let up_in: Vec<Vec<f32>> = (0..tokens).map(|t| quant_dequant(&x[t * hidden..(t + 1) * hidden]).0).collect();
    let mut expected = vec![0f32; pairs * hidden];
    for p in 0..pairs {
        let e = idx[p] as usize;
        let xin = &up_in[p / PAIRS];
        let act: Vec<f32> = (0..inter)
            .map(|r| {
                let dot = |w: &Stack| -> f32 { (0..hidden).map(|c| w.dequant[(e * inter + r) * hidden + c] * xin[c]).sum() };
                silu(dot(&gate)) * dot(&up)
            })
            .collect();
        let mid = quant_dequant(&act).0;
        for r in 0..hidden {
            expected[p * hidden + r] = (0..inter).map(|c| down.dequant[(e * hidden + r) * inter + c] * mid[c]).sum();
        }
    }
    let d = |v: &[u8]| stream.clone_htod(v).unwrap();
    let df = |v: &[f32]| stream.clone_htod(v).unwrap();
    let (gw, gg) = (d(&gate.values), df(&gate.grid));
    let (uw, ug) = (d(&up.values), df(&up.grid));
    let (dw, dg) = (d(&down.values), df(&down.grid));
    let xd = df(&x);
    let idx_d = stream.clone_htod(&idx).unwrap();
    let mut offsets: CudaSlice<i32> = stream.alloc_zeros(SLOTS + 2).unwrap();
    let mut sorted: CudaSlice<i32> = stream.alloc_zeros(pairs).unwrap();
    let max_tiles = SLOTS + pairs / expected_bm;
    let mut tiles: CudaSlice<i32> = stream.alloc_zeros(2 * max_tiles).unwrap();
    let mut xq: CudaSlice<u8> = stream.alloc_zeros(tokens * hidden).unwrap();
    let mut xs: CudaSlice<f32> = stream.alloc_zeros(tokens * hidden / 32).unwrap();
    let mut act: CudaSlice<f32> = stream.alloc_zeros(pairs * inter).unwrap();
    let mut aq: CudaSlice<u8> = stream.alloc_zeros(pairs * inter).unwrap();
    let mut asc: CudaSlice<f32> = stream.alloc_zeros(pairs * inter / 32).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(pairs * hidden).unwrap();
    let cfg = |g: u32, b: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    let count = pairs as i32;
    let kernel = f("moe_permute");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&idx_d).arg(&mut offsets).arg(&mut sorted).arg(&count);
    unsafe { b.launch(cfg(1, 1024)) }.unwrap();
    let (bm_i, max_i) = (expected_bm as i32, max_tiles as i32);
    let kernel = f("moe_tiles_f8");
    let mut b = stream.launch_builder(&kernel);
    b.arg(&offsets).arg(&mut tiles).arg(&bm_i).arg(&max_i);
    unsafe { b.launch(cfg(1, 32)) }.unwrap();
    let quant = |x: &CudaSlice<f32>, q: &mut CudaSlice<u8>, s: &mut CudaSlice<f32>, rows: usize, width: usize| {
        let w = width as i32;
        let kernel = f("moe_quant_f8");
        let mut b = stream.launch_builder(&kernel);
        b.arg(x).arg(q).arg(s).arg(&w);
        let config = LaunchConfig { grid_dim: (rows as u32, (width / BLOCK) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }.unwrap();
    };
    quant(&xd, &mut xq, &mut xs, tokens, hidden);
    let small = expected_bm == MOE_MMA_BM_SMALL;
    let kernel = f(if small { "moe_mma_up_f8_s" } else { "moe_mma_up_f8" });
    let mut b = stream.launch_builder(&kernel);
    b.arg(&gw).arg(&gg).arg(&uw).arg(&ug).arg(&xq).arg(&xs).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut act);
    let config = LaunchConfig { grid_dim: ((inter / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma8_smem(expected_bm, 2) };
    unsafe { b.launch(config) }.unwrap();
    quant(&act, &mut aq, &mut asc, pairs, inter);
    let kernel = f(if small { "moe_mma_down_f8_s" } else { "moe_mma_down_f8" });
    let mut b = stream.launch_builder(&kernel);
    b.arg(&dw).arg(&dg).arg(&dw).arg(&dg).arg(&aq).arg(&asc).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut out);
    let config = LaunchConfig { grid_dim: ((hidden / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma8_smem(expected_bm, 1) };
    unsafe { b.launch(config) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(&expected).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("tokens {tokens} bm {expected_bm}: max err {err:.3e} of {scale:.3}");
    assert!(err <= 0.02 * scale, "{err} of {scale}");
}

#[test]
#[ignore = "needs an sm_89 or newer GPU"]
fn grouped_mma_fp8_matches_cpu_at_an_expert_width_of_640() {
    run_shape(48, MOE_MMA_BM, 512, 640);
    run_shape(8, MOE_MMA_BM_SMALL, 512, 640);
}

#[test]
#[ignore = "needs an sm_89 or newer GPU"]
fn grouped_mma_fp8_matches_cpu_with_big_tiles() {
    run(48, MOE_MMA_BM);
}

#[test]
#[ignore = "needs an sm_89 or newer GPU"]
fn grouped_mma_fp8_matches_cpu_with_small_tiles() {
    run(8, MOE_MMA_BM_SMALL);
}

#[test]
fn fp8_mma_path_is_selected_for_fp8_moe_on_sm89_and_up() {
    let s = spec();
    assert!(specrt::codegen::moe_mma_fp8_enabled(&s, &kir::Target::cuda(12, 0, 48, 99 * 1024)));
    assert!(!specrt::codegen::moe_mma_fp8_enabled(&s, &kir::Target::cuda(8, 0, 48, 99 * 1024)));
    let mut off = s.clone();
    off.mlp.act_quant = ActQuant::None;
    assert!(!specrt::codegen::moe_mma_fp8_enabled(&off, &kir::Target::cuda(12, 0, 48, 99 * 1024)));
}
