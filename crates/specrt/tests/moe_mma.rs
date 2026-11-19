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

const HIDDEN: usize = 256;
const INTER: usize = 256;
const EXPERTS: usize = 16;
const TOP_K: usize = 4;
const PAIRS: usize = TOP_K + 1;
const SLOTS: usize = EXPERTS + 1;

fn spec(hidden: usize, inter: usize) -> Spec {
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

fn mx_quantize(w: &[f32]) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut packed = vec![0u8; w.len() / 2];
    let mut scales = Vec::new();
    let mut deq = Vec::new();
    for (g, group) in w.as_chunks::<32>().0.iter().enumerate() {
        let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let bits = (amax / 6.0).to_bits();
        let e = (((bits >> 23) + u32::from(bits & 0x7fffff != 0)).clamp(1, 253)) as u8;
        scales.push(e);
        let scale = f32::from_bits(u32::from(e) << 23);
        for (i, v) in group.iter().enumerate() {
            let q = specrt::nvfp4::f32_to_e2m1(v / scale);
            let at = g * 32 + i;
            packed[at / 2] |= if at % 2 == 0 { q } else { q << 4 };
            deq.push(e2m1_to_f32(q) * scale);
        }
    }
    (packed, scales, deq)
}

fn mx_stack(rows: usize, cols: usize, rng: &mut Lcg) -> Stack {
    let mut s = Stack { packed: vec![], scales: vec![], inv_global: vec![], dequant: vec![] };
    let padded = specrt::dequant::MXFP4.padded_cols(cols);
    for _ in 0..SLOTS {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.float() * 0.5).collect();
        let (p, sc, d) = mx_quantize(&w);
        s.packed.extend(specrt::dequant::MXFP4.pad_rows(&p, cols / 2, padded / 2));
        s.scales.extend(specrt::dequant::MXFP4.pad_rows(&sc, cols / 32, padded / 32));
        s.dequant.extend(d);
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
    run_with(tokens, expected_bm, false, HIDDEN, INTER);
}

fn run_with(tokens: usize, expected_bm: usize, mx: bool, hidden: usize, inter: usize) {
    let mut spec = spec(hidden, inter);
    if mx {
        let m = spec.moe.as_mut().unwrap();
        m.expert_block = Some(specrt::dequant::MXFP4);
        m.expert_bias = true;
    }
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
    let (ph, pi) = (hidden.next_multiple_of(128), inter.next_multiple_of(128));
    let mut rng = Lcg(644);
    let make = |rows, cols, rng: &mut Lcg| if mx { mx_stack(rows, cols, rng) } else { stack(rows, cols, rng) };
    let gate = make(inter, hidden, &mut rng);
    let up = make(inter, hidden, &mut rng);
    let down = make(hidden, inter, &mut rng);
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
    let bias_up: Vec<f32> = (0..EXPERTS * 2 * inter).map(|_| if mx { rng.float() } else { 0.0 }).collect();
    let bias_down: Vec<f32> = (0..EXPERTS * hidden).map(|_| if mx { rng.float() } else { 0.0 }).collect();
    let eb = |v: &[f32], e: usize, which: usize, r: usize, width: usize| if e < EXPERTS { v[(e * 2 + which) * width + r] } else { 0.0 };
    let igs_up = 300.0f32;
    let igs_down = 120.0f32;
    let qd = |v: &[f32], igs: f32| -> Vec<f32> { if mx { v.chunks(32).flat_map(|c| mx_quantize(c).2).collect() } else { quant_dequant(v, igs) } };
    let up_in: Vec<Vec<f32>> = (0..tokens).map(|t| qd(&x[t * hidden..(t + 1) * hidden], igs_up)).collect();
    let mut expected = vec![0f32; pairs * hidden];
    for p in 0..pairs {
        let e = idx[p] as usize;
        let xin = &up_in[p / PAIRS];
        let act: Vec<f32> = (0..inter)
            .map(|r| {
                let dot = |w: &Stack| -> f32 { (0..hidden).map(|c| w.dequant[(e * inter + r) * hidden + c] * xin[c]).sum() };
                silu(dot(&gate) + eb(&bias_up, e, 0, r, inter)) * (dot(&up) + eb(&bias_up, e, 1, r, inter))
            })
            .collect();
        let mid = qd(&act, igs_down);
        for r in 0..hidden {
            expected[p * hidden + r] = (0..inter).map(|c| down.dequant[(e * hidden + r) * inter + c] * mid[c]).sum::<f32>() + if e < EXPERTS { bias_down[e * hidden + r] } else { 0.0 };
        }
    }
    let d = |v: &[u8]| stream.clone_htod(v).unwrap();
    let df = |v: &[f32]| stream.clone_htod(v).unwrap();
    let one = |s: &Stack| if mx { df(&[1.0]) } else { df(&s.inv_global) };
    let (gw, gs, gg) = (d(&gate.packed), d(&gate.scales), one(&gate));
    let (uw, us, ug) = (d(&up.packed), d(&up.scales), one(&up));
    let (dw, ds, dg) = (d(&down.packed), d(&down.scales), one(&down));
    let bias_u = if mx { df(&bias_up) } else { df(&[0.0]) };
    let bias_d = if mx { df(&bias_down) } else { df(&[0.0]) };
    let stem = if mx { "moe_mma_mx" } else { "moe_mma" };
    let quant_name = if mx { "moe_quant_mx" } else { "moe_quant_f32" };
    let xd = df(&x);
    let idx_d = stream.clone_htod(&idx).unwrap();
    let mut offsets: CudaSlice<i32> = stream.alloc_zeros(SLOTS + 2).unwrap();
    let mut sorted: CudaSlice<i32> = stream.alloc_zeros(pairs).unwrap();
    let max_tiles = SLOTS + pairs / expected_bm;
    let mut tiles: CudaSlice<i32> = stream.alloc_zeros(2 * max_tiles).unwrap();
    let mut xq: CudaSlice<u8> = stream.alloc_zeros(tokens * ph / 2).unwrap();
    let mut xs: CudaSlice<u8> = stream.alloc_zeros(tokens * ph / 16).unwrap();
    let mut act: CudaSlice<f32> = stream.alloc_zeros(pairs * inter).unwrap();
    let mut aq: CudaSlice<u8> = stream.alloc_zeros(pairs * pi / 2).unwrap();
    let mut asc: CudaSlice<u8> = stream.alloc_zeros(pairs * pi / 16).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(pairs * hidden).unwrap();
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
    let w = hidden as i32;
    let kernel = f(quant_name);
    let mut b = stream.launch_builder(&kernel);
    b.arg(&xd).arg(&mut xq).arg(&mut xs).arg(&w);
    if !mx {
        b.arg(&igs_up);
    }
    unsafe { b.launch(cfg(tokens as u32, 256)) }.unwrap();
    let inv_up = 1.0 / igs_up;
    let small = expected_bm == MOE_MMA_BM_SMALL;
    let kernel = f(&format!("{stem}_up{}", if small { "_s" } else { "" }));
    let mut b = stream.launch_builder(&kernel);
    b.arg(&gw).arg(&gs).arg(&gg).arg(&uw).arg(&us).arg(&ug).arg(&bias_u).arg(&inv_up).arg(&xq).arg(&xs).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut act);
    let config = LaunchConfig { grid_dim: ((inter / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(expected_bm, 2) };
    unsafe { b.launch(config) }.unwrap();
    let w = inter as i32;
    let kernel = f(quant_name);
    let mut b = stream.launch_builder(&kernel);
    b.arg(&act).arg(&mut aq).arg(&mut asc).arg(&w);
    if !mx {
        b.arg(&igs_down);
    }
    unsafe { b.launch(cfg(pairs as u32, 256)) }.unwrap();
    let inv_down = 1.0 / igs_down;
    let kernel = f(&format!("{stem}_down{}", if small { "_s" } else { "" }));
    let mut b = stream.launch_builder(&kernel);
    b.arg(&dw).arg(&ds).arg(&dg).arg(&dw).arg(&ds).arg(&dg).arg(&bias_d).arg(&inv_down).arg(&aq).arg(&asc).arg(&offsets).arg(&sorted).arg(&tiles).arg(&mut out);
    let config = LaunchConfig { grid_dim: ((hidden / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(expected_bm, 1) };
    unsafe { b.launch(config) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(&expected).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("tokens {tokens} bm {expected_bm}: max err {err:.3e} of {scale:.3}");
    assert!(err <= 0.02 * scale, "{err} of {scale}");
    let _ = bf16_to_f32(0);
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

#[test]
#[ignore = "needs an sm_120 GPU"]
fn mx_grouped_mma_matches_cpu_with_big_tiles() {
    run_with(48, MOE_MMA_BM, true, HIDDEN, INTER);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn mx_grouped_mma_matches_cpu_with_small_tiles() {
    run_with(8, MOE_MMA_BM_SMALL, true, HIDDEN, INTER);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn mx_grouped_mma_handles_a_k_tail_with_big_tiles() {
    run_with(48, MOE_MMA_BM, true, 2880, 448);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn mx_grouped_mma_handles_a_k_tail_with_small_tiles() {
    run_with(8, MOE_MMA_BM_SMALL, true, 2880, 448);
}
