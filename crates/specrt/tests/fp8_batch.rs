use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use specrt::codegen::{F8B_REDUCE_BLOCK, F8B_ROWS, F8B_WARPS, FP8_BLOCK, Module, Tile, f8b_smem, f8b_splits, f8b_tokens};
use specrt::jit::Jit;
use specrt::spec::Spec;
use specrt::weights::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, fp8_scales};
use std::path::Path;
use std::sync::Arc;

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

struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<u8>,
    row_scale: Vec<f32>,
}

impl Matrix {
    fn random(rows: usize, cols: usize, rng: &mut Lcg) -> Matrix {
        let values = (0..rows * cols)
            .map(|_| {
                let b = rng.next() as u8;
                if b & 0x7f == 0x7f { b & 0x80 } else { b }
            })
            .collect();
        let row_scale = (0..rows).map(|_| 1.0 / 2000.0 * (1.0 + rng.float() * 0.5)).collect();
        Matrix { rows, cols, values, row_scale }
    }

    fn device_scales(&self) -> Vec<f32> {
        let kb = self.cols / FP8_BLOCK;
        let grid = self.row_scale.iter().flat_map(|&s| std::iter::repeat_n(s, kb)).collect::<Vec<_>>();
        let map = (0..self.rows).map(|r| (r * kb) as u32).collect::<Vec<_>>();
        fp8_scales(&grid, &map)
    }
}

fn planes(tokens: usize, cols: usize, rng: &mut Lcg) -> (Vec<f32>, Vec<u16>, Vec<u16>) {
    let x: Vec<f32> = (0..tokens * cols).map(|_| rng.float() * 3.0).collect();
    let hi: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
    let lo: Vec<u16> = x.iter().zip(&hi).map(|(&v, &h)| f32_to_bf16(v - bf16_to_f32(h))).collect();
    let exact = hi.iter().zip(&lo).map(|(&h, &l)| bf16_to_f32(h) + bf16_to_f32(l)).collect();
    (exact, hi, lo)
}

fn cpu(m: &Matrix, x: &[f32], tokens: usize) -> Vec<f64> {
    let mut out = vec![0f64; tokens * m.rows];
    for r in 0..m.rows {
        let w: Vec<f64> = (0..m.cols).map(|c| e4m3_to_f32(m.values[r * m.cols + c]) as f64 * m.row_scale[r] as f64).collect();
        for t in 0..tokens {
            let xr = &x[t * m.cols..(t + 1) * m.cols];
            out[t * m.rows + r] = w.iter().zip(xr).map(|(a, &b)| a * b as f64).sum();
        }
    }
    out
}

struct Gpu {
    stream: Arc<CudaStream>,
    tiny: CudaFunction,
    batch: [CudaFunction; 2],
    reduce: CudaFunction,
    tiny_smem: u32,
    sms: usize,
}

fn gpu(linear: &str) -> Gpu {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.toml");
    let module = Module::generate(&Spec::load(&path).unwrap(), 256).unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), specrt::jit::default_cache_dir()).unwrap();
    let loaded = jit.load(&module).unwrap();
    let f = |name: &str| loaded.function(name).unwrap();
    let tiny_name = format!("{linear}_mma_tiny");
    let tiny_smem = module.smem[tiny_name.as_str()];
    let tiny = f(&tiny_name);
    let batch = [8, 16].map(|t| f(&format!("{linear}_f8b{t}")));
    use cudarc::driver::sys::CUfunction_attribute as A;
    tiny.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, tiny_smem as i32).unwrap();
    for (b, t) in batch.iter().zip([8, 16]) {
        b.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f8b_smem(t) as i32).unwrap();
    }
    let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap();
    Gpu { stream: ctx.default_stream(), tiny, batch, reduce: f(&format!("{linear}_f8b_reduce")), tiny_smem, sms: sms as usize }
}

struct Run {
    out: Vec<f32>,
    micros: f64,
}

struct Device {
    w: CudaSlice<u8>,
    s: CudaSlice<f32>,
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
    out: CudaSlice<f32>,
    oh: CudaSlice<u16>,
    ol: CudaSlice<u16>,
    part: CudaSlice<f32>,
    dummy: CudaSlice<f32>,
}

const REPEATS: usize = 10;
const ROUNDS: usize = 20;
const ERROR_BOUND: f64 = 1e-4;

fn time(g: &Gpu, d: &mut Device, mut launch: impl FnMut(&Device)) -> f64 {
    g.stream.memset_zeros(&mut d.out).unwrap();
    launch(d);
    g.stream.synchronize().unwrap();
    let first = g.stream.clone_dtoh(&d.out).unwrap();
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let started = std::time::Instant::now();
        for _ in 0..REPEATS {
            launch(d);
        }
        g.stream.synchronize().unwrap();
        best = best.min(started.elapsed().as_secs_f64() * 1e6 / REPEATS as f64);
    }
    g.stream.memcpy_htod(&first, &mut d.out).unwrap();
    best
}

fn read(g: &Gpu, d: &Device, act: bool, n: usize) -> Vec<f32> {
    if act {
        let hi = g.stream.clone_dtoh(&d.oh).unwrap();
        let lo = g.stream.clone_dtoh(&d.ol).unwrap();
        hi.iter().zip(&lo).take(n).map(|(&h, &l)| bf16_to_f32(h) + bf16_to_f32(l)).collect()
    } else {
        g.stream.clone_dtoh(&d.out).unwrap()[..n].to_vec()
    }
}

fn run_tiny(g: &Gpu, d: &mut Device, rows: usize, tokens: usize, act: bool) -> Run {
    let t = Tile::tiny();
    let n = tokens as i32;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(t.bn) as u32, tokens.div_ceil(t.bm) as u32, 1),
        block_dim: (t.threads(), 1, 1),
        shared_mem_bytes: g.tiny_smem,
    };
    let micros = time(g, d, |d| {
        let mut b = g.stream.launch_builder(&g.tiny);
        b.arg(&d.w).arg(&d.s).arg(&d.dummy).arg(&d.hi).arg(&d.lo).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&n);
        unsafe { b.launch(config) }.unwrap();
    });
    let len = if act { tokens * rows / 2 } else { tokens * rows };
    Run { out: read(g, d, act, len), micros }
}

fn run_batch(g: &Gpu, d: &mut Device, rows: usize, cols: usize, tokens: usize, act: bool) -> Run {
    let splits = f8b_splits(rows, cols, g.sms);
    let (n, s) = (tokens as i32, splits as i32);
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(F8B_ROWS) as u32, splits as u32, 1),
        block_dim: ((F8B_WARPS * 32) as u32, 1, 1),
        shared_mem_bytes: f8b_smem(tokens),
    };
    let outs = if act { tokens * rows / 2 } else { tokens * rows };
    let reduce = LaunchConfig {
        grid_dim: (outs.div_ceil(F8B_REDUCE_BLOCK) as u32, 1, 1),
        block_dim: (F8B_REDUCE_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let kernel = &g.batch[usize::from(f8b_tokens(tokens) == 16)];
    let micros = time(g, d, |d| {
        let mut b = g.stream.launch_builder(kernel);
        b.arg(&d.w).arg(&d.s).arg(&d.dummy).arg(&d.hi).arg(&d.lo).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&d.part).arg(&n).arg(&s);
        unsafe { b.launch(config) }.unwrap();
        if splits > 1 {
            let mut b = g.stream.launch_builder(&g.reduce);
            b.arg(&d.part).arg(&d.dummy).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&n).arg(&s);
            unsafe { b.launch(reduce) }.unwrap();
        }
    });
    let len = if act { tokens * rows / 2 } else { tokens * rows };
    Run { out: read(g, d, act, len), micros }
}

fn swiglu(v: &[f64], tokens: usize, rows: usize) -> Vec<f64> {
    (0..tokens * rows / 2)
        .map(|i| {
            let (gate, up) = (v[(i / (rows / 2)) * rows + 2 * (i % (rows / 2))], v[(i / (rows / 2)) * rows + 2 * (i % (rows / 2)) + 1]);
            gate / (1.0 + (-gate).exp()) * up
        })
        .collect()
}

fn max_err(ours: &[f32], exact: &[f64]) -> (f64, f64) {
    let scale = exact.iter().fold(0f64, |m, v| m.max(v.abs()));
    let err = ours.iter().zip(exact).fold(0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()));
    (err, scale)
}

fn compare(linear: &str, rows: usize, cols: usize, batches: &[usize]) {
    let mut rng = Lcg(375);
    let m = Matrix::random(rows, cols, &mut rng);
    let most = *batches.iter().max().unwrap();
    let (x, hi, lo) = planes(most, cols, &mut rng);
    let exact = cpu(&m, &x, most);
    let g = gpu(linear);
    let act = linear == "gate_up";
    let splits = f8b_splits(rows, cols, g.sms);
    let mut d = Device {
        w: g.stream.clone_htod(&m.values).unwrap(),
        s: g.stream.clone_htod(&m.device_scales()).unwrap(),
        hi: g.stream.clone_htod(&hi).unwrap(),
        lo: g.stream.clone_htod(&lo).unwrap(),
        out: g.stream.alloc_zeros(most * rows).unwrap(),
        oh: g.stream.alloc_zeros(most * rows / 2).unwrap(),
        ol: g.stream.alloc_zeros(most * rows / 2).unwrap(),
        part: g.stream.alloc_zeros(splits * most * rows).unwrap(),
        dummy: g.stream.alloc_zeros(1).unwrap(),
    };
    let bytes = (m.values.len() + rows * 4) as f64;
    for &tokens in batches {
        let exact = &exact[..tokens * rows];
        let exact = if act { swiglu(exact, tokens, rows) } else { exact.to_vec() };
        let old = run_tiny(&g, &mut d, rows, tokens, act);
        let new = run_batch(&g, &mut d, rows, cols, tokens, act);
        let (old_err, scale) = max_err(&old.out, &exact);
        let (new_err, _) = max_err(&new.out, &exact);
        eprintln!(
            "{linear} {rows}x{cols} batch {tokens}: tile {:.1} us ({:.0} GB/s, max err {old_err:.2e}), batched {:.1} us ({:.0} GB/s, max err {new_err:.2e}, splits {splits}), max abs {scale:.1}",
            old.micros,
            bytes / old.micros / 1e3,
            new.micros,
            bytes / new.micros / 1e3,
        );
        assert!(old_err <= ERROR_BOUND * scale, "tile path {old_err} of {scale}");
        assert!(new_err <= ERROR_BOUND * scale, "batched path {new_err} of {scale}");
    }
}

const BATCHES: [usize; 7] = [1, 2, 3, 4, 8, 13, 16];

#[test]
#[ignore = "needs an sm_120 GPU"]
fn batched_fp8_lin_in_matches_the_tile_path_on_a_random_27b_matrix() {
    compare("lin_in", 10240 + 6144, 5120, &BATCHES);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn batched_fp8_lin_out_matches_the_tile_path_on_a_random_27b_matrix() {
    compare("lin_out", 5120, 6144, &BATCHES);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn batched_fp8_qkv_matches_the_tile_path_on_a_random_27b_matrix() {
    compare("qkv", 12288 + 2048, 5120, &BATCHES);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn batched_fp8_o_matches_the_tile_path_on_a_random_27b_matrix() {
    compare("o", 5120, 6144, &BATCHES);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn batched_fp8_gate_up_matches_the_tile_path_on_a_random_27b_matrix() {
    compare("gate_up", 2 * 17408, 5120, &BATCHES);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn batched_fp8_down_matches_the_tile_path_on_a_random_27b_matrix() {
    compare("down", 5120, 17408, &BATCHES);
}
