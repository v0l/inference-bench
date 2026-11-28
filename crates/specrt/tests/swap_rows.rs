use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use specrt::codegen::{BATCH_REDUCE_BLOCK, FP8_BLOCK, GemmPath, KernelChoice, Module};
use specrt::gemm::WeightKind;
use specrt::jit::Loaded;
use specrt::nvfp4::{GROUP as FP4_GROUP, SCALE_HEADER};
use specrt::spec::{Quant, Spec};
use specrt::weights::{bf16_to_f32, f32_to_bf16, fp8_scales};
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

fn spec(name: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.toml"))).unwrap()
}

enum Weights {
    Fp8 { values: CudaSlice<u8>, scales: CudaSlice<f32> },
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8> },
}

impl Weights {
    fn new(stream: &Arc<CudaStream>, fp4: bool, block: bool, n: usize, k: usize, rng: &mut Lcg) -> Weights {
        if fp4 {
            let mut s = (1.0f32 / 3000.0).to_le_bytes().to_vec();
            s.extend(0.75f32.to_le_bytes());
            s.resize(SCALE_HEADER, 0);
            s.extend((0..n * k / FP4_GROUP).map(|_| 0x28 + (rng.next() % 0x40) as u8));
            let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
            Weights::Fp4 { packed: stream.clone_htod(&packed).unwrap(), scales: stream.clone_htod(&s).unwrap() }
        } else {
            let kb = if block { k / FP8_BLOCK } else { 1 };
            let grid: Vec<f32> = (0..n * kb).map(|_| (1 + rng.next() % 9) as f32 / 4000.0).collect();
            let map: Vec<u32> = (0..n).map(|r| (r * kb) as u32).collect();
            let values: Vec<u8> = (0..n * k).map(|_| rng.next() as u8 & 0xf7).collect();
            Weights::Fp8 { values: stream.clone_htod(&values).unwrap(), scales: stream.clone_htod(&fp8_scales(&grid, &map)).unwrap() }
        }
    }
}

struct Bufs {
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
    q: CudaSlice<u8>,
    qscale: CudaSlice<f32>,
    qs4: CudaSlice<u8>,
    part: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

fn cfg(grid: (u32, u32), block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

struct Gpu {
    stream: Arc<CudaStream>,
    target: kir::Target,
    loaded: Loaded,
    module: Module,
}

impl Gpu {
    fn func(&self, name: &str, smem: u32) -> CudaFunction {
        let f = self.loaded.function(name).unwrap();
        use cudarc::driver::sys::CUfunction_attribute as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
        f
    }
}

fn weight_args<'a>(b: &mut cudarc::driver::LaunchArgs<'a>, w: &'a Weights) {
    match w {
        Weights::Fp8 { values, scales } => b.arg(values).arg(scales),
        Weights::Fp4 { packed, scales } => b.arg(packed).arg(scales),
    };
}

fn quantise(g: &Gpu, spec: &Spec, c: &KernelChoice, w: &Weights, d: &Bufs, k: usize, tokens: usize) {
    let quant = g.func(c.quant.unwrap(), 0);
    let width = k as i32;
    let mut b = g.stream.launch_builder(&quant);
    match w {
        Weights::Fp4 { scales, .. } => {
            b.arg(&d.hi).arg(&d.lo).arg(scales).arg(&d.q).arg(&d.qs4).arg(&width);
            unsafe { b.launch(cfg(((k / FP4_GROUP).div_ceil(128) as u32, tokens as u32), 128, 0)) }.unwrap();
        }
        Weights::Fp8 { .. } => {
            b.arg(&d.hi).arg(&d.lo).arg(&d.q).arg(&d.qscale).arg(&width);
            let qc = if spec.quant == Quant::Fp8Channel { cfg((tokens as u32, 1), 256, 0) } else { cfg(((k / FP8_BLOCK) as u32, tokens as u32), (FP8_BLOCK / 2) as u32, 0) };
            unsafe { b.launch(qc) }.unwrap();
        }
    }
}

fn scales_arg<'a>(b: &mut cudarc::driver::LaunchArgs<'a>, d: &'a Bufs, fp4: bool) {
    b.arg(&d.q);
    if fp4 {
        b.arg(&d.qs4);
    } else {
        b.arg(&d.qscale);
    }
}

fn run(g: &Gpu, spec: &Spec, c: &KernelChoice, w: &Weights, d: &Bufs, out: &(CudaSlice<f32>, CudaSlice<u16>, CudaSlice<u16>), n: usize, k: usize, tokens: usize) {
    quantise(g, spec, c, w, d, k, tokens);
    let fp4 = matches!(w, Weights::Fp4 { .. });
    let tokens_i = tokens as i32;
    let s = &g.stream;
    if c.path.swapped() {
        let shape = c.kernel.as_ref().unwrap().swap_shape().unwrap();
        let splits = shape.splits(&g.target, n, k);
        let splits_i = splits as i32;
        let f = g.func(&c.name, 0);
        let mut b = s.launch_builder(&f);
        weight_args(&mut b, w);
        if !fp4 {
            b.arg(&d.bias);
        }
        scales_arg(&mut b, d, fp4);
        b.arg(&out.0).arg(&out.1).arg(&out.2).arg(&d.part).arg(&tokens_i).arg(&splits_i);
        unsafe { b.launch(cfg((n.div_ceil(shape.rows()) as u32, splits as u32), shape.threads(), 0)) }.unwrap();
        if splits > 1 {
            let reduce = g.func(c.reduce.as_ref().unwrap(), 0);
            let mut b = s.launch_builder(&reduce);
            b.arg(&d.part);
            if !fp4 {
                b.arg(&d.bias);
            }
            b.arg(&out.0).arg(&out.1).arg(&out.2).arg(&tokens_i).arg(&splits_i);
            unsafe { b.launch(cfg(((tokens * n).div_ceil(BATCH_REDUCE_BLOCK) as u32, 1), BATCH_REDUCE_BLOCK as u32, 0)) }.unwrap();
        }
    } else {
        let smem = g.module.smem[c.name.as_str()];
        let f = g.func(&c.name, smem);
        let mut b = s.launch_builder(&f);
        weight_args(&mut b, w);
        if !fp4 {
            b.arg(&d.bias);
        }
        scales_arg(&mut b, d, fp4);
        b.arg(&out.0).arg(&out.1).arg(&out.2).arg(&tokens_i);
        unsafe { b.launch(cfg((n.div_ceil(c.tile.bn) as u32, tokens.div_ceil(c.tile.bm) as u32), c.tile.threads(), smem)) }.unwrap();
    }
}

fn values(g: &Gpu, out: &(CudaSlice<f32>, CudaSlice<u16>, CudaSlice<u16>), planes: bool) -> Vec<f32> {
    if planes {
        let hi = g.stream.clone_dtoh(&out.1).unwrap();
        let lo = g.stream.clone_dtoh(&out.2).unwrap();
        hi.iter().zip(&lo).map(|(&h, &l)| bf16_to_f32(h) + bf16_to_f32(l)).collect()
    } else {
        g.stream.clone_dtoh(&out.0).unwrap()
    }
}

fn check(spec_name: &str, linear: &str, kind: WeightKind) {
    let spec = spec(spec_name);
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let module = Module::generate_kv(&spec, 256, 2, &target).unwrap();
    let jit = specrt::jit::Jit::new(ctx.clone(), specrt::jit::default_cache_dir()).unwrap();
    let loaded = jit.load(&module).unwrap();
    let stream = ctx.default_stream();
    let g = Gpu { stream: stream.clone(), target, loaded, module };
    let shape = specrt::gemm::linear_shape(&spec, linear);
    let (n, k) = (shape.n, shape.k);
    let fp4 = kind == WeightKind::Fp4;
    let mut rng = Lcg(0x5eed ^ n as u64 ^ (k as u64) << 20);
    let w = Weights::new(&stream, fp4, spec.quant == Quant::Fp8Block128, n, k, &mut rng);
    let most = 64;
    let x: Vec<f32> = (0..most * k).map(|_| rng.float() * 2.5).collect();
    let hi: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
    let lo: Vec<u16> = x.iter().zip(&hi).map(|(&v, &h)| f32_to_bf16(v - bf16_to_f32(h))).collect();
    let bias: Vec<f32> = (0..n).map(|_| rng.float()).collect();
    let d = Bufs {
        hi: stream.clone_htod(&hi).unwrap(),
        lo: stream.clone_htod(&lo).unwrap(),
        q: stream.alloc_zeros(most * k).unwrap(),
        qscale: stream.alloc_zeros(most * k / FP8_BLOCK).unwrap(),
        qs4: stream.alloc_zeros(most * k / FP4_GROUP).unwrap(),
        part: stream.alloc_zeros(kir::gemm::batch_part_len(&g.target, n, k).max(1)).unwrap(),
        bias: stream.clone_htod(&bias).unwrap(),
    };
    let planes = matches!(shape.epi, kir::gemm::Epilogue::Act | kir::gemm::Epilogue::Map);
    let start: Vec<f32> = (0..most * n).map(|_| rng.float()).collect();
    let fresh = || (stream.clone_htod(&start).unwrap(), stream.alloc_zeros::<u16>(most * n).unwrap(), stream.alloc_zeros::<u16>(most * n).unwrap());
    let mut checked = 0;
    for tokens in [2, 3, 8, 9, 12, 17, 24, 40, 48, 64] {
        let all = specrt::gemm::candidates(&spec, &g.target, linear, tokens, kind);
        let reference = all.iter().find(|c| matches!(c.path, GemmPath::F4a | GemmPath::F8a) && c.name.ends_with("_narrow")).unwrap_or_else(|| panic!("{linear} {tokens}: no act tile"));
        let ref_out = fresh();
        run(&g, &spec, reference, &w, &d, &ref_out, n, k, tokens);
        let want = values(&g, &ref_out, planes);
        let swaps: Vec<&KernelChoice> = all.iter().filter(|c| c.path.swapped()).collect();
        assert!(!swaps.is_empty(), "{linear} {tokens}: no swapped rows candidate");
        for c in swaps {
            let o = fresh();
            run(&g, &spec, c, &w, &d, &o, n, k, tokens);
            let got = values(&g, &o, planes);
            let width = if planes { n / 2 } else { n };
            let (mut worst, mut scale) = (0f32, 0f32);
            for i in 0..tokens * width {
                worst = worst.max((got[i] - want[i]).abs());
                scale = scale.max(want[i].abs());
            }
            for i in tokens * width..most * width {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "{spec_name} {} wrote past {tokens} rows", c.name);
            }
            assert!(scale > 0.0);
            assert!(worst <= 1e-4 * scale, "{spec_name} {linear} {} at {tokens} rows: max diff {worst} against {} (scale {scale})", c.name, reference.name);
            checked += 1;
        }
    }
    eprintln!("{spec_name} {linear}: {checked} swapped kernels agree with the act quant tile");
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn swapped_fp4_rows_match_the_act_quant_tile_on_gate_up_and_down() {
    check("qwen3.8-27b-nvfp4", "gate_up", WeightKind::Fp4);
    check("qwen3.8-27b-nvfp4", "down", WeightKind::Fp4);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn swapped_fp8_channel_rows_match_the_act_quant_tile() {
    for linear in ["lin_in", "qkv", "lin_out", "o"] {
        check("qwen3.8-27b-nvfp4", linear, WeightKind::Fp8);
    }
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn swapped_fp8_block_rows_match_the_act_quant_tile() {
    for linear in ["lin_in", "gate_up", "down"] {
        check("qwen3.8-27b", linear, WeightKind::Fp8);
    }
}
