use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::{Module, WeightKind};
use specrt::dequant::{BlockFormat, Elem, ScaleKind};
use specrt::gemm::GemmPath;
use specrt::jit::Jit;
use specrt::nvfp4::{Fp4, e2m1_to_f32};
use specrt::spec::{Quant, Spec};
use specrt::weights::{bf16_to_f32, e4m3_to_f32, f32_to_bf16};
use std::borrow::Cow;
use std::path::Path;

const NVFP4: BlockFormat = BlockFormat { elem: Elem::E2m1, scale: ScaleKind::E4m3, group: 16 };

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

struct Case {
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    weight_scale_2: f32,
}

impl Case {
    fn new(rows: usize, cols: usize) -> Case {
        let mut rng = Lcg(1190);
        let packed = (0..rows * cols / 2).map(|_| rng.next() as u8).collect();
        let scales = (0..rows * cols / 16).map(|_| 0x28 + (rng.next() % 0x40) as u8).collect();
        Case { rows, cols, packed, scales, weight_scale_2: 1.0 / 1500.0 }
    }

    fn reference(&self, r: usize, c: usize) -> f32 {
        let b = self.packed[(r * self.cols + c) / 2];
        let nibble = if c.is_multiple_of(2) { b & 15 } else { b >> 4 };
        bf16_to_f32(f32_to_bf16(e2m1_to_f32(nibble) * (e4m3_to_f32(self.scales[(r * self.cols + c) / 16]) * self.weight_scale_2)))
    }
}

#[test]
fn the_w4a16_decode_is_the_nvfp4_block_format_times_weight_scale_2() {
    let c = Case::new(32, 64);
    let fp4 = Fp4::new(Cow::Borrowed(&c.packed), Cow::Borrowed(&c.scales), 1.0 / c.weight_scale_2, c.rows, c.cols).unwrap();
    let mut host = Vec::new();
    NVFP4.decode(&c.packed, &c.scales, &mut host).unwrap();
    for r in 0..c.rows {
        let row = fp4.dequant_row(r);
        for col in 0..c.cols {
            let want = c.reference(r, col);
            assert_eq!(bf16_to_f32(f32_to_bf16(row[col])), want, "Fp4 row {r} col {col}");
            assert_eq!(bf16_to_f32(f32_to_bf16(host[r * c.cols + col] * c.weight_scale_2)), want);
        }
    }
    let k = NVFP4.kernel(&kir::Target::sm_120()).unwrap();
    let src = kir::cuda::emit_block_dequant(&k).unwrap();
    assert!(src.contains("e2m1_f32(b & 15u) * sc") && src.contains("return e4m3_f32(b);"));
    assert_eq!(k.group_bytes(), 8);
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn w4a16_nvfp4_decoded_by_kir_then_the_bf16_gemm_matches_the_bf16_reference() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.quant = Quant::Bf16;
    let (n, k) = (spec.dims.hidden, spec.o_dim());
    let c = Case::new(n, k);
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let cache = std::env::temp_dir().join(format!("specrt-w4a16-{}", std::process::id()));
    let w = NVFP4.decode_on_device(&stream, &cache, &c.packed, &c.scales, c.weight_scale_2).unwrap();
    let decoded = stream.clone_dtoh(&w).unwrap();
    for (i, &v) in decoded.iter().enumerate() {
        assert_eq!(bf16_to_f32(v), c.reference(i / k, i % k), "decoded element {i}");
    }
    let module = Module::generate(&spec, 256).unwrap();
    let loaded = Jit::new(ctx.clone(), cache.clone()).unwrap().load(&module).unwrap();
    let mut rng = Lcg(7);
    let tokens = 33;
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.float() * 3.0).collect();
    let hi: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
    let lo: Vec<u16> = x.iter().zip(&hi).map(|(&v, &h)| f32_to_bf16(v - bf16_to_f32(h))).collect();
    let xs: Vec<f64> = if spec.split_activations() {
        hi.iter().zip(&lo).map(|(&h, &l)| (bf16_to_f32(h) + bf16_to_f32(l)) as f64).collect()
    } else {
        hi.iter().map(|&h| bf16_to_f32(h) as f64).collect()
    };
    let choice = specrt::gemm::select(&spec, &target, "o", tokens, WeightKind::Bf16).unwrap();
    assert_eq!(choice.path, GemmPath::Mma, "{}", choice.name);
    assert!(choice.quant.is_none(), "activations stay bf16");
    let f = loaded.function(&choice.name).unwrap();
    let smem = module.smem[choice.name.as_str()];
    use cudarc::driver::sys::CUfunction_attribute as A;
    f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
    let dh = stream.clone_htod(&hi).unwrap();
    let dl = stream.clone_htod(&lo).unwrap();
    let dummy = stream.alloc_zeros::<f32>(1).unwrap();
    let oh = stream.alloc_zeros::<u16>(1).unwrap();
    let ol = stream.alloc_zeros::<u16>(1).unwrap();
    let mut out = stream.alloc_zeros::<f32>(tokens * n).unwrap();
    let t = tokens as i32;
    let tile = choice.tile;
    let cfg = LaunchConfig { grid_dim: (n.div_ceil(tile.bn) as u32, tokens.div_ceil(tile.bm) as u32, 1), block_dim: (tile.threads(), 1, 1), shared_mem_bytes: smem };
    let mut b = stream.launch_builder(&f);
    b.arg(&w).arg(&dummy).arg(&dummy).arg(&dh).arg(&dl).arg(&mut out).arg(&oh).arg(&ol).arg(&t);
    unsafe { b.launch(cfg) }.unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    let (mut err, mut scale) = (0f64, 0f64);
    for tok in 0..tokens {
        for r in 0..n {
            let want: f64 = (0..k).map(|i| c.reference(r, i) as f64 * xs[tok * k + i]).sum();
            scale = scale.max(want.abs());
            err = err.max((got[tok * n + r] as f64 - want).abs());
        }
    }
    eprintln!("{}: max err {err:.3e} of {scale:.3e}", choice.name);
    assert!(err <= 1e-5 * scale, "{err} of {scale}");
    std::fs::remove_dir_all(&cache).ok();
}
