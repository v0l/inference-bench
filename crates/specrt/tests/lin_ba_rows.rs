use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::{Module, linear_dims};
use specrt::jit::Jit;
use specrt::spec::Spec;
use specrt::weights::{bf16_to_f32, f32_to_bf16};
use std::path::Path;

struct Lcg(u64);

impl Lcg {
    fn float(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    }
}

fn fixture_spec(name: &str) -> Spec {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if name.ends_with(".toml") {
        return Spec::load(&root.join(name)).unwrap();
    }
    let dir = root.join(name);
    Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn check(spec: &Spec, label: &str, token_counts: &[usize]) -> Option<()> {
    let ctx = CudaContext::new(0).ok()?;
    let (rows, cols, _, _) = linear_dims(spec, "lin_ba");
    let module = Module::generate(spec, 256).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-lin-ba-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let loaded = Jit::new(ctx.clone(), dir.clone()).unwrap().load(&module).unwrap();
    let stream = ctx.default_stream();
    let most = *token_counts.iter().max().unwrap();
    let mut rng = Lcg(604);
    let w: Vec<u16> = (0..rows * cols).map(|_| f32_to_bf16(rng.float() * 0.2)).collect();
    let x: Vec<f32> = (0..most * cols).map(|_| rng.float() * 3.0).collect();
    let hi: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
    let lo: Vec<u16> = x.iter().zip(&hi).map(|(&v, &h)| f32_to_bf16(v - bf16_to_f32(h))).collect();
    let dw = stream.clone_htod(&w).unwrap();
    let dh = stream.clone_htod(&hi).unwrap();
    let dl = stream.clone_htod(&lo).unwrap();
    let dummy = stream.clone_htod(&[0.0f32]).unwrap();
    let dummy16 = stream.clone_htod(&[0u16]).unwrap();
    for &tokens in token_counts {
        let mut exact = vec![0f64; tokens * rows];
        for t in 0..tokens {
            for r in 0..rows {
                exact[t * rows + r] = (0..cols)
                    .map(|c| {
                        let xv = bf16_to_f32(hi[t * cols + c]) as f64 + bf16_to_f32(lo[t * cols + c]) as f64;
                        bf16_to_f32(w[r * cols + c]) as f64 * xv
                    })
                    .sum();
            }
        }
        let n = tokens as i32;
        let out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
        let f = loaded.function("lin_ba_rows").unwrap();
        let mut b = stream.launch_builder(&f);
        b.arg(&dw).arg(&dummy).arg(&dummy).arg(&dh).arg(&dl).arg(&out).arg(&dummy16).arg(&dummy16).arg(&n);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (rows as u32, tokens.div_ceil(32) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }
        .unwrap();
        let got = stream.clone_dtoh(&out).unwrap();

        let suffix = if tokens <= 16 { "_tiny" } else { "_narrow" };
        let name = format!("lin_ba_mma{suffix}");
        let tile = specrt::gemm::variant_tile(spec, suffix);
        let smem = module.smem[name.as_str()];
        let tf = loaded.function(&name).unwrap();
        use cudarc::driver::sys::CUfunction_attribute as A;
        tf.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
        let tout = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
        let mut b = stream.launch_builder(&tf);
        b.arg(&dw).arg(&dummy).arg(&dummy).arg(&dh).arg(&dl).arg(&tout).arg(&dummy16).arg(&dummy16).arg(&n);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(tile.bn) as u32, tokens.div_ceil(tile.bm) as u32, 1),
                block_dim: (tile.threads(), 1, 1),
                shared_mem_bytes: smem,
            })
        }
        .unwrap();
        let tile_out = stream.clone_dtoh(&tout).unwrap();

        let scale = exact.iter().fold(0f64, |a, v| a.max(v.abs()));
        let err = |v: &[f32]| v.iter().zip(&exact).fold(0f64, |a, (&g, &e)| a.max((g as f64 - e).abs()));
        let (rows_err, tile_err) = (err(&got), err(&tile_out));
        eprintln!("{label} {rows}x{cols} tokens {tokens}: rows kernel {rows_err:.2e}, tile {tile_err:.2e}, max abs {scale:.2}");
        assert!(rows_err <= 1e-4 * scale, "{label} tokens {tokens}: lin_ba_rows is {rows_err} off, max abs {scale}");
        assert!(tile_err <= 1e-4 * scale, "{label} tokens {tokens}: tile path is {tile_err} off");
        let between = got.iter().zip(&tile_out).fold(0f32, |a, (g, t)| a.max((g - t).abs()));
        assert!(between as f64 <= 2e-4 * scale, "{label} tokens {tokens}: rows and tile differ by {between}");
        assert!(got.iter().all(|v| v.is_finite()) && got.iter().any(|v| *v != 0.0), "{label} tokens {tokens}: degenerate");
    }
    std::fs::remove_dir_all(&dir).ok();
    Some(())
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn lin_ba_rows_matches_the_tile_path_on_a_random_27b_in_proj_ba() {
    let spec = fixture_spec("qwen3.8-27b-nvfp4.toml");
    assert_eq!(linear_dims(&spec, "lin_ba").0, 96);
    check(&spec, "27b", &[1, 2, 8, 16, 17, 24, 32, 33, 36, 48, 64]).expect("a CUDA device");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn lin_ba_rows_matches_the_tile_path_on_the_tiny_fixture() {
    let spec = fixture_spec("tiny-qwen3.5");
    check(&spec, "tiny", &[1, 17, 24, 32, 36, 64]).expect("a CUDA device");
}
