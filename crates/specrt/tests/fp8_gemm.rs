use cudarc::driver::{CudaContext, PushKernelArg, LaunchConfig};
use specrt::codegen::{FP8_BLOCK, Module, Tile};
use specrt::jit::Jit;
use specrt::spec::{ActQuant, Quant, Spec};
use specrt::weights::{e4m3_to_f32, f32_to_e4m3, fp8_scales};
use std::path::Path;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn spec(quant: Quant) -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-fp8");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.quant = quant;
    spec.act_quant = ActQuant::Fp8;
    spec
}

fn run(quant: Quant, linear: &str, suffix: &str, tokens: usize) -> (Vec<f32>, Vec<f64>) {
    run_with(spec(quant), linear, suffix, tokens, None)
}

fn run_with(spec: Spec, linear: &str, suffix: &str, tokens: usize, raster: Option<usize>) -> (Vec<f32>, Vec<f64>) {
    let quant = spec.quant;
    let (n, k) = match linear {
        "qkv" => (spec.qkv_rows(), spec.dims.hidden),
        _ => (spec.dims.hidden, spec.q_dim()),
    };
    let kb = k / FP8_BLOCK;
    let mut rng = Lcg(7 + tokens as u64);
    let wq: Vec<u8> = (0..n * k).map(|_| f32_to_e4m3(rng.next() * 300.0)).collect();
    let (grid, map): (Vec<f32>, Vec<u32>) = if quant == Quant::Fp8Channel {
        let rows: Vec<f32> = (0..n).map(|_| 0.002 + 0.001 * rng.next().abs()).collect();
        (rows.iter().flat_map(|&s| std::iter::repeat_n(s, kb)).collect(), (0..n).map(|r| (r * kb) as u32).collect())
    } else {
        let rb = n.div_ceil(FP8_BLOCK);
        ((0..rb * kb).map(|_| 0.002 + 0.001 * rng.next().abs()).collect(), (0..n).map(|r| (r / FP8_BLOCK * kb) as u32).collect())
    };
    let scales = fp8_scales(&grid, &map);
    let x: Vec<f32> = (0..tokens * k).map(|i| rng.next() * (1.0 + (i % 7) as f32)).collect();
    let per_token = quant == Quant::Fp8Channel;
    let xs_width = if per_token { 1 } else { kb };
    let mut xq = vec![0u8; tokens * k];
    let mut xs = vec![0f32; tokens * xs_width];
    for t in 0..tokens {
        let span = if per_token { k } else { FP8_BLOCK };
        for b in 0..k / span {
            let r = t * k + b * span..t * k + (b + 1) * span;
            let amax = x[r.clone()].iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            xs[t * xs_width + b] = s;
            for i in r {
                xq[i] = f32_to_e4m3(x[i] / s);
            }
        }
    }
    let mut expect = vec![0f64; tokens * n];
    for t in 0..tokens {
        for c in 0..n {
            let wscale_at = |b: usize| grid[map[c] as usize + b] as f64;
            let mut sum = 0f64;
            for b in 0..kb {
                let mut dot = 0f64;
                for i in b * FP8_BLOCK..(b + 1) * FP8_BLOCK {
                    dot += e4m3_to_f32(xq[t * k + i]) as f64 * e4m3_to_f32(wq[c * k + i]) as f64;
                }
                let sa = if per_token { xs[t] } else { xs[t * kb + b] } as f64;
                sum += dot * sa * wscale_at(b);
            }
            expect[t * n + c] = sum;
        }
    }
    let ctx = CudaContext::new(0).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-fp8-gemm-{quant:?}-{}", std::process::id()));
    let jit = Jit::new(ctx.clone(), dir.clone()).unwrap();
    let mut module = Module::generate(&spec, 256).unwrap();
    if let Some(group) = raster {
        let (g, c) = ("constexpr int RASTER_GROUP = 32;", "constexpr int RASTER_MIN_COLS = 256;");
        assert!(module.source.contains(g) && module.source.contains(c));
        module.source = module
            .source
            .replace(g, &format!("constexpr int RASTER_GROUP = {group};"))
            .replace(c, "constexpr int RASTER_MIN_COLS = 1;");
    }
    let loaded = jit.load(&module).unwrap();
    let kernel = format!("{linear}_f8{suffix}");
    let f = loaded.function(&kernel).unwrap();
    let smem = module.smem[kernel.as_str()];
    use cudarc::driver::sys::CUfunction_attribute as A;
    f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
    let stream = ctx.default_stream();
    let dw = stream.clone_htod(&wq).unwrap();
    let ds = stream.clone_htod(&scales).unwrap();
    let dxq = stream.clone_htod(&xq).unwrap();
    let dxs = stream.clone_htod(&xs).unwrap();
    let bias = stream.alloc_zeros::<f32>(1).unwrap();
    let out = stream.alloc_zeros::<f32>(tokens * n).unwrap();
    let dummy = stream.alloc_zeros::<u16>(1).unwrap();
    let tile = Tile::variant(suffix, &spec.runtime);
    let tokens_i = tokens as i32;
    let mut b = stream.launch_builder(&f);
    b.arg(&dw).arg(&ds).arg(&bias).arg(&dxq).arg(&dxs).arg(&out).arg(&dummy).arg(&dummy).arg(&tokens_i);
    let config = LaunchConfig {
        grid_dim: (n.div_ceil(tile.bn) as u32, tokens.div_ceil(tile.bm) as u32, 1),
        block_dim: (tile.threads(), 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe { b.launch(config) }.unwrap();
    let got = stream.clone_dtoh(&out).unwrap();
    (got, expect)
}

fn check(quant: Quant) {
    for linear in ["qkv", "o"] {
        for suffix in ["", "_big", "_narrow", "_huge"] {
            for tokens in [1, 64, 1024] {
                let (got, expect) = run(quant, linear, suffix, tokens);
                let scale = expect.iter().fold(0f64, |m, v| m.max(v.abs()));
                let error = got.iter().zip(&expect).fold(0f64, |m, (g, e)| m.max((*g as f64 - e).abs()));
                assert!(error <= 1e-5 * scale, "{quant:?} {linear}{suffix} {tokens} rows: {error} against {scale}");
            }
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU with FP8 mma"]
fn fp8_gemm_with_grouped_tile_order_matches_the_f64_product() {
    for quant in [Quant::Fp8Channel, Quant::Fp8Block128] {
        for tokens in [300, 1300] {
            let (got, expect) = run_with(spec(quant), "qkv", "_huge", tokens, Some(3));
            let scale = expect.iter().fold(0f64, |m, v| m.max(v.abs()));
            let error = got.iter().zip(&expect).fold(0f64, |m, (g, e)| m.max((*g as f64 - e).abs()));
            assert!(error <= 1e-5 * scale, "{quant:?} grouped qkv {tokens} rows: {error} against {scale}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU with FP8 mma"]
fn fp8_gemm_with_per_token_scales_matches_the_f64_product() {
    check(Quant::Fp8Channel);
}

#[test]
#[ignore = "needs a CUDA GPU with FP8 mma"]
fn fp8_gemm_with_block_128_scales_matches_the_f64_product() {
    check(Quant::Fp8Block128);
}
