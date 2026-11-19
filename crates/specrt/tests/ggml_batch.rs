use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::{BATCH_REDUCE_BLOCK, BatchShape, Module, linear_dims};
use specrt::gguf::{DeviceRows, GgmlType};
use specrt::jit::Jit;
use specrt::spec::Spec;
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

fn half(v: f32) -> [u8; 2] {
    let b = v.to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (b >> 13) & 0x3ff;
    (((b >> 16) & 0x8000) as u16 | ((exp.clamp(1, 30) as u16) << 10) | mant as u16).to_le_bytes()
}

fn blocks(kind: GgmlType, cols: usize, rng: &mut Lcg) -> Vec<u8> {
    let (elems, bytes) = kind.block().unwrap();
    let count = cols / elems;
    let mut out = Vec::with_capacity(count * bytes);
    for _ in 0..count {
        let mut b: Vec<u8> = (0..bytes).map(|_| rng.next() as u8).collect();
        match kind {
            GgmlType::Q4K => {
                b[..2].copy_from_slice(&half(0.01 + rng.float().abs() * 0.01));
                b[2..4].copy_from_slice(&half(0.01 + rng.float().abs() * 0.01));
            }
            GgmlType::Q8_0 => b[..2].copy_from_slice(&half(0.001 + rng.float().abs() * 0.001)),
            _ => b[208..210].copy_from_slice(&half(0.001 + rng.float().abs() * 0.001)),
        }
        out.extend(b);
    }
    out
}

fn matrix(kinds: &[GgmlType], rows: usize, cols: usize, rng: &mut Lcg) -> DeviceRows<'static> {
    let mut all: Option<DeviceRows<'static>> = None;
    let mut table = Vec::new();
    for r in 0..rows {
        let kind = kinds[r % kinds.len()];
        let data = blocks(kind, cols, rng);
        let one = DeviceRows::pack(kind, &data, 1, cols).unwrap();
        let one = DeviceRows { parts: one.parts.into_iter().map(|p| std::borrow::Cow::Owned(p.into_owned())).collect(), rows: one.rows, cols };
        match all.as_mut() {
            None => {
                table.extend(one.rows.clone());
                all = Some(one);
            }
            Some(a) => table.extend(a.append(one)),
        }
    }
    let mut a = all.unwrap();
    a.rows = table;
    a
}

fn check(linear: &str, kinds: &[GgmlType], token_counts: &[usize]) -> Option<()> {
    let ctx = CudaContext::new(0).ok()?;
    let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b-gguf.toml")).unwrap();
    let (rows, cols, _, _) = linear_dims(&spec, linear);
    let module = Module::generate(&spec, 256).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-ggr-{}-{linear}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let loaded = Jit::new(ctx.clone(), dir.clone()).unwrap().load(&module).unwrap();
    let stream = ctx.default_stream();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let mut rng = Lcg(814);
    let m = matrix(kinds, rows, cols, &mut rng);
    let most = *token_counts.iter().max().unwrap();
    let hi: Vec<u16> = (0..most * cols).map(|_| f32_to_bf16(rng.float() * 3.0)).collect();
    let lo = vec![0u16; hi.len()];
    let x: Vec<f64> = hi.iter().map(|&v| bf16_to_f32(v) as f64).collect();
    let wf: Vec<Vec<f32>> = (0..rows).map(|r| m.row_f32(r)).collect();
    let data = stream.clone_htod(m.data().as_ref()).unwrap();
    let table = stream.clone_htod(&m.rows).unwrap();
    let dhi = stream.clone_htod(&hi).unwrap();
    let dlo = stream.clone_htod(&lo).unwrap();
    let part = specrt::gemm::ggml_part_len(&spec, &target);
    let dpart = stream.alloc_zeros::<f32>(part).unwrap();
    let dummy = stream.alloc_zeros::<f32>(1).unwrap();
    let dummy16 = stream.alloc_zeros::<u16>(1).unwrap();
    for &tokens in token_counts {
        let shape = BatchShape::for_ggml(tokens);
        let splits = shape.ggml_splits(&target, rows, cols);
        let name = format!("{linear}_ggr{}", shape.tokens);
        let f = loaded.function(&name).unwrap();
        use cudarc::driver::sys::CUfunction_attribute as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shape.smem() as i32).unwrap();
        let out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
        let (n, s) = (tokens as i32, splits as i32);
        let mut b = stream.launch_builder(&f);
        b.arg(&data).arg(&table).arg(&dummy).arg(&dhi).arg(&dlo).arg(&out).arg(&dummy16).arg(&dummy16).arg(&dpart).arg(&n).arg(&s);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(shape.rows()) as u32, splits as u32, 1),
                block_dim: (shape.threads(), 1, 1),
                shared_mem_bytes: shape.smem(),
            })
        }
        .unwrap();
        if splits > 1 {
            let reduce = loaded.function(&format!("{linear}_ggr_reduce")).unwrap();
            let mut b = stream.launch_builder(&reduce);
            b.arg(&dpart).arg(&dummy).arg(&out).arg(&dummy16).arg(&dummy16).arg(&n).arg(&s);
            let outs = tokens * rows;
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (outs.div_ceil(BATCH_REDUCE_BLOCK) as u32, 1, 1),
                    block_dim: (BATCH_REDUCE_BLOCK as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        let got = stream.clone_dtoh(&out).unwrap();
        let mut worst = 0f64;
        let mut scale = 0f64;
        for t in 0..tokens {
            for r in 0..rows {
                let want: f64 = wf[r].iter().zip(&x[t * cols..(t + 1) * cols]).map(|(&a, &b)| a as f64 * b).sum();
                scale = scale.max(want.abs());
                worst = worst.max((got[t * rows + r] as f64 - want).abs());
            }
        }
        eprintln!("{linear} {kinds:?} tokens {tokens} splits {splits}: max err {worst:.3e} of {scale:.3}");
        assert!(worst <= 0.01 * scale, "{linear} tokens {tokens}: error {worst} of {scale}");
    }
    std::fs::remove_dir_all(&dir).ok();
    Some(())
}

const TOKENS: [usize; 7] = [1, 3, 8, 9, 16, 24, 32];

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn ggml_batch_rows_match_the_dequantised_reference_on_q4k_rows() {
    check("qkv", &[GgmlType::Q4K], &TOKENS).expect("a CUDA device");
    check("down", &[GgmlType::Q4K], &TOKENS).expect("a CUDA device");
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn ggml_batch_rows_match_the_dequantised_reference_on_q6k_rows() {
    check("qkv", &[GgmlType::Q6K], &TOKENS).expect("a CUDA device");
    check("down", &[GgmlType::Q6K], &TOKENS).expect("a CUDA device");
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn ggml_batch_rows_match_the_dequantised_reference_on_mixed_and_q8_rows() {
    check("o", &[GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0], &TOKENS).expect("a CUDA device");
}
