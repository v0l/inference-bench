use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use specrt::codegen::{BATCH_REDUCE_BLOCK, BatchShape, FP8_BLOCK, Module, WeightKind};
use specrt::gemm::GemmPath;
use specrt::jit::Jit;
use specrt::spec::{ActQuant, Quant, Spec};
use specrt::weights::{Checkpoint, Matrix, bf16_to_f32, e4m3_to_f32, f32_to_bf16};
use std::path::{Path, PathBuf};

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

fn spec() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-fp8");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.quant = Quant::Fp8Block128;
    spec.act_quant = ActQuant::None;
    spec
}

struct Case {
    rows: usize,
    cols: usize,
    codes: Vec<u8>,
    grid: Vec<f32>,
}

impl Case {
    fn new(rows: usize, cols: usize) -> Case {
        let mut rng = Lcg(1190);
        let codes = (0..rows * cols)
            .map(|_| {
                let b = rng.next() as u8;
                if b & 0x7f == 0x7f { b & 0x80 } else { b }
            })
            .collect();
        let grid = (0..rows.div_ceil(FP8_BLOCK) * cols.div_ceil(FP8_BLOCK)).map(|i| (1.0 + i as f32) / 1000.0).collect();
        Case { rows, cols, codes, grid }
    }

    fn bf16_weight(&self, r: usize, c: usize) -> f64 {
        let s = self.grid[r / FP8_BLOCK * self.cols.div_ceil(FP8_BLOCK) + c / FP8_BLOCK];
        bf16_to_f32(f32_to_bf16(e4m3_to_f32(self.codes[r * self.cols + c]) * s)) as f64
    }

    fn checkpoint(&self, tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("specrt-fp8pbwo-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let scale: Vec<u8> = self.grid.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (rb, cb) = (self.rows.div_ceil(FP8_BLOCK), self.cols.div_ceil(FP8_BLOCK));
        let embed = vec![0u8; self.rows * self.cols * 2];
        let norm = vec![0u8; self.cols * 4];
        let views = vec![
            ("model.embed_tokens.weight".to_string(), TensorView::new(Dtype::BF16, vec![self.rows, self.cols], &embed).unwrap()),
            ("model.norm.weight".to_string(), TensorView::new(Dtype::F32, vec![self.cols], &norm).unwrap()),
            ("lm_head.weight".to_string(), TensorView::new(Dtype::F8_E4M3, vec![self.rows, self.cols], &self.codes).unwrap()),
            ("lm_head.weight_scale".to_string(), TensorView::new(Dtype::F32, vec![rb, 1, cb, 1], &scale).unwrap()),
        ];
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
        dir
    }

    fn load(&self, quant: Quant) -> Matrix<'static> {
        let dir = self.checkpoint(&format!("{quant:?}"));
        let ck = Box::leak(Box::new(Checkpoint::open(&dir).unwrap()));
        let mut s = spec();
        s.dims.vocab = self.rows;
        s.dims.hidden = self.cols;
        s.dims.tie_embeddings = false;
        s.head_quant = Some(quant);
        let (_, m) = ck.head(&s).unwrap();
        m
    }
}

#[test]
fn an_fp8_pb_wo_scale_loads_as_the_128_block_grid() {
    let c = Case::new(200, 256);
    match c.load(Quant::Fp8Block128) {
        Matrix::Fp8 { values, scales } => {
            assert_eq!(&values[..], &c.codes[..]);
            assert_eq!(&scales[c.rows..], &c.grid[..], "the grid is the squeezed [r/128, c/128] scale");
            assert_eq!(scales[199].to_bits() as usize, c.rows + 2, "row 199 reads block row 1");
        }
        _ => panic!("expected fp8"),
    }
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn the_weight_only_fp8_block_gemm_matches_the_bf16_reference() {
    let s = spec();
    let (n, k) = (s.dims.hidden, s.o_dim());
    let c = Case::new(n, k);
    let Matrix::Fp8 { values, scales } = c.load(Quant::Fp8Block128) else { panic!("expected fp8") };
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let module = Module::generate(&s, 256).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-fp8pbwo-jit-{}", std::process::id()));
    let loaded = Jit::new(ctx.clone(), dir.clone()).unwrap().load(&module).unwrap();
    let stream = ctx.default_stream();
    let mut rng = Lcg(7);
    let most = 96;
    let x: Vec<f32> = (0..most * k).map(|_| rng.float() * 3.0).collect();
    let hi: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
    let lo: Vec<u16> = x.iter().zip(&hi).map(|(&v, &h)| f32_to_bf16(v - bf16_to_f32(h))).collect();
    let xs: Vec<f64> = hi.iter().zip(&lo).map(|(&h, &l)| (bf16_to_f32(h) + bf16_to_f32(l)) as f64).collect();
    let w = stream.clone_htod(&values[..]).unwrap();
    let sc = stream.clone_htod(&scales).unwrap();
    let dh = stream.clone_htod(&hi).unwrap();
    let dl = stream.clone_htod(&lo).unwrap();
    let dummy = stream.alloc_zeros::<f32>(1).unwrap();
    let oh = stream.alloc_zeros::<u16>(most * n).unwrap();
    let ol = stream.alloc_zeros::<u16>(most * n).unwrap();
    let part = stream.alloc_zeros::<f32>(kir::gemm::batch_part_len(&target, n, k).max(1)).unwrap();
    use cudarc::driver::sys::CUfunction_attribute as A;
    let mut paths = Vec::new();
    for tokens in [1usize, 5, 64, 96] {
        let choice = specrt::gemm::select(&s, &target, "o", tokens, WeightKind::Fp8).unwrap();
        assert!(choice.quant.is_none(), "weight only: no activation quant kernel for {}", choice.name);
        let f = loaded.function(&choice.name).unwrap();
        let mut out = stream.alloc_zeros::<f32>(tokens * n).unwrap();
        let t = tokens as i32;
        match choice.path {
            GemmPath::F8b => {
                let shape = BatchShape::for_tokens(tokens);
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shape.smem() as i32).unwrap();
                let splits = shape.splits(&target, n, k);
                let sp = splits as i32;
                let cfg = LaunchConfig { grid_dim: (n.div_ceil(shape.rows()) as u32, splits as u32, 1), block_dim: (shape.threads(), 1, 1), shared_mem_bytes: shape.smem() };
                let mut b = stream.launch_builder(&f);
                b.arg(&w).arg(&sc).arg(&dummy).arg(&dh).arg(&dl).arg(&mut out).arg(&oh).arg(&ol).arg(&part).arg(&t).arg(&sp);
                unsafe { b.launch(cfg) }.unwrap();
                if splits > 1 {
                    let r = loaded.function(choice.reduce.as_deref().unwrap()).unwrap();
                    let cfg = LaunchConfig { grid_dim: ((tokens * n).div_ceil(BATCH_REDUCE_BLOCK) as u32, 1, 1), block_dim: (BATCH_REDUCE_BLOCK as u32, 1, 1), shared_mem_bytes: 0 };
                    let mut b = stream.launch_builder(&r);
                    b.arg(&part).arg(&dummy).arg(&mut out).arg(&oh).arg(&ol).arg(&t).arg(&sp);
                    unsafe { b.launch(cfg) }.unwrap();
                }
            }
            GemmPath::Mma => {
                let smem = module.smem[choice.name.as_str()];
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
                let tile = choice.tile;
                let cfg = LaunchConfig { grid_dim: (n.div_ceil(tile.bn) as u32, tokens.div_ceil(tile.bm) as u32, 1), block_dim: (tile.threads(), 1, 1), shared_mem_bytes: smem };
                let mut b = stream.launch_builder(&f);
                b.arg(&w).arg(&sc).arg(&dummy).arg(&dh).arg(&dl).arg(&mut out).arg(&oh).arg(&ol).arg(&t);
                unsafe { b.launch(cfg) }.unwrap();
            }
            p => panic!("{} takes path {p:?}, expected a weight only fp8 path", choice.name),
        }
        let got = stream.clone_dtoh(&out).unwrap();
        let mut scale = 0f64;
        let mut err = 0f64;
        for t in 0..tokens {
            for r in 0..n {
                let want: f64 = (0..k).map(|i| c.bf16_weight(r, i) * xs[t * k + i]).sum();
                scale = scale.max(want.abs());
                err = err.max((got[t * n + r] as f64 - want).abs());
            }
        }
        eprintln!("{} ({:?}) tokens {tokens}: max err {err:.3e} of {scale:.3e}", choice.name, choice.path);
        assert!(err <= 4e-3 * scale, "{}: {err} of {scale}", choice.name);
        paths.push(choice.path);
    }
    assert!(paths.contains(&GemmPath::F8b) && paths.contains(&GemmPath::Mma), "{paths:?}");
    std::fs::remove_dir_all(&dir).ok();
}
