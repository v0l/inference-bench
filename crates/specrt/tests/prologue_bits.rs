use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use specrt::codegen::Module;
use specrt::spec::{Quant, Spec};
use specrt::weights::{Checkpoint, Matrix, Mixer};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FP4_THREADS: u32 = 512;
const GEMV_THREADS: u32 = 256;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn hf(dir: &str) -> Spec {
    let d = fixtures().join(dir);
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    s.name = dir.into();
    s.repo = d.display().to_string();
    s
}

fn with_fp4_head(mut s: Spec) -> Spec {
    s.head_quant = Some(Quant::Nvfp4G16);
    s.name = format!("{}-head-nvfp4", s.name);
    s
}

fn four_b() -> Option<Spec> {
    let dir = std::env::var("SPECRT_4B_DIR").ok().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/specrt-models/Qwen3.5-4B-NVFP4"));
    if !dir.join("model.safetensors.index.json").is_file() {
        return None;
    }
    let text = include_str!("../scripts/qwen3.5-4b-nvfp4.toml.in").replace("@CHECKPOINT@", &dir.display().to_string());
    Some(Spec::parse(&text).unwrap())
}

fn specs() -> Vec<Spec> {
    let mut out = vec![
        hf("tiny-qwen3.5"),
        hf("tiny-qwen3.5-fp8"),
        hf("tiny-qwen3.5-nvfp4"),
        with_fp4_head(hf("tiny-qwen3.5-nvfp4")),
        hf("tiny-qwen3.5-27b"),
        Spec::from_gguf_file(&fixtures().join("qwen35-gguf/tiny-mtp-Q8_0.gguf")).unwrap(),
    ];
    match four_b() {
        Some(s) => {
            out.push(with_fp4_head(s.clone()));
            out.push(s);
        }
        None => eprintln!("no Qwen3.5-4B NVFP4 checkpoint, the 4B cases are skipped"),
    }
    out
}

fn decoder(m: &Matrix) -> &'static str {
    match m {
        Matrix::Bf16(_) => "DecBf16",
        Matrix::Q8 { .. } => "DecQ8",
        Matrix::Fp8 { .. } => "DecFp8",
        Matrix::Fp4(_) => "DecFp4",
        Matrix::Ggml(_) => "DecGgml",
    }
}

fn rows_of(m: &Matrix) -> usize {
    match m {
        Matrix::Ggml(r) => r.rows.len(),
        Matrix::Fp4(f) => f.packed.len() * 2 / f.cols,
        _ => 0,
    }
}

struct Case<'a> {
    fused: &'static str,
    matrix: &'a Matrix<'a>,
    norm: &'a [f32],
    n: usize,
    gated: bool,
}

fn cases<'a>(spec: &Spec, layers: &'a [specrt::weights::Layer<'a>], head: &'a (Vec<f32>, Matrix<'a>)) -> Vec<Case<'a>> {
    let mut out = Vec::new();
    for l in layers {
        match &l.mixer {
            Mixer::Attention(a) => out.push(Case { fused: "qkv_proj_norm", matrix: &a.qkv, norm: &l.ln1, n: spec.qkv_rows(), gated: false }),
            Mixer::Linear(g) => {
                let la = spec.linear_attention.as_ref().unwrap();
                out.push(Case { fused: "lin_in_norm", matrix: &g.in_proj, norm: &l.ln1, n: la.in_proj_rows(), gated: false });
                assert!(matches!(g.ba, Matrix::Bf16(_)), "lin_ba_norm reads bf16 rows");
                out.push(Case { fused: "lin_ba_norm", matrix: &g.ba, norm: &l.ln1, n: la.ba_rows(), gated: false });
            }
            _ => {}
        }
    }
    let l = &layers[0];
    let pre = l.ln_pre_mlp.as_deref().unwrap_or(&l.ln2);
    let fp4 = matches!(l.gate_up, Matrix::Fp4(_));
    out.push(Case { fused: if fp4 { "gate_up_act_f4" } else { "gate_up_act" }, matrix: &l.gate_up, norm: pre, n: 2 * spec.dims.intermediate, gated: true });
    let fp4_head = matches!(head.1, Matrix::Fp4(_));
    out.push(Case { fused: if fp4_head { "lm_head_norm_f4" } else { "lm_head_norm" }, matrix: &head.1, norm: &head.0, n: spec.head_rows().len(), gated: false });
    out
}

fn reference_source(cases: &[Case]) -> String {
    let mut s = String::from("\n");
    if cases.iter().any(|c| matches!(c.matrix, Matrix::Fp4(_))) {
        s.push_str("extern \"C\" __global__ void prologue_bits_fp4_t(const float* x, float* xt) {\n    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xt[fp4_t(i, HIDDEN / FP4_GROUP)] = x[i];\n}\n");
    }
    for c in cases {
        let d = decoder(c.matrix);
        s.push_str(&format!(
            "extern \"C\" __global__ void {}_ref(const void* w, const float* s, const float* b, const float* x, float* out) {{\n    linear_w<{d}, {}, HIDDEN, false, false>(w, s, b, x, out);\n}}\n",
            c.fused, c.n
        ));
    }
    s
}

fn ptx_for(spec: &Spec, module: &Module) -> Ptx {
    let dir = PathBuf::from(std::env::var("PROLOGUE_PTX").unwrap_or_else(|_| std::env::temp_dir().join("specrt-prologue-bits").display().to_string()));
    std::fs::create_dir_all(&dir).unwrap();
    let key = specrt::jit::cache_key(&module.source, "prologue", &[]);
    let path = dir.join(format!("{}-{}.ptx", spec.name.replace('/', "_"), &key[..16]));
    if let Ok(text) = std::fs::read_to_string(&path) {
        return Ptx::from_src(text);
    }
    let arch = if module.arch_features { "compute_120a" } else { "compute_120" };
    let opts = CompileOptions { arch: Some(arch), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = compile_ptx_with_opts(&module.source, opts).unwrap_or_else(|e| panic!("{}: nvrtc {e:?}", spec.name)).to_src();
    std::fs::write(&path, &ptx).unwrap();
    Ptx::from_src(ptx)
}

struct Dev {
    w: CudaSlice<u8>,
    s: CudaSlice<u8>,
}

fn upload(stream: &Arc<CudaStream>, m: &Matrix) -> Dev {
    let bytes = |v: &[u8]| stream.clone_htod(v).unwrap();
    let f32s = |v: &[f32]| bytes(&v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
    match m {
        Matrix::Bf16(w) => Dev { w: bytes(&w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()), s: bytes(&[0; 4]) },
        Matrix::Q8 { values, scales } => Dev { w: bytes(&values.iter().map(|v| *v as u8).collect::<Vec<u8>>()), s: f32s(scales) },
        Matrix::Fp8 { values, scales } => Dev { w: bytes(values), s: f32s(scales) },
        Matrix::Fp4(f) => Dev { w: bytes(&f.packed), s: bytes(&f.device_scales()) },
        Matrix::Ggml(r) => Dev { w: bytes(&r.data()), s: bytes(&r.rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()) },
    }
}

fn cfg(grid: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

fn fp4_plain(v: &[f32]) -> Vec<f32> {
    let kg = v.len() / 16;
    (0..v.len()).map(|k| v[(k & 15) * kg + (k >> 4)]).collect()
}

struct Outcome {
    same_width: bool,
    differing: usize,
    max_rel: f32,
}

fn run_case(stream: &Arc<CudaStream>, m: &Arc<cudarc::driver::CudaModule>, spec: &Spec, c: &Case, x: &[f32]) -> Outcome {
    let f = |name: &str| m.load_function(name).unwrap_or_else(|e| panic!("{}: {name}: {e:?}", spec.name));
    let fp4 = matches!(c.matrix, Matrix::Fp4(_));
    let (fused_fn, norm_fn, ref_fn, act_fn) = (f(c.fused), f("rmsnorm"), f(&format!("{}_ref", c.fused)), f("swiglu"));
    let perm_fn = fp4.then(|| f("prologue_bits_fp4_t"));
    let hidden = spec.dims.hidden;
    let inter = spec.dims.intermediate;
    let threads = if fp4 { FP4_THREADS } else { GEMV_THREADS };
    let warps = threads as usize / 32;
    let dev = upload(stream, c.matrix);
    let dx = stream.clone_htod(x).unwrap();
    let dn = stream.clone_htod(c.norm).unwrap();
    let dummy = stream.clone_htod(&[0.0f32]).unwrap();
    let out_len = if c.gated { inter } else { c.n };
    let fused = stream.alloc_zeros::<f32>(out_len).unwrap();
    let pair = if decoder(c.matrix) == "DecGgml" && c.n.is_multiple_of(2) { 2 } else { 1 };
    let mut b = stream.launch_builder(&fused_fn);
    b.arg(&dev.w).arg(&dev.s);
    if !c.gated {
        b.arg(&dummy);
    }
    b.arg(&dx).arg(&dn).arg(&fused);
    let fused_rows = match (c.fused, c.gated) {
        ("lm_head_norm_f4", _) => warps * if c.n.is_multiple_of(2) { 2 } else { 1 },
        (_, true) => warps,
        _ => warps * pair,
    };
    let grid_rows = if c.gated { inter } else { c.n };
    unsafe { b.launch(cfg(grid_rows.div_ceil(fused_rows), threads)) }.unwrap();
    let fused = stream.clone_dtoh(&fused).unwrap();
    let fused = if c.gated && fp4 { fp4_plain(&fused) } else { fused };
    let chain = |norm_threads: u32| -> Vec<f32> {
        let xn = stream.alloc_zeros::<f32>(hidden).unwrap();
        let mut b = stream.launch_builder(&norm_fn);
        b.arg(&dx).arg(&dn).arg(&xn);
        unsafe { b.launch(cfg(1, norm_threads)) }.unwrap();
        let input = if let Some(perm) = &perm_fn {
            let xt = stream.alloc_zeros::<f32>(hidden).unwrap();
            let mut b = stream.launch_builder(perm);
            b.arg(&xn).arg(&xt);
            unsafe { b.launch(cfg(1, 256)) }.unwrap();
            xt
        } else {
            xn
        };
        let lin = stream.alloc_zeros::<f32>(c.n).unwrap();
        let mut b = stream.launch_builder(&ref_fn);
        b.arg(&dev.w).arg(&dev.s).arg(&dummy).arg(&input).arg(&lin);
        unsafe { b.launch(cfg(c.n.div_ceil(warps * pair), threads)) }.unwrap();
        if !c.gated {
            return stream.clone_dtoh(&lin).unwrap();
        }
        let act = stream.alloc_zeros::<f32>(inter).unwrap();
        let mut b = stream.launch_builder(&act_fn);
        b.arg(&lin).arg(&act);
        unsafe { b.launch(cfg(inter.div_ceil(256), 256)) }.unwrap();
        stream.clone_dtoh(&act).unwrap()
    };
    let same = chain(threads);
    let runtime = chain(1024);
    assert!(fused.iter().any(|v| *v != 0.0) && fused.iter().all(|v| v.is_finite()), "{} {}: fused output is degenerate", spec.name, c.fused);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let scale = runtime.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
    let differing = fused.iter().zip(&runtime).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max_rel = fused.iter().zip(&runtime).map(|(a, b)| (a - b).abs() / scale).fold(0.0, f32::max);
    Outcome { same_width: bits(&fused) == bits(&same), differing, max_rel }
}

fn prepare(spec: &Spec) -> (Module, String) {
    let ck = Checkpoint::for_repo(&spec.repo).unwrap();
    let layers = layer_picks(spec).into_iter().map(|i| ck.layer(spec, i).unwrap()).collect::<Vec<_>>();
    let head = ck.head(spec).unwrap();
    let cs = cases(spec, &layers, &head);
    let refs = reference_source(&cs);
    let mut module = Module::generate(spec, 256).unwrap();
    module.source.edit_text(|s| s.push_str(&refs));
    (module, refs)
}

fn layer_picks(spec: &Spec) -> Vec<usize> {
    let first = |f: &dyn Fn(usize) -> bool| (0..spec.layers.count).find(|&i| f(i));
    let mut picks: Vec<usize> = [first(&|i| spec.block(i).is_linear()), first(&|i| spec.block(i).is_attention())].into_iter().flatten().collect();
    picks.sort();
    picks.dedup();
    if picks.first() != Some(&0) {
        picks.insert(0, 0);
    }
    picks
}

#[test]
#[ignore = "needs NVRTC; with PROLOGUE_COMPILE_ONLY it compiles the PTX without a GPU, otherwise it needs a CUDA GPU"]
fn the_prologue_gemvs_equal_the_unfused_chain_at_the_same_norm_width() {
    let compile_only = std::env::var_os("PROLOGUE_COMPILE_ONLY").is_some();
    let specs = specs();
    let mut compiled = Vec::new();
    for spec in &specs {
        let (module, _) = prepare(spec);
        compiled.push(ptx_for(spec, &module));
    }
    if compile_only {
        eprintln!("compiled {} modules", compiled.len());
        return;
    }
    let ctx = CudaContext::new(0).expect("a CUDA device");
    let stream = ctx.default_stream();
    let mut report = Vec::new();
    for (spec, ptx) in specs.iter().zip(compiled) {
        let m = ctx.load_module(ptx).unwrap();
        let ck = Checkpoint::for_repo(&spec.repo).unwrap();
        let layers = layer_picks(spec).into_iter().map(|i| ck.layer(spec, i).unwrap()).collect::<Vec<_>>();
        let head = ck.head(spec).unwrap();
        let mut seed = 0x9e3779b97f4a7c15u64;
        let x: Vec<f32> = (0..spec.dims.hidden)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
            })
            .collect();
        for c in cases(spec, &layers, &head) {
            let o = run_case(&stream, &m, spec, &c, &x);
            let line = format!(
                "{} {} {} n={} rows={}: same width {}, runtime 1024-thread norm {} of {} differ, max rel {:.2e}",
                spec.name,
                c.fused,
                decoder(c.matrix),
                c.n,
                rows_of(c.matrix),
                if o.same_width { "bit identical" } else { "DIFFERS" },
                o.differing,
                if c.gated { spec.dims.intermediate } else { c.n },
                o.max_rel
            );
            eprintln!("{line}");
            report.push((line, o));
        }
    }
    assert!(report.iter().all(|(_, o)| o.same_width), "a fused kernel differs from the unfused chain at the same norm width");
    assert!(report.iter().all(|(_, o)| o.max_rel < 1e-4), "a fused kernel is far from the runtime's unfused chain");
}
