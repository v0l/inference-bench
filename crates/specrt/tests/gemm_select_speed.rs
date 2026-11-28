use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use specrt::codegen::{BATCH_REDUCE_BLOCK, FP8_BLOCK, GemmPath, KernelChoice, Module};
use specrt::gemm::WeightKind;
use specrt::jit::{Jit, Loaded};
use specrt::nvfp4::{GROUP as FP4_GROUP, SCALE_HEADER};
use specrt::spec::{Quant, Spec};
use specrt::weights::fp8_scales;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

const TOKENS: &str = "1,2,3,4,6,8,9,12,16,17,24,32,48,64,96";
const LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
const ROTATE_BYTES: usize = 2 << 30;
const CHAIN: usize = 32;
const REPS: usize = 7;

fn fixture_spec(name: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.toml"))).unwrap()
}

fn four_b(act: bool) -> Spec {
    let text = include_str!("../scripts/qwen3.5-4b-nvfp4.toml.in").replace("@CHECKPOINT@", "qwen3.5-4b-nvfp4");
    let text = if act { text } else { text.lines().filter(|l| !l.starts_with("act_quant")).collect::<Vec<_>>().join("\n") };
    let mut s = Spec::parse(&text).unwrap();
    if !act {
        s.name = "qwen3.5-4b-nvfp4-exact".into();
    }
    s
}

fn list(var: &str, default: &str) -> Vec<usize> {
    std::env::var(var).unwrap_or_else(|_| default.into()).split(',').map(|t| t.parse().unwrap()).collect()
}

struct Rng(u64);

impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 24) as u8
    }
}

fn fp8_values(len: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..len).map(|_| r.byte() & 0xf7).collect()
}

fn planes(len: usize, seed: u64) -> Vec<u16> {
    let mut r = Rng(seed | 1);
    (0..len).map(|_| specrt::weights::f32_to_bf16((r.byte() as f32 - 128.0) / 40.0)).collect()
}

enum Weights {
    Fp8 { values: CudaSlice<u8>, scales: CudaSlice<f32> },
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8> },
}

impl Weights {
    fn new(stream: &Arc<CudaStream>, fp4: bool, n: usize, k: usize, seed: u64) -> Weights {
        if fp4 {
            let mut s = (1.0f32 / 3000.0).to_le_bytes().to_vec();
            s.extend(1.0f32.to_le_bytes());
            s.resize(SCALE_HEADER, 0);
            let mut r = Rng(seed + 7);
            s.extend((0..n * k / FP4_GROUP).map(|_| 0x28 + r.byte() % 0x50));
            let mut r = Rng(seed);
            let packed: Vec<u8> = (0..n * k / 2).map(|_| r.byte()).collect();
            Weights::Fp4 { packed: stream.clone_htod(&packed).unwrap(), scales: stream.clone_htod(&s).unwrap() }
        } else {
            let kb = k / FP8_BLOCK;
            let grid = vec![1.0f32 / 2000.0; n * kb];
            let map: Vec<u32> = (0..n).map(|r| (r * kb) as u32).collect();
            Weights::Fp8 { values: stream.clone_htod(&fp8_values(n * k, seed)).unwrap(), scales: stream.clone_htod(&fp8_scales(&grid, &map)).unwrap() }
        }
    }

    fn bytes(fp4: bool, n: usize, k: usize) -> usize {
        if fp4 { n * k / 2 + n * k / FP4_GROUP } else { n * k + n * k / FP8_BLOCK * 4 }
    }
}

struct Bufs {
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
    x: CudaSlice<f32>,
    nw: CudaSlice<f32>,
    out: CudaSlice<f32>,
    oh: CudaSlice<u16>,
    ol: CudaSlice<u16>,
    part: CudaSlice<f32>,
    q: CudaSlice<u8>,
    qscale: CudaSlice<f32>,
    qs4: CudaSlice<u8>,
    dummy: CudaSlice<f32>,
}

struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    target: kir::Target,
    loaded: Loaded,
    module: Module,
    funcs: std::cell::RefCell<BTreeMap<String, CudaFunction>>,
}

impl Gpu {
    fn func(&self, name: &str, smem: u32) -> CudaFunction {
        self.funcs
            .borrow_mut()
            .entry(name.to_string())
            .or_insert_with(|| {
                let f = self.loaded.function(name).unwrap();
                use cudarc::driver::sys::CUfunction_attribute as A;
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
                f
            })
            .clone()
    }
}

fn cfg(grid: (u32, u32), block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

fn weight_args<'a>(b: &mut cudarc::driver::LaunchArgs<'a>, w: &'a Weights) {
    match w {
        Weights::Fp8 { values, scales } => b.arg(values).arg(scales),
        Weights::Fp4 { packed, scales } => b.arg(packed).arg(scales),
    };
}

fn launch_choice(g: &Gpu, spec: &Spec, c: &KernelChoice, w: &Weights, d: &Bufs, n: usize, k: usize, tokens: usize) {
    let s = &g.stream;
    let tokens_i = tokens as i32;
    match c.path {
        GemmPath::F8a => {
            let quant = g.func(c.quant.unwrap(), 0);
            let width = k as i32;
            let mut b = s.launch_builder(&quant);
            b.arg(&d.hi).arg(&d.lo).arg(&d.q).arg(&d.qscale).arg(&width);
            let qc = if spec.quant == Quant::Fp8Channel { cfg((tokens as u32, 1), 256, 0) } else { cfg(((k / FP8_BLOCK) as u32, tokens as u32), (FP8_BLOCK / 2) as u32, 0) };
            unsafe { b.launch(qc) }.unwrap();
            let smem = g.module.smem[c.name.as_str()];
            let f = g.func(&c.name, smem);
            let mut b = s.launch_builder(&f);
            weight_args(&mut b, w);
            b.arg(&d.dummy).arg(&d.q).arg(&d.qscale).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&tokens_i);
            unsafe { b.launch(cfg((n.div_ceil(c.tile.bn) as u32, tokens.div_ceil(c.tile.bm) as u32), c.tile.threads(), smem)) }.unwrap();
        }
        GemmPath::F4a => {
            let Weights::Fp4 { packed, scales } = w else { unreachable!() };
            let quant = g.func(c.quant.unwrap(), 0);
            let width = k as i32;
            let mut b = s.launch_builder(&quant);
            b.arg(&d.hi).arg(&d.lo).arg(scales).arg(&d.q).arg(&d.qs4).arg(&width);
            unsafe { b.launch(cfg(((k / FP4_GROUP).div_ceil(128) as u32, tokens as u32), 128, 0)) }.unwrap();
            let smem = g.module.smem[c.name.as_str()];
            let f = g.func(&c.name, smem);
            let mut b = s.launch_builder(&f);
            b.arg(packed).arg(scales).arg(&d.q).arg(&d.qs4).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&tokens_i);
            unsafe { b.launch(cfg((n.div_ceil(c.tile.bn) as u32, tokens.div_ceil(c.tile.bm) as u32), c.tile.threads(), smem)) }.unwrap();
        }
        GemmPath::F8s | GemmPath::F4s => {
            let shape = c.kernel.as_ref().unwrap().swap_shape().unwrap();
            let fp4 = c.path == GemmPath::F4s;
            let quant = g.func(c.quant.unwrap(), 0);
            let width = k as i32;
            let mut b = s.launch_builder(&quant);
            if fp4 {
                let Weights::Fp4 { scales, .. } = w else { unreachable!() };
                b.arg(&d.hi).arg(&d.lo).arg(scales).arg(&d.q).arg(&d.qs4).arg(&width);
                unsafe { b.launch(cfg(((k / FP4_GROUP).div_ceil(128) as u32, tokens as u32), 128, 0)) }.unwrap();
            } else {
                b.arg(&d.hi).arg(&d.lo).arg(&d.q).arg(&d.qscale).arg(&width);
                let qc = if spec.quant == Quant::Fp8Channel { cfg((tokens as u32, 1), 256, 0) } else { cfg(((k / FP8_BLOCK) as u32, tokens as u32), (FP8_BLOCK / 2) as u32, 0) };
                unsafe { b.launch(qc) }.unwrap();
            }
            let splits = shape.splits(&g.target, n, k);
            let splits_i = splits as i32;
            let f = g.func(&c.name, 0);
            let mut b = s.launch_builder(&f);
            weight_args(&mut b, w);
            if !fp4 {
                b.arg(&d.dummy);
            }
            b.arg(&d.q);
            if fp4 {
                b.arg(&d.qs4);
            } else {
                b.arg(&d.qscale);
            }
            b.arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&d.part).arg(&tokens_i).arg(&splits_i);
            unsafe { b.launch(cfg((n.div_ceil(shape.rows()) as u32, splits as u32), shape.threads(), 0)) }.unwrap();
            if splits > 1 {
                let reduce = g.func(c.reduce.as_ref().unwrap(), 0);
                let mut b = s.launch_builder(&reduce);
                b.arg(&d.part);
                if !fp4 {
                    b.arg(&d.dummy);
                }
                b.arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&tokens_i).arg(&splits_i);
                unsafe { b.launch(cfg(((tokens * n).div_ceil(BATCH_REDUCE_BLOCK) as u32, 1), BATCH_REDUCE_BLOCK as u32, 0)) }.unwrap();
            }
        }
        GemmPath::F8b | GemmPath::F4b => {
            let shape = c.kernel.as_ref().unwrap().batch_shape().unwrap();
            let splits = shape.splits(&g.target, n, k);
            let splits_i = splits as i32;
            let f = g.func(&c.name, shape.smem());
            let mut b = s.launch_builder(&f);
            weight_args(&mut b, w);
            if c.path == GemmPath::F8b {
                b.arg(&d.dummy);
            }
            b.arg(&d.hi).arg(&d.lo).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&d.part).arg(&tokens_i).arg(&splits_i);
            unsafe { b.launch(cfg((n.div_ceil(shape.rows()) as u32, splits as u32), shape.threads(), shape.smem())) }.unwrap();
            if splits > 1 {
                let reduce = g.func(c.reduce.as_ref().unwrap(), 0);
                let mut b = s.launch_builder(&reduce);
                b.arg(&d.part);
                if c.path == GemmPath::F8b {
                    b.arg(&d.dummy);
                }
                b.arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&tokens_i).arg(&splits_i);
                unsafe { b.launch(cfg(((tokens * n).div_ceil(BATCH_REDUCE_BLOCK) as u32, 1), BATCH_REDUCE_BLOCK as u32, 0)) }.unwrap();
            }
        }
        _ => {
            let smem = g.module.smem[c.name.as_str()];
            let f = g.func(&c.name, smem);
            let mut b = s.launch_builder(&f);
            weight_args(&mut b, w);
            b.arg(&d.dummy).arg(&d.hi).arg(&d.lo).arg(&d.out).arg(&d.oh).arg(&d.ol).arg(&tokens_i);
            unsafe { b.launch(cfg((n.div_ceil(c.tile.bn) as u32, tokens.div_ceil(c.tile.bm) as u32), c.tile.threads(), smem)) }.unwrap();
        }
    }
}

fn launch_gemv(g: &Gpu, spec: &Spec, linear: &str, kind: WeightKind, w: &Weights, d: &Bufs) -> Option<String> {
    let (normed, l) = [false, true].into_iter().find_map(|normed| specrt::gemm::gemv(spec, linear, kind, normed).map(|l| (normed, l)))?;
    let f = g.func(l.name, 0);
    let mut b = g.stream.launch_builder(&f);
    weight_args(&mut b, w);
    if l.bias {
        b.arg(&d.dummy);
    }
    b.arg(&d.x);
    if normed {
        b.arg(&d.nw);
    }
    b.arg(&d.out);
    unsafe { b.launch(cfg((l.grid(g.target.sm_count), 1), l.threads, 0)) }.unwrap();
    Some(l.name.to_string())
}

fn time_chain(g: &Gpu, copies: usize, mut launch: impl FnMut(usize)) -> f64 {
    for c in 0..copies {
        launch(c);
    }
    g.stream.synchronize().unwrap();
    let ev = || {
        let e = g.ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
        e.record(&g.stream).unwrap();
        e
    };
    let mut t: Vec<f64> = (0..REPS)
        .map(|_| {
            let a = ev();
            for i in 0..CHAIN {
                launch(i % copies);
            }
            let b = ev();
            a.elapsed_ms(&b).unwrap() as f64 * 1e3 / CHAIN as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn weight_kinds(spec: &Spec, linear: &str) -> Vec<WeightKind> {
    let mut out = vec![WeightKind::Fp8];
    if matches!(linear, "gate_up" | "down") && spec.mlp.quant == Some(Quant::Nvfp4G16) {
        out.insert(0, WeightKind::Fp4);
    }
    out
}

struct Row {
    model: &'static str,
    linear: &'static str,
    weight: &'static str,
    tokens: usize,
    kernel: String,
    us: f64,
}

fn kind_label(k: WeightKind) -> &'static str {
    if k == WeightKind::Fp4 { "nvfp4" } else { "fp8" }
}

fn sweep(model: &'static str, spec: &Spec, ctx: &Arc<CudaContext>, rows: &mut Vec<Row>) {
    let target = specrt::codegen::cuda_target(ctx).unwrap();
    let module = Module::generate_kv(spec, 256, 2, &target).unwrap();
    let jit = Jit::new(ctx.clone(), specrt::jit::default_cache_dir()).unwrap();
    let loaded = jit.load(&module).unwrap();
    let stream = ctx.default_stream();
    let g = Gpu { ctx: ctx.clone(), stream: stream.clone(), target, loaded, module, funcs: Default::default() };
    let tokens_list = list("SPECRT_SWEEP_TOKENS", TOKENS);
    let most = *tokens_list.iter().max().unwrap();
    for linear in LINEARS {
        let s = specrt::gemm::linear_shape(spec, linear);
        let (n, k) = (s.n, s.k);
        let part = kir::gemm::batch_part_len(&g.target, n, k);
        let d = Bufs {
            hi: stream.clone_htod(&planes(most * k, 1)).unwrap(),
            lo: stream.clone_htod(&planes(most * k, 2)).unwrap(),
            x: stream.clone_htod(&vec![0.01f32; k]).unwrap(),
            nw: stream.clone_htod(&vec![1.0f32; k]).unwrap(),
            out: stream.alloc_zeros(most * n).unwrap(),
            oh: stream.alloc_zeros(most * n).unwrap(),
            ol: stream.alloc_zeros(most * n).unwrap(),
            part: stream.alloc_zeros(part.max(1)).unwrap(),
            q: stream.alloc_zeros(most * k).unwrap(),
            qscale: stream.alloc_zeros(most * k / FP8_BLOCK).unwrap(),
            qs4: stream.alloc_zeros(most * k / FP4_GROUP).unwrap(),
            dummy: stream.alloc_zeros(1).unwrap(),
        };
        for kind in weight_kinds(spec, linear) {
            let fp4 = kind == WeightKind::Fp4;
            let copies = ROTATE_BYTES.div_ceil(Weights::bytes(fp4, n, k)).clamp(2, 64);
            let ws: Vec<Weights> = (0..copies).map(|c| Weights::new(&stream, fp4, n, k, 11 + c as u64)).collect();
            for &tokens in &tokens_list {
                if tokens == 1 {
                    let mut name = None;
                    let us = time_chain(&g, copies, |c| name = launch_gemv(&g, spec, linear, kind, &ws[c], &d));
                    if let Some(kernel) = name {
                        rows.push(Row { model, linear, weight: kind_label(kind), tokens, kernel, us });
                    }
                }
                for c in specrt::gemm::candidates(spec, &g.target, linear, tokens, kind) {
                    let us = time_chain(&g, copies, |i| launch_choice(&g, spec, &c, &ws[i], &d, n, k, tokens));
                    eprintln!("{model} {linear} {} m={tokens} {}: {us:.1} us", kind_label(kind), c.name);
                    rows.push(Row { model, linear, weight: kind_label(kind), tokens, kernel: c.name.clone(), us });
                }
            }
        }
    }
}

fn eligible(spec: &Spec, t: &kir::Target, linear: &str, kind: WeightKind, tokens: usize) -> Vec<String> {
    specrt::gemm::candidates(spec, t, linear, tokens, kind).into_iter().map(|c| c.name).collect()
}

fn report(rows: &[Row], specs: &[(&'static str, Spec, Spec)], t: &kir::Target) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "model\tlinear\tweight\ttokens\tkernel\tus");
    for r in rows {
        let _ = writeln!(out, "{}\t{}\t{}\t{}\t{}\t{:.1}", r.model, r.linear, r.weight, r.tokens, r.kernel, r.us);
    }
    let _ = writeln!(out);
    for (model, act, exact) in specs {
        for (label, spec) in [("act", act), ("exact", exact)] {
            let mut keys: Vec<(&str, &str, usize)> = rows.iter().filter(|r| r.model == *model).map(|r| (r.linear, r.weight, r.tokens)).collect();
            keys.dedup();
            for (linear, weight, tokens) in keys {
                let kind = if weight == "nvfp4" { WeightKind::Fp4 } else { WeightKind::Fp8 };
                let ok = eligible(spec, t, linear, kind, tokens);
                let here: Vec<&Row> = rows.iter().filter(|r| r.model == *model && r.linear == linear && r.weight == weight && r.tokens == tokens && ok.contains(&r.kernel)).collect();
                let Some(best) = here.iter().min_by(|a, b| a.us.total_cmp(&b.us)) else { continue };
                let pick = specrt::gemm::select(spec, t, linear, tokens, kind).map(|c| c.name).unwrap_or_default();
                let picked = here.iter().find(|r| r.kernel == pick).map_or(f64::NAN, |r| r.us);
                let _ = writeln!(out, "{model}\t{label}\t{linear}\t{weight}\t{tokens}\tbest {} {:.1}\tselect {pick} {picked:.1}\tratio {:.3}", best.kernel, best.us, picked / best.us);
            }
        }
    }
    out
}

#[test]
#[ignore = "needs an sm_120 GPU; timing only"]
fn gemm_family_sweep_at_decode_row_counts() {
    let ctx = CudaContext::new(0).unwrap();
    let picks = specrt::runtime::load_picks(&specrt::runtime::tune_path(&specrt::jit::default_cache_dir()));
    eprintln!("{picks} measured picks loaded from the kernel cache");
    let specs: Vec<(&'static str, Spec, Spec)> = vec![
        ("27b", fixture_spec("qwen3.8-27b-nvfp4"), fixture_spec("qwen3.8-27b-nvfp4-exact")),
        ("4b", four_b(true), four_b(false)),
    ];
    let mut rows = Vec::new();
    for (model, act, _) in &specs {
        sweep(model, act, &ctx, &mut rows);
    }
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let text = report(&rows, &specs, &target);
    eprintln!("{text}");
    if let Ok(path) = std::env::var("SPECRT_SWEEP_OUT") {
        std::fs::write(path, &text).unwrap();
    }
}
