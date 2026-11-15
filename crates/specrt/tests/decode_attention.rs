use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;
use std::sync::Arc;

fn list(var: &str, default: &str) -> Vec<usize> {
    std::env::var(var).unwrap_or_else(|_| default.into()).split(',').map(|v| v.parse().unwrap()).collect()
}

fn one_layer_27b() -> Spec {
    let mut spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    spec.layers.count = spec.layers.pattern.len();
    spec.mtp = None;
    assert_eq!(spec.kv_layers(), 1);
    spec
}

fn tiny() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.name = "tiny-qwen3.5".into();
    spec.layers.count = spec.layers.pattern.len();
    spec.mtp = None;
    assert_eq!(spec.kv_layers(), 1);
    spec
}

fn one_layer_122b() -> Spec {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.5-122b-a10b-nvfp4.config.json");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(path).unwrap()).unwrap();
    spec.name = "qwen3.5-122b-a10b".into();
    spec.layers.count = spec.layers.pattern.len();
    spec.mtp = None;
    assert_eq!((spec.attention.heads, spec.attention.kv_heads, spec.attention.head_dim), (32, 2, 256));
    assert_eq!(spec.kv_layers(), 1);
    spec
}

fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as f32;
    if exp == 0 {
        return sign * man * 2f32.powi(-24);
    }
    sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn half(&mut self) -> u16 {
        let r = self.next();
        ((r & 1) << 15) as u16 | ((12 + (r >> 1) % 3) << 10) as u16 | ((r >> 4) & 0x3ff) as u16
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

struct Harness {
    spec: Spec,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    attn: CudaFunction,
    combine: CudaFunction,
    smem: u32,
    max_ctx: usize,
    window: std::cell::Cell<i32>,
}

struct Batch {
    rows: usize,
    qlen: usize,
    qkv: CudaSlice<f32>,
    pos: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    table: CudaSlice<i32>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    part_acc: CudaSlice<f32>,
    part_ml: CudaSlice<f32>,
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
}

impl Harness {
    fn new(spec: Spec, max_ctx: usize) -> Harness {
        let ctx = CudaContext::new(0).unwrap();
        let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-decode-attn-{}", std::process::id()))).unwrap();
        let stream = ctx.default_stream();
        let mut module = Module::generate_kv(&spec, max_ctx, 1, &specrt::codegen::cuda_target(&ctx).unwrap()).unwrap();
        let mut smem = module.smem["batch_decode_attention"];
        for (name, var) in [("DM_BK", "SPECRT_DECODE_BK"), ("DM_STAGES", "SPECRT_DECODE_STAGES"), ("DM_CHUNK", "SPECRT_DECODE_CHUNK")] {
            if let Ok(v) = std::env::var(var) {
                let at = module.source.find(&format!("constexpr int {name} = ")).unwrap();
                let end = at + module.source[at..].find(';').unwrap();
                module.source.replace_range(at..end, &format!("constexpr int {name} = {v}"));
            }
        }
        if let (Ok(bk), Ok(st)) = (std::env::var("SPECRT_DECODE_BK"), std::env::var("SPECRT_DECODE_STAGES")) {
            smem = (st.parse::<usize>().unwrap() * 2 * bk.parse::<usize>().unwrap() * spec.attention.head_dim * 2) as u32;
        }
        let loaded = jit.load(&module).unwrap();
        let attn = loaded.function("batch_decode_attention").unwrap();
        let combine = loaded.function("batch_decode_combine").unwrap();
        use cudarc::driver::sys::CUfunction_attribute as A;
        attn.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
        Harness { spec, ctx, stream, attn, combine, smem, max_ctx, window: std::cell::Cell::new(0) }
    }

    fn pages(&self) -> usize {
        self.max_ctx.div_ceil(specrt::codegen::PAGE)
    }

    fn block_elems(&self) -> usize {
        let a = &self.spec.attention;
        a.kv_heads * specrt::codegen::PAGE * a.head_dim
    }

    fn batch(&self, qkv: &[f32], pos: &[i32], slots: &[i32], qlen: usize, table: &[i32], k: &[u16], v: &[u16]) -> Batch {
        let s = &self.stream;
        self.batch_on(qkv, pos, slots, qlen, table, s.clone_htod(k).unwrap(), s.clone_htod(v).unwrap())
    }

    fn batch_on(&self, qkv: &[f32], pos: &[i32], slots: &[i32], qlen: usize, table: &[i32], k: CudaSlice<u16>, v: CudaSlice<u16>) -> Batch {
        let a = &self.spec.attention;
        let rows = pos.len();
        let parts = rows * a.heads * specrt::codegen::decode_splits(self.max_ctx);
        let s = &self.stream;
        Batch {
            rows,
            qlen,
            qkv: s.clone_htod(qkv).unwrap(),
            pos: s.clone_htod(pos).unwrap(),
            slots: s.clone_htod(slots).unwrap(),
            table: s.clone_htod(table).unwrap(),
            k,
            v,
            part_acc: s.alloc_zeros(parts * a.head_dim).unwrap(),
            part_ml: s.alloc_zeros(parts * 2).unwrap(),
            hi: s.alloc_zeros(rows * self.spec.q_dim()).unwrap(),
            lo: s.alloc_zeros(rows * self.spec.q_dim()).unwrap(),
        }
    }

    fn launch(&self, b: &Batch) {
        let a = &self.spec.attention;
        let layer = 0i32;
        let window = self.window.get();
        let qlen = b.qlen as i32;
        let pages = self.pages() as i32;
        let splits = specrt::codegen::decode_splits(self.max_ctx) as i32;
        let mut l = self.stream.launch_builder(&self.attn);
        l.arg(&b.qkv).arg(&b.k).arg(&b.v).arg(&b.table).arg(&b.pos).arg(&b.slots).arg(&layer).arg(&qlen).arg(&pages).arg(&splits).arg(&b.part_acc).arg(&b.part_ml).arg(&window);
        let config = LaunchConfig {
            grid_dim: (a.kv_heads as u32, (b.rows / b.qlen) as u32, specrt::codegen::decode_splits(self.max_ctx) as u32),
            block_dim: (32 * specrt::codegen::decode_warps(a.heads / a.kv_heads, b.qlen) as u32, 1, 1),
            shared_mem_bytes: self.smem,
        };
        unsafe { l.launch(config) }.unwrap();
        let mut l = self.stream.launch_builder(&self.combine);
        l.arg(&b.part_acc).arg(&b.part_ml).arg(&b.qkv).arg(&b.pos).arg(&qlen).arg(&splits).arg(&b.hi).arg(&b.lo);
        let config = LaunchConfig { grid_dim: (a.heads as u32, b.rows as u32, 1), block_dim: (a.head_dim as u32 / 2, 1, 1), shared_mem_bytes: 0 };
        unsafe { l.launch(config) }.unwrap();
    }

    fn output(&self, b: &Batch) -> Vec<f32> {
        let hi = self.stream.clone_dtoh(&b.hi).unwrap();
        let lo = self.stream.clone_dtoh(&b.lo).unwrap();
        hi.iter().zip(&lo).map(|(&h, &l)| bf16_value(h) + bf16_value(l)).collect()
    }

    fn time(&self, b: &Batch) -> (f32, f32) {
        self.launch(b);
        let mut times = Vec::new();
        for _ in 0..8 {
            let e0 = self.ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            e0.record(&self.stream).unwrap();
            self.launch(b);
            let e1 = self.ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            e1.record(&self.stream).unwrap();
            self.stream.synchronize().unwrap();
            times.push(e0.elapsed_ms(&e1).unwrap());
        }
        times.sort_by(f32::total_cmp);
        (times[0], times[4])
    }
}

fn reference(spec: &Spec, window: usize, qkv: &[f32], pos: &[i32], slots: &[i32], table: &[i32], pages: usize, k: &[u16], v: &[u16]) -> Vec<f32> {
    let a = &spec.attention;
    let (hd, group, qrow) = (a.head_dim, a.heads / a.kv_heads, spec.qkv_rows());
    let page = specrt::codegen::PAGE;
    let gate_off = spec.q_dim() + 2 * spec.kv_dim();
    let at = |slot: i32, t: usize, kvh: usize| -> usize {
        let block = table[slot as usize * pages + t / page] as usize;
        ((block * a.kv_heads + kvh) * page + t % page) * hd
    };
    let mut out = vec![0.0f32; pos.len() * spec.q_dim()];
    for (row, (&p, &slot)) in pos.iter().zip(slots).enumerate() {
        for head in 0..a.heads {
            let kvh = head / group;
            let q = &qkv[row * qrow + head * hd..row * qrow + (head + 1) * hd];
            let scale = a.query_scalar.map_or(1.0 / (hd as f64).sqrt(), |s| 1.0 / s.sqrt());
            let cap = |s: f64| a.softcap.map_or(s, |c| c * (s / c).tanh());
            let scores: Vec<f64> = (0..=p as usize)
                .map(|t| {
                    let kr = &k[at(slot, t, kvh)..at(slot, t, kvh) + hd];
                    let dot = q.iter().zip(kr).map(|(&x, &y)| x as f64 * f16_value(y) as f64).sum::<f64>() * scale;
                    if window > 0 && (t as i64) <= p as i64 - window as i64 { f64::NEG_INFINITY } else { cap(dot) }
                })
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| if *s == f64::NEG_INFINITY { 0.0 } else { (s - m).exp() }).collect();
            let l: f64 = w.iter().sum();
            for d in 0..hd {
                let acc: f64 = w.iter().enumerate().map(|(t, &wt)| wt * f16_value(v[at(slot, t, kvh) + d]) as f64).sum();
                let gate = if a.output_gate { 1.0 / (1.0 + (-(qkv[row * qrow + gate_off + head * hd + d] as f64)).exp()) } else { 1.0 };
                out[row * spec.q_dim() + head * hd + d] = (acc / l * gate) as f32;
            }
        }
    }
    out
}

struct Case {
    lens: Vec<usize>,
    qlen: usize,
}

fn rows_of(case: &Case) -> (Vec<i32>, Vec<i32>) {
    let mut pos = Vec::new();
    let mut slots = Vec::new();
    for (s, &len) in case.lens.iter().enumerate() {
        for j in 0..case.qlen {
            pos.push((len - case.qlen + j) as i32);
            slots.push(s as i32);
        }
    }
    (pos, slots)
}

fn check_against_reference(spec: Spec, max_ctx: usize, cases: &[Case]) {
    check_windowed(spec, max_ctx, 0, cases)
}

fn check_windowed(spec: Spec, max_ctx: usize, window: usize, cases: &[Case]) {
    let h = Harness::new(spec.clone(), max_ctx);
    h.window.set(window as i32);
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for case in cases {
        let seqs = case.lens.len();
        let pages = h.pages();
        let blocks = seqs * pages;
        let mut table: Vec<i32> = (0..blocks as i32).collect();
        for i in (1..table.len()).rev() {
            let j = rng.next() as usize % (i + 1);
            table.swap(i, j);
        }
        let k: Vec<u16> = (0..blocks * h.block_elems()).map(|_| rng.half()).collect();
        let v: Vec<u16> = (0..blocks * h.block_elems()).map(|_| rng.half()).collect();
        let (pos, slots) = rows_of(case);
        let qkv: Vec<f32> = (0..pos.len() * spec.qkv_rows()).map(|_| rng.unit()).collect();
        let b = h.batch(&qkv, &pos, &slots, case.qlen, &table, &k, &v);
        h.launch(&b);
        let got = h.output(&b);
        let want = reference(&spec, window, &qkv, &pos, &slots, &table, pages, &k, &v);
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        let scale = want.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        eprintln!("{} lens={:?} qlen={}: max abs error {worst:.2e} (largest output {scale:.3})", spec.name, case.lens, case.qlen);
        assert!(worst <= 2e-4 * scale.max(1.0), "{} lens={:?} qlen={}: max abs error {worst}", spec.name, case.lens, case.qlen);
        if case.qlen > 1 {
            let single = h.batch(&qkv, &pos, &slots, 1, &table, &k, &v);
            h.launch(&single);
            let alone = h.output(&single);
            let q = spec.q_dim();
            for (row, &p) in pos.iter().enumerate() {
                let group = &pos[row / case.qlen * case.qlen..(row / case.qlen + 1) * case.qlen];
                let longest = *group.iter().max().unwrap() as usize + 1;
                let same_chunk = specrt::codegen::decode_chunk(p as usize + 1, max_ctx) == specrt::codegen::decode_chunk(longest, max_ctx);
                let identical = got[row * q..(row + 1) * q].iter().zip(&alone[row * q..(row + 1) * q]).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    identical || !same_chunk,
                    "{} lens={:?} row {row}: a verify group row differs from the same row run alone",
                    spec.name,
                    case.lens
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn decode_attention_matches_the_reference_across_splits_mixed_lengths_and_verify_groups() {
    let cases = [
        Case { lens: vec![1], qlen: 1 },
        Case { lens: vec![3, 17, 255, 256, 257], qlen: 1 },
        Case { lens: vec![3, 40, 600], qlen: 3 },
        Case { lens: vec![4, 1000, 5000, 129], qlen: 4 },
        Case { lens: vec![16384, 9000, 18], qlen: 3 },
        Case { lens: vec![16397, 7], qlen: 2 },
        Case { lens: vec![16385, 255], qlen: 3 },
    ];
    check_against_reference(one_layer_27b(), 16384 + 256, &cases);
    check_against_reference(tiny(), 16384 + 256, &cases);
    check_against_reference(one_layer_122b(), 16384 + 256, &cases);
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn batch_decode_attention_speed_on_a_27b_shaped_layer() {
    let spec = one_layer_27b();
    let a = spec.attention.clone();
    let max_ctx = list("SPECRT_DECODE_MAX_CTX", "131072")[0];
    let h = Harness::new(spec.clone(), max_ctx);
    let page = specrt::codegen::PAGE;
    let pages = h.pages();
    let pool_mb = list("SPECRT_DECODE_POOL_MB", "1280")[0];
    let pool_cap = pool_mb * (1 << 20) / (2 * h.block_elems() * 2);
    for seqs in list("SPECRT_DECODE_SEQS", "1,4,12") {
        for ctx_len in list("SPECRT_DECODE_CTX", "8192,32768,131072") {
            let needed = seqs * ctx_len.div_ceil(page);
            let pool = needed.min(pool_cap);
            let mut rng = Rng(7);
            let tile: Vec<u16> = (0..h.block_elems() * 61).map(|_| rng.half()).collect();
            let fill = || -> CudaSlice<u16> {
                let mut dev = h.stream.alloc_zeros::<u16>(pool * h.block_elems()).unwrap();
                for chunk in (0..pool).collect::<Vec<_>>().chunks(61) {
                    let at = chunk[0] * h.block_elems();
                    let n = chunk.len() * h.block_elems();
                    h.stream.memcpy_htod(&tile[..n], &mut dev.slice_mut(at..at + n)).unwrap();
                }
                dev
            };
            let (mut k, mut v) = (fill(), fill());
            let used = ctx_len.div_ceil(page);
            let table: Vec<i32> = (0..seqs * pages)
                .map(|i| {
                    let linear = (i / pages * used + (i % pages).min(used - 1)) as u64;
                    ((linear * 1_000_003) % pool as u64) as i32
                })
                .collect();
            for qlen in list("SPECRT_DECODE_QLEN", "1,3") {
                let case = Case { lens: vec![ctx_len; seqs], qlen };
                let (pos, slots) = rows_of(&case);
                let qkv: Vec<f32> = (0..pos.len() * spec.qkv_rows()).map(|_| rng.unit() * 0.2).collect();
                let b = h.batch_on(&qkv, &pos, &slots, qlen, &table, k, v);
                let (min, median) = h.time(&b);
                let bytes = (seqs * ctx_len * a.kv_heads * a.head_dim * 2 * 2) as f64;
                eprintln!(
                    "decode seqs={seqs} ctx={ctx_len} qlen={qlen}{}: min {min:.3} ms median {median:.3} ms, {:.0} GB/s of unique KV at min",
                    if pool < needed { " (aliased pool)" } else { "" },
                    bytes / min as f64 / 1e6
                );
                (k, v) = (b.k, b.v);
            }
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn decode_attention_applies_window_and_softcap_like_the_reference() {
    let cases = [
        Case { lens: vec![3, 40, 600], qlen: 1 },
        Case { lens: vec![200, 1000, 5000], qlen: 3 },
        Case { lens: vec![16397, 70], qlen: 2 },
    ];
    for window in [0usize, 16, 128] {
        let mut spec = tiny();
        spec.attention.softcap = Some(30.0);
        spec.attention.query_scalar = Some(48.0);
        check_windowed(spec, 16384 + 256, window, &cases);
    }
}
