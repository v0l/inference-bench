use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use specrt::family::ParamSource;
use specrt::kir_kernel::Registry;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const NAME: &str = "tiny-qwen4-exp";
const KERNELS: [&str; 7] = ["ple_ngram_hash", "ple_embed_gather", "ple_key_proj", "ple_value_proj", "ple_gate", "ple_conv", "ple_add"];
const TOLERANCE: f32 = 1e-3;

const PARAMS: &str = r#"
streams = { config = "hc_count" }
hidden = { config = "hidden_size" }
vocab = { config = "vocab_size" }
norm_eps = { config = "rms_norm_eps" }
ngram_size = { config = "ngram_size", default = 3.0 }
heads_per_ngram = { config = "heads_per_ngram", default = 8.0 }
ngram_heads = { expr = "(ngram_size - 1) * heads_per_ngram" }
ngram_base = { config = "ngram_vocab_size_base", default = 20000000.0 }
ngram_seed = { config = "seed", default = 1234.0 }
ngram_eos = { config = "eos_token_id" }
ple_embed_dim = { config = "ple_embed_dim", alt = "hidden_size" }
ple_head_dim = { expr = "ple_embed_dim / ngram_heads" }
ple_conv_taps = { config = "ple_conv_kernel_size", default = 4.0 }
ple_conv_state = { expr = "(ple_conv_taps - 1) * ngram_size" }
ple_gate_scale = { expr = "1 / sqrt(hidden)" }
"#;

const STATES: &str = r#"
[[state]]
name = "ngram_history"
per = "sequence"
layers = "linear"
shape = ["ngram_size - 1"]
dtype = "f32"
rollback = "checkpoint"
owner = "layer:1"

[[state]]
name = "ple_conv"
per = "sequence"
layers = "linear"
shape = ["streams * hidden", "ple_conv_state"]
dtype = "f32"
rollback = "checkpoint"
owner = "layer:1"
"#;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(NAME)
}

fn text_config() -> serde_json::Value {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    v["text_config"].clone()
}

fn params(config: &serde_json::Value, ple_layer: usize) -> BTreeMap<String, f64> {
    let sources: BTreeMap<String, ParamSource> = toml::from_str(PARAMS).unwrap();
    let mut p = specrt::family::read_params(&sources, config).unwrap();
    let ids: Vec<u64> = config["ple_layer_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    p.insert("ple_index".into(), ids.iter().position(|&l| l as usize == ple_layer + 1).unwrap() as f64);
    p.insert("ple_layers".into(), 1.0);
    p
}

fn shape(p: &BTreeMap<String, f64>) -> kir::desc::Shape {
    p.iter().fold(kir::desc::Shape::new(), |s, (k, v)| s.num(k, *v))
}

fn floats(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("reference").join(file)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn reference() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(format!("{NAME}.ple.json"))).unwrap()).unwrap()
}

struct Weights {
    table: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
}

fn weights(layer: usize) -> Weights {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |leaf: &str| -> Vec<f32> {
        let t = st.tensor(&format!("model.language_model.layers.{layer}.ple.{leaf}")).unwrap();
        match t.dtype() {
            safetensors::Dtype::BF16 => t.data().as_chunks::<2>().0.iter().map(|b| specrt::weights::bf16_to_f32(u16::from_le_bytes(*b))).collect(),
            safetensors::Dtype::F32 => t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            d => panic!("{leaf} is {d:?}"),
        }
    };
    Weights {
        table: get("ple_embedding.ngram_embedding.weight"),
        key: get("key_proj.weight"),
        value: get("value_proj.weight"),
        norm_key: get("norm_key.weight"),
        norm_query: get("norm_query.weight"),
        norm_conv: get("norm_conv.weight"),
        conv: get("conv1d.weight"),
    }
}

struct Dims {
    hidden: usize,
    streams: usize,
    heads: usize,
    head_dim: usize,
    embed: usize,
    taps: usize,
    dilation: usize,
    eps: f64,
}

impl Dims {
    fn of(p: &BTreeMap<String, f64>) -> Dims {
        let u = |k: &str| p[k] as usize;
        Dims { hidden: u("hidden"), streams: u("streams"), heads: u("ngram_heads"), head_dim: u("ple_head_dim"), embed: u("ple_embed_dim"), taps: u("ple_conv_taps"), dilation: u("ngram_size"), eps: p["norm_eps"] }
    }

    fn width(&self) -> usize {
        self.streams * self.hidden
    }

    fn span(&self) -> usize {
        (self.taps - 1) * self.dilation
    }
}

struct HostSeq {
    history: Vec<i64>,
    conv: Vec<Vec<f64>>,
}

impl HostSeq {
    fn new(d: &Dims, eos: i64) -> HostSeq {
        HostSeq { history: vec![eos; d.dilation - 1], conv: vec![vec![0.0; d.span()]; d.width()] }
    }
}

fn host_ple(d: &Dims, w: &Weights, hash: &kir::hash::TokenHash, seq: &mut HostSeq, tokens: &[i64], x: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let ids = hash.eval(tokens, &mut seq.history).unwrap();
    let norm = |v: &[f64], weight: &[f32]| -> Vec<f64> {
        let inv = 1.0 / (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64 + d.eps).sqrt();
        v.iter().zip(weight).map(|(a, w)| a * inv * (1.0 + *w as f64)).collect()
    };
    let mut out = Vec::new();
    for (r, _) in tokens.iter().enumerate() {
        let emb: Vec<f64> = (0..d.heads).flat_map(|h| (0..d.head_dim).map(move |j| (h, j))).map(|(h, j)| w.table[ids[r * d.heads + h] as usize * d.head_dim + j] as f64).collect();
        let proj = |m: &[f32], rows: usize| -> Vec<f64> { (0..rows).map(|o| (0..d.embed).map(|i| m[o * d.embed + i] as f64 * emb[i]).sum()).collect() };
        let key = proj(&w.key, d.width());
        let value = proj(&w.value, d.hidden);
        let xr: Vec<f64> = x[r * d.width()..(r + 1) * d.width()].iter().map(|v| *v as f64).collect();
        let mut gated = vec![0.0; d.width()];
        let mut normed = vec![0.0; d.width()];
        for s in 0..d.streams {
            let g = s * d.hidden..(s + 1) * d.hidden;
            let kn = norm(&key[g.clone()], &w.norm_key[g.clone()]);
            let qn = norm(&xr[g.clone()], &w.norm_query[g.clone()]);
            let dot = kn.iter().zip(&qn).map(|(a, b)| a * b).sum::<f64>() / (d.hidden as f64).sqrt();
            let gate = dot.abs().max(1e-6).sqrt() * dot.signum() * f64::from(dot != 0.0);
            let sig = 1.0 / (1.0 + (-gate).exp());
            let gv: Vec<f64> = value.iter().map(|v| sig * v).collect();
            let gn = norm(&gv, &w.norm_conv[g.clone()]);
            gated[g.clone()].copy_from_slice(&gv);
            normed[g].copy_from_slice(&gn);
        }
        for c in 0..d.width() {
            let mut win = seq.conv[c].clone();
            win.push(normed[c]);
            let v: f64 = (0..d.taps).map(|j| w.conv[c * d.taps + j] as f64 * win[j * d.dilation]).sum();
            seq.conv[c] = win[1..].to_vec();
            out.push((gated[c] + v / (1.0 + (-v).exp())) as f32);
        }
    }
    (ids, out)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

struct Phase {
    seqs: Vec<(usize, Vec<i64>, Vec<f32>)>,
}

struct Gpu {
    stream: Arc<CudaStream>,
    module: Arc<cudarc::driver::CudaModule>,
    threads: BTreeMap<&'static str, u32>,
    d: Dims,
    table: CudaSlice<f32>,
    key_w: CudaSlice<f32>,
    value_w: CudaSlice<f32>,
    norm_key: CudaSlice<f32>,
    norm_query: CudaSlice<f32>,
    norm_conv: CudaSlice<f32>,
    conv_w: CudaSlice<f32>,
    history: CudaSlice<f32>,
    conv_state: CudaSlice<f32>,
}

struct Out {
    ids: Vec<i32>,
    ple: Vec<f32>,
    after: Vec<f32>,
}

impl Gpu {
    fn new(p: &BTreeMap<String, f64>, w: &Weights, slots: usize) -> Gpu {
        let ctx = CudaContext::new(0).unwrap();
        let target = specrt::codegen::cuda_target(&ctx).unwrap();
        let registry = Registry::embedded();
        let kernels: Vec<kir::Kernel> = KERNELS.iter().map(|n| registry.get(n).unwrap_or_else(|| panic!("{n} is not embedded")).build(&target, &shape(p)).unwrap_or_else(|e| panic!("{e}"))).collect();
        let threads = KERNELS.iter().zip(&kernels).map(|(n, k)| (*n, k.threads)).collect();
        let module = specrt::jit::load_uncached(&ctx, &kir::cuda::emit_module(&kernels).unwrap()).unwrap();
        let stream = ctx.default_stream();
        let d = Dims::of(p);
        let up = |v: &[f32]| stream.clone_htod(v).unwrap();
        Gpu {
            table: up(&w.table),
            key_w: up(&w.key),
            value_w: up(&w.value),
            norm_key: up(&w.norm_key),
            norm_query: up(&w.norm_query),
            norm_conv: up(&w.norm_conv),
            conv_w: up(&w.conv),
            history: stream.alloc_zeros(slots * (d.dilation - 1)).unwrap(),
            conv_state: stream.alloc_zeros(slots * d.width() * d.span()).unwrap(),
            stream,
            module,
            threads,
            d,
        }
    }

    fn reset(&mut self, slot: usize) {
        let (h, c) = (self.d.dilation - 1, self.d.width() * self.d.span());
        self.stream.memset_zeros(&mut self.history.slice_mut(slot * h..(slot + 1) * h)).unwrap();
        self.stream.memset_zeros(&mut self.conv_state.slice_mut(slot * c..(slot + 1) * c)).unwrap();
    }

    fn run(&mut self, phase: &Phase) -> Out {
        let d = &self.d;
        let r = phase.seqs[0].1.len();
        assert!(phase.seqs.iter().all(|s| s.1.len() == r));
        let seqs = phase.seqs.len();
        let n = seqs * r;
        let tokens: Vec<i32> = phase.seqs.iter().flat_map(|s| s.1.iter().map(|t| *t as i32)).collect();
        let x: Vec<f32> = phase.seqs.iter().flat_map(|s| s.2.iter().copied()).collect();
        let slots: Vec<i32> = phase.seqs.iter().map(|s| s.0 as i32).collect();
        let s = &self.stream;
        let (tok_d, x_d, slots_d) = (s.clone_htod(&tokens).unwrap(), s.clone_htod(&x).unwrap(), s.clone_htod(&slots).unwrap());
        let mut idx: CudaSlice<i32> = s.alloc_zeros(n * d.heads).unwrap();
        let mut emb: CudaSlice<f32> = s.alloc_zeros(n * d.embed).unwrap();
        let mut key: CudaSlice<f32> = s.alloc_zeros(n * d.width()).unwrap();
        let mut value: CudaSlice<f32> = s.alloc_zeros(n * d.hidden).unwrap();
        let mut gated: CudaSlice<f32> = s.alloc_zeros(n * d.width()).unwrap();
        let mut normed: CudaSlice<f32> = s.alloc_zeros(n * d.width()).unwrap();
        let mut conv: CudaSlice<f32> = s.alloc_zeros(n * d.width()).unwrap();
        let mut ple: CudaSlice<f32> = s.alloc_zeros(n * d.width()).unwrap();
        let mut after: CudaSlice<f32> = s.alloc_zeros(n * d.width()).unwrap();
        let (layer, slot_base, slot_step, per) = (0i32, slots[0], 1i32, r as i32);
        let f = |name: &str| self.module.load_function(name).unwrap();
        let cfg = |name: &str, grid: (u32, u32)| LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (self.threads[name], 1, 1), shared_mem_bytes: 0 };
        let hash = f("ple_ngram_hash");
        let mut b = s.launch_builder(&hash);
        b.arg(&tok_d).arg(&mut idx).arg(&mut self.history).arg(&slots_d).arg(&layer).arg(&slot_base).arg(&slot_step).arg(&per);
        unsafe { b.launch(cfg("ple_ngram_hash", (seqs as u32, 1))) }.unwrap();
        let gather = f("ple_embed_gather");
        let mut b = s.launch_builder(&gather);
        b.arg(&idx).arg(&self.table).arg(&mut emb);
        unsafe { b.launch(cfg("ple_embed_gather", (d.heads as u32, n as u32))) }.unwrap();
        let kp = f("ple_key_proj");
        let mut b = s.launch_builder(&kp);
        b.arg(&emb).arg(&self.key_w).arg(&mut key);
        unsafe { b.launch(cfg("ple_key_proj", (1, n as u32))) }.unwrap();
        let vp = f("ple_value_proj");
        let mut b = s.launch_builder(&vp);
        b.arg(&emb).arg(&self.value_w).arg(&mut value);
        unsafe { b.launch(cfg("ple_value_proj", (1, n as u32))) }.unwrap();
        let gate = f("ple_gate");
        let mut b = s.launch_builder(&gate);
        b.arg(&key).arg(&value).arg(&x_d).arg(&self.norm_key).arg(&self.norm_query).arg(&self.norm_conv).arg(&mut gated).arg(&mut normed);
        unsafe { b.launch(cfg("ple_gate", (d.streams as u32, n as u32))) }.unwrap();
        let cv = f("ple_conv");
        let mut b = s.launch_builder(&cv);
        b.arg(&normed).arg(&self.conv_w).arg(&mut conv).arg(&mut self.conv_state).arg(&slots_d).arg(&layer).arg(&slot_base).arg(&slot_step).arg(&per);
        unsafe { b.launch(cfg("ple_conv", (d.width().div_ceil(self.threads["ple_conv"] as usize) as u32, seqs as u32))) }.unwrap();
        let add = f("ple_add");
        let mut b = s.launch_builder(&add);
        b.arg(&x_d).arg(&gated).arg(&conv).arg(&mut ple).arg(&mut after);
        unsafe { b.launch(cfg("ple_add", (d.streams as u32, n as u32))) }.unwrap();
        s.synchronize().unwrap();
        Out { ids: s.clone_dtoh(&idx).unwrap(), ple: s.clone_dtoh(&ple).unwrap(), after: s.clone_dtoh(&after).unwrap() }
    }
}

struct Fixture {
    p: BTreeMap<String, f64>,
    w: Weights,
    hash: kir::hash::TokenHash,
    tokens: Vec<i64>,
    streams: Vec<f32>,
    want: Vec<f32>,
    ids: Vec<i64>,
    prefill: usize,
    eos: i64,
}

fn load() -> Fixture {
    let config = text_config();
    let m = reference();
    let layer = m["layer"].as_u64().unwrap() as usize;
    let p = params(&config, layer);
    let registry = Registry::embedded();
    let hash = registry.get("ple_ngram_hash").unwrap().token_hash(&shape(&p)).unwrap();
    Fixture {
        w: weights(layer),
        hash,
        tokens: m["tokens"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect(),
        streams: floats(&format!("{NAME}.ple.streams.bin")),
        want: floats(&format!("{NAME}.ple.out.bin")),
        ids: m["ngram_ids"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_i64().unwrap())).collect(),
        prefill: m["prefill"].as_u64().unwrap() as usize,
        eos: config["eos_token_id"].as_i64().unwrap(),
        p,
    }
}

#[test]
fn the_ple_params_read_from_the_tiny_and_the_real_config() {
    let tiny = params(&text_config(), 1);
    assert_eq!((tiny["ngram_heads"], tiny["ple_head_dim"], tiny["ple_conv_state"], tiny["ngram_eos"], tiny["ngram_seed"]), (4.0, 32.0, 9.0, 1.0, 1234.0));
    let real: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/popular/Qwen__Qwen3.8-Flash-Next/config.json")).unwrap()).unwrap();
    let real = params(&real["text_config"], 1);
    assert_eq!((real["ngram_heads"], real["ple_head_dim"], real["ple_conv_state"], real["ngram_eos"], real["ngram_seed"], real["ngram_base"]), (16.0, 160.0, 9.0, 248044.0, 1234.0, 20_000_000.0));
}

#[test]
fn every_ple_kernel_is_an_embedded_description_that_builds_for_the_real_and_the_tiny_model() {
    let registry = Registry::embedded();
    let real: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/popular/Qwen__Qwen3.8-Flash-Next/config.json")).unwrap()).unwrap();
    for p in [params(&text_config(), 1), params(&real["text_config"], 1)] {
        let kernels: Vec<kir::Kernel> = KERNELS.iter().map(|n| registry.get(n).unwrap_or_else(|| panic!("{n}")).build(&kir::Target::sm_120(), &shape(&p)).unwrap_or_else(|e| panic!("{e}"))).collect();
        kir::cuda::emit_module(&kernels).unwrap();
    }
}

#[derive(serde::Deserialize)]
struct States {
    state: Vec<specrt::state::State>,
}

#[test]
fn the_ple_state_table_sizes_the_history_and_the_dilated_conv_window() {
    let states: States = toml::from_str(STATES).unwrap();
    let mut spec = specrt::spec::Spec::describe_hf_config(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    spec.attention.params.extend(params(&text_config(), 1));
    spec.state.extend(states.state);
    specrt::state::validate(&spec).unwrap();
    let dims = |n: &str| specrt::state::named(&spec, n).unwrap().dims(&spec).unwrap();
    assert_eq!((dims("ngram_history"), dims("ple_conv")), (vec![2], vec![512, 9]));
}

#[test]
fn the_host_ple_reference_matches_transformers_and_the_fixture_hash() {
    let fx = load();
    let d = Dims::of(&fx.p);
    let mut seq = HostSeq::new(&d, fx.eos);
    let (ids, out) = host_ple(&d, &fx.w, &fx.hash, &mut seq, &fx.tokens[..fx.prefill], &fx.streams[..fx.prefill * d.width()]);
    let mut ids = ids;
    let mut out = out;
    for t in fx.prefill..fx.tokens.len() {
        let (i, o) = host_ple(&d, &fx.w, &fx.hash, &mut seq, &fx.tokens[t..t + 1], &fx.streams[t * d.width()..(t + 1) * d.width()]);
        ids.extend(i);
        out.extend(o);
    }
    assert_eq!(ids, fx.ids);
    let diff = max_diff(&out, &fx.want);
    eprintln!("host ple vs transformers max_abs {diff}");
    assert!(diff < 1e-4, "{diff}");
}

fn check(label: &str, fx: &Fixture, got: &Out, rows: std::ops::Range<usize>) {
    let w = Dims::of(&fx.p).width();
    let heads = fx.hash.heads as usize;
    let ids: Vec<i64> = got.ids.iter().map(|v| *v as i64).collect();
    assert_eq!(ids, fx.ids[rows.start * heads..rows.end * heads], "{label}: n-gram ids");
    let want = &fx.want[rows.start * w..rows.end * w];
    let diff = max_diff(&got.ple, want);
    let after: Vec<f32> = fx.streams[rows.start * w..rows.end * w].iter().zip(want).map(|(a, b)| a + b).collect();
    let diff_after = max_diff(&got.after, &after);
    eprintln!("{label}: ple max_abs {diff}, streams after max_abs {diff_after}");
    assert!(diff < TOLERANCE && diff_after < TOLERANCE, "{label}: {diff} {diff_after}");
}

fn rows(fx: &Fixture, slot: usize, r: std::ops::Range<usize>) -> (usize, Vec<i64>, Vec<f32>) {
    let w = Dims::of(&fx.p).width();
    (slot, fx.tokens[r.clone()].to_vec(), fx.streams[r.start * w..r.end * w].to_vec())
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_ple_layer_matches_transformers_for_prefill_and_decode_steps_one_to_four() {
    let fx = load();
    let mut gpu = Gpu::new(&fx.p, &fx.w, 3);
    let prefill = gpu.run(&Phase { seqs: vec![rows(&fx, 2, 0..fx.prefill)] });
    check("prefill", &fx, &prefill, 0..fx.prefill);
    for t in fx.prefill..fx.tokens.len() {
        let step = gpu.run(&Phase { seqs: vec![rows(&fx, 2, t..t + 1)] });
        check(&format!("decode step {}", t - fx.prefill + 1), &fx, &step, t..t + 1);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn chunked_prefill_and_slot_reuse_give_the_same_ple_output() {
    let fx = load();
    let mut gpu = Gpu::new(&fx.p, &fx.w, 2);
    for round in 0..2 {
        let mut start = 0;
        for chunk in [7usize, 7, 7, 7, 2] {
            let got = gpu.run(&Phase { seqs: vec![rows(&fx, 1, start..start + chunk)] });
            check(&format!("round {round} chunk at {start}"), &fx, &got, start..start + chunk);
            start += chunk;
        }
        for t in fx.prefill..fx.tokens.len() {
            let got = gpu.run(&Phase { seqs: vec![rows(&fx, 1, t..t + 1)] });
            check(&format!("round {round} decode {t}"), &fx, &got, t..t + 1);
        }
        gpu.reset(1);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_sequences_decode_side_by_side_with_their_own_history() {
    let fx = load();
    let d = Dims::of(&fx.p);
    let w = d.width();
    let mut gpu = Gpu::new(&fx.p, &fx.w, 4);
    let mut other_tokens: Vec<i64> = fx.tokens.iter().rev().copied().collect();
    other_tokens[3] = fx.eos;
    other_tokens[fx.prefill] = fx.eos;
    let other_streams: Vec<f32> = (0..fx.tokens.len()).rev().flat_map(|r| fx.streams[r * w..(r + 1) * w].iter().map(|v| v * 0.5)).collect();
    let mut host = HostSeq::new(&d, fx.eos);
    let (_, other_prefill) = host_ple(&d, &fx.w, &fx.hash, &mut host, &other_tokens[..fx.prefill], &other_streams[..fx.prefill * w]);
    gpu.run(&Phase { seqs: vec![rows(&fx, 3, 0..fx.prefill)] });
    let got = gpu.run(&Phase { seqs: vec![(0, other_tokens[..fx.prefill].to_vec(), other_streams[..fx.prefill * w].to_vec())] });
    let diff = max_diff(&got.ple, &other_prefill);
    assert!(diff < TOLERANCE, "second prefill {diff}");
    for t in fx.prefill..fx.tokens.len() {
        let mine = rows(&fx, 3, t..t + 1);
        let theirs = (0, other_tokens[t..t + 1].to_vec(), other_streams[t * w..(t + 1) * w].to_vec());
        let (ids, want) = host_ple(&d, &fx.w, &fx.hash, &mut host, &theirs.1, &theirs.2);
        let got = gpu.run(&Phase { seqs: vec![mine, theirs] });
        let heads = fx.hash.heads as usize;
        let first = Out { ids: got.ids[..heads].to_vec(), ple: got.ple[..w].to_vec(), after: got.after[..w].to_vec() };
        check(&format!("side by side step {t}"), &fx, &first, t..t + 1);
        assert_eq!(got.ids[heads..].iter().map(|v| *v as i64).collect::<Vec<_>>(), ids, "second sequence ids at {t}");
        let diff = max_diff(&got.ple[w..], &want);
        eprintln!("second sequence step {t}: ple max_abs {diff}");
        assert!(diff < TOLERANCE, "second sequence {t}: {diff}");
    }
}
