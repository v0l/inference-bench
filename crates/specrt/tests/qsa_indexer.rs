use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use serde_json::Value;
use specrt::codegen::Module;
use specrt::graph::Bind;
use specrt::jit::Jit;
use specrt::plan::Weight;
use specrt::spec::Spec;
use specrt::weights::{Checkpoint, Matrix, bf16_to_f32};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-qwen4-exp";
const LAYER: usize = 3;
const MAX_CTX: usize = 64;
const PAGE: usize = 16;
const TOLERANCE: f32 = 1e-3;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(NAME)
}

fn floats(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("reference").join(file)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(format!("{NAME}.indexer.json"))).unwrap()).unwrap()
}

fn qsa_spec() -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap().replace("\"model_type\": \"qwen4_exp\"", "\"model_type\": \"qwen3_5\"");
    let mut s = Spec::from_hf_config(&json).unwrap();
    s.name = NAME.into();
    s.repo = fixture().display().to_string();
    let fragment = std::fs::read_to_string(fixture().join("qsa-indexer.toml")).unwrap();
    let states: String = specrt::state::stock().iter().map(|st| format!("[[state]]\n{}\n", toml::to_string(st).unwrap())).collect();
    Spec::parse(&format!("{}\n{states}{fragment}", s.to_toml().unwrap())).unwrap_or_else(|e| panic!("{e:#}"))
}

struct Index {
    heads: usize,
    dim: usize,
    ratio: usize,
    groups: usize,
    out: usize,
    rot: usize,
    theta: f64,
    eps: f32,
}

impl Index {
    fn of(spec: &Spec) -> Index {
        let p = |n: &str| spec.attention.params[n] as usize;
        Index { heads: p("index_heads"), dim: p("index_head_dim"), ratio: p("index_ratio"), groups: p("index_groups"), out: p("index_out"), rot: spec.rotary_dim(), theta: spec.theta(), eps: spec.norm.eps as f32 }
    }

    fn norm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + self.eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v * inv * (1.0 + w)).collect()
    }

    fn rope(&self, x: &mut [f32], pos: usize) {
        let half = self.rot / 2;
        for i in 0..half {
            let angle = pos as f64 / self.theta.powf(2.0 * i as f64 / self.rot as f64);
            let (c, s) = (angle.cos() as f32, angle.sin() as f32);
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * c - b * s;
            x[i + half] = b * c + a * s;
        }
    }

    fn pooled(&self, raw: &[Vec<f32>], norm: &[f32], group: usize) -> Vec<f32> {
        let start = group * self.ratio;
        let mean: Vec<f32> = (0..self.dim).map(|d| raw[start..start + self.ratio].iter().map(|r| r[d]).sum::<f32>() / self.ratio as f32).collect();
        let mut k = self.norm(&mean, norm);
        self.rope(&mut k, start);
        k
    }

    fn query(&self, q: &[f32], norm: &[f32], pos: usize) -> Vec<f32> {
        q.chunks(self.dim)
            .flat_map(|h| {
                let mut v = self.norm(h, norm);
                self.rope(&mut v, pos);
                v
            })
            .collect()
    }

    fn select(&self, q: &[f32], keys: &[Vec<f32>], p: usize) -> Vec<i32> {
        let n = (p + 1) / self.ratio;
        let scale = 1.0 / (self.dim as f32).sqrt();
        let mut scores: Vec<(f32, usize)> = (0..n).map(|g| (q.chunks(self.dim).map(|h| h.iter().zip(&keys[g]).map(|(a, b)| a * b).sum::<f32>().max(0.0)).sum::<f32>() * scale, g)).collect();
        scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut out: Vec<i32> = scores.iter().take(self.groups).flat_map(|&(_, g)| (g * self.ratio..(g + 1) * self.ratio).map(|t| t as i32)).collect();
        out.extend((n * self.ratio..=p).map(|t| t as i32));
        out.resize(self.out, -1);
        out
    }
}

struct Attention {
    qkv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    extra: Vec<(String, Vec<f32>)>,
}

fn attention(spec: &Spec) -> Attention {
    let ck = Checkpoint::open(&fixture()).unwrap();
    let a = ck.layer_attention(spec, LAYER).unwrap();
    let Matrix::Bf16(w) = &a.qkv else { panic!("the fixture qkv is bf16") };
    Attention { qkv: w.iter().map(|b| bf16_to_f32(*b)).collect(), q_norm: a.q_norm.clone().unwrap(), k_norm: a.k_norm.clone().unwrap(), extra: a.extra.clone() }
}

impl Attention {
    fn extra(&self, name: &str) -> &[f32] {
        &self.extra.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no {name}")).1
    }

    fn project(&self, spec: &Spec, input: &[f32]) -> Vec<f32> {
        let h = spec.dims.hidden;
        input.chunks(h).flat_map(|x| self.qkv.chunks(h).map(|w| w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()).collect::<Vec<_>>()).collect()
    }
}

fn reference_attn() -> (usize, Vec<f32>) {
    let m = meta();
    let scale = m["attn_scale"].as_f64().unwrap() as f32;
    let bytes = std::fs::read(fixture().join("reference").join(format!("{NAME}.indexer.attn.i16"))).unwrap();
    (m["attn_from"].as_u64().unwrap() as usize, bytes.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 * scale).collect())
}

fn reference_rows() -> Vec<Vec<i32>> {
    let width = meta()["width"].as_u64().unwrap() as usize;
    floats(&format!("{NAME}.indexer.selected.bin")).chunks(width).map(|r| r.iter().map(|v| *v as i32).collect()).collect()
}

fn sorted(row: &[i32]) -> Vec<i32> {
    let mut v: Vec<i32> = row.iter().copied().filter(|t| *t >= 0).collect();
    v.sort();
    v
}

fn groups(row: &[i32], p: usize, ratio: usize) -> Vec<i32> {
    let complete = ((p + 1) / ratio * ratio) as i32;
    let mut g: Vec<i32> = row.iter().filter(|t| **t >= 0 && **t < complete).map(|t| t / ratio as i32).collect();
    g.sort();
    g.dedup();
    g
}

fn cpu_selection(spec: &Spec) -> Vec<Vec<i32>> {
    let ix = Index::of(spec);
    let attn = attention(spec);
    let qkv = attn.project(spec, &floats(&format!("{NAME}.indexer.input.bin")));
    let (row, at) = (spec.qkv_rows(), spec.appended_offset());
    let raw: Vec<Vec<f32>> = qkv.chunks(row).map(|r| r[at + ix.heads * ix.dim..at + (ix.heads + 1) * ix.dim].to_vec()).collect();
    let (qn, kn) = (attn.extra("indexer.q_layernorm.weight"), attn.extra("indexer.k_layernorm.weight"));
    let keys: Vec<Vec<f32>> = (0..raw.len() / ix.ratio).map(|g| ix.pooled(&raw, kn, g)).collect();
    qkv.chunks(row).enumerate().map(|(p, r)| ix.select(&ix.query(&r[at..at + ix.heads * ix.dim], qn, p), &keys, p)).collect()
}

#[test]
fn the_selection_keeps_the_best_scoring_groups_not_the_most_recent() {
    let ix = Index { heads: 2, dim: 8, ratio: 4, groups: 2, out: 11, rot: 8, theta: 10000.0, eps: 1e-6 };
    let mut q = vec![0f32; 16];
    q[0] = 1.0;
    q[8] = -1.0;
    let raw: Vec<Vec<f32>> = (0..28).map(|t| (0..8).map(|d| if d == 0 && (t / 4 == 0 || t / 4 == 2) { 1.0 } else { 0.0 }).collect()).collect();
    let keys: Vec<Vec<f32>> = (0..7).map(|g| ix.pooled(&raw, &[0.0; 8], g)).collect();
    let recency: Vec<i32> = (16..24).collect();
    let chosen = ix.select(&q, &keys, 23);
    assert_eq!(chosen, vec![0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1]);
    assert_ne!(sorted(&chosen), recency, "a recency window would keep groups 4 and 5");
    assert_eq!(ix.select(&q, &keys, 25), vec![0, 1, 2, 3, 8, 9, 10, 11, 24, 25, -1], "the incomplete trailing group is always kept");
    assert_eq!(ix.select(&q, &keys, 6), vec![0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1], "fewer complete groups than the budget keeps them all");
    assert_eq!(ix.select(&q, &keys, 2), vec![0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1], "no complete group keeps only the tail");
}

#[test]
fn the_cpu_reference_reproduces_the_transformers_selection_of_the_fixture() {
    let spec = qsa_spec();
    let want = reference_rows();
    let got = cpu_selection(&spec);
    assert_eq!((got.len(), meta()["prompt"].as_u64().unwrap()), (34, 30));
    for (p, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(sorted(g), sorted(w), "row {p}");
    }
    let ratio = Index::of(&spec).ratio;
    let sparse: Vec<usize> = (0..want.len()).filter(|&p| sorted(&want[p]).len() < p + 1).collect();
    assert!(sparse.contains(&29) && sparse.contains(&33), "selection is active at the prompt end and in decode: {sparse:?}");
    assert_ne!(groups(&want[30], 30, ratio), vec![3, 4, 5, 6], "row 30 keeps groups other than the most recent four");
    assert_ne!(groups(&want[31], 31, ratio), groups(&want[30], 30, ratio), "decode step 2 completes group 7 and moves the selection");
}

#[test]
fn the_indexer_spec_appends_index_qk_proj_to_the_qkv_rows_and_builds_its_kernels() {
    let spec = qsa_spec();
    let gated = 2 * spec.q_dim() + 2 * spec.kv_dim();
    assert_eq!((spec.appended_offset(), spec.qkv_rows()), (gated, gated + 5 * 64));
    assert!(Weight::Qkv.tensors(&spec).contains(&"self_attn.indexer.index_qk_proj.weight".to_string()));
    let attn = attention(&spec);
    let h = spec.dims.hidden;
    assert_eq!(attn.qkv.len(), spec.qkv_rows() * h);
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = st.tensor(&format!("model.language_model.layers.{LAYER}.self_attn.indexer.index_qk_proj.weight")).unwrap();
    let index: Vec<f32> = t.data().as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect();
    assert_eq!(attn.qkv[gated * h..], index[..], "index_qk_proj rows follow q, k, v and the gate");
    let module = Module::generate_kv(&spec, MAX_CTX, 2, &kir::Target::sm_120()).unwrap();
    for k in ["qsa_index_raw_write", "qsa_index_pool", "qsa_index_select", "qsa_attention", "qk_norm_rope"] {
        assert!(module.kernels.contains(&k), "{k}");
    }
    let src = module.source.as_str();
    assert!(src.contains("constexpr int MAX_GROUPS = 16;") && src.contains("constexpr int TOPK = 4;") && src.contains("(1.0f + norm[d])"), "the budget and the unit offset norm come from the spec");
    assert!(src.contains("sel_hit |= idx[(size_t)row * 19 + sel_j] == t"), "the attention reads the selected token list");
    let mut refused = spec.clone();
    refused.attention.qkv_bias = true;
    assert!(Spec::parse(&refused.to_toml().unwrap()).unwrap_err().to_string().contains("append"));
}

#[test]
fn attention_over_the_selected_tokens_reproduces_the_fixture_output_on_cpu() {
    let spec = qsa_spec();
    let a = &spec.attention;
    let ix = Index::of(&spec);
    let attn = attention(&spec);
    let qkv = attn.project(&spec, &floats(&format!("{NAME}.indexer.input.bin")));
    let (row, hd, qd, kvd) = (spec.qkv_rows(), a.head_dim, spec.q_dim(), spec.kv_dim());
    let rope = Index { rot: spec.rotary_dim(), ..ix };
    let head = |r: &[f32], at: usize, w: &[f32], p: usize| {
        let inv = 1.0 / (r[at..at + hd].iter().map(|v| v * v).sum::<f32>() / hd as f32 + rope.eps).sqrt();
        let mut v: Vec<f32> = r[at..at + hd].iter().zip(w).map(|(x, w)| x * inv * w).collect();
        rope.rope(&mut v, p);
        v
    };
    let rows: Vec<&[f32]> = qkv.chunks(row).collect();
    let keys: Vec<Vec<Vec<f32>>> = rows.iter().enumerate().map(|(p, r)| (0..a.kv_heads).map(|k| head(r, qd + k * hd, &attn.k_norm, p)).collect()).collect();
    let (from, want) = reference_attn();
    let selected = reference_rows();
    let mut worst = 0f32;
    for (p, r) in rows.iter().enumerate().skip(from) {
        let chosen: Vec<usize> = selected[p].iter().filter(|t| **t >= 0).map(|t| *t as usize).collect();
        for h in 0..a.heads {
            let kv = h / (a.heads / a.kv_heads);
            let q = head(r, h * hd, &attn.q_norm, p);
            let scores: Vec<f32> = chosen.iter().map(|&t| q.iter().zip(&keys[t][kv]).map(|(x, y)| x * y).sum::<f32>() / (hd as f32).sqrt()).collect();
            let top = scores.iter().copied().fold(f32::MIN, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let total: f32 = e.iter().sum();
            for d in 0..hd {
                let v: f32 = chosen.iter().zip(&e).map(|(&t, w)| w * rows[t][qd + kvd + kv * hd + d]).sum::<f32>() / total;
                let gate = 1.0 / (1.0 + (-r[qd + 2 * kvd + h * hd + d]).exp());
                worst = worst.max((v * gate - want[(p - from) * qd + h * hd + d]).abs());
            }
        }
    }
    assert!(worst <= TOLERANCE, "gated attention over the selected tokens differs by {worst}");
}

#[test]
#[ignore = "NVRTC compile of the indexer module for sm_120, no GPU"]
fn the_indexer_module_compiles_with_nvrtc() {
    let module = Module::generate_kv(&qsa_spec(), MAX_CTX, 2, &kir::Target::sm_120()).unwrap();
    let cubin = specrt::jit::cubin_for(&module.source, "sm_120", 1).unwrap_or_else(|e| panic!("{e:#}"));
    assert!(!cubin.is_empty());
}

struct Gpu {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    loaded: specrt::jit::Loaded,
    described: Vec<Option<kir::Kernel>>,
    weights: Vec<(String, CudaSlice<f32>)>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    k_cache: CudaSlice<u16>,
    v_cache: CudaSlice<u16>,
    states: Vec<(String, CudaSlice<u16>)>,
    table: CudaSlice<i32>,
    dummy: CudaSlice<i32>,
    pages: i32,
}

struct Step {
    selected: Vec<i32>,
    attn: Vec<f32>,
}

impl Gpu {
    fn run(&self, spec: &Spec, qkv: &[f32], start: usize, rows: usize, decode: bool) -> Step {
        let a = &spec.attention;
        let ix = Index::of(spec);
        let stream = self.stream.clone();
        let qkv_d = stream.clone_htod(qkv).unwrap();
        let pos = stream.clone_htod(&[start as i32]).unwrap();
        let slots: CudaSlice<i32> = stream.alloc_zeros(rows).unwrap();
        let idx: CudaSlice<i32> = stream.alloc_zeros(rows * ix.out).unwrap();
        let out_hi: CudaSlice<u16> = stream.alloc_zeros(rows * spec.q_dim()).unwrap();
        let out_lo: CudaSlice<u16> = stream.alloc_zeros(rows * spec.q_dim()).unwrap();
        let (pos_step, row_step) = if decode { (1i32, 0i32) } else { (0, 1) };
        let (layer, per_row, slot_base, mm_stride) = (0i32, i32::from(decode), 0i32, MAX_CTX as i32);
        let f = self.loaded.function("qk_norm_rope").unwrap();
        let mut b = stream.launch_builder(&f);
        b.arg(&qkv_d).arg(&self.q_norm).arg(&self.k_norm).arg(&self.k_cache).arg(&self.v_cache).arg(&self.table).arg(&pos).arg(&slots).arg(&layer).arg(&per_row).arg(&self.pages).arg(&mm_stride).arg(&self.dummy).arg(&self.dummy).arg(&slot_base);
        unsafe { b.launch(LaunchConfig { grid_dim: ((a.heads + 2 * a.kv_heads) as u32, rows as u32, 1), block_dim: (a.head_dim.max(32) as u32, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
        for (i, node) in a.node.iter().enumerate() {
            let Some(k) = self.described[i].as_ref() else { continue };
            let values: Vec<(i32, f32)> = k
                .params
                .iter()
                .map(|p| {
                    let v = match p.name {
                        "layer" => 0.0,
                        "pages" => self.pages as f64,
                        "pos_step" => pos_step as f64,
                        "row_step" => row_step as f64,
                        n => match node.bind.get(n) {
                            Some(Bind::Name(name)) => a.params[name],
                            Some(Bind::Number(v)) => *v,
                            _ => 0.0,
                        },
                    };
                    (v as i32, v as f32)
                })
                .collect();
            let f = self.loaded.function(k.name).unwrap();
            let mut b = stream.launch_builder(&f);
            let state_of = |name: &str| &self.states.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no state {name}")).1;
            for (j, p) in k.params.iter().enumerate() {
                match (p.role, p.name, node.bind.get(p.name)) {
                    (kir::Role::Value, _, _) => match p.ty {
                        kir::Scalar::F32 => b.arg(&values[j].1),
                        _ => b.arg(&values[j].0),
                    },
                    (kir::Role::Weight, _, Some(Bind::Weight { weight, .. })) => b.arg(&self.weights.iter().find(|(n, _)| n == weight).unwrap().1),
                    (_, _, Some(Bind::Buffer { buffer })) if buffer == "qkv" => b.arg(&qkv_d),
                    (_, _, Some(Bind::Buffer { .. })) => b.arg(&idx),
                    (_, _, Some(Bind::State { state })) => b.arg(state_of(state)),
                    (_, "qkv", None) => b.arg(&qkv_d),
                    (_, "k_cache", None) => b.arg(&self.k_cache),
                    (_, "v_cache", None) => b.arg(&self.v_cache),
                    (_, "block_table", None) => b.arg(&self.table),
                    (_, "pos", None) => b.arg(&pos),
                    (_, "slots", None) => b.arg(&slots),
                    (_, "out_hi", None) => b.arg(&out_hi),
                    (_, "out_lo", None) => b.arg(&out_lo),
                    (_, n, _) => panic!("{} param {n} has no buffer", k.name),
                };
            }
            let grid: Vec<u32> = node.grid.iter().map(|g| match g.as_str() {
                "rows" => rows as u32,
                "heads" => a.heads as u32,
                "kv_heads" => a.kv_heads as u32,
                n => n.parse().unwrap(),
            }).collect();
            unsafe { b.launch(LaunchConfig { grid_dim: (grid[0], grid[1], 1), block_dim: (k.threads, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
        }
        stream.synchronize().unwrap();
        let (hi, lo) = (stream.clone_dtoh(&out_hi).unwrap(), stream.clone_dtoh(&out_lo).unwrap());
        let plane = |b: u16| f32::from_bits((b as u32) << 16);
        Step { selected: stream.clone_dtoh(&idx).unwrap(), attn: hi.iter().zip(&lo).map(|(h, l)| plane(*h) + plane(*l)).collect() }
    }
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn the_indexed_layer_matches_the_fixture_at_prefill_and_across_decode_group_boundaries() {
    let spec = qsa_spec();
    let ix = Index::of(&spec);
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let module = Module::generate_kv(&spec, MAX_CTX, 2, &target).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-qsa-{}", std::process::id()))).unwrap();
    let loaded = jit.load(&module).unwrap();
    let described = specrt::codegen::described_kernels(&spec, 2, MAX_CTX, &target).unwrap();
    let stream = ctx.default_stream();
    let attn = attention(&spec);
    let pages = MAX_CTX / PAGE;
    let kv = pages * spec.kv_layers().max(1) * spec.attention.kv_heads * PAGE * spec.attention.head_dim * 2;
    let state = pages * spec.kv_layers().max(1) * PAGE * ix.dim * 2;
    let table: Vec<i32> = (0..pages as i32).rev().collect();
    let gpu = Gpu {
        stream: stream.clone(),
        loaded,
        described,
        weights: attn.extra.iter().map(|(n, w)| (n.clone(), stream.clone_htod(w).unwrap())).collect(),
        q_norm: stream.clone_htod(&attn.q_norm).unwrap(),
        k_norm: stream.clone_htod(&attn.k_norm).unwrap(),
        k_cache: stream.alloc_zeros(kv).unwrap(),
        v_cache: stream.alloc_zeros(kv).unwrap(),
        states: ["index_raw", "index_pool"].iter().map(|n| (n.to_string(), stream.alloc_zeros(state).unwrap())).collect(),
        table: stream.clone_htod(&table).unwrap(),
        dummy: stream.alloc_zeros(2 * MAX_CTX).unwrap(),
        pages: pages as i32,
    };
    let qkv = attn.project(&spec, &floats(&format!("{NAME}.indexer.input.bin")));
    let row = spec.qkv_rows();
    let prompt = meta()["prompt"].as_u64().unwrap() as usize;
    let total = qkv.len() / row;
    let (from, want_attn) = reference_attn();
    assert!(from < prompt, "the attention reference starts at the first sparse prompt row");
    let want = reference_rows();
    let cpu = cpu_selection(&spec);
    let qd = spec.q_dim();
    let mut steps = vec![(0, gpu.run(&spec, &qkv[..prompt * row], 0, prompt, false))];
    for p in prompt..total {
        steps.push((p, gpu.run(&spec, &qkv[p * row..(p + 1) * row], p, 1, true)));
    }
    let mut worst = 0f32;
    for (start, step) in &steps {
        for (r, got) in step.selected.chunks(ix.out).enumerate() {
            let p = start + r;
            assert_eq!(sorted(got), sorted(&want[p]), "row {p} selected tokens");
            assert_eq!(groups(got, p, ix.ratio), groups(&want[p], p, ix.ratio), "row {p} group ids");
            assert_eq!(sorted(got), sorted(&cpu[p]), "row {p} matches the CPU reference");
            if p < from {
                continue;
            }
            let diff = step.attn[r * qd..(r + 1) * qd].iter().zip(&want_attn[(p - from) * qd..(p - from + 1) * qd]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            worst = worst.max(diff);
            assert!(diff <= TOLERANCE, "row {p} attention output differs by {diff}");
        }
    }
    eprintln!("qsa indexed layer: {total} rows ({prompt} prefill, {} decode), selection exact, attention max abs {worst:e}", total - prompt);
}
