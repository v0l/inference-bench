use serde_json::Value;
use specrt::kir_kernel::Registry;
use specrt::spec::Spec;
use specrt::weights::{Checkpoint, Stacked, bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3};
use std::path::{Path, PathBuf};

const TABLE: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight";

fn here() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn tiny() -> PathBuf {
    here().join("tests/fixtures/tiny-qwen4-exp")
}

fn headers() -> Value {
    serde_json::from_str(&std::fs::read_to_string(here().join("tests/fixtures/qwen4-exp-real/headers.json")).unwrap()).unwrap()
}

fn real_spec(repo: &str) -> Spec {
    let json = std::fs::read_to_string(here().join("tests/fixtures/popular").join(repo).join("config.json")).unwrap();
    Spec::from_hf_config_serving(&json).unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("specrt-qwen4-ckpt-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_real_shard_names_join_into_the_table_the_family_reads() {
    for (repo, entry) in headers().as_object().unwrap() {
        let tensors = entry["tensors"].as_object().unwrap();
        let names: Vec<String> = tensors.keys().cloned().collect();
        let joined = specrt::weights::joined_names(names.iter(), |n| tensors.contains_key(n));
        assert_eq!(joined.len(), 1, "{repo}: {:?}", joined.keys());
        let parts = &joined[TABLE];
        assert_eq!(parts.len(), 128, "{repo}");
        assert!(parts.iter().enumerate().all(|(i, p)| p == &TABLE.replace("ngram_embedding.weight", &format!("ngram_embedding.shard_{i}.weight"))), "{repo}: parts in shard order");
        let rows: u64 = parts.iter().map(|p| tensors[p][1][0].as_u64().unwrap()).sum();
        assert!(parts.iter().all(|p| tensors[p][1][1] == 160), "{repo}");
        let spec = real_spec(repo);
        let pre = spec.prelude.as_ref().unwrap();
        assert_eq!(rows as f64, pre.params["ngram_rows"], "{repo}: 128 x 2500012 rows are the folded ngram_rows");
        let weight = pre.weights().into_iter().find(|(w, _)| TABLE.ends_with(w.as_str())).expect("the gather binds the joined name");
        assert_eq!(spec.weight_len(&weight.1).unwrap(), rows as usize * 160);
        assert_eq!(pre.stored_as(&weight.0).map(|(p, _)| p), Some("ple_table"));
        let dtype = tensors[&parts[0]][0].as_str().unwrap();
        assert!(parts.iter().all(|p| tensors[p][0] == dtype));
        let scale = tensors.get(&format!("{TABLE}_scale"));
        match repo.as_str() {
            "Qwen__Qwen3.8-Flash-Next" => assert!(dtype == "BF16" && scale.is_none()),
            _ => assert!(dtype == "F8_E4M3" && scale.is_some_and(|s| s[0] == "BF16" && s[1] == serde_json::json!([1])), "{repo}"),
        }
    }
}

#[test]
fn the_folded_hash_matches_the_real_checkpoint_buffers() {
    let registry = Registry::embedded();
    for (repo, entry) in headers().as_object().unwrap() {
        let spec = real_spec(repo);
        let mut shape = kir::desc::Shape::new().num("vocab", spec.dims.vocab as f64);
        for (n, v) in spec.prelude.as_ref().unwrap().symbols(0) {
            shape = shape.num(&n, v);
        }
        let h = registry.get("ple_ngram_hash").unwrap().token_hash(&shape).unwrap();
        let values = &entry["values"];
        let ints = |leaf: &str| -> Vec<u64> { values[format!("model.language_model.layers.1.ple.ple_embedding.{leaf}")].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect() };
        let table = |n: &str| h.per_head.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(table("size"), ints("ngram_heads_vocab_sizes"), "{repo}");
        assert_eq!(table("offset"), ints("ngram_heads_offsets"), "{repo}");
        let mult = ints("layer_multipliers");
        assert_eq!(mult.len(), 3);
        let row = |t: [i64; 3]| -> Vec<i64> {
            let mut history = vec![t[0], t[1]];
            h.eval(&[t[2]], &mut history).unwrap()
        };
        let sizes = table("size");
        let offsets = table("offset");
        for t in [[5i64, 77, 1000], [248000, 3, 17]] {
            let got = row(t);
            let (a, b, c) = (t[2] as u64, t[1] as u64, t[0] as u64);
            let mixed2 = a.wrapping_mul(mult[0]) ^ b.wrapping_mul(mult[1]);
            let mixed3 = mixed2 ^ c.wrapping_mul(mult[2]);
            for (head, id) in got.iter().enumerate() {
                let mixed = if head >= 8 { mixed3 } else { mixed2 };
                let want = (mixed as i64).rem_euclid(sizes[head] as i64) + offsets[head] as i64;
                assert_eq!(*id, want, "{repo}: tokens {t:?} head {head}");
            }
        }
        if let Some(s) = values.get(format!("{TABLE}_scale")) {
            assert!((s.as_f64().unwrap() - 1.9931793212890625e-4).abs() < 1e-12, "{repo}");
        }
    }
}

struct Table {
    rows: usize,
    width: usize,
    values: Vec<f32>,
}

fn tiny_table() -> Table {
    let bytes = std::fs::read(tiny().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = st.tensor(TABLE).unwrap();
    let values = t.data().as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect();
    Table { rows: t.shape()[0], width: t.shape()[1], values }
}

fn quantised(t: &Table) -> (Vec<u8>, f32) {
    let max = t.values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = f32_to_bf16(max / 448.0);
    let scale = bf16_to_f32(scale);
    (t.values.iter().map(|v| f32_to_e4m3(v / scale)).collect(), scale)
}

enum Stored {
    Bf16,
    Fp8,
    Decoded,
}

fn sharded_fixture(tag: &str, stored: Stored, parts: usize) -> PathBuf {
    let dir = scratch(tag);
    for f in ["config.json", "qsa-indexer.toml"] {
        std::fs::copy(tiny().join(f), dir.join(f)).unwrap();
    }
    let bytes = std::fs::read(tiny().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let table = tiny_table();
    let (fp8, scale) = quantised(&table);
    let scale_bits = f32_to_bf16(scale).to_le_bytes().to_vec();
    let per = table.rows / parts;
    assert_eq!(per * parts, table.rows);
    let mut owned: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, view) in st.tensors() {
        if name == TABLE {
            continue;
        }
        owned.push((name, view.dtype(), view.shape().to_vec(), view.data().to_vec()));
    }
    let raw = st.tensor(TABLE).unwrap().data().to_vec();
    for i in 0..parts {
        let name = TABLE.replace("ngram_embedding.weight", &format!("ngram_embedding.shard_{i}.weight"));
        let rows = i * per..(i + 1) * per;
        let (dtype, data) = match stored {
            Stored::Bf16 => (safetensors::Dtype::BF16, raw[rows.start * table.width * 2..rows.end * table.width * 2].to_vec()),
            Stored::Fp8 => (safetensors::Dtype::F8_E4M3, fp8[rows.start * table.width..rows.end * table.width].to_vec()),
            Stored::Decoded => (safetensors::Dtype::F32, fp8[rows.start * table.width..rows.end * table.width].iter().flat_map(|b| (e4m3_to_f32(*b) * scale).to_le_bytes()).collect()),
        };
        owned.push((name, dtype, vec![per, table.width], data));
    }
    if matches!(stored, Stored::Fp8) {
        owned.push((format!("{TABLE}_scale"), safetensors::Dtype::BF16, vec![1], scale_bits));
    }
    let views: Vec<(String, safetensors::tensor::TensorView)> = owned.iter().map(|(n, d, s, b)| (n.clone(), safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap())).collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    dir
}

fn spec_for(dir: &Path) -> Spec {
    let mut s = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    s.repo = dir.display().to_string();
    s.runtime.split_activations = Some(true);
    s
}

#[test]
fn an_fp8_sharded_table_loads_through_the_name_map_in_its_stored_dtype() {
    let table = tiny_table();
    let (fp8, scale) = quantised(&table);
    let dir = sharded_fixture("fp8", Stored::Fp8, 4);
    let ck = Checkpoint::open(&dir).unwrap();
    assert!(ck.has(TABLE) && ck.is_part(&TABLE.replace("ngram_embedding.weight", "ngram_embedding.shard_2.weight")));
    assert_eq!(ck.joined(), vec![TABLE.to_string()]);
    assert_eq!(ck.shape_of(TABLE).unwrap(), vec![table.rows, table.width]);
    assert_eq!(ck.dtype_of(TABLE).unwrap(), safetensors::Dtype::F8_E4M3);
    let mut s = spec_for(&dir);
    ck.complete_prelude(&mut s, 0).unwrap();
    let pre = s.prelude.as_ref().unwrap();
    assert_eq!(pre.words["ple_table_dtype"], "fp8");
    assert_eq!(pre.params["ple_table_scale"] as f32, scale);
    s.runnable().unwrap();
    let stored = ck.stored_prelude(&s, 1, 1).unwrap();
    assert_eq!(stored.len(), 1);
    let (name, parts) = &stored[0];
    assert_eq!((name.as_str(), parts.elem, parts.parts.len()), ("ple_embedding.ngram_embedding.weight", 1, 4));
    let joined: Vec<u8> = parts.parts.concat();
    assert_eq!(joined, fp8, "the parts are the table rows in order, never widened to f32");
    let back: Vec<f32> = joined.iter().map(|b| e4m3_to_f32(*b) * scale).collect();
    let worst = back.iter().zip(&table.values).map(|(a, b)| (a - b).abs() / b.abs().max(1e-2)).fold(0.0f32, f32::max);
    assert!(worst < 0.07, "e4m3 keeps three mantissa bits: {worst}");
    let layer = ck.layer(&s, 1).unwrap();
    assert!(layer.prelude.iter().all(|(n, _)| n != "ple_embedding.ngram_embedding.weight"), "the f32 path skips the stored table");
    assert_eq!(layer.prelude.len(), 6);
    let plain = Checkpoint::open(&tiny()).unwrap();
    let mut f = spec_for(&tiny());
    plain.complete_prelude(&mut f, 0).unwrap();
    assert_eq!(f.prelude.as_ref().unwrap().words["ple_table_dtype"], "bf16");
}

#[test]
fn a_spec_with_a_stored_dtype_it_does_not_bind_is_refused() {
    let dir = sharded_fixture("bad", Stored::Bf16, 2);
    let mut s = spec_for(&dir);
    s.prelude.as_mut().unwrap().words.insert("ple_table_dtype".into(), "int4".into());
    assert!(format!("{:#}", s.runnable().unwrap_err()).contains("int4"));
    let mut s = spec_for(&dir);
    s.prelude.as_mut().unwrap().stored.insert("other".into(), "nowhere.weight".into());
    assert!(format!("{:#}", s.runnable().unwrap_err()).contains("nowhere.weight"));
}

#[test]
fn a_gap_in_the_shard_numbers_leaves_the_parts_unjoined() {
    let names: Vec<String> = ["a.shard_0.weight", "a.shard_2.weight", "b.shard_0.weight", "b.shard_1.weight", "c.weight", "c.shard_0.weight"].iter().map(|s| s.to_string()).collect();
    let joined = specrt::weights::joined_names(names.iter(), |n| names.iter().any(|m| m == n));
    assert_eq!(joined.keys().collect::<Vec<_>>(), vec!["b.weight"]);
}

fn fused_fixture(tag: &str) -> PathBuf {
    let dir = scratch(tag);
    for f in ["config.json", "qsa-indexer.toml"] {
        std::fs::copy(tiny().join(f), dir.join(f)).unwrap();
    }
    let bytes = std::fs::read(tiny().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut owned: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut layers: Vec<String> = Vec::new();
    let (e, inter, h) = (8usize, 32usize, 128usize);
    for (name, view) in st.tensors() {
        match name.split_once(".mlp.experts.") {
            Some((layer, _)) => {
                if !layers.iter().any(|l| l == layer) {
                    layers.push(layer.to_string());
                }
            }
            None => owned.push((name, view.dtype(), view.shape().to_vec(), view.data().to_vec())),
        }
    }
    for layer in &layers {
        let get = |x: usize, w: &str| st.tensor(&format!("{layer}.mlp.experts.{x}.{w}_proj.weight")).unwrap().data().to_vec();
        let mut gate_up = Vec::new();
        let mut down = Vec::new();
        for x in 0..e {
            gate_up.extend(get(x, "gate"));
            gate_up.extend(get(x, "up"));
            down.extend(get(x, "down"));
        }
        owned.push((format!("{layer}.mlp.experts.gate_up_proj"), safetensors::Dtype::BF16, vec![e, 2 * inter, h], gate_up));
        owned.push((format!("{layer}.mlp.experts.down_proj"), safetensors::Dtype::BF16, vec![e, h, inter], down));
    }
    let views: Vec<(String, safetensors::tensor::TensorView)> = owned.iter().map(|(n, d, s, b)| (n.clone(), safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap())).collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn fused_bf16_experts_without_a_weight_suffix_load_like_the_per_expert_copy() {
    let real = &headers()["Qwen__Qwen3.8-Flash-Next"]["tensors"];
    let spec = real_spec("Qwen__Qwen3.8-Flash-Next");
    let m = spec.moe.as_ref().unwrap();
    assert!(m.fused_experts);
    let (e, inter, h) = (m.experts as u64, m.expert_intermediate as u64, spec.dims.hidden as u64);
    assert_eq!(real["model.language_model.layers.1.mlp.experts.gate_up_proj"][1], serde_json::json!([e, 2 * inter, h]), "gate rows first, then up, per expert");
    assert_eq!(real["model.language_model.layers.1.mlp.experts.down_proj"][1], serde_json::json!([e, h, inter]));
    let dir = fused_fixture("fused");
    let fused = Checkpoint::open(&dir).unwrap();
    let per = Checkpoint::open(&tiny()).unwrap();
    let s = spec_for(&tiny());
    assert!(s.moe.as_ref().unwrap().fused_experts);
    for layer in 0..4 {
        let a = fused.layer(&s, layer).unwrap().moe.unwrap();
        let b = per.layer(&s, layer).unwrap().moe.unwrap();
        for (x, y, what) in [(&a.gate, &b.gate, "gate"), (&a.up, &b.up, "up"), (&a.down, &b.down, "down")] {
            let (Stacked::Bf16(x), Stacked::Bf16(y)) = (x, y) else { panic!("bf16 stacks") };
            assert_eq!(x.len(), 9 * 32 * 128, "layer {layer} {what}: 8 routed experts and the shared one");
            assert_eq!(x, y, "layer {layer} {what}");
        }
    }
}

fn gather_shape(dtype: &str, scale: f32) -> kir::desc::Shape {
    let s = spec_for(&tiny());
    let mut shape = kir::desc::Shape::new();
    for (n, v) in s.prelude.as_ref().unwrap().symbols(0) {
        shape = shape.num(&n, v);
    }
    let shape = shape.word("ple_table_dtype", dtype);
    if dtype == "fp8" { shape.num("ple_table_scale", scale as f64) } else { shape }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_fp8_gather_matches_the_f32_gather_of_the_same_table_on_the_gpu() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    let table = tiny_table();
    let (fp8, scale) = quantised(&table);
    let dequant: Vec<f32> = fp8.iter().map(|b| e4m3_to_f32(*b) * scale).collect();
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let d = Registry::embedded();
    let d = d.get("ple_embed_gather").unwrap();
    let build = |dtype: &str| d.build(&target, &gather_shape(dtype, scale)).unwrap_or_else(|e| panic!("{e}"));
    let (k8, k32, k16) = (build("fp8"), build("f32"), build("bf16"));
    let stream = ctx.default_stream();
    let s = spec_for(&tiny());
    let p = &s.prelude.as_ref().unwrap().params;
    let (heads, head_dim, embed) = (p["ngram_heads"] as usize, p["ple_head_dim"] as usize, p["ple_embed_dim"] as usize);
    let rows = 37usize;
    let mut idx: Vec<i32> = (0..rows * heads).map(|i| ((i * 2654435761usize) % table.rows) as i32).collect();
    idx[0] = 0;
    idx[1] = table.rows as i32 - 1;
    let idx_d = stream.clone_htod(&idx).unwrap();
    fn run<T: cudarc::driver::DeviceRepr>(ctx: &std::sync::Arc<CudaContext>, k: &kir::Kernel, idx: &cudarc::driver::CudaSlice<i32>, table: &[T], grid: (u32, u32), out: usize) -> Vec<f32> {
        let stream = ctx.default_stream();
        let module = specrt::jit::load_uncached(ctx, &kir::cuda::emit_module(std::slice::from_ref(k)).unwrap()).unwrap();
        let f = module.load_function("ple_embed_gather").unwrap();
        let t = stream.clone_htod(table).unwrap();
        let mut emb = stream.alloc_zeros::<f32>(out).unwrap();
        let mut b = stream.launch_builder(&f);
        b.arg(idx).arg(&t).arg(&mut emb);
        unsafe { b.launch(LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (k.threads, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
        stream.synchronize().unwrap();
        stream.clone_dtoh(&emb).unwrap()
    }
    let grid = (heads as u32, rows as u32);
    let bf: Vec<u16> = table.values.iter().map(|v| f32_to_bf16(*v)).collect();
    let got8 = run(&ctx, &k8, &idx_d, &fp8, grid, rows * embed);
    let got32 = run(&ctx, &k32, &idx_d, &dequant, grid, rows * embed);
    let got16 = run(&ctx, &k16, &idx_d, &bf, grid, rows * embed);
    let want: Vec<f32> = (0..rows).flat_map(|r| (0..heads).flat_map(move |h| (0..head_dim).map(move |j| (r, h, j)))).map(|(r, h, j)| dequant[idx[r * heads + h] as usize * head_dim + j]).collect();
    assert_eq!(got32, want, "the f32 gather is the host reference");
    assert_eq!(got8, got32, "the fp8 gather decodes each byte and its scale exactly like the f32 table");
    let orig: Vec<f32> = (0..rows).flat_map(|r| (0..heads).flat_map(move |h| (0..head_dim).map(move |j| (r, h, j)))).map(|(r, h, j)| table.values[idx[r * heads + h] as usize * head_dim + j]).collect();
    assert_eq!(got16, orig, "the bf16 gather reads the stored table bit for bit");
    let worst = got8.iter().zip(&orig).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max = orig.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!("fp8 gather against the bf16 table: max abs {worst:e}, table max {max:e}");
    assert!(worst <= max / 16.0, "{worst}");
}

fn served_logits(dir: &Path) -> (Vec<f32>, Vec<u32>) {
    use specrt::runtime::{DecodeRow, Runtime, Sampling};
    let s = spec_for(dir);
    let m: Value = serde_json::from_str(&std::fs::read_to_string(tiny().join("reference/tiny-qwen4-exp.ref.json")).unwrap()).unwrap();
    let tokens: Vec<u32> = m["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mut rt = Runtime::load_batched(&s, 256, 1).unwrap_or_else(|e| panic!("{e:#}"));
    let trace = rt.trace_streams(&tokens, &Default::default()).unwrap();
    let mut st = rt.stepper().unwrap();
    let mut out = vec![rt.prefill_slot(&mut st, 0, &tokens, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
    for step in 0..4 {
        let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: tokens.len() + step, sampling: Sampling::GREEDY, step: step as u32 + 1 };
        out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
    }
    (trace.logits, out)
}

fn reference_logits() -> (Vec<f32>, Vec<u32>) {
    let bytes = std::fs::read(tiny().join("reference/tiny-qwen4-exp.ref.bin")).unwrap();
    let m: Value = serde_json::from_str(&std::fs::read_to_string(tiny().join("reference/tiny-qwen4-exp.ref.json")).unwrap()).unwrap();
    (bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(), m["greedy"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect())
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_engine_serves_bf16_and_fp8_sharded_tables_and_fused_experts() {
    let (want, greedy) = reference_logits();
    let (plain, plain_greedy) = served_logits(&tiny());
    let bf16 = served_logits(&sharded_fixture("engine-bf16", Stored::Bf16, 4));
    let fused = served_logits(&fused_fixture("engine-fused"));
    let fp8 = served_logits(&sharded_fixture("engine-fp8", Stored::Fp8, 4));
    let decoded = served_logits(&sharded_fixture("engine-decoded", Stored::Decoded, 4));
    let d = max_diff(&plain, &want);
    eprintln!("f32 table vs transformers {d:e}, bf16 shards vs f32 table {:e}, fused vs per expert {:e}, fp8 shards vs the decoded f32 table {:e}, fp8 vs bf16 table {:e}", max_diff(&bf16.0, &plain), max_diff(&fused.0, &plain), max_diff(&fp8.0, &decoded.0), max_diff(&fp8.0, &plain));
    assert!(d <= 1e-3, "{d}");
    assert_eq!(plain_greedy, greedy);
    assert_eq!(bf16.0, plain, "a bf16 table read in bf16 gives the same logits as the f32 copy");
    assert_eq!(bf16.1, greedy);
    assert_eq!(fused.0, plain, "fused experts give the same logits as per expert names");
    assert_eq!(fused.1, greedy);
    assert_eq!(fp8.0, decoded.0, "the fp8 table served in fp8 gives the logits of its decoded f32 copy");
    assert_eq!(fp8.1, decoded.1);
}
