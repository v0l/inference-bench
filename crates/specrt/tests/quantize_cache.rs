use specrt::quant_cache::{Attached, MANIFEST, attach, dir_for, key};
use specrt::spec::{LayerQuant, Quant, QuantPlan, QuantizeChoice, Spec};
use specrt::weights::{Checkpoint, Matrix, Mixer};
use std::path::{Path, PathBuf};

fn fixture(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn hub_copy(dir: &str, tag: &str) -> (PathBuf, PathBuf) {
    let hub = std::env::temp_dir().join(format!("quant-cache-{}-{tag}", std::process::id()));
    std::fs::remove_dir_all(&hub).ok();
    let snapshot = hub.join(format!("models--acme--{dir}/snapshots/0123abcd"));
    std::fs::create_dir_all(&snapshot).unwrap();
    for entry in std::fs::read_dir(fixture(dir)).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() {
            std::os::unix::fs::symlink(&p, snapshot.join(p.file_name().unwrap())).unwrap();
        }
    }
    (hub, snapshot)
}

fn spec_at(snapshot: &Path, dir: &str) -> Spec {
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(fixture(dir).join("config.json")).unwrap()).unwrap();
    spec.name = dir.into();
    spec.repo = snapshot.display().to_string();
    spec.quant_keep.first_layers = 0;
    spec.quant_keep.last_layers = 0;
    spec.quant_plan = QuantPlan { layers: (0..spec.layers.count).map(|l| if l == 1 && spec.moe.is_none() { LayerQuant::new(Quant::Fp8Channel, true) } else { LayerQuant::new(Quant::Nvfp4G16, l == 0) }).collect(), kernels: Vec::new() };
    spec.runtime.quantize_cache = true;
    spec.apply_quantize().unwrap();
    spec
}

fn opened(spec: &Spec) -> (Checkpoint, &'static str) {
    let ck = Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors);
    let attached = attach(spec, &ck.files, ck.is_safetensors(), true).unwrap();
    let state = match &attached {
        Attached::Off => "off",
        Attached::Hit(_) => "hit",
        Attached::Miss(_) => "miss",
    };
    (ck.with_quant_cache(attached), state)
}

fn bytes(m: &Matrix) -> Vec<u8> {
    match m {
        Matrix::Fp4(f) => [f.packed.as_ref(), f.scales.as_ref(), &f.global.to_le_bytes()].concat(),
        Matrix::Fp8 { values, scales } => [values.as_ref(), &scales.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>()].concat(),
        Matrix::Bf16(w) => w.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => panic!("unexpected matrix"),
    }
}

fn layer_bytes(ck: &Checkpoint, spec: &Spec, i: usize) -> Vec<Vec<u8>> {
    let l = ck.layer(spec, i).unwrap();
    let mut out = vec![bytes(&l.gate_up), bytes(&l.down)];
    match &l.mixer {
        Mixer::Attention(a) => out.extend([bytes(&a.qkv), bytes(&a.o)]),
        Mixer::Linear(g) => out.extend([bytes(&g.in_proj), bytes(&g.out), bytes(&g.ba)]),
        _ => {}
    }
    if let Some(m) = &l.moe {
        out.push(match &m.gate {
            specrt::weights::Stacked::Fp4 { packed, scales, .. } => [packed.as_slice(), scales.as_slice()].concat(),
            _ => panic!("experts are not nvfp4"),
        });
    }
    out
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() { out.extend(files_under(&p)) } else { out.push(p) }
    }
    out.sort();
    out
}

#[test]
fn the_second_load_reads_the_quantised_weights_back_bit_for_bit_and_skips_the_bf16_source() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-moe"] {
        let (hub, snapshot) = hub_copy(dir, dir);
        let spec = spec_at(&snapshot, dir);
        let before = files_under(&snapshot);
        let (first, state) = opened(&spec);
        assert_eq!(state, "miss", "{dir}");
        let fresh: Vec<Vec<Vec<u8>>> = (0..spec.layers.count)
            .map(|i| {
                let b = layer_bytes(&first, &spec, i);
                first.quant_writer().unwrap().flush(&format!("layer-{i}"));
                b
            })
            .collect();
        let usage = first.quant_writer().unwrap().finish().unwrap().expect("something was written");
        assert_eq!(usage.dir, dir_for(&spec, &first.files).unwrap());
        assert!(usage.dir.starts_with(hub.join(format!("models--acme--{dir}/specrt-quant"))), "{}", usage.dir.display());
        assert_eq!(files_under(&snapshot), before, "{dir}: nothing is written beside the original files");
        assert!(usage.dir.join(MANIFEST).is_file());
        let on_disk: u64 = files_under(&usage.dir).iter().filter(|p| p.extension().is_some_and(|e| e == "safetensors")).map(|p| p.metadata().unwrap().len()).sum();
        assert_eq!(usage.bytes, on_disk);
        drop(first);
        let (second, state) = opened(&spec);
        assert_eq!(state, "hit", "{dir}");
        let store = second.quant_store().unwrap();
        let weight = if spec.moe.is_some() { "model.layers.1.mlp.experts.0.down_proj.weight" } else { "model.layers.0.mlp.down_proj.weight" };
        assert!(store.covers(weight), "{dir}: {:?}", store.manifest.sources);
        assert!(!store.covers("model.layers.0.input_layernorm.weight"), "{dir}: norms are read from the checkpoint");
        for (i, want) in fresh.iter().enumerate() {
            assert_eq!(&layer_bytes(&second, &spec, i), want, "{dir}: layer {i} from the cache equals the conversion");
        }
        assert_eq!(second.quantized().tensors, 0, "{dir}: a cache hit converts nothing");
        std::fs::remove_dir_all(&hub).ok();
    }
}

#[test]
fn the_cache_key_follows_the_plan_the_source_files_and_the_format_but_not_serving_options() {
    let (hub, snapshot) = hub_copy("tiny-qwen3.5", "key");
    let spec = spec_at(&snapshot, "tiny-qwen3.5");
    let files = Checkpoint::for_repo(&spec.repo).unwrap().files;
    let base = key(&spec, &files).unwrap();
    let mut serving = spec.clone();
    serving.runtime.kv_dtype = specrt::spec::KvDtype::Fp8;
    serving.runtime.graphs = false;
    assert_eq!(key(&serving, &files).unwrap(), base, "serving options do not change the weights");
    let mut plain = spec.clone();
    plain.quant_plan.layers[0] = LayerQuant::new(Quant::Nvfp4G16, false);
    assert_ne!(key(&plain, &files).unwrap(), base, "a different recipe is a different cache");
    let mut keep = spec.clone();
    keep.quant_keep.sensitive.push("mlp.down_proj".into());
    assert_ne!(key(&keep, &files).unwrap(), base);
    let shard = snapshot.join("model.safetensors");
    let target = std::fs::read_link(&shard).unwrap();
    std::fs::remove_file(&shard).unwrap();
    let copy = hub.join("copy.safetensors");
    std::fs::copy(&target, &copy).unwrap();
    std::fs::OpenOptions::new().append(true).open(&copy).unwrap().set_len(std::fs::metadata(&target).unwrap().len() + 1).unwrap();
    std::os::unix::fs::symlink(&copy, &shard).unwrap();
    assert_ne!(key(&spec, &files).unwrap(), base, "changed source weights are a different cache");
    std::fs::remove_dir_all(&hub).ok();
}

#[test]
#[ignore = "needs a GPU"]
fn a_cached_start_gives_the_logits_of_a_converting_start_and_reads_less() {
    use specrt::runtime::Runtime;
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-moe"] {
        let (hub, snapshot) = hub_copy(dir, &format!("gpu-{dir}"));
        let spec = spec_at(&snapshot, dir);
        let prompt: Vec<u32> = (0..64u32).map(|i| (i * 13 + 5) % 1024).collect();
        let run = |s: &Spec| {
            let mut rt = Runtime::load(s, 256).unwrap();
            (rt.prompt_logits(&prompt).unwrap(), rt.quant_cache.clone(), rt.bytes_read, rt.quantized.tensors, rt.weight_bytes)
        };
        let mut uncached = spec.clone();
        uncached.runtime.quantize_cache = false;
        let (want, none, plain_read, converted, weights) = run(&uncached);
        assert!(none.is_none() && converted > 0);
        let (first, written, first_read, _, first_weights) = run(&spec);
        let written = written.expect("the first start writes the cache");
        assert!(!written.hit && written.bytes > 0);
        let (second, read, second_read, second_converted, second_weights) = run(&spec);
        let read = read.expect("the second start reads the cache");
        assert!(read.hit && read.dir == written.dir && read.bytes == written.bytes);
        assert_eq!(first, want, "{dir}: writing the cache changes no logits");
        assert_eq!(second, want, "{dir}: a cached start gives the logits of a converting start");
        assert_eq!((first_weights, second_weights), (weights, weights));
        assert_eq!(second_converted, 0, "{dir}: nothing is converted on a hit");
        assert!(second_read < first_read && first_read == plain_read, "{dir}: bytes read {plain_read} {first_read} {second_read}");
        eprintln!("{dir}: cache {} bytes in {}, source bytes read {first_read} then {second_read}", written.bytes, written.dir.display());
        std::fs::remove_dir_all(&hub).ok();
    }
}

#[test]
fn a_checkpoint_outside_the_hub_caches_under_the_cache_dir_and_off_writes_nothing() {
    let dir = fixture("tiny-qwen3.5");
    let mut spec = spec_at(&dir, "tiny-qwen3.5");
    let cache = std::env::temp_dir().join(format!("quant-cache-dir-{}", std::process::id()));
    spec.runtime.cache_dir = Some(cache.clone());
    let files = Checkpoint::for_repo(&spec.repo).unwrap().files;
    assert!(dir_for(&spec, &files).unwrap().starts_with(cache.join("quant/tiny-qwen3.5")));
    let mut off = spec.clone();
    off.runtime.quantize_cache = false;
    assert!(matches!(attach(&off, &files, true, true).unwrap(), Attached::Off));
    let mut bf16 = spec.clone();
    bf16.quant_plan = QuantPlan::default();
    bf16.runtime.quantize = QuantizeChoice::Off;
    assert!(matches!(attach(&bf16, &files, true, true).unwrap(), Attached::Off), "nothing to cache without quantisation");
    assert!(matches!(attach(&spec, &files, true, false).unwrap(), Attached::Off), "a pipeline stage converts at load");
    assert!(!cache.exists());
}
