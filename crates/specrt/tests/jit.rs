use specrt::codegen::Module;
use specrt::jit::{Jit, cache_key};
use specrt::spec::Spec;
use std::path::Path;

fn spec(file: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs").join(file)).unwrap()
}

#[test]
fn cache_key_covers_source_arch_and_options() {
    let opts = vec!["--std=c++17".to_string()];
    let base = cache_key("src", "compute_121", &opts);
    assert_eq!(base, cache_key("src", "compute_121", &opts));
    assert_ne!(base, cache_key("src2", "compute_121", &opts));
    assert_ne!(base, cache_key("src", "compute_120", &opts));
    assert_ne!(base, cache_key("src", "compute_121", &[]));
    assert_eq!(base.len(), 64);
}

#[test]
fn cache_key_changes_with_nvrtc_and_driver_versions() {
    use specrt::jit::{ToolVersions, versioned_key};
    let opts = vec!["--std=c++17".to_string()];
    let base = ToolVersions { nvrtc: (13, 2), driver: 13000 };
    let key = versioned_key("src", "compute_121", &opts, base);
    assert_eq!(key, versioned_key("src", "compute_121", &opts, base));
    assert_ne!(key, versioned_key("src", "compute_121", &opts, ToolVersions { nvrtc: (13, 5), ..base }));
    assert_ne!(key, versioned_key("src", "compute_121", &opts, ToolVersions { driver: 13010, ..base }));
}

#[test]
fn eviction_removes_oldest_files_until_under_the_bound() {
    use specrt::jit::evict_oldest;
    let dir = std::env::temp_dir().join(format!("specrt-evict-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let now = std::time::SystemTime::now();
    for (i, name) in ["a.cubin", "b.ptx", "c.cubin", "d.cubin"].iter().enumerate() {
        let path = dir.join(name);
        std::fs::write(&path, vec![0u8; 100]).unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_modified(now - std::time::Duration::from_secs(1000 - i as u64 * 100)).unwrap();
    }
    std::fs::write(dir.join("notes.txt"), vec![0u8; 500]).unwrap();
    let removed = evict_oldest(&dir, 250, Some(&dir.join("d.cubin")));
    assert_eq!(removed, 200);
    assert!(!dir.join("a.cubin").exists() && !dir.join("b.ptx").exists());
    assert!(dir.join("c.cubin").exists() && dir.join("d.cubin").exists() && dir.join("notes.txt").exists());
    assert_eq!(evict_oldest(&dir, 0, Some(&dir.join("d.cubin"))), 100);
    assert!(dir.join("d.cubin").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn second_load_reads_the_cubin_without_nvrtc() {
    let dir = std::env::temp_dir().join(format!("specrt-jit-cubin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx, dir.clone()).unwrap();
    let module = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    let first = jit.load(&module).unwrap();
    assert!(!first.cache_hit && first.binary);
    assert!(dir.join(format!("{}.cubin", first.key)).exists());
    let second = jit.load(&module).unwrap();
    assert!(second.cache_hit && second.binary);
    assert!(second.compile_secs < 0.05, "read took {}", second.compile_secs);
    for name in &module.kernels {
        second.function(name).unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_gguf_module_renders_the_ggml_kernels() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b-gguf.toml");
    let module = Module::generate(&Spec::load(&path).unwrap(), 256).unwrap();
    if let Some(out) = std::env::var_os("SPECRT_DUMP_SOURCE").filter(|o| !o.is_empty()) {
        std::fs::write(out, &module.source).unwrap();
    }
    assert!(module.source.contains("linear_w<DecGgml,"));
    assert!(module.source.contains("gate_up_act_norm<DecGgml>"));
    assert!(module.kernels.contains(&"gg_dequant_rows"));
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn compiles_and_caches_both_models() {
    let dir = std::env::temp_dir().join(format!("specrt-jit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx, dir.clone()).unwrap();
    for file in ["qwen3-0.6b.toml", "qwen2.5-0.5b.toml"] {
        let module = Module::generate(&spec(file), 256).unwrap();
        let started = std::time::Instant::now();
        let first = jit.load(&module).unwrap();
        let wall = started.elapsed().as_secs_f64();
        assert!(!first.cache_hit);
        assert!(first.compile_secs > 0.0 && first.module_secs > 0.0);
        let sum = first.compile_secs + first.module_secs;
        assert!((sum - wall).abs() <= 0.1 * wall, "split {sum} vs wall {wall}");
        let second = jit.load(&module).unwrap();
        assert!(second.cache_hit);
        for name in &module.kernels {
            second.function(name).unwrap();
        }
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn qwen38_27b_module_compiles_through_nvrtc() {
    let dir = std::env::temp_dir().join(format!("specrt-jit27-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx, dir.clone()).unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-exact.toml");
    let module = Module::generate(&Spec::load(&path).unwrap(), 4096).unwrap();
    let loaded = jit.load(&module).unwrap();
    for name in &module.kernels {
        loaded.function(name).unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
