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
#[ignore = "needs a CUDA GPU with NVRTC"]
fn compiles_and_caches_both_models() {
    let dir = std::env::temp_dir().join(format!("specrt-jit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx, dir.clone()).unwrap();
    for file in ["qwen3-0.6b.toml", "qwen2.5-0.5b.toml"] {
        let module = Module::generate(&spec(file), 256).unwrap();
        let first = jit.load(&module).unwrap();
        assert!(!first.cache_hit);
        let second = jit.load(&module).unwrap();
        assert!(second.cache_hit);
        for name in &module.kernels {
            second.function(name).unwrap();
        }
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}
