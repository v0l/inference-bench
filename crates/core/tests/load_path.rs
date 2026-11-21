use std::path::Path;

const LOAD_PATH: &[&str] = &[
    "specrt/src/serve.rs",
    "specrt/src/tp_exec.rs",
    "specrt/src/pipeline.rs",
    "specrt/src/stage.rs",
    "specrt/src/solve.rs",
    "specrt/src/host.rs",
    "specrt/src/cluster.rs",
    "core/src/engine.rs",
    "core/src/serving.rs",
    "core/src/cluster.rs",
    "core/src/inventory.rs",
    "core/src/native.rs",
    "ui/src/load.rs",
    "ui/src/hardware.rs",
    "ui/src/main.rs",
];

const DEFAULT_DEVICE: &[&str] = &["CudaContext::new(0)", "device: 0,", "device: 0 ", "slots, 0)", "slots, 0, "];

fn non_test_source(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split("#[cfg(test)]").next().unwrap_or_default().to_string()
}

fn default_device_uses(source: &str) -> Vec<String> {
    source.lines().enumerate().filter(|(_, l)| DEFAULT_DEVICE.iter().any(|p| l.contains(p))).map(|(n, l)| format!("{}: {}", n + 1, l.trim())).collect()
}

#[test]
fn no_load_path_file_falls_back_to_gpu_0() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let found: Vec<String> = LOAD_PATH.iter().flat_map(|f| default_device_uses(&non_test_source(&crates.join(f))).into_iter().map(move |l| format!("{f}:{l}"))).collect();
    assert!(found.is_empty(), "the load path opens a default GPU instead of the planned one:\n{}", found.join("\n"));
}

#[test]
fn the_guard_catches_a_default_gpu() {
    assert_eq!(default_device_uses("let ctx = CudaContext::new(0)?;").len(), 1);
    assert_eq!(default_device_uses("Ok(RuntimeStage { spec, device: 0, api })").len(), 1);
    assert_eq!(default_device_uses("RuntimeStage::unloaded_on(&spec, max_ctx, slots, 0)").len(), 1);
    assert!(default_device_uses("let ctx = CudaContext::new(range.device)?;").is_empty());
}
