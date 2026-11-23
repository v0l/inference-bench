use specrt::spec::Spec;
use std::path::Path;

#[test]
#[ignore = "writes every fixture module source to SOURCE_DUMP"]
fn dump_module_sources() {
    let Some(out) = std::env::var_os("SOURCE_DUMP") else { return };
    let out = Path::new(&out);
    std::fs::create_dir_all(out).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut specs: Vec<(String, Spec)> = Vec::new();
    for e in std::fs::read_dir(root.join("tests/fixtures")).unwrap().flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        let json = if p.is_dir() { p.join("config.json") } else if name.ends_with(".config.json") { p.clone() } else { continue };
        let Ok(text) = std::fs::read_to_string(&json) else { continue };
        if let Ok(s) = Spec::from_hf_config(&text) {
            specs.push((format!("hf-{name}"), s));
        }
        if let Ok(s) = Spec::from_hf_config_serving(&text) {
            specs.push((format!("serving-{name}"), s));
        }
    }
    for dir in [root.join("tests/fixtures"), root.join("specs")] {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".toml")
                && let Ok(s) = Spec::load(&e.path())
            {
                specs.push((format!("toml-{name}"), s));
            }
        }
    }
    let mut n = 0;
    for (name, spec) in specs {
        for (tag, t) in [("sm120", kir::Target::sm_120()), ("sm80", kir::Target::sm_80())] {
            let text = match specrt::codegen::Module::generate_kv(&spec, 256, 1, &t) {
                Ok(m) => {
                    let only = std::env::var("PTX_ONLY").ok();
                    let picked = only.as_deref().map_or(name.contains("27b") || name.contains("qwen3-0.6b") || name.contains("4b"), |o| name == o);
                    if std::env::var_os("PTX_DUMP").is_some() && tag == "sm120" && picked {
                        let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
                        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&m.source, opts).map(|p| p.to_src()).unwrap_or_else(|e| format!("nvrtc error {e:?}"));
                        std::fs::write(out.join(format!("{name}.{tag}.ptx")), ptx).unwrap();
                    }
                    format!("{}\n{}", m.kir_hash, m.source)
                }
                Err(e) => format!("error {e:#}"),
            };
            std::fs::write(out.join(format!("{name}.{tag}.cu")), text).unwrap();
            n += 1;
        }
    }
    eprintln!("dumped {n}");
}
