use specrt::kir_kernel::Registry;
use specrt::spec::{Checked, Spec, check};
use std::path::Path;

fn targets() -> Vec<kir::Target> {
    vec![kir::Target::sm_120(), kir::Target::sm_80()]
}

fn attention_shape(spec: &Spec, kv_planes: usize, max_ctx: usize) -> kir::attention::AttentionShape {
    specrt::codegen::attention_shape(spec, kv_planes, max_ctx)
}

fn families() -> [(kir::attention::Family, &'static [&'static str]); 5] {
    [
        (kir::attention::Family::RopeKv, &["qk_norm_rope", "kv_norm_rope"]),
        (kir::attention::Family::SplitDecode, &["attention_split", "attention_reduce"]),
        (kir::attention::Family::FlashLegacy, &["flash_attention"]),
        (kir::attention::Family::FlashMma, &["flash_mma"]),
        (kir::attention::Family::GroupedDecode, &["batch_decode_attention", "batch_decode_combine"]),
    ]
}

fn specs() -> Vec<Spec> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for (file, _) in [
        ("specs/qwen3-0.6b.toml", 1),
        ("tests/fixtures/tiny-qwen3.5-27b/config.json", 1),
        ("tests/fixtures/tiny-qwen3.5-fp8/config.json", 1),
    ] {
        let mut spec = if file.ends_with(".toml") {
            Spec::load(&root.join(file)).unwrap()
        } else {
            let mut s = Spec::from_hf_config(&std::fs::read_to_string(root.join(file)).unwrap()).unwrap();
            s.name = file.rsplit('/').nth(1).unwrap().to_string();
            s
        };
        if file.contains("tiny") {
            spec.layers.count = spec.layers.pattern.len();
            spec.mtp = None;
        }
        out.push(spec.for_kv_context(8192));
    }
    out
}

#[test]
fn every_attention_kernel_equals_its_description_per_target() {
    let registry = Registry::embedded();
    for spec in specs() {
        let shape = attention_shape(&spec, 1, 8192);
        for t in targets() {
            for (family, names) in families() {
                for name in names {
                    let rust = kir::attention::AttentionKernel::new(&t, family, shape.clone()).unwrap();
                    let d = registry.get(name).unwrap_or_else(|| panic!("no description {name}"));
                    let data = d.build_attention(&t, &specrt::codegen::attention_symbols(&spec, 1, 8192)).unwrap();
                    assert_eq!(data, rust, "{name} {}", spec.name);
                    assert_eq!(data.hash(), rust.hash(), "{name} {}", spec.name);
                }
            }
        }
    }
}

#[test]
fn codegen_attention_kernels_come_from_descriptions() {
    let target = kir::Target::sm_120();
    for spec in specs() {
        for planes in [1usize, 2] {
            let want = specrt::codegen::attention_kernels(&spec, planes, 8192, &target).unwrap();
            let shape = attention_shape(&spec, planes, 8192);
            assert_eq!(want.len(), 8);
            for k in &want {
                assert_eq!(k.shape, shape, "{} planes {planes}", k.family.block());
            }
            for f in kir::attention::Family::ALL {
                assert!(want.iter().any(|k| k.family == f), "missing {f:?}");
            }
        }
    }
}

#[test]
fn embedded_kernel_descriptions_parse_and_are_delivered() {
    let registry = Registry::embedded();
    assert!(registry.descriptions().count() >= 8);
    for name in ["qk_norm_rope", "kv_norm_rope", "attention_split", "attention_reduce", "flash_attention", "flash_mma", "batch_decode_attention", "batch_decode_combine"] {
        assert!(registry.get(name).is_some(), "{name}");
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("specs");
    for name in ["qk_norm_rope", "kv_norm_rope", "attention_split", "attention_reduce", "batch_decode_attention", "batch_decode_combine"] {
        let text = std::fs::read_to_string(dir.join(format!("{name}.toml"))).unwrap();
        assert!(matches!(check(&text), Ok(Checked::Kernel(_))), "{name}");
    }
}

#[test]
fn a_kernel_description_is_never_a_model_spec() {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/attention_split.toml")).unwrap();
    assert!(Spec::parse(&text).is_err());
    assert!(matches!(check(&text), Ok(Checked::Kernel(_))));
}

#[test]
fn a_downloaded_kernel_description_installs_into_the_registry() {
    let extra = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/batch_decode_combine.toml")).unwrap();
    let registry = Registry::from_texts([extra.as_str()]).unwrap();
    assert_eq!(registry.descriptions().count(), 1);
    assert!(registry.get("batch_decode_combine").is_some());
    assert!(registry.get("attention_split").is_none());
}
