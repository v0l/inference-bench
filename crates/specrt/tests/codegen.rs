use specrt::codegen::{GDN_LINEARS, MMA_LINEARS, Module, WeightKind, kernel_for, linear_dims, render};
use specrt::mtp::MTP_LINEARS;
use specrt::spec::{Quant, Spec, TileChoice};
use std::collections::BTreeMap;
use std::path::Path;

fn spec(file: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs").join(file)).unwrap()
}

#[test]
fn render_substitutes_every_slot() {
    let vars = BTreeMap::from([("A", "1".to_string()), ("B", "two".to_string())]);
    assert_eq!(render("x{{A}}y{{B}}{{A}}", &vars).unwrap(), "x1ytwo1");
}

#[test]
fn render_rejects_unknown_slot() {
    let vars = BTreeMap::from([("A", "1".to_string())]);
    assert!(render("{{A}}{{MISSING}}", &vars).is_err());
}

#[test]
fn module_bakes_spec_constants() {
    let m = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(!m.source.contains("{{"));
    assert!(m.source.contains("constexpr int HEAD_DIM = 128;"));
    assert!(m.source.contains("constexpr int GROUP = 2;"));
    assert!(m.source.contains("constexpr bool QK_NORM = true;"));
    assert!(m.source.contains("constexpr int MAX_CTX = 256;"));
    assert!(m.source.contains("linear_bf16<4096, 1024, false, false>"));
    assert!(m.source.contains("linear_bf16<151936, 1024, false, false>"));
}

#[test]
fn second_model_differs_only_by_spec() {
    let a = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    let b = Module::generate(&spec("qwen2.5-0.5b.toml"), 256).unwrap();
    assert_ne!(a.source, b.source);
    assert_eq!(a.kernels, b.kernels);
    assert!(b.source.contains("constexpr bool QK_NORM = false;"));
    assert!(b.source.contains("linear_bf16<1152, 896, true, false>"));
}

#[test]
fn q8_spec_selects_q8_linears() {
    let mut s = spec("qwen3-0.6b.toml");
    s.quant = Quant::Q8;
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("linear_q8<4096, 1024, false, false>"));
    assert!(m.source.contains("linear_q8<151936, 1024, false, false>"));
    assert!(!m.source.contains("linear_bf16<"));
}

#[test]
fn bf16_prefill_splits_activations_and_q8_does_not() {
    let bf16 = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(bf16.source.contains("EPI_ACT, false, 0, true>"));
    let mut s = spec("qwen3-0.6b.toml");
    s.quant = Quant::Q8;
    let q8 = Module::generate(&s, 256).unwrap();
    assert!(q8.source.contains("EPI_ACT, false, 1, false>"));
    assert!(!q8.source.contains(", true>(w, s, b, xh"));
}

#[test]
fn fp8_spec_uses_fp8_projections_and_keeps_lm_head_and_gates_bf16() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml");
    let s = Spec::load(&path).unwrap();
    let m = Module::generate(&s, 4096).unwrap();
    assert!(!m.source.contains("{{"));
    assert!(m.source.contains("linear_fp8<5120, 6144, false, true>"));
    assert!(m.source.contains("linear_bf16<248320, 5120, false, false>"));
    assert!(m.source.contains("linear_norm<2, LA_IN_ROWS, false>"));
    assert!(m.source.contains("constexpr int LA_IN_ROWS = LA_CONV_DIM + LA_VALUE_DIM;"));
    assert!(m.source.contains("linear_norm<WF_BF16, LA_BA_ROWS, false>"));
    let tc: Vec<&str> = m.source.lines().filter(|l| l.contains("gemm_tc<")).collect();
    let has = |s: &str| tc.iter().any(|l| l.contains(s));
    assert!(has("gemm_tc<34816, 5120, 64, 128, 2, 4, 64, 2, EPI_ACT, false, 2, true>"), "{tc:#?}");
    assert!(has("gemm_tc<248320, 5120, 64, 128, 2, 4, 32, 2, EPI_STORE, false, 0, true>"));
    assert!(has("gemm_tc<96, 5120, 64, 64, 2, 2, 32, 2, EPI_STORE, false, 0, true>"));
    assert!(m.kernels.contains(&"lin_ba_mma"));
}

#[test]
fn head_quant_picks_the_head_kernels_and_bf16_stays_the_master_source() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.toml");
    let plain = Spec::load(&path).unwrap();
    let base = Module::generate(&plain, 4096).unwrap();
    let mut bf16 = plain.clone();
    bf16.head_quant = Some(Quant::Bf16);
    assert_eq!(Module::generate(&bf16, 4096).unwrap(), base, "head_quant = bf16 is the default");
    assert!(base.source.contains("linear_norm<0, 248320, false>"));
    assert!(!base.source.contains("lm_head_norm_f4"));
    let mut fp8 = plain.clone();
    fp8.head_quant = Some(Quant::Fp8Channel);
    let m = Module::generate(&fp8, 4096).unwrap();
    assert!(m.source.contains("linear_norm<2, 248320, false>"));
    assert!(m.source.contains("gemm_tc<248320, 5120, 64, 128, 2, 4, 64, 2, EPI_STORE, false, 2, true>"));
    let mut fp4 = plain.clone();
    fp4.head_quant = Some(Quant::Nvfp4G16);
    let m = Module::generate(&fp4, 4096).unwrap();
    assert!(!m.source.contains("{{"));
    for k in specrt::codegen::FP4_HEAD_KERNELS {
        assert!(m.kernels.contains(k), "{k}");
        assert!(m.source.contains(&format!(") {k}(")), "{k}");
        if *k != "lm_head_norm_f4" {
            assert!(m.smem.contains_key(k), "{k}");
        }
    }
    assert!(m.source.contains("gemm_f4w<248320, 5120, 64, 128, 2, 4, 3, EPI_STORE>"));
    assert!(m.source.contains("constexpr int HEAD_F4_ROWS = 2;"));
    let dense = spec("qwen3-0.6b.toml");
    let mut dense_fp4 = dense.clone();
    dense_fp4.head_quant = Some(Quant::Nvfp4G16);
    let m = Module::generate(&dense_fp4, 256).unwrap();
    assert!(m.kernels.contains(&"lm_head_norm_f4") && !m.kernels.contains(&"gate_up_act_f4"));
    assert!(!m.source.contains("gate_up_f4w"));
}

#[test]
fn generation_is_deterministic() {
    let s = spec("qwen3-0.6b.toml");
    assert_eq!(
        Module::generate(&s, 256).unwrap().source,
        Module::generate(&s, 256).unwrap().source
    );
}

#[test]
fn module_has_batched_gemm_for_every_linear() {
    let m = Module::generate(&spec("qwen2.5-0.5b.toml"), 256).unwrap();
    for name in ["qkv_gemm", "o_gemm", "gate_up_gemm", "down_gemm", "lm_head_gemm"] {
        assert!(m.kernels.contains(&name), "{name}");
        assert!(m.source.contains(&format!("void {name}(")), "{name}");
    }
    assert!(m.source.contains("gemm<1152, 896, true, false, 0>"));
}

#[test]
fn hybrid_spec_adds_the_delta_net_kernels_and_dense_specs_do_not() {
    let json = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5/config.json"),
    )
    .unwrap();
    let hybrid = Module::generate(&Spec::from_hf_config(&json).unwrap(), 256).unwrap();
    assert!(!hybrid.source.contains("{{"));
    for name in specrt::codegen::GDN_KERNELS {
        assert!(hybrid.kernels.contains(name), "{name}");
        assert!(hybrid.source.contains(&format!("void {name}(")) || hybrid.source.contains(&format!(") {name}(")), "{name}");
    }
    assert!(hybrid.source.contains("constexpr int QKV_ROW = 1536;"));
    assert!(hybrid.source.contains("constexpr int ROT = 32;"));
    assert!(hybrid.source.contains("constexpr bool OUT_GATE = true;"));
    assert!(hybrid.source.contains("constexpr int LA_VH = 6;"));
    assert!(hybrid.smem.contains_key("gdn_chunk_prep"));
    let dense = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(!dense.source.contains("gdn_step"));
    assert!(dense.kernels.iter().all(|k| !k.starts_with("gdn_") && !k.starts_with("lin_")));
    assert!(dense.source.contains("constexpr int ROT = 128;"));
    assert!(dense.source.contains("constexpr bool OUT_GATE = false;"));
}

fn dispatch_specs() -> Vec<(String, Spec)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    let mut files = Vec::new();
    for dir in [root.join("specs"), root.join("tests/fixtures")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            files.push(if path.is_dir() { path.join("config.json") } else { path });
        }
    }
    files.sort();
    for path in files {
        let name = path.strip_prefix(root).unwrap().display().to_string();
        if name.ends_with(".toml") {
            out.push((name, Spec::load(&path).unwrap()));
        } else if name.ends_with("config.json") && !name.contains("tokenizer") && path.exists() {
            let json = std::fs::read_to_string(&path).unwrap();
            out.push((name, Spec::from_hf_config(&json).unwrap()));
        }
    }
    out
}

fn dispatch_variants(base: &Spec) -> Vec<Spec> {
    use specrt::spec::ActQuant;
    let heads = [None, Some(Quant::Bf16), Some(Quant::Fp8Channel), Some(Quant::Nvfp4G16)];
    let mut out = Vec::new();
    for head in heads {
        for act in [ActQuant::None, ActQuant::Fp8] {
            for mlp_act in [ActQuant::None, ActQuant::Fp4] {
                for tile in [TileChoice::Auto, TileChoice::Small, TileChoice::Big, TileChoice::Narrow] {
                    let mut s = base.clone();
                    s.head_quant = head;
                    s.act_quant = act;
                    s.mlp.act_quant = mlp_act;
                    s.runtime.tile = tile;
                    s.runtime.gemm_fp32 = false;
                    if s.validate().is_ok() {
                        out.push(s);
                    }
                }
            }
        }
    }
    out
}

fn weight_kinds(spec: &Spec, linear: &str) -> Vec<WeightKind> {
    let kind = |q: Quant| match q {
        Quant::Bf16 => WeightKind::Bf16,
        Quant::Q8 => WeightKind::Q8,
        Quant::Fp8Block128 | Quant::Fp8Channel => WeightKind::Fp8,
        Quant::Nvfp4G16 => WeightKind::Fp4,
    };
    let mut out: Vec<WeightKind> = match linear {
        "lm_head" => vec![kind(spec.lm_head_quant())],
        "lin_ba" => vec![WeightKind::Bf16],
        l if l.starts_with("mtp_") => vec![WeightKind::Bf16],
        "gate_up" | "down" => (0..spec.layers.count).map(|l| kind(spec.mlp_quant(l))).collect(),
        _ => vec![kind(spec.quant)],
    };
    out.dedup();
    out
}

#[test]
fn every_dispatched_kernel_is_generated() {
    let tokens = [1, 8, 9, 16, 17, 32, 64, 128, 256, 512, 1024, 2048];
    let mut checked = 0;
    let mut mtp_specs = 0;
    let mut missing = std::collections::BTreeSet::new();
    for (file, base) in dispatch_specs() {
        for spec in dispatch_variants(&base) {
            let m = Module::generate(&spec, 256).unwrap();
            let gdn: &[&str] = if spec.linear_attention.is_some() { &GDN_LINEARS } else { &[] };
            let mtp: &[&str] = if spec.linear_attention.is_some() && spec.speculative().is_some() { &MTP_LINEARS } else { &[] };
            if !mtp.is_empty() {
                mtp_specs += 1;
            }
            for &linear in MMA_LINEARS.iter().chain(gdn).chain(mtp) {
                let rows = if linear.starts_with("mtp_") { specrt::mtp::dims(&spec, linear).0 } else { linear_dims(&spec, linear).0 };
                for weight in weight_kinds(&spec, linear) {
                    for &n in &tokens {
                        for sms in [48, 84, 188] {
                            let c = kernel_for(&spec, linear, rows, n, sms, weight);
                            let mut need = vec![c.name.clone()];
                            need.extend(c.quant.map(str::to_string));
                            need.extend(c.reduce.clone());
                            for k in &need {
                                if !m.kernels.contains(&k.as_str()) {
                                    missing.insert(format!("{file} {:?} {k} not generated", spec.runtime.tile));
                                }
                            }
                            if !m.smem.contains_key(c.name.as_str()) {
                                missing.insert(format!("{file} {} has no smem entry", c.name));
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "{checked}");
    assert!(mtp_specs > 0);
    assert!(missing.is_empty(), "{missing:#?}");
}

#[test]
fn rope_scaling_and_activation_are_baked() {
    let mut s = spec("qwen3-0.6b.toml");
    let plain = Module::generate(&s, 256).unwrap();
    assert!(plain.source.contains("constexpr int ROPE_KIND = 0;"));
    assert!(plain.source.contains("constexpr int ACT_KIND = 0;"));
    s.rope.scaling = Some(specrt::spec::RopeScaling {
        kind: specrt::spec::RopeScalingKind::Linear,
        factor: 4.0,
        low_freq_factor: None,
        high_freq_factor: None,
        original_max_positions: None,
    });
    s.mlp.activation = specrt::spec::Activation::Geglu;
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("constexpr int ROPE_KIND = 1;"));
    assert!(m.source.contains("constexpr int ACT_KIND = 1;"));
    assert!(!m.source.contains("{{"));
}
