use specrt::codegen::{GDN_LINEARS, MMA_LINEARS, Module, WeightKind, select};
use specrt::mtp::MTP_LINEARS;
use specrt::spec::{Quant, Spec, TileChoice};
use std::path::Path;

fn spec(file: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs").join(file)).unwrap()
}

const KIR_SNIPPETS: &[&str] = &[
    "cuda/prelude.cu",
    "cuda/quant_fp4.cu",
    "cuda/quant_fp8_block.cu",
    "cuda/quant_fp8_token.cu",
];


const LIBRARY_ONLY: &[&str] = &["cuda/prelude.cu"];

const VERBATIM_LINEAR: &[(&str, &str)] = &[
    ("cuda/linear/common.rs", "device functions by design, shared by every linear family"),
    ("cuda/linear/chunk_lib.rs", "mma split tile and fragment device functions by design"),
    ("cuda/linear/chunk_prep.rs", "triangular solve and WY prep is verbatim text, to port onto tile prims (#794)"),
];

const VERBATIM_ATTENTION: &[(&str, &str)] = &[];

const PRIM_LIBRARY_ATTENTION: &[(&str, &str)] = &[
    ("cuda/attention/mma.rs", "mma fragment and exp2 prims, device functions by design"),
    ("cuda/attention/decode_common.rs", "cp.async and fp8 convert prims plus decode chunk helpers, device functions by design"),
];

const VERBATIM_QUANT: &[(&str, &str)] = &[
    ("cuda/gemm/fp4.rs", "fp4 gemv and dequant text, to port onto prims (#793)"),
    ("cuda/gemm/fp4_head.rs", "head norm gemv text, to port onto prims (#793)"),
    ("cuda/ggml/decode.rs", "ggml Q4_K, Q5_K, Q6_K, Q8_0 row decoders, to port onto prims (#793)"),
    ("cuda/gemm/prim.rs", "the prim library itself"),
];

const VERBATIM_MOE: &[(&str, &str)] = &[
    ("cuda/moe/route.rs", "top-k routing, permute, combine are MoE-specific; the logits block sum keeps its warp-sum plus smem loop because block_reduce adds a __syncthreads and changes the PTX"),
    ("cuda/moe/gemv.rs", "expert row dots read per-expert fp4/fp8/bf16 rows; porting onto the shared row dots changes the PTX (#793)"),
    ("cuda/moe/prim.rs", "re-exports the shared gemm prims plus MoE constants and launch wrappers"),
];

#[test]
fn moe_builders_use_shared_prims_unless_listed() {
    let kir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kir/src");
    let mma = std::fs::read_to_string(kir.join("cuda/moe/mma.rs")).unwrap();
    for p in ["prim::mma_fp4", "prim::e2m1x2", "prim::cp8_zfill", "prim::swizzle", "prim::zero_frag", "prim::pipeline_prologue"] {
        assert!(mma.contains(p), "cuda/moe/mma.rs must build from {p}");
    }
    let gemv = std::fs::read_to_string(kir.join("cuda/moe/gemv.rs")).unwrap();
    assert!(gemv.contains("prim::e2m1_to_f32") && gemv.contains("prim::warp_ids"));
    assert!(!kir.join("cuda/moe.cu").exists() && !kir.join("cuda/moe_mma.cu").exists());
    for (f, why) in VERBATIM_MOE {
        assert!(kir.join(f).exists() && !why.is_empty(), "stale allowlist entry {f}");
    }
}

#[test]
fn gemm_and_ggml_builders_use_prims_unless_listed() {
    let kir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kir/src");
    for sub in ["cuda/gemm", "cuda/ggml"] {
        for e in std::fs::read_dir(kir.join(sub)).unwrap() {
            let path = e.unwrap().path();
            let rel = path.strip_prefix(&kir).unwrap().display().to_string();
            let text = std::fs::read_to_string(&path).unwrap();
            if !VERBATIM_QUANT.iter().any(|(f, _)| *f == rel) {
                assert!(text.contains("prim::") || text.contains("warp_reduce"), "{rel} is verbatim kernel text: build it from prims or list it with a reason");
            }
        }
    }
    let quant = std::fs::read_to_string(kir.join("cuda/quant.rs")).unwrap();
    assert!(quant.contains("block_reduce") && quant.contains("pack_bf16x8"));
    for (f, why) in VERBATIM_QUANT {
        assert!(kir.join(f).exists() && !why.is_empty(), "stale allowlist entry {f}");
    }
}

#[test]
fn allreduce_builders_use_prims() {
    let kir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kir/src");
    let family = std::fs::read_to_string(kir.join("cuda/allreduce.rs")).unwrap();
    assert!(!family.contains("include_str!"), "allreduce kernels are built, not included from text");
    for p in ["prim::rank_sum", "prim::grid_stride", "prim::sync_point", "prim::finish", "prim::library"] {
        assert!(family.contains(p), "cuda/allreduce.rs must build its bodies from {p}");
    }
    let lib = std::fs::read_to_string(kir.join("cuda/allreduce/prim.rs")).unwrap();
    assert!(!lib.contains("__global__"), "cuda/allreduce/prim.rs is a device library and must not define kernels");
}

const BUILT_IN_RUST: &[&str] = &["embed", "splice_rows", "rms_split", "argmax_rows", "apply_mask", "sample_rows", "argmax_advance", "bump", "copy_rows", "split_rows", "rope_inv_freq", "lin_ba_rows", "lin_in_norm", "lin_out_proj", "qkv_proj", "lm_head_norm"];

fn kernel_files(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries {
            let path = e.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "cu" || x == "wgsl") {
                out.push(path.strip_prefix(dir).unwrap().display().to_string());
            }
        }
    }
    out.sort();
    out
}

#[test]
fn every_kernel_comes_from_kir() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stray = kernel_files(&root.join("templates"));
    assert!(stray.is_empty(), "kernel text belongs in kir: {stray:?}");
    for file in ["src/codegen.rs", "src/wgsl.rs"] {
        let text = std::fs::read_to_string(root.join(file)).unwrap();
        assert!(!text.contains("render("), "{file} must not template kernel text");
        assert!(!text.contains("{{"), "{file} must not carry template slots");
    }
    let kir = root.join("../kir/src");
    let files = kernel_files(&kir);
    let unknown: Vec<&String> = files.iter().filter(|f| !KIR_SNIPPETS.contains(&f.as_str())).collect();
    assert!(unknown.is_empty(), "new kernel text in kir must be a builder, not a file: {unknown:?}");
    let wgsl: Vec<&String> = files.iter().filter(|f| f.ends_with(".wgsl")).collect();
    assert!(wgsl.is_empty(), "wgpu kernels come from kir builders: {wgsl:?}");
    for f in &files {
        let text = std::fs::read_to_string(kir.join(f)).unwrap();
        if LIBRARY_ONLY.contains(&f.as_str()) {
            assert!(!text.contains("__global__"), "{f} is a device library and must not define kernels");
        }
        for name in BUILT_IN_RUST {
            for def in [format!("void {name}("), format!("float {name}(")] {
                assert!(!text.contains(&def), "{name} is generated by a kir builder, not text in {f}");
            }
        }
    }
}

#[test]
fn attention_builders_use_prims_unless_listed() {
    let kir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kir/src");
    let dir = kir.join("cuda/attention");
    let mut seen = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let path = e.unwrap().path();
        if path.extension().is_none_or(|x| x != "rs") || path.file_name().is_some_and(|n| n == "prim.rs") {
            continue;
        }
        seen += 1;
        let rel = path.strip_prefix(&kir).unwrap().display().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        let listed = VERBATIM_ATTENTION.iter().chain(PRIM_LIBRARY_ATTENTION).any(|(f, _)| *f == rel);
        if !listed {
            assert!(text.contains("prim::"), "{rel} is verbatim kernel text: build it from prims or list it with a reason");
        }
    }
    assert!(seen >= 9);
    for (f, why) in VERBATIM_ATTENTION.iter().chain(PRIM_LIBRARY_ATTENTION) {
        assert!(kir.join(f).exists() && !why.is_empty(), "stale allowlist entry {f}");
    }
}

#[test]
fn linear_builders_use_prims_unless_listed() {
    let kir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kir/src");
    let dir = kir.join("cuda/linear");
    let mut seen = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let path = e.unwrap().path();
        if path.extension().is_none_or(|x| x != "rs") || path.file_name().is_some_and(|n| n == "prim.rs") {
            continue;
        }
        seen += 1;
        let rel = path.strip_prefix(&kir).unwrap().display().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        if !VERBATIM_LINEAR.iter().any(|(f, _)| *f == rel) {
            assert!(text.contains("prim::") || text.contains("chunk_"), "{rel} is verbatim kernel text: build it from prims or list it with a reason");
        }
    }
    assert!(seen >= 8);
    for (f, why) in VERBATIM_LINEAR {
        assert!(kir.join(f).exists() && !why.is_empty(), "stale allowlist entry {f}");
    }
}

#[test]
fn module_bakes_spec_constants() {
    let m = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(!m.source.contains("{{"));
    assert!(m.source.contains("constexpr int HEAD_DIM = 128;"));
    assert!(m.source.contains("constexpr int GROUP = 2;"));
    assert!(m.source.contains("constexpr bool QK_NORM = true;"));
    assert!(!m.source.contains("MAX_CTX"));
    assert!(m.source.contains("linear_w<DecBf16, 4096, 1024, false, false>"));
    assert!(m.source.contains("linear_w<DecBf16, 151936, 1024, false, false>"));
}

#[test]
fn second_model_differs_only_by_spec() {
    let a = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    let b = Module::generate(&spec("qwen2.5-0.5b.toml"), 256).unwrap();
    assert_ne!(a.source, b.source);
    assert_eq!(a.kernels, b.kernels);
    assert!(b.source.contains("constexpr bool QK_NORM = false;"));
    assert!(b.source.contains("linear_w<DecBf16, 1152, 896, true, false>"));
}

#[test]
fn q8_spec_selects_q8_linears() {
    let mut s = spec("qwen3-0.6b.toml");
    s.quant = Quant::Q8;
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("linear_w<DecQ8, 4096, 1024, false, false>"));
    assert!(m.source.contains("linear_w<DecQ8, 151936, 1024, false, false>"));
    assert!(!m.source.contains("linear_w<DecBf16,"));
}

#[test]
fn bf16_prefill_splits_activations_and_q8_does_not() {
    let bf16 = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(bf16.source.contains("EPI_ACT, false, DecBf16, true, false>"));
    let mut s = spec("qwen3-0.6b.toml");
    s.quant = Quant::Q8;
    let q8 = Module::generate(&s, 256).unwrap();
    assert!(q8.source.contains("EPI_ACT, false, DecQ8, false, false>"));
    assert!(!q8.source.contains(", true, false>(w, s, b, xh"));
}

#[test]
fn fp8_spec_uses_fp8_projections_and_keeps_lm_head_and_gates_bf16() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-exact.toml");
    let s = Spec::load(&path).unwrap();
    let m = Module::generate(&s, 4096).unwrap();
    assert!(!m.source.contains("{{"));
    assert!(m.source.contains("linear_w<DecFp8, 5120, 6144, false, true>"));
    assert!(m.source.contains("linear_w<DecBf16, 248320, 5120, false, false>"));
    assert!(m.source.contains("linear_norm<DecFp8, 16384, false>"));
    assert!(m.source.contains("linear_norm<DecBf16, 96, false>"));
    let tc: Vec<&str> = m.source.lines().filter(|l| l.contains("gemm_tc<")).collect();
    let has = |s: &str| tc.iter().any(|l| l.contains(s));
    assert!(has("gemm_tc<34816, 5120, 128, 128, 2, 4, 64, 2, EPI_ACT, false, DecFp8, true, false>"), "{tc:#?}");
    assert!(has("gemm_tc<248320, 5120, 128, 128, 2, 4, 32, 2, EPI_STORE, false, DecBf16, true, false>"));
    assert!(has("gemm_tc<96, 5120, 64, 64, 2, 2, 32, 2, EPI_STORE, false, DecBf16, true, false>"));
    assert!(m.kernels.contains(&"lin_ba_mma"));
}

#[test]
fn head_quant_picks_the_head_kernels_and_bf16_stays_the_master_source() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml");
    let plain = Spec::load(&path).unwrap();
    let base = Module::generate(&plain, 4096).unwrap();
    let mut bf16 = plain.clone();
    bf16.head_quant = Some(Quant::Bf16);
    assert_eq!(Module::generate(&bf16, 4096).unwrap(), base, "head_quant = bf16 is the default");
    assert!(base.source.contains("linear_norm<DecBf16, 248320, false>"));
    assert!(!base.source.contains("lm_head_norm_f4"));
    let mut fp8 = plain.clone();
    fp8.head_quant = Some(Quant::Fp8Channel);
    let m = Module::generate(&fp8, 4096).unwrap();
    assert!(m.source.contains("linear_norm<DecFp8, 248320, false>"));
    assert!(m.source.contains("gemm_tc<248320, 5120, 128, 128, 2, 4, 64, 2, EPI_STORE, false, DecFp8, true, false>"));
    let mut fp4 = plain.clone();
    fp4.head_quant = Some(Quant::Nvfp4G16);
    let m = Module::generate(&fp4, 4096).unwrap();
    assert!(!m.source.contains("{{"));
    for k in ["lm_head_norm_f4", "lm_head_f4w", "lm_head_f4w_big", "lm_head_f4w_narrow", "lm_head_f4w_tiny"] {
        assert!(m.kernels.contains(&k), "{k}");
        assert!(m.source.contains(&format!(") {k}(")), "{k}");
        if k != "lm_head_norm_f4" {
            assert!(m.smem.contains_key(k), "{k}");
        }
    }
    assert!(m.source.contains("gemm_tc<248320, 5120, 128, 128, 2, 4, 64, 2, EPI_STORE, false, DecFp4, true, false>"));
    assert!(m.source.contains("head_norm_f4<248320, 5120, 2>"));
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
    assert!(m.source.contains("gemm<1152, 896, true, false, DecBf16>"));
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
            if name.contains("family-") {
                continue;
            }
            if std::fs::read_to_string(&path).map(|t| t.contains("kind = \"kir-kernel\"")).unwrap_or(false) {
                continue;
            }
            out.push((name, Spec::load(&path).unwrap()));
        } else if name.ends_with("config.json") && !name.contains("tokenizer") && path.exists() {
            let json = std::fs::read_to_string(&path).unwrap();
            if json.contains("\"num_experts\"") || json.contains("\"glm5_next\"") {
                continue;
            }
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
        Quant::Gguf => WeightKind::Ggml,
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
            for target in [kir::Target::sm_120(), kir::Target::sm_80()] {
            let m = Module::generate_kv(&spec, 256, 2, &target).unwrap();
            let gemms = specrt::gemm::gemm_kernels(&spec, &target).unwrap();
            let gdn: &[&str] = if spec.linear_attention.is_some() { &GDN_LINEARS } else { &[] };
            let mtp: &[&str] = if spec.linear_attention.is_some() && spec.speculative().is_some() { &MTP_LINEARS } else { &[] };
            if !mtp.is_empty() {
                mtp_specs += 1;
            }
            for &linear in MMA_LINEARS.iter().chain(gdn).chain(mtp) {
                for weight in weight_kinds(&spec, linear) {
                    for &n in &tokens {
                        for sms in [48, 84, 188] {
                            let t = kir::Target { sm_count: sms, ..target };
                            let c = select(&spec, &t, linear, n, weight).unwrap();
                            let mut need = vec![c.name.clone()];
                            need.extend(c.quant.map(str::to_string));
                            need.extend(c.reduce.clone());
                            for k in &need {
                                if !m.kernels.contains(&k.as_str()) {
                                    missing.insert(format!("{file} {:?} {k} not generated on {}", spec.runtime.tile, target.describe()));
                                }
                            }
                            if !m.smem.contains_key(c.name.as_str()) {
                                missing.insert(format!("{file} {} has no smem entry", c.name));
                            }
                            if let Some(k) = &c.kernel {
                                let built = gemms.kernels.iter().find(|g| g.name == k.name);
                                if built.is_none_or(|g| g.kind != k.kind || g.smem_bytes != k.smem_bytes) {
                                    missing.insert(format!("{file} {} dispatched differs from the generated instance", k.name));
                                }
                            }
                            checked += 1;
                        }
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
        yarn: None,
        longrope: None,
    });
    s.mlp.activation = specrt::spec::Activation::Geglu;
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("constexpr int ROPE_KIND = 1;"));
    assert!(m.source.contains("constexpr int ACT_KIND = 1;"));
    assert!(!m.source.contains("{{"));
}

#[test]
fn kir_kernels_are_emitted_and_hashed_into_the_module() {
    let s = spec("qwen3-0.6b.toml");
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("kir_block_sum"));
    assert!(m.source.contains("extern \"C\" __global__ void __launch_bounds__(1024) rmsnorm("));
    assert!(m.source.contains("constexpr int WIDTH = 1024;"));
    let other = Module::generate(&spec("qwen2.5-0.5b.toml"), 256).unwrap();
    assert_ne!(m.kir_hash, other.kir_hash);
    assert_eq!(m.kir_hash, Module::generate(&s, 256).unwrap().kir_hash);
}

#[test]
fn kir_kernels_follow_the_target() {
    let s = spec("qwen3-0.6b.toml");
    let tree = specrt::codegen::kir_kernels(&s, &kir::Target::wgpu_baseline()).unwrap();
    let sub = specrt::codegen::kir_kernels(&s, &kir::Target::wgpu_baseline().with_subgroups(32)).unwrap();
    assert_ne!(kir::module_hash(&tree), kir::module_hash(&sub));
}

#[test]
fn vision_builders_use_prims() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kir/src/cuda/vision");
    let text = std::fs::read_to_string(dir.join("kernels.rs")).unwrap();
    let prim = std::fs::read_to_string(dir.join("prim.rs")).unwrap();
    assert!(text.contains("attn::zero_scores") && text.contains("attn::rescale_acc") && text.contains("attn::pack_rows") && text.contains("attn::online_softmax_with"), "vision builders must call the shared attention prims");
    let mut found = Vec::new();
    for line in text.lines().chain(prim.lines()) {
        if let Some(rest) = line.split("__global__ void ").nth(1) {
            let rest = rest.strip_prefix("__launch_bounds__(GM_THREADS) ").or_else(|| rest.strip_prefix("__launch_bounds__(32 * AT_WARPS) ")).unwrap_or(rest);
            found.push(rest.split('(').next().unwrap().to_string());
        }
    }
    assert!(found.len() >= 8, "{found:?}");
    for name in &found {
        let section: String = text.split("extern \"C\" __global__ void").skip(1).find(|s| s.split('\n').next().is_some_and(|l| l.contains(&format!(" {name}(")))).unwrap_or("").to_string();
        if section.is_empty() {
            assert!(prim.contains(&format!("void {name}(")), "{name} is not a builder in kernels.rs or a prim in prim.rs");
            continue;
        }
        let templated = section.contains("@") || section.contains("{{") || section.contains("prim::");
        assert!(templated, "{name} is verbatim kernel text: build it from prims");
    }
}
