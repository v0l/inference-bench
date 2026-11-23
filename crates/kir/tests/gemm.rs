use kir::compose::Stage;
use kir::expr::Expr;
use kir::gemm::{
    BatchShape, Epilogue, Fp4Gemv, Fp8Scale, GemmPath, GemmShape, KernelInstance, MmaShape, Policy, Request, Tile, TileChoice,
    Tuning, Variant, WeightFormat, select,
};
use kir::heavy::{self, Family, Heavy, HeavyError, Slot};
use kir::{Dialect, Feature, Target};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/gemm").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn gate_up() -> GemmShape {
    GemmShape::new(34816, 5120, Epilogue::Act, false)
}

fn down() -> GemmShape {
    GemmShape::new(5120, 17408, Epilogue::Residual, false)
}

fn qkv() -> GemmShape {
    GemmShape::new(14336, 5120, Epilogue::Store, true)
}

fn up_map() -> GemmShape {
    GemmShape::new(15680, 4480, Epilogue::Map, false)
}

#[test]
fn the_mapped_epilogue_stores_one_activation_per_column_into_planes() {
    let t = Target::sm_120();
    let lib = kir::cuda::emit_gemms(&[
        KernelInstance::gemm(&t, "up_map_mma", up_map(), WeightFormat::Bf16, Tuning::default().tile(Variant::Small), true, false).unwrap(),
        KernelInstance::batch(&t, "up_map_f8b16", up_map(), WeightFormat::Fp8, 16).unwrap(),
    ])
    .unwrap();
    assert!(lib.contains("constexpr int EPI_MAP = 3;"));
    assert!(lib.contains("mapped_act(acc[i][j][h * 2])"), "the mma epilogue maps each column");
    assert!(lib.contains("split_hi_lo(mapped_act(v), hi, lo);"), "the batch store maps each column");
    assert!(KernelInstance::gemm(&Target::sm_120(), "odd", GemmShape::new(15, 4480, Epilogue::Map, false), WeightFormat::Bf16, Tile::SMALL, true, false).is_err());
}

fn cuda(k: &KernelInstance) -> String {
    format!("// {}\n{}", k.describe().replace('\n', "\n// "), kir::cuda::emit_gemm(k).unwrap())
}

fn wgsl() -> Target {
    Target::wgpu_baseline().with_subgroups(32).with_feature(Feature::ShaderF16)
}

fn families(t: &Target) -> Vec<KernelInstance> {
    let tuning = Tuning::default();
    let mut out = vec![
        KernelInstance::gemm(t, "qkv_mma_big", qkv(), WeightFormat::Bf16, tuning.tile(Variant::Big), true, false).unwrap(),
        KernelInstance::gemm(t, "gate_up_mma", gate_up(), WeightFormat::Fp8, tuning.tile(Variant::Small), true, false).unwrap(),
        KernelInstance::gemm(t, "up_map_mma", up_map(), WeightFormat::Bf16, tuning.tile(Variant::Small), true, false).unwrap(),
        KernelInstance::batch(t, "up_map_f8b16", up_map(), WeightFormat::Fp8, 16).unwrap(),
        KernelInstance::gemm(t, "down_mma_tiny", down(), WeightFormat::Q8, tuning.tile(Variant::Tiny), false, false).unwrap(),
        KernelInstance::gemm(t, "qkv_ggb_narrow", qkv(), WeightFormat::Bf16, tuning.tile(Variant::Narrow), true, true).unwrap(),
        KernelInstance::gemm(t, "gate_up_f4w_big", gate_up(), WeightFormat::Fp4, tuning.tile(Variant::Big), true, false).unwrap(),
        KernelInstance::batch(t, "gate_up_f4b16", gate_up(), WeightFormat::Fp4, 16).unwrap(),
        KernelInstance::batch_reduce(t, "gate_up_f4b_reduce", gate_up(), WeightFormat::Fp4).unwrap(),
        KernelInstance::ggml_batch(t, "down_ggr8", down(), 8, false).unwrap(),
        KernelInstance::batch_reduce(t, "down_ggr_reduce", down(), WeightFormat::Ggml).unwrap(),
        KernelInstance::ggml_gemv(t, "down_ggv", GemmShape::new(1024, 3072, Epilogue::Residual, false), false).unwrap(),
        KernelInstance::ggml_gemv(t, "gate_up_ggv", GemmShape::new(6144, 1024, Epilogue::Act, false), true).unwrap(),
        KernelInstance::batch(t, "qkv_f8b64", qkv(), WeightFormat::Fp8, 64).unwrap(),
        KernelInstance::batch_reduce(t, "qkv_f8b_reduce", qkv(), WeightFormat::Fp8).unwrap(),
        KernelInstance::fp4_gemv(t, "gate_up_act_f4", Fp4Gemv::GateUpAct, 34816, 5120).unwrap(),
        KernelInstance::fp4_gemv(t, "down_proj_f4", Fp4Gemv::Down, 5120, 17408).unwrap(),
        KernelInstance::fp4_head_norm(t, "lm_head_norm_f4", 248320, 5120).unwrap(),
    ];
    if t.has(Feature::Fp8Mma) {
        out.push(KernelInstance::quant_fp8(t, Fp8Scale::Block).unwrap());
        out.push(KernelInstance::fp8_act(t, "gate_up_f8_huge", gate_up(), Tile::FP8_HUGE, Fp8Scale::Channel).unwrap());
    }
    if t.has(Feature::BlockScaledFp4) {
        out.push(KernelInstance::quant_fp4(t).unwrap());
        out.push(KernelInstance::fp4_act(t, "down_f4a_big", down(), tuning.fp4_tile(Variant::Big), &tuning).unwrap());
    }
    out
}

#[test]
fn cuda_gemm_goldens_per_family_and_target() {
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        for k in families(&t) {
            golden(&format!("{}_{tag}.cu", k.name), &cuda(&k));
        }
    }
}

#[test]
fn wgsl_dequant_goldens_per_format() {
    let t = wgsl();
    for (name, shape, wf) in [
        ("qkv_mma", qkv(), WeightFormat::Bf16),
        ("down_mma", down(), WeightFormat::Fp8),
        ("gate_up_f4w", gate_up(), WeightFormat::Fp4),
    ] {
        let k = KernelInstance::gemm(&t, name, shape, wf, Tile::SMALL, true, false).unwrap();
        let src = kir::wgsl::emit_gemm(&k).unwrap();
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::SHADER_FLOAT16)
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        golden(&format!("{name}.wgsl"), &src);
    }
}

#[test]
fn the_bf16_activation_stage_splits_hi_and_lo_planes_by_tile_size_not_by_total() {
    let t = Target::sm_120();
    let k = KernelInstance::gemm(&t, "qkv_mma", qkv(), WeightFormat::Bf16, Tuning::default().tile(Variant::Small), true, false).unwrap();
    let src = kir::cuda::emit_gemms(&[k]).unwrap();
    assert!(src.contains("int p = idx / (BM * A_CHUNKS);"), "plane index must divide by one plane's chunk count");
    assert!(src.contains("int rem = idx % (BM * A_CHUNKS);"));
    assert!(!src.contains("idx / (A_TOTAL)"));
}

#[test]
fn sm120_gets_the_block_scaled_fp4_mma_and_sm80_does_not() {
    let k = KernelInstance::fp4_act(&Target::sm_120(), "down_f4a", down(), Tile::SMALL, &Tuning::default()).unwrap();
    assert!(kir::cuda::emit_gemms(&[k]).unwrap().contains("kind::mxf4nvf4.block_scale"));
    let err = KernelInstance::fp4_act(&Target::sm_80(), "down_f4a", down(), Tile::SMALL, &Tuning::default()).unwrap_err();
    assert!(err.is_unsupported());
    assert!(err.to_string().contains("block BlockScaledFp4Mma") && err.to_string().contains("BlockScaledFp4"), "{err}");
    let err = KernelInstance::fp8_act(&Target::sm_80(), "down_f8", down(), Tile::SMALL, Fp8Scale::Block).unwrap_err();
    assert!(err.to_string().contains("block Fp8ActMma"), "{err}");
}

fn policy(fp8: bool, fp4: bool) -> Policy {
    Policy { tuning: Tuning::default(), fp8_weights: fp8.then_some(Fp8Scale::Channel), fp8_act: fp8, fp4_act: fp4 }
}

fn req(linear: &str, shape: GemmShape, tokens: usize, weight: WeightFormat) -> Request<'_> {
    Request { linear, shape, tokens, weight, split: true }
}

#[test]
fn nvfp4_prefill_falls_back_to_dequant_bf16_mma_without_block_scaled_fp4() {
    let p = policy(true, true);
    let on_120 = select(&Target::sm_120(), &p, &req("down", down(), 2048, WeightFormat::Fp4)).unwrap();
    assert_eq!((on_120.path, on_120.name.as_str(), on_120.quant), (GemmPath::F4a, "down_f4a_big", Some("quant_rows_fp4")));
    let on_80 = select(&Target::sm_80(), &p, &req("down", down(), 2048, WeightFormat::Fp4)).unwrap();
    assert_eq!((on_80.path, on_80.name.as_str()), (GemmPath::F4w, "down_f4w_big"));
    let k = on_80.kernel.unwrap();
    assert!(kir::cuda::emit_gemms(&[k]).unwrap().contains("DecFp4, true, false>"));
}

#[test]
fn selection_reproduces_the_decode_and_prefill_paths() {
    let t = Target::sm_120();
    let p = policy(true, true);
    let f8b = select(&t, &p, &req("qkv", qkv(), 9, WeightFormat::Fp8)).unwrap();
    assert_eq!((f8b.path, f8b.name.as_str(), f8b.reduce.as_deref()), (GemmPath::F8b, "qkv_f8b16", Some("qkv_f8b_reduce")));
    let f4b = select(&t, &policy(true, false), &req("gate_up", gate_up(), 33, WeightFormat::Fp4)).unwrap();
    assert_eq!(f4b.name, "gate_up_f4b64");
    let f4a = select(&t, &p, &req("gate_up", gate_up(), 33, WeightFormat::Fp4)).unwrap();
    assert_eq!(f4a.path, GemmPath::F4a);
    let huge = select(&t, &p, &req("gate_up", gate_up(), 4096, WeightFormat::Fp8)).unwrap();
    assert_eq!((huge.name.as_str(), huge.tile, huge.quant), ("gate_up_f8_huge", Tile::FP8_HUGE, Some("quant_rows_fp8_token")));
    let rows = select(&t, &p, &req("lin_ba", GemmShape::new(96, 5120, Epilogue::Store, false), 8, WeightFormat::Bf16)).unwrap();
    assert_eq!(rows.path, GemmPath::Rows);
    let head = select(&t, &p, &req("lm_head", GemmShape::new(248320, 5120, Epilogue::Store, false), 24, WeightFormat::Bf16)).unwrap();
    assert_eq!(head.name, "lm_head_mma_narrow");
    let tiny = select(&t, &p, &req("o", down(), 4, WeightFormat::Fp8)).unwrap();
    assert_eq!(tiny.path, GemmPath::F8b);
    let small = Policy { tuning: Tuning { tile: TileChoice::Small, ..Tuning::default() }, ..p };
    assert_eq!(select(&t, &small, &req("o", down(), 4, WeightFormat::Fp8)).unwrap().name, "o_f8");
    let plain = Policy { fp8_act: false, ..small };
    assert_eq!(select(&t, &plain, &req("o", down(), 4, WeightFormat::Fp8)).unwrap().name, "o_mma");
}

#[test]
fn small_ggml_batches_pick_the_ggml_batch_rows_and_large_ones_the_bf16_scratch() {
    let t = Target::sm_120();
    let p = policy(false, false);
    let ggr = select(&t, &p, &req("gate_up", gate_up(), 8, WeightFormat::Ggml)).unwrap();
    assert_eq!((ggr.path, ggr.name.as_str(), ggr.reduce.as_deref()), (GemmPath::Ggr, "gate_up_ggr8", Some("gate_up_ggr_reduce")));
    assert_eq!(select(&t, &p, &req("down", down(), 20, WeightFormat::Ggml)).unwrap().name, "down_ggr32");
    assert_eq!(select(&t, &p, &req("down", down(), 33, WeightFormat::Ggml)).unwrap().path, GemmPath::Ggb);
    let head = select(&t, &p, &req("lm_head", GemmShape::new(151936, 1024, Epilogue::Store, false), 4, WeightFormat::Ggml)).unwrap();
    assert_eq!(head.name, "lm_head_ggr8");
    let one = select(&t, &p, &req("down", GemmShape::new(1024, 3072, Epilogue::Residual, false), 1, WeightFormat::Ggml)).unwrap();
    assert_eq!((one.path, one.name.as_str(), one.reduce), (GemmPath::Ggv, "down_ggv", None));
    let head_one = select(&t, &p, &req("lm_head", GemmShape::new(151936, 1024, Epilogue::Store, false), 1, WeightFormat::Ggml)).unwrap();
    assert_eq!(head_one.name, "lm_head_ggv");
    assert_eq!(select(&t, &p, &req("down", down(), 1, WeightFormat::Ggml)).unwrap().path, GemmPath::Ggr, "a row too long for the smem vector keeps the batch path");
    assert_eq!(select(&t, &p, &req("down", GemmShape::new(1024, 3072, Epilogue::Residual, false), 2, WeightFormat::Ggml)).unwrap().path, GemmPath::Ggr);
}

#[test]
fn batch_splits_follow_the_target_sm_count() {
    let s = BatchShape::for_tokens(16);
    assert_eq!(s.tokens, 16);
    let few = Target { sm_count: 48, ..Target::sm_120() };
    assert!(s.splits(&few, 5120, 17408) <= s.splits(&Target::sm_120(), 5120, 17408));
    assert_eq!(BatchShape::for_tokens(9).tokens, 16);
    assert_eq!(BatchShape::for_tokens(100).tokens, 64);
    assert_eq!(s.blocks_per_sm(&Target::sm_120()), 4);
}

#[test]
fn fp4_tile_stages_fit_the_target_shared_memory() {
    let t = Target::sm_120();
    let big = MmaShape::new(&t, Tile::FP4_BIG, 5120, WeightFormat::Fp4, true);
    assert!(big.smem_bytes() <= t.smem_limit);
    let small_smem = Target { smem_limit: 48 * 1024, ..t };
    let s = MmaShape::new(&small_smem, Tile::FP4_BIG, 5120, WeightFormat::Fp4, true);
    assert_eq!(s.stages, 2);
    let tuning = Tuning { fp4_stages: Some(8), ..Tuning::default() };
    assert_eq!(tuning.f4a_stages(&t, Tile::FP4_BIG), t.smem_limit as usize / ((128 + 256) * 72));
}

#[test]
fn wgsl_refuses_what_it_cannot_lower_and_names_the_block() {
    let no_f16 = Target::wgpu_baseline();
    let err = KernelInstance::gemm(&no_f16, "qkv_mma", qkv(), WeightFormat::Bf16, Tile::SMALL, true, false).unwrap_err();
    assert!(err.to_string().contains("block DequantF16") && err.to_string().contains("ShaderF16"), "{err}");
    let err = KernelInstance::gemm(&wgsl(), "qkv_mma", qkv(), WeightFormat::Q8, Tile::SMALL, true, false).unwrap_err();
    assert!(err.to_string().contains("block DequantF16"), "{err}");
    let err = KernelInstance::batch(&wgsl(), "qkv_f8b8", qkv(), WeightFormat::Fp8, 8).unwrap_err();
    assert!(err.to_string().contains("block BatchRowsMma"), "{err}");
    let err = KernelInstance::fp4_gemv(&wgsl(), "down_proj_f4", Fp4Gemv::Down, 5120, 17408).unwrap_err();
    assert!(err.to_string().contains("block Fp4Gemv"), "{err}");
}

#[test]
fn bad_shapes_are_refused() {
    let t = Target::sm_120();
    let odd = GemmShape::new(64, 100, Epilogue::Store, false);
    assert!(KernelInstance::gemm(&t, "x_mma", odd, WeightFormat::Bf16, Tile::SMALL, true, false).is_err());
    let bad_tile = Tile { bm: 48, bn: 64, wm: 2, wn: 2 };
    assert!(KernelInstance::gemm(&t, "x_mma", qkv(), WeightFormat::Bf16, bad_tile, true, false).is_err());
    assert!(KernelInstance::gemm(&t, "x-mma", qkv(), WeightFormat::Bf16, Tile::SMALL, true, false).is_err());
}

#[test]
fn instance_hash_tracks_shape_target_and_tuning() {
    let t = Target::sm_120();
    let a = KernelInstance::gemm(&t, "qkv_mma", qkv(), WeightFormat::Bf16, Tile::SMALL, true, false).unwrap();
    assert_eq!(a.hash(), a.clone().hash());
    let b = KernelInstance::gemm(&Target::sm_80(), "qkv_mma", qkv(), WeightFormat::Bf16, Tile::SMALL, true, false).unwrap();
    assert_ne!(a.hash(), b.hash());
    let c = KernelInstance::gemm(&t, "qkv_mma", qkv(), WeightFormat::Bf16, Tile::NARROW, true, false).unwrap();
    assert_ne!(a.hash(), c.hash());
}

#[test]
fn nvfp4_act_rows_between_17_and_32_take_the_skinny_tiles() {
    let t = Target::sm_120();
    let p = policy(true, true);
    let gu = select(&t, &p, &req("gate_up", gate_up(), 24, WeightFormat::Fp4)).unwrap();
    assert_eq!((gu.path, gu.name.as_str(), gu.tile), (GemmPath::F4a, "gate_up_f4a_skinnywide", Tile::SKINNY_WIDE));
    let dn = select(&t, &p, &req("down", down(), 24, WeightFormat::Fp4)).unwrap();
    assert_eq!((dn.name.as_str(), dn.tile), ("down_f4a_skinny", Tile::SKINNY));
    let wide = select(&t, &p, &req("down", down(), 33, WeightFormat::Fp4)).unwrap();
    assert!(!wide.name.contains("skinny"), "{}", wide.name);
}

fn heavy_targets() -> Vec<Target> {
    vec![Target::sm_120(), Target::sm_80()]
}

#[test]
fn every_slotted_family_lifts_to_heavy_and_lowers_to_the_same_kernel_and_golden() {
    let mut lifted = 0;
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        for k in families(&t) {
            let Some(h) = Heavy::lift(&k).unwrap() else { continue };
            let back = h.compile(&t, "test").unwrap();
            assert_eq!(back, k, "{}", k.name);
            golden(&format!("{}_{tag}.cu", k.name), &cuda(&back));
            lifted += 1;
        }
    }
    let w = wgsl();
    for (name, shape, wf) in [("qkv_mma", qkv(), WeightFormat::Bf16), ("down_mma", down(), WeightFormat::Fp8)] {
        let k = KernelInstance::gemm(&w, name, shape, wf, Tile::SMALL, true, false).unwrap();
        let back = Heavy::lift(&k).unwrap().unwrap().lower(&w).unwrap();
        golden(&format!("{name}.wgsl"), &kir::wgsl::emit_gemm(&back).unwrap());
    }
    assert!(lifted >= 26, "only {lifted} kernels lifted");
}

#[test]
fn the_four_epilogues_are_stage_lists_the_emitters_spell() {
    for (epi, bias) in [(Epilogue::Store, false), (Epilogue::Store, true), (Epilogue::Residual, false), (Epilogue::Residual, true), (Epilogue::Act, false), (Epilogue::Map, false)] {
        let stages = heavy::epilogue(epi, bias, 64).unwrap();
        assert!(matches!(stages.last(), Some(Stage::TileStore { .. })), "{epi:?}");
        assert_eq!(heavy::spelled(&stages, 64).unwrap(), (epi, bias));
    }
    assert!(heavy::epilogue(Epilogue::Act, true, 64).is_err());
    let mut odd = heavy::epilogue(Epilogue::Residual, false, 64).unwrap();
    odd.insert(1, Stage::Epilogue { dst: "s".into(), expr: Expr::parse("acc * 2.0").unwrap(), mat: Default::default() });
    assert!(heavy::spelled(&odd, 64).unwrap_err().contains("no emitter spells"));
}

#[test]
fn heavy_records_slot_and_mainloop_and_wgsl_refuses_the_module_functions() {
    let t = Target::sm_120();
    let k = KernelInstance::gemm(&t, "gate_up_mma", gate_up(), WeightFormat::Fp8, Tile::SMALL, true, false).unwrap();
    let h = Heavy::lift(&k).unwrap().unwrap();
    assert_eq!(h.family, Family::Tile { wf: WeightFormat::Fp8, split: true, token_major: false });
    assert_eq!((h.problem.n, h.problem.k, h.tile), (34816, 5120, Some(Tile::SMALL)));
    assert!(h.prologue.is_empty());
    assert!(Heavy::lift(&KernelInstance::quant_fp4(&t).unwrap()).unwrap().is_none(), "quant rows are light, not heavy");
    let gated = Expr::parse("gated(1.0, 2.0)").unwrap();
    assert_eq!(gated.lower(Dialect::Cuda, &[]).unwrap(), "gated_act(1.0f, 2.0f)");
    assert!(gated.lower(Dialect::Wgsl, &[]).is_err());
    assert!(Expr::parse("mapped(a, b)").is_err());
}

fn store_only(t: &Target, name: &str, shape: GemmShape, wf: WeightFormat, tile: Tile) -> Heavy {
    let bare = GemmShape { epi: Epilogue::Store, ..shape };
    Heavy::lift(&KernelInstance::gemm(t, name, bare, wf, tile, true, false).unwrap()).unwrap().unwrap()
}

#[test]
fn attaching_the_residual_and_act_stages_rebuilds_the_hand_epilogues() {
    for t in heavy_targets() {
        let residual = vec![
            Stage::TileLoad { dst: "r".into(), src: heavy::OUT.into(), tile: None, mat: Default::default() },
            Stage::Epilogue { dst: "y".into(), expr: Expr::parse("acc + r").unwrap(), mat: Default::default() },
            Stage::TileStore { dst: heavy::OUT.into(), value: "y".into(), tile: None },
        ];
        let o = store_only(&t, "o_mma", down(), WeightFormat::Bf16, Tile::SMALL).attach(&t, "epilogue_attach", Slot::Epilogue, residual.clone()).unwrap();
        let hand = KernelInstance::gemm(&t, "o_mma", down(), WeightFormat::Bf16, Tile::SMALL, true, false).unwrap();
        assert_eq!(o.lower(&t).unwrap(), hand);
        assert_eq!(kir::cuda::emit_gemm(&o.lower(&t).unwrap()).unwrap(), kir::cuda::emit_gemm(&hand).unwrap());
        let biased = GemmShape { bias: true, ..down() };
        let ob = store_only(&t, "o_mma", biased, WeightFormat::Bf16, Tile::SMALL).attach(&t, "epilogue_attach", Slot::Epilogue, residual).unwrap();
        assert_eq!(ob.lower(&t).unwrap(), KernelInstance::gemm(&t, "o_mma", biased, WeightFormat::Bf16, Tile::SMALL, true, false).unwrap());
        let act = heavy::epilogue(Epilogue::Act, false, gate_up().n).unwrap();
        let gu = store_only(&t, "gate_up_mma", gate_up(), WeightFormat::Fp8, Tile::SMALL).attach(&t, "epilogue_attach", Slot::Epilogue, act).unwrap();
        assert_eq!(gu.lower(&t).unwrap(), KernelInstance::gemm(&t, "gate_up_mma", gate_up(), WeightFormat::Fp8, Tile::SMALL, true, false).unwrap());
        let map = vec![
            Stage::Epilogue { dst: "y".into(), expr: Expr::parse("mapped(acc)").unwrap(), mat: Default::default() },
            Stage::TileStore { dst: heavy::PLANES.into(), value: "y".into(), tile: None },
        ];
        let up = store_only(&t, "up_map_mma", up_map(), WeightFormat::Bf16, Tile::SMALL).attach(&t, "epilogue_attach", Slot::Epilogue, map).unwrap();
        assert_eq!(up.lower(&t).unwrap(), KernelInstance::gemm(&t, "up_map_mma", up_map(), WeightFormat::Bf16, Tile::SMALL, true, false).unwrap());
    }
}

#[test]
fn an_epilogue_no_emitter_spells_is_refused_naming_the_pass() {
    let t = Target::sm_120();
    let scale = vec![
        Stage::Epilogue { dst: "y".into(), expr: Expr::parse("acc * 0.5").unwrap(), mat: Default::default() },
        Stage::TileStore { dst: heavy::OUT.into(), value: "y".into(), tile: None },
    ];
    let err = store_only(&t, "qkv_mma", qkv(), WeightFormat::Bf16, Tile::SMALL).attach(&t, "epilogue_attach", Slot::Epilogue, scale).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("pass epilogue_attach") && msg.contains("qkv_mma") && msg.contains("no emitter spells"), "{msg}");
    let clash = vec![Stage::TileLoad { dst: "b".into(), src: heavy::OUT.into(), tile: None, mat: Default::default() }];
    let biased = GemmShape { bias: true, ..qkv() };
    let err = store_only(&t, "qkv_mma", biased, WeightFormat::Bf16, Tile::SMALL).attach(&t, "epilogue_attach", Slot::Epilogue, clash).unwrap_err();
    assert!(err.to_string().contains("already defined"), "{err}");
}

#[test]
fn the_fused_kernel_budget_is_checked_against_the_target_and_names_the_pass() {
    let t = Target::sm_120();
    let h = Heavy::lift(&KernelInstance::gemm(&t, "qkv_mma_big", qkv(), WeightFormat::Bf16, Tuning::default().tile(Variant::Big), true, false).unwrap()).unwrap().unwrap();
    let b = h.budget(&t, "slot_fit").unwrap();
    assert_eq!(b.regs, heavy::acc_regs(128 * 128, 256) + 2);
    assert!(b.smem > 0 && b.smem <= t.smem_limit);
    let tight = Target { max_regs: b.regs - 1, ..t };
    let err = h.budget(&tight, "slot_fit").unwrap_err();
    assert_eq!(err, HeavyError::Budget { pass: "slot_fit".into(), kernel: "qkv_mma_big".into(), what: "registers per thread", need: b.regs as u64, limit: b.regs as u64 - 1 });
    let norm = vec![
        Stage::TileLoad { dst: "x".into(), src: "x".into(), tile: None, mat: Default::default() },
        Stage::BlockSum { dst: "total".into(), expr: Expr::parse("x * x").unwrap() },
    ];
    let err = h.clone().attach(&t, "prologue_attach", Slot::Prologue, norm.clone()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.starts_with("pass prologue_attach: kernel qkv_mma_big needs") && msg.contains("bytes of shared memory"), "{msg}");
    let gemv = Heavy::lift(&KernelInstance::ggml_gemv(&t, "down_ggv", GemmShape::new(1024, 3072, Epilogue::Residual, false), false).unwrap()).unwrap().unwrap();
    let err = gemv.attach(&t, "prologue_attach", Slot::Prologue, norm).unwrap_err();
    assert!(err.to_string().contains("pass prologue_attach") && err.to_string().contains("no prologue emitter"), "{err}");
}
