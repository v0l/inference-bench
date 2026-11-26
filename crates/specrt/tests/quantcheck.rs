use specrt::quantcheck::{Budget, LayerOption, Measured, Probe, Recipe, choose, estimated_kl, layer_options, recipe_from, total_bytes};
use specrt::spec::{LayerQuant, Quant, Spec};
use std::path::Path;

fn kl(mean: f64, max: f64) -> Measured {
    Measured { mean_kl: mean, max_kl: max, top1: 1.0 - mean }
}

fn probe(layer: usize, format: Quant, search: bool, bytes: u64, mean: f64, max: f64) -> Probe {
    Probe { layer, quant: LayerQuant::new(format, search), bytes, kl: kl(mean, max) }
}

fn three_layers() -> Vec<Vec<LayerOption>> {
    let probes = [
        probe(0, Quant::Nvfp4G16, true, 30, 0.010, 0.25),
        probe(0, Quant::Nvfp4G16, false, 30, 0.012, 0.2),
        probe(0, Quant::Fp8Channel, true, 55, 0.001, 0.01),
        probe(1, Quant::Nvfp4G16, true, 30, 0.050, 6.0),
        probe(1, Quant::Nvfp4G16, false, 30, 0.060, 1.0),
        probe(1, Quant::Fp8Channel, true, 55, 0.004, 0.05),
        probe(2, Quant::Nvfp4G16, true, 30, 0.002, 0.03),
        probe(2, Quant::Fp8Channel, true, 55, 0.0002, 0.002),
    ];
    layer_options(&probes, &[100, 100, 100])
}

#[test]
fn each_layer_offers_its_formats_by_size_and_keeps_the_rounding_with_the_better_measured_logits() {
    let o = three_layers();
    let formats: Vec<Quant> = o[0].iter().map(|x| x.quant.format).collect();
    assert_eq!(formats, [Quant::Nvfp4G16, Quant::Fp8Channel, Quant::Bf16]);
    assert!(o[0][0].quant.search, "layer 0: search has the lower mean and its max cost is small");
    assert!(!o[1][0].quant.search, "layer 1: the search raised the worst position from 1.0 to 6.0, plain rounding wins");
    assert_eq!(o[2].len(), 3);
    assert_eq!(o[1][2].kl, Measured { mean_kl: 0.0, max_kl: 0.0, top1: 1.0 });
}

#[test]
fn greedy_starts_at_the_lowest_bit_format_and_moves_the_most_damaging_layer_up() {
    let o = three_layers();
    let all_low = choose(&o, 7, Budget { kl: 1.0, bytes: None });
    assert_eq!(all_low, [0, 0, 0]);
    assert_eq!(total_bytes(&o, &all_low, 7), 97);
    let at = choose(&o, 7, Budget { kl: 0.02, bytes: None });
    assert_eq!(at, [0, 1, 0], "layer 1 is the most damaging, so it moves to fp8 first");
    assert!(estimated_kl(&o, &at) <= 0.02);
    let tight = choose(&o, 7, Budget { kl: 0.0, bytes: None });
    assert_eq!(tight, [2, 2, 2], "a zero budget ends with everything bf16");
    let capped = choose(&o, 7, Budget { kl: 0.0, bytes: Some(7 + 55 + 55 + 30) });
    assert_eq!(total_bytes(&o, &capped, 7), 147, "the memory budget stops the climb");
    assert_eq!(capped, [1, 1, 0]);
}

#[test]
fn a_recipe_is_a_few_kb_of_toml_and_round_trips_into_a_quant_plan() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b.config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.repo = "Qwen/Qwen3-0.6B".into();
    let probes: Vec<Probe> = (0..spec.layers.count)
        .flat_map(|l| [probe(l, Quant::Nvfp4G16, l % 3 != 0, 9_000_000, 0.004 * (l + 1) as f64, 0.3), probe(l, Quant::Fp8Channel, true, 17_000_000, 0.0005, 0.02)])
        .collect();
    let options = layer_options(&probes, &vec![32_000_000; spec.layers.count]);
    let budget = Budget { kl: 0.05, bytes: None };
    let at = choose(&options, 311_000_000, budget);
    let recipe = recipe_from(&spec, &options, &at, 311_000_000, budget, 353);
    let text = recipe.to_toml().unwrap();
    assert!(text.len() < 8192, "{} bytes", text.len());
    assert!(text.contains("[[layer]]") && text.contains("prompt_set = ") && text.contains("engine = "));
    let path = std::env::temp_dir().join(format!("quantcheck-{}.quantrecipe.toml", std::process::id()));
    recipe.save(&path).unwrap();
    let back = Recipe::load(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back, recipe);
    let plan = back.plan();
    assert_eq!(plan.layers.len(), spec.layers.count);
    assert_eq!(plan.layers[27].format, Quant::Fp8Channel, "the most damaging layers moved up");
    assert_eq!(plan.layers[0], LayerQuant::new(Quant::Nvfp4G16, false));
    assert!(recipe.estimated_kl <= 0.05 && recipe.bytes < recipe.bf16_bytes);
    assert_eq!(recipe.prompt_set, specrt::quantize::prompt_set_hash());
    let mut planned = spec.clone();
    planned.quant_plan = plan;
    planned.apply_quantize().unwrap();
    assert_eq!(planned.linear_quant("qkv", 27), Quant::Fp8Channel);
    assert_eq!(planned.served_mlp_quant(1), Quant::Nvfp4G16);
    let mut bumped = recipe.to_toml().unwrap();
    bumped = bumped.replacen("version = 1", "version = 99", 1);
    std::fs::write(&path, bumped).unwrap();
    assert!(Recipe::load(&path).is_err());
    std::fs::remove_file(&path).ok();
}
