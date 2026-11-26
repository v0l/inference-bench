use specrt::quantize::{fp8_channel_search, nvfp4_search, squared_error_fp8_channel, squared_error_nvfp4};
use specrt::weights::{quantize_fp8_channel, quantize_nvfp4};
use std::path::Path;

fn case() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quant-search/case.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn ints(v: &serde_json::Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}

fn weights(c: &serde_json::Value) -> (Vec<u16>, usize) {
    (ints(&c["bf16"]).into_iter().map(|v| v as u16).collect(), c["cols"].as_u64().unwrap() as usize)
}

#[test]
fn nvfp4_scale_search_matches_the_numpy_reference_bit_for_bit() {
    let c = case();
    let (w, cols) = weights(&c);
    let got = nvfp4_search(&w, cols);
    let want = &c["nvfp4_g16"];
    assert_eq!(got.global, want["global"].as_f64().unwrap() as f32);
    assert_eq!(got.scales.iter().map(|&s| s as u64).collect::<Vec<_>>(), ints(&want["scales"]));
    assert_eq!(got.packed.iter().map(|&s| s as u64).collect::<Vec<_>>(), ints(&want["packed"]));
}

#[test]
fn fp8_channel_scale_search_matches_the_numpy_reference_bit_for_bit() {
    let c = case();
    let (w, cols) = weights(&c);
    let (values, scales) = fp8_channel_search(&w, cols);
    let want = &c["fp8_channel"];
    let rows = w.len() / cols;
    let row_scales: Vec<f32> = (0..rows).map(|r| scales[scales[r].to_bits() as usize]).collect();
    let want_scales: Vec<f32> = want["scales"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    assert_eq!(row_scales, want_scales);
    assert_eq!(values.iter().map(|&v| v as u64).collect::<Vec<_>>(), ints(&want["values"]));
}

#[test]
fn the_search_never_does_worse_than_round_to_nearest_from_the_absmax() {
    let c = case();
    let (w, cols) = weights(&c);
    let (searched, plain) = (nvfp4_search(&w, cols), quantize_nvfp4(&w, cols));
    let (e_search, e_plain) = (squared_error_nvfp4(&w, &searched), squared_error_nvfp4(&w, &plain));
    assert!(e_search <= e_plain, "nvfp4 search {e_search} against plain {e_plain}");
    assert!(e_search < e_plain * 0.95, "the search should find better scales on this tensor: {e_search} against {e_plain}");
    let (sv, ss) = fp8_channel_search(&w, cols);
    let (pv, ps) = quantize_fp8_channel(&w, cols);
    let (e_search, e_plain) = (squared_error_fp8_channel(&w, cols, &sv, &ss), squared_error_fp8_channel(&w, cols, &pv, &ps));
    assert!(e_search <= e_plain, "fp8 search {e_search} against plain {e_plain}");
}

#[test]
fn the_quantcheck_prompt_set_parses_and_its_hash_is_stable() {
    let set = specrt::quantize::prompt_set();
    assert_eq!(set.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["short", "code", "science", "story"]);
    let lens: Vec<usize> = set.iter().map(|p| p.text.len()).collect();
    assert!(lens[0] * 10 < lens[3], "{lens:?}");
    assert_eq!(specrt::quantize::prompt_set_hash().len(), 16);
}

#[test]
fn logit_stats_counts_kl_and_top1_per_position() {
    let a = [0.0f32, 1.0, 2.0, 3.0, 0.0, 0.0];
    let same = specrt::quantize::logit_stats(&[&a], &[&a], 3);
    assert_eq!((same.positions, same.mean_kl, same.max_kl, same.top1), (2, 0.0, 0.0, 1.0));
    let b = [0.0f32, 1.0, 2.0, 0.0, 0.0, 3.0];
    let s = specrt::quantize::logit_stats(&[&a], &[&b], 3);
    assert_eq!(s.positions, 2);
    assert_eq!(s.top1, 0.5);
    assert!(s.max_kl > 1.0 && (s.mean_kl - s.max_kl / 2.0).abs() < 1e-12, "{s:?}");
}
