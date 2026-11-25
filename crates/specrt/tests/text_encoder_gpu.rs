use specrt::runtime::Runtime;
use specrt::sampler::{Pipeline, stack_layers};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const HIDDEN_TOLERANCE: f32 = 1e-4;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference/tiny-flux2-klein.json")).unwrap()).unwrap()
}

fn floats(key: &str) -> Vec<f32> {
    let file = meta()["files"][key]["file"].as_str().unwrap().to_string();
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn encoder() -> Runtime {
    let dir = fixture().join("text_encoder");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec.runtime.split_activations = Some(true);
    Runtime::load(&spec, 64).unwrap()
}

fn prompt(key: &str) -> (Vec<u32>, usize) {
    let ids: Vec<u32> = floats(&format!("{key}.ids")).iter().map(|v| *v as u32).collect();
    let real = floats(&format!("{key}.mask")).iter().filter(|m| **m == 1.0).count();
    (ids, real)
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test text_encoder_gpu -- --ignored"]
fn the_engine_text_encoder_gives_the_diffusers_hidden_states_with_pad_keys_masked() {
    let m = meta();
    let layers: Vec<usize> = m["text_encoder_out_layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap() as usize).collect();
    let mut rt = encoder();
    let (ids, real) = prompt("text");
    assert_eq!(real, m["prompt_tokens"].as_u64().unwrap() as usize);
    let states = rt.hidden_states(&ids, real, &layers).unwrap();
    let want = floats("text.hidden");
    let d = max_diff(&states, &want);
    let pad_rows: f32 = (0..layers.len()).map(|l| max_diff(&states[(l * 16 + real) * 32..(l + 1) * 16 * 32], &want[(l * 16 + real) * 32..(l + 1) * 16 * 32])).fold(0.0, f32::max);
    eprintln!("text.hidden max diff {d:e} (pad rows {pad_rows:e})");
    assert!(d < HIDDEN_TOLERANCE, "{d}");
    let embeds = stack_layers(&states, layers.len(), 16, 32);
    let e = max_diff(&embeds, &floats("text.embeds"));
    eprintln!("text.embeds max diff {e:e}");
    assert!(e < HIDDEN_TOLERANCE, "{e}");
    let (neg, neg_real) = prompt("negative");
    let n = stack_layers(&rt.hidden_states(&neg, neg_real, &layers).unwrap(), layers.len(), 16, 32);
    let nd = max_diff(&n, &floats("negative.embeds"));
    eprintln!("negative.embeds max diff {nd:e}");
    assert!(nd < HIDDEN_TOLERANCE, "{nd}");
    let again = rt.hidden_states(&ids, real, &layers).unwrap();
    assert_eq!(again, states, "a second prompt run is independent of the first");
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test text_encoder_gpu -- --ignored"]
fn unmasked_pad_keys_change_the_padded_rows() {
    let layers = Pipeline::from_dir(&fixture()).unwrap().text_layers;
    let mut rt = encoder();
    let (ids, real) = prompt("text");
    let masked = rt.hidden_states(&ids, real, &layers).unwrap();
    let open = rt.hidden_states(&ids, ids.len(), &layers).unwrap();
    let h = 32;
    let prompt_rows = max_diff(&masked[..real * h], &open[..real * h]);
    let all = max_diff(&open, &floats("text.hidden"));
    eprintln!("without the key mask: prompt rows move {prompt_rows:e}, the padded rows end {all:e} from the reference");
    assert_eq!(prompt_rows, 0.0, "causal prompt rows never see pad keys");
    assert!(all > 10.0 * HIDDEN_TOLERANCE, "{all}");
}
