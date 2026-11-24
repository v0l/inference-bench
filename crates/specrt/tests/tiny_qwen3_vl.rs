use serde_json::Value;
use specrt::runtime::Runtime;
use specrt::sched::MediaInput;
use specrt::spec::Spec;
use specrt::vision::{Patches, Processed};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-vl")
}

fn reference_dir() -> PathBuf {
    fixture().join("reference")
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(reference_dir().join(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(reference_dir().join(name)).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn load_spec() -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let mut s = Spec::from_hf_config(&json).unwrap();
    s.name = "tiny-qwen3-vl".into();
    s.repo = fixture().display().to_string();
    s.runtime.vision.precision = specrt::vision_encoder::VisionPrecision::Exact;
    s
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

fn mean_kl(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows)
        .map(|r| {
            let (a, b) = (log_softmax(&p[r * vocab..(r + 1) * vocab]), log_softmax(&q[r * vocab..(r + 1) * vocab]));
            a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

fn top1(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows).filter(|&r| argmax(&p[r * vocab..(r + 1) * vocab]) == argmax(&q[r * vocab..(r + 1) * vocab])).count() as f64 / rows as f64
}

fn relative(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    err / b.iter().fold(0f32, |m, v| m.max(v.abs()))
}

#[test]
fn the_fixture_is_the_qwen3_vl_shape_in_miniature() {
    let s = load_spec();
    assert_eq!(s.layers.count, 2);
    assert!(s.attention.qk_norm);
    assert!(s.moe.is_none());
    let m = s.rope.mrope.as_ref().unwrap();
    assert_eq!(m.section, [24, 20, 20]);
    assert!(m.interleaved);
    let v = s.vision.as_ref().unwrap();
    assert_eq!((v.config.depth, v.config.spatial_merge_size, v.config.out_hidden_size), (2, 2, s.dims.hidden));
    assert_eq!(v.config.deepstack_visual_indexes, [0]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn text_logits_match_transformers() {
    let s = load_spec();
    let m = meta("tinyvl-text.json");
    let prompt = ids(&m["prompt"]);
    let want = floats("tinyvl-text.logits.bin");
    let mut rt = Runtime::load(&s, 256).unwrap();
    let ours = rt.prompt_logits(&prompt).unwrap();
    let vocab = s.dims.vocab;
    let (kl, agree) = (mean_kl(&want, &ours, vocab), top1(&want, &ours, vocab));
    eprintln!("text: KL {kl:e}, top-1 {agree}");
    assert!(kl <= 1e-4, "KL {kl}");
    assert_eq!(agree, 1.0);
}

fn vision_case(rt: &mut Runtime, case: &str) {
    let m = meta(&format!("tinyvl-vision-{case}.json"));
    let patch_dim = m["patch_dim"].as_u64().unwrap() as usize;
    let all = floats(&format!("tinyvl-vision-{case}.patches.bin"));
    let mut at = 0;
    let media: Vec<MediaInput> = m["grids"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let grid = [g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize];
            let n = grid.iter().product::<usize>() * patch_dim;
            let data = all[at..at + n].to_vec();
            at += n;
            MediaInput { processed: Processed::Image(Patches { grid, data, layout: [0, 0] }), hash: [(case.len() * 8 + i) as u8 + 1; 32] }
        })
        .collect();
    let prompt = ids(&m["ids"]);
    let plan = rt.plan_media(&prompt, &media).unwrap();
    let want_embeds = floats(&format!("tinyvl-vision-{case}.embeds.bin"));
    let err = relative(&plan.embeds, &want_embeds);
    eprintln!("{case}: vision embeddings max relative {err:e}");
    assert!(err <= 2e-3, "{case}: {err}");
    let want_deep = floats(&format!("tinyvl-vision-{case}.deepstack.bin"));
    assert_eq!(plan.deepstack.len(), 1, "{case}: one deepstack level");
    let err = relative(&plan.deepstack[0], &want_deep);
    eprintln!("{case}: deepstack max relative {err:e}");
    assert!(err <= 2e-3, "{case}: deepstack {err}");
    rt.set_media(0, Some(&plan)).unwrap();
    let ours = rt.prompt_logits(&prompt).unwrap();
    let want = floats(&format!("tinyvl-vision-{case}.logits.bin"));
    let vocab = rt.spec().dims.vocab;
    let (kl, agree) = (mean_kl(&want, &ours, vocab), top1(&want, &ours, vocab));
    eprintln!("{case}: logits KL {kl:e}, top-1 {agree}");
    assert!(kl <= 1e-4, "{case}: KL {kl}");
    assert_eq!(agree, 1.0, "{case}");
    rt.set_media(0, None).unwrap();
    let text_only = rt.prompt_logits(&prompt).unwrap();
    assert!(mean_kl(&want, &text_only, vocab) > 10.0 * kl.max(1e-6), "{case}: the image changes the logits");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn vision_embeddings_deepstack_and_mrope_logits_match_transformers() {
    let mut rt = Runtime::load(&load_spec(), 256).unwrap();
    assert!(rt.has_vision());
    vision_case(&mut rt, "image");
    vision_case(&mut rt, "images2");
}
