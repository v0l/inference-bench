use specrt::engine::encode_media_prompt_with;
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use specrt::vision::{MediaItem, MediaKind};
use std::path::{Path, PathBuf};

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_TILED_VISION_DIR").expect("SPECRT_TILED_VISION_DIR points at a tiled vision checkpoint snapshot"))
}

fn refs() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR holds the output of scripts/llama4_real_vision_reference.py"))
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn spec() -> Spec {
    let dir = checkpoint();
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec
}

#[test]
#[ignore = "needs a real tiled vision checkpoint in SPECRT_TILED_VISION_DIR, scripts/llama4_real_vision_reference.py output in SPECRT_REF_DIR and about 70 GB of free GPU memory"]
fn real_tiled_vision_embeddings_match_transformers_and_the_model_answers() {
    let image = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision/house.png")).unwrap();
    let items = [MediaItem::new(MediaKind::Image, image)];
    let messages = vec![serde_json::json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}, {"type": "text", "text": "Describe this image in detail."}]})];
    let prompt = encode_media_prompt_with(spec(), &messages, Some(&serde_json::json!({})), &items).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refs().join("house.json")).unwrap()).unwrap();
    eprintln!("prompt tokens {} vision tokens {} reference {meta}", prompt.ids.len(), prompt.vision_tokens);
    assert_eq!(prompt.vision_tokens, meta["rows"].as_u64().unwrap() as usize);

    let mut rt = Runtime::load(&spec(), 4096).unwrap();
    assert!(rt.has_vision());
    let plan = rt.plan_media(&prompt.ids, &prompt.media).unwrap();
    let want = floats(&refs().join("house.embeds.bin"));
    assert_eq!(plan.embeds.len(), want.len());
    let num: f64 = plan.embeds.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt();
    eprintln!("image embeddings relative L2 {rel:e}");

    rt.set_media(0, Some(&plan)).unwrap();
    let out = rt.generate(&prompt.ids, 48, Sampling::GREEDY).unwrap();
    rt.set_media(0, None).unwrap();
    let tok = tokenizers::Tokenizer::from_file(checkpoint().join("tokenizer.json")).unwrap();
    eprintln!("answer: {}", tok.decode(&out.tokens, true).unwrap());
    assert!(rel < 0.01, "embeddings differ by {rel}");
}
