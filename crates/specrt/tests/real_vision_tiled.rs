use specrt::engine::encode_media_prompt_with;
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use specrt::vision::{MediaItem, MediaKind};
use std::path::{Path, PathBuf};

const BF16_ACTIVATION_NOISE_MARGIN: f64 = 1.5;

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_TILED_VISION_DIR").expect("SPECRT_TILED_VISION_DIR points at a tiled vision checkpoint snapshot"))
}

fn refs() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR holds the output of scripts/llama4_real_vision_reference.py"))
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

fn spec() -> Spec {
    let dir = checkpoint();
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec
}

#[test]
#[ignore = "needs a real tiled vision checkpoint in SPECRT_TILED_VISION_DIR, scripts/llama4_real_vision_reference.py output in SPECRT_REF_DIR (house.png and portrait.png) and about 70 GB of free GPU memory"]
fn real_tiled_vision_embeddings_stay_within_the_bf16_noise_of_transformers() {
    let mut rt = Runtime::load(&spec(), 4096).unwrap();
    assert!(rt.has_vision());
    let tok = tokenizers::Tokenizer::from_file(checkpoint().join("tokenizer.json")).unwrap();
    for name in ["house", "portrait"] {
        let image = std::fs::read(refs().join(format!("{name}.png"))).unwrap();
        let items = [MediaItem::new(MediaKind::Image, image)];
        let messages = vec![serde_json::json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}, {"type": "text", "text": "Describe this image in detail."}]})];
        let prompt = encode_media_prompt_with(spec(), &messages, Some(&serde_json::json!({})), &items).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refs().join(format!("{name}.json"))).unwrap()).unwrap();
        assert_eq!(prompt.vision_tokens, meta["rows"].as_u64().unwrap() as usize, "{name}");
        let plan = rt.plan_media(&prompt.ids, &prompt.media).unwrap();
        let f32_ref = floats(&refs().join(format!("{name}.embeds.bin")));
        let bf16_ref = floats(&refs().join(format!("{name}.embeds.bf16gpu.bin")));
        let ours_vs_f32 = rel(&plan.embeds, &f32_ref);
        let bf16_vs_f32 = rel(&bf16_ref, &f32_ref);
        let ours_vs_bf16 = rel(&plan.embeds, &bf16_ref);
        eprintln!("{name}: transformers bf16 gpu vs f32 {bf16_vs_f32:e}, ours vs f32 {ours_vs_f32:e}, ours vs transformers bf16 gpu {ours_vs_bf16:e}, meta {meta}");
        rt.set_media(0, Some(&plan)).unwrap();
        let out = rt.generate(&prompt.ids, 48, Sampling::GREEDY).unwrap();
        rt.set_media(0, None).unwrap();
        eprintln!("{name} answer: {}", tok.decode(&out.tokens, true).unwrap());
        assert!(ours_vs_f32 <= bf16_vs_f32 * BF16_ACTIVATION_NOISE_MARGIN, "{name}: ours vs f32 {ours_vs_f32} exceeds the bf16 floor {bf16_vs_f32}");
    }
}
