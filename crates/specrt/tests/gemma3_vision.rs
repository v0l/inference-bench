use serde_json::{Value, json};
use specrt::engine::{complete_spec, encode_media_prompt_with};
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use specrt::vision::{MediaItem, MediaKind};
use std::path::{Path, PathBuf};

fn ref_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR holds the output of scripts/gemma3_vision.py"))
}

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("GEMMA3_4B_DIR").expect("GEMMA3_4B_DIR holds google/gemma-3-4b-it or unsloth/gemma-3-4b-it"))
}

fn house() -> Vec<u8> {
    std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision/house.png")).unwrap()
}

fn floats(path: PathBuf) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn spec(pan: bool) -> Spec {
    let dir = model_dir();
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.name = "gemma3-4b".into();
    spec.repo = dir.display().to_string();
    complete_spec(&mut spec).unwrap();
    spec.runtime.vision.pan_and_scan = pan;
    if let Some(p) = std::env::var_os("SPECRT_VISION_PRECISION") {
        spec.runtime.vision.precision = serde_json::from_value(json!(p.to_string_lossy())).expect("fast or exact");
    }
    spec
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

fn top1(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn relative(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale.max(1e-6)
}

fn l2(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut d, mut n, mut x, mut y, mut dot) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (p, q) in a.iter().zip(b) {
        let (p, q) = (*p as f64, *q as f64);
        d += (p - q) * (p - q);
        n += q * q;
        x += p * p;
        y += q * q;
        dot += p * q;
    }
    ((d / n).sqrt(), dot / (x.sqrt() * y.sqrt()))
}

fn case(name: &str, pan: bool) {
    let dir = ref_dir();
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap()).unwrap();
    let bytes = if pan { std::fs::read(dir.join("wide.png")).unwrap() } else { house() };
    let items = vec![MediaItem::new(MediaKind::Image, bytes)];
    let messages = vec![json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": meta["question"]}]})];
    let s = spec(pan);
    let prompt = encode_media_prompt_with(s.clone(), &messages, Some(&json!({})), &items).unwrap();
    let want_ids = ids(&meta["ids"]);
    assert_eq!(prompt.ids, want_ids, "{name}: the prompt must tokenise to the processor's ids");

    let patches = floats(dir.join(format!("{name}.patches.bin")));
    let ours_patches: Vec<f32> = prompt.media.iter().flat_map(|m| m.processed.patches().data.clone()).collect();
    assert_eq!(ours_patches.len(), patches.len(), "{name}: patch tensor size");
    let perr = ours_patches.iter().zip(&patches).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("{name}: patch tensor max abs {perr:e}");
    assert!(perr < 0.05, "{name}: preprocessing differs by {perr}");

    let mut rt = Runtime::load(&s, 4096).unwrap();
    assert!(rt.has_vision());
    let plan = rt.plan_media(&prompt.ids, &prompt.media).unwrap();
    let want_embeds = floats(dir.join(format!("{name}.embeds.bin")));
    assert_eq!(plan.embeds.len(), want_embeds.len(), "{name}: embedding count");
    let err = relative(&plan.embeds, &want_embeds);
    let (rel, cos) = l2(&plan.embeds, &want_embeds);
    eprintln!("{name}: image embeddings against transformers bf16: max relative {err:e}, relative L2 {rel:e}, cosine {cos:.5}");
    let exact = floats(dir.join(format!("{name}.embeds32.bin")));
    let (vs_exact, cos_exact) = l2(&plan.embeds, &exact);
    let (bf16_noise, _) = l2(&want_embeds, &exact);
    eprintln!("{name}: against the fp32 tower: ours {vs_exact:e} (cosine {cos_exact:.6}), transformers bf16 {bf16_noise:e}");
    assert!(vs_exact < 0.01 && cos_exact > 0.9999, "{name}: embeddings differ from the fp32 tower by {vs_exact}");
    assert!(rel < 0.15 && cos > 0.99, "{name}: embeddings differ from transformers bf16 by L2 {rel}, cosine {cos}");

    rt.set_media(0, Some(&plan)).unwrap();
    let vocab = meta["vocab"].as_u64().unwrap() as usize;
    let rows = meta["rows"].as_u64().unwrap() as usize;
    let ours = rt.prompt_logits(&prompt.ids).unwrap();
    let want = floats(dir.join(format!("{name}.logits.bin")));
    let first = prompt.ids.len() - rows;
    let mut worst = 0f64;
    for r in 0..rows {
        let a = &ours[(first + r) * vocab..(first + r + 1) * vocab];
        let b = &want[r * vocab..(r + 1) * vocab];
        let (la, lb) = (log_softmax(a), log_softmax(b));
        let kl: f64 = lb.iter().zip(&la).map(|(q, p)| q.exp() * (q - p)).sum();
        eprintln!("{name}: row {r} kl {kl:e}");
        worst = worst.max(kl);
        assert_eq!(top1(a), top1(b), "{name}: row {r} top-1");
    }
    assert!(worst < 0.05, "{name}: logits KL {worst}");

    let greedy = ids(&meta["greedy"]);
    let mut seq = prompt.ids.clone();
    seq.extend(&greedy);
    rt.set_media(0, Some(&plan)).unwrap();
    let forced = rt.prompt_logits(&seq).unwrap();
    let ref_margins: Vec<f64> = meta["margins"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
    let (mut agree, mut margins) = (0usize, Vec::new());
    for (i, &t) in greedy.iter().enumerate() {
        let row = &forced[(prompt.ids.len() + i - 1) * vocab..(prompt.ids.len() + i) * vocab];
        let best = top1(row);
        if best == t as usize {
            agree += 1;
        } else {
            margins.push((row[best] - row[t as usize], ref_margins[i] as f32));
        }
    }
    eprintln!("{name}: teacher-forced top-1 {agree}/{}, (our margin, reference margin) on the misses {margins:?}", greedy.len());
    assert!(margins.iter().all(|(m, theirs)| *m < 0.5 || *theirs < 0.5), "{name}: a miss where neither side is close: {margins:?}");
    assert!(agree * 10 >= greedy.len() * 9, "{name}: teacher-forced top-1 {agree} of {}", greedy.len());

    rt.set_media(0, Some(&plan)).unwrap();
    let got = rt.generate(&prompt.ids, greedy.len(), Sampling::GREEDY).unwrap().tokens;
    let same = got.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
    eprintln!("{name}: greedy prefix {same}/{}", greedy.len());
    if !pan {
        assert_eq!(same, greedy.len(), "{name}: greedy diverged after {same}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU, GEMMA3_4B_DIR and scripts/gemma3_vision.py output in SPECRT_REF_DIR"]
fn gemma3_4b_image_matches_transformers() {
    case("gemma3-image", false);
}

#[test]
#[ignore = "needs a CUDA GPU, GEMMA3_4B_DIR and scripts/gemma3_vision.py output in SPECRT_REF_DIR"]
fn gemma3_4b_pan_and_scan_matches_transformers() {
    case("gemma3-pan", true);
}
