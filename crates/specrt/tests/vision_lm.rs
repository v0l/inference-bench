use serde_json::{Value, json};
use specrt::engine::{complete_spec, encode_media_prompt};
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use specrt::vision::{MediaItem, MediaKind};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ref_dir() -> PathBuf {
    std::env::var_os("SPECRT_VISION_REF_DIR").map(PathBuf::from).expect("SPECRT_VISION_REF_DIR holds the output of scripts/vllm_vision.py")
}

fn spec() -> Spec {
    spec_named("qwen3.8-27b")
}

fn spec_named(name: &str) -> Spec {
    let mut spec = Spec::load(&fixtures().join(format!("{name}.toml"))).unwrap();
    complete_spec(&mut spec).unwrap();
    assert!(spec.vision.is_some() && spec.rope.mrope.is_some());
    spec
}

struct Case {
    name: &'static str,
    kind: MediaKind,
    file: &'static str,
}

const CASES: [Case; 2] =
    [Case { name: "image", kind: MediaKind::Image, file: "house.png" }, Case { name: "video", kind: MediaKind::Video, file: "clip.mp4" }];

fn record(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(ref_dir().join(format!("vllm-{name}.json"))).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn prompt(case: &Case, rec: &Value) -> specrt::engine::MediaPrompt {
    prompt_for("Qwen/Qwen3.8-27B-FP8", case, rec)
}

fn prompt_for(repo: &str, case: &Case, rec: &Value) -> specrt::engine::MediaPrompt {
    let bytes = std::fs::read(fixtures().join("vision").join(case.file)).unwrap();
    let item = MediaItem::new(case.kind, bytes);
    let part = match case.kind {
        MediaKind::Image => json!({"type": "image_url", "image_url": {"url": "x"}}),
        MediaKind::Video => json!({"type": "video_url", "video_url": {"url": "x"}}),
    };
    let messages = vec![json!({"role": "user", "content": [part, {"type": "text", "text": rec["question"]}]})];
    encode_media_prompt(repo, &messages, Some(&json!({"enable_thinking": false})), &[item]).unwrap()
}

#[test]
#[ignore = "needs the 27B NVFP4 checkpoint, about 28 GB of free GPU memory and scripts/vllm_vision.py output in SPECRT_VISION_REF_DIR"]
fn media_prompts_and_logprobs_agree_with_vllm() {
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let repo = spec.repo.clone();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    assert!(rt.has_vision());
    let vocab = rt.spec().dims.vocab;
    for case in &CASES {
        let rec = record(case.name);
        let p = prompt_for(&repo, case, &rec);
        let theirs_prompt = ids(&rec["prompt_ids"]);
        assert_eq!(p.ids, theirs_prompt, "{}: the templated prompt must tokenise to vLLM's ids", case.name);
        let theirs = ids(&rec["greedy"]);
        let plan = rt.plan_media(&p.ids, &p.media).unwrap();
        rt.set_media(0, Some(&plan)).unwrap();
        let ours = rt.generate(&p.ids, theirs.len(), Sampling::GREEDY).unwrap().tokens;
        let prefix = ours.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
        let mut seq = p.ids.clone();
        seq.extend(&theirs);
        rt.set_media(0, Some(&plan)).unwrap();
        let logits = rt.prompt_logits(&seq).unwrap();
        let (mut kls, mut top1) = (Vec::new(), 0usize);
        for (step, row) in rec["logprobs"].as_array().unwrap().iter().enumerate() {
            let at = (p.ids.len() + step - 1) * vocab;
            let l = &logits[at..at + vocab];
            let max = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = max + l.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
            let top: Vec<(usize, f64)> = row.as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_f64().unwrap())).collect();
            let p_mass: f64 = top.iter().map(|(_, lp)| lp.exp()).sum();
            let mut kl = 0.0;
            for &(t, lp) in &top {
                kl += lp.exp() * (lp - (l[t] - lse) as f64);
            }
            let rest_p = (1.0 - p_mass).max(1e-12);
            let rest_q = (1.0 - top.iter().map(|&(t, _)| ((l[t] - lse) as f64).exp()).sum::<f64>()).max(1e-12);
            kl += rest_p * (rest_p / rest_q).ln();
            kls.push(kl);
            let best = (0..vocab).fold(0, |b, i| if l[i] > l[b] { i } else { b });
            if best == theirs[step] as usize {
                top1 += 1;
            }
        }
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        eprintln!(
            "{}: prompt {} tokens, {} vision, greedy prefix {prefix}/{}, top1 {top1}/{}, mean KL {mean:.5}, max KL {:.5}, ours {:?}",
            case.name,
            p.ids.len(),
            p.vision_tokens,
            theirs.len(),
            kls.len(),
            kls.iter().cloned().fold(0.0, f64::max),
            rt.spec().name
        );
        let (min_top1, max_kl) = match case.kind {
            MediaKind::Image => (0.9, 0.05),
            MediaKind::Video => (0.8, 0.1),
        };
        assert!(top1 as f64 >= min_top1 * kls.len() as f64, "{}: teacher-forced top-1 {top1}/{}", case.name, kls.len());
        assert!(mean <= max_kl, "{}: mean KL {mean}", case.name);
        rt.set_media(0, None).unwrap();
    }
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint and about 36 GB of free GPU memory"]
fn greedy_answer_names_the_object_in_the_fixture_photo() {
    let mut rt = Runtime::load(&spec(), 2048).unwrap();
    let rec = json!({"question": "What is in this image? Answer in one short sentence."});
    let p = prompt(&CASES[0], &rec);
    let plan = rt.plan_media(&p.ids, &p.media).unwrap();
    rt.set_media(0, Some(&plan)).unwrap();
    let tokens = rt.generate(&p.ids, 24, Sampling::GREEDY).unwrap().tokens;
    let tok = tokenizers::Tokenizer::from_file(specrt::weights::find_in_snapshot("Qwen/Qwen3.8-27B-FP8", "tokenizer.json").unwrap()).unwrap();
    let text = tok.decode(&tokens, true).unwrap().to_lowercase();
    eprintln!("answer: {text}");
    assert!(["barn", "house", "building", "cabin", "shed"].iter().any(|w| text.contains(w)), "answer was {text}");
}
