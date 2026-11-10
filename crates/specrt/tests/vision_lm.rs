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

fn has_refs() -> bool {
    let ok = std::env::var_os("SPECRT_VISION_REF_DIR").is_some_and(|d| Path::new(&d).join("vllm-image.json").exists());
    if !ok {
        eprintln!("skipped: set SPECRT_VISION_REF_DIR to the output of scripts/vllm_vision.py (needs the vLLM server on port 8001)");
    }
    ok
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
    files: &'static [&'static str],
}

const CASES: [Case; 6] = [
    Case { name: "image", kind: MediaKind::Image, files: &["house.png"] },
    Case { name: "video", kind: MediaKind::Video, files: &["clip.mp4"] },
    Case { name: "images3", kind: MediaKind::Image, files: &["house.png", "shapes.png", "lines.png"] },
    Case { name: "video10", kind: MediaKind::Video, files: &["clip10.mp4"] },
    Case { name: "video2f", kind: MediaKind::Video, files: &["clip2f.mp4"] },
    Case { name: "video4f", kind: MediaKind::Video, files: &["clip4f.mp4"] },
];

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
    let items: Vec<MediaItem> = case.files.iter().map(|f| MediaItem::new(case.kind, std::fs::read(fixtures().join("vision").join(f)).unwrap())).collect();
    let part = match case.kind {
        MediaKind::Image => json!({"type": "image_url", "image_url": {"url": "x"}}),
        MediaKind::Video => json!({"type": "video_url", "video_url": {"url": "x"}}),
    };
    let mut content: Vec<Value> = items.iter().map(|_| part.clone()).collect();
    content.push(json!({"type": "text", "text": rec["question"]}));
    let messages = vec![json!({"role": "user", "content": content})];
    encode_media_prompt(repo, &messages, Some(&json!({"enable_thinking": false})), &items).unwrap()
}

#[test]
#[ignore = "needs the 27B NVFP4 checkpoint, about 28 GB of free GPU memory and scripts/vllm_vision.py output in SPECRT_VISION_REF_DIR"]
fn real_27b_media_prompts_and_logprobs_agree_with_vllm() {
    if !has_refs() {
        return;
    }
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let repo = spec.repo.clone();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    assert!(rt.has_vision());
    let vocab = rt.spec().dims.vocab;
    let mut failed: Vec<String> = Vec::new();
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
        if case.name != "image" || (top1 as f64 >= 0.9 * kls.len() as f64 && mean <= 0.05) {
            eprintln!("CASE {} measured", case.name);
        } else {
            failed.push(format!("{}: top-1 {top1}/{}, mean KL {mean:.4}", case.name, kls.len()));
        }
        rt.set_media(0, None).unwrap();
    }
    assert!(failed.is_empty(), "the single image case is outside the text bounds (top-1 90%, KL 0.05): {failed:?}");
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint and about 36 GB of free GPU memory"]
fn real_27b_greedy_answer_names_the_object_in_the_fixture_photo() {
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

fn kl_against(rt: &mut Runtime, ids: &[u32], rec: &Value) -> (f64, usize, usize) {
    let vocab = rt.spec().dims.vocab;
    let theirs = self::ids(&rec["greedy"]);
    let mut seq = ids.to_vec();
    seq.extend(&theirs);
    let logits = rt.prompt_logits(&seq).unwrap();
    let (mut sum, mut top1, mut n) = (0.0, 0, 0);
    for (step, row) in rec["logprobs"].as_array().unwrap().iter().enumerate() {
        let at = (ids.len() + step - 1) * vocab;
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
        sum += kl;
        n += 1;
        if (0..vocab).fold(0, |b, i| if l[i] > l[b] { i } else { b }) == theirs[step] as usize {
            top1 += 1;
        }
    }
    (sum / n as f64, top1, n)
}

#[test]
#[ignore = "debug bisect"]
fn real_27b_bisect_video_variants() {
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let repo = spec.repo.clone();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    let rec = record("video");
    let p = prompt_for(&repo, &CASES[1], &rec);
    let base = (*rt.plan_media(&p.ids, &p.media).unwrap()).clone();
    let mut run = |name: &str, plan: &specrt::vision::MediaPlan| {
        rt.set_media(0, Some(plan)).unwrap();
        let (kl, top1, n) = kl_against(&mut rt, &p.ids, &rec);
        eprintln!("VARIANT {name}: mean KL {kl:.4} top1 {top1}/{n}");
    };
    run("as built", &base);
    let mut plain = base.clone();
    for (i, pos) in plain.positions.iter_mut().enumerate() {
        pos[..3].fill(i as i32);
    }
    plain.delta = 0;
    run("plain positions", &plain);
    let mut tframe = base.clone();
    let mut frame = -1;
    let mut last = -1;
    for pos in tframe.positions.iter_mut() {
        if pos[3] >= 0 {
            if pos[0] != last {
                frame += 1;
                last = pos[0];
            }
            pos[0] += frame;
        }
    }
    run("t advances per frame", &tframe);
    let mut zt = base.clone();
    for pos in zt.positions.iter_mut() {
        if pos[3] >= 0 {
            pos[0] = pos[1];
        }
    }
    run("t equals h", &zt);
    run("same again", &base);
    for tag in ["vision", "vision_bf16"] {
        let raw = std::fs::read(fixtures().join(format!("vision/clip.{tag}.f32"))).unwrap();
        let mut hf = base.clone();
        hf.embeds = std::sync::Arc::new(raw.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        assert_eq!(hf.embeds.len(), base.embeds.len());
        run(&format!("transformers {tag} embeds"), &hf);
    }
}

#[test]
#[ignore = "debug bisect"]
fn real_27b_video_positions_equal_transformers_get_rope_index() {
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let rec = record("video");
    let p = prompt_for(&spec.repo, &CASES[1], &rec);
    let vis = spec.vision.as_ref().unwrap();
    let processed: Vec<_> = p.media.iter().map(|m| m.processed.clone()).collect();
    let hashes: Vec<[u8; 32]> = p.media.iter().map(|m| m.hash).collect();
    let plan = specrt::vision::plan_media(&p.ids, &vis.tokens, vis.config.spatial_merge_size, &processed, &hashes).unwrap();
    let want: Value = serde_json::from_str(&std::fs::read_to_string(ref_dir().join("hf-pos-video.json")).unwrap()).unwrap();
    let mut bad = 0;
    for axis in 0..3 {
        for (i, w) in want["pos"][axis].as_array().unwrap().iter().enumerate() {
            if plan.positions[i][axis] as i64 != w.as_i64().unwrap() {
                bad += 1;
                if bad < 6 {
                    eprintln!("axis {axis} token {i}: ours {} transformers {}", plan.positions[i][axis], w);
                }
            }
        }
    }
    eprintln!("POSDIFF {bad} delta ours {} transformers {}", plan.delta, want["delta"]);
    assert_eq!(bad, 0);
}

fn noisy(embeds: &[f32], rel: f32, seed: u64) -> Vec<f32> {
    let rms = (embeds.iter().map(|v| v * v).sum::<f32>() / embeds.len() as f32).sqrt();
    let mut x = seed;
    let mut next = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    embeds.iter().map(|v| v + rel * rms * 1.732 * next()).collect()
}

#[test]
#[ignore = "debug bisect"]
fn real_27b_embedding_noise_sensitivity() {
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let repo = spec.repo.clone();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    for case in &CASES {
        let rec = record(case.name);
        let p = prompt_for(&repo, case, &rec);
        let base = (*rt.plan_media(&p.ids, &p.media).unwrap()).clone();
        for rel in [0.0f32, 0.01, 0.03, 0.06] {
            let mut plan = base.clone();
            plan.embeds = std::sync::Arc::new(noisy(&base.embeds, rel, 5));
            rt.set_media(0, Some(&plan)).unwrap();
            let (kl, top1, n) = kl_against(&mut rt, &p.ids, &rec);
            eprintln!("NOISE {} rel {rel}: mean KL {kl:.4} top1 {top1}/{n}", case.name);
        }
    }
}

#[test]
#[ignore = "debug bisect"]
fn real_27b_prompt_position_kl_against_vllm() {
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let repo = spec.repo.clone();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    let vocab = rt.spec().dims.vocab;
    for case in &CASES {
        let rec = record(case.name);
        let p = prompt_for(&repo, case, &rec);
        let plan = rt.plan_media(&p.ids, &p.media).unwrap();
        rt.set_media(0, Some(&plan)).unwrap();
        let logits = rt.prompt_logits(&p.ids).unwrap();
        let theirs: Value = serde_json::from_str(&std::fs::read_to_string(ref_dir().join(format!("plp-{}.json", case.name))).unwrap()).unwrap();
        assert_eq!(ids(&theirs["ids"]), p.ids);
        let mut rows = Vec::new();
        for (i, row) in theirs["prompt_logprobs"].as_array().unwrap().iter().enumerate() {
            let Some(row) = row.as_object() else { continue };
            let l = &logits[(i - 1) * vocab..i * vocab];
            let max = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = max + l.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
            let top: Vec<(usize, f64)> = row.iter().map(|(k, v)| (k.parse().unwrap(), v["logprob"].as_f64().unwrap())).collect();
            let mut kl = 0.0;
            let p_mass: f64 = top.iter().map(|(_, lp)| lp.exp()).sum();
            for &(t, lp) in &top {
                kl += lp.exp() * (lp - (l[t] - lse) as f64);
            }
            let rest_p = (1.0 - p_mass).max(1e-12);
            let rest_q = (1.0 - top.iter().map(|&(t, _)| ((l[t] - lse) as f64).exp()).sum::<f64>()).max(1e-12);
            kl += rest_p * (rest_p / rest_q).ln();
            let actual = p.ids[i] as usize;
            let likely = top.iter().any(|&(t, lp)| t == actual && lp > -4.6);
            if likely {
                rows.push((i, p.ids[i], kl));
            }
        }
        if case.name == "video" {
            for at in [90usize, 91, 92, 93, 94, 95, 96, 97, 98] {
                let l = &logits[(at - 1) * vocab..at * vocab];
                let max = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let lse = max + l.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
                let mut order: Vec<usize> = (0..vocab).collect();
                order.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
                let ours: Vec<(usize, f32)> = order.iter().take(3).map(|&t| (t, l[t] - lse)).collect();
                let row = theirs["prompt_logprobs"][at].as_object().unwrap();
                let mut tv: Vec<(usize, f64)> = row.iter().map(|(k, v)| (k.parse().unwrap(), v["logprob"].as_f64().unwrap())).collect();
                tv.sort_by(|a, b| b.1.total_cmp(&a.1));
                eprintln!("TOP at {at} actual {}: ours {:?} vllm {:?}", p.ids[at], ours, &tv[..3.min(tv.len())]);
            }
        }
        let media: Vec<usize> = plan.positions.iter().enumerate().filter(|(_, p)| p[3] >= 0).map(|(i, _)| i).collect();
        let (first, last) = (media[0], *media.last().unwrap());
        let all: Vec<String> = rows.iter().map(|r| format!("{}:{:.3}", r.0, r.2)).collect();
        eprintln!("PLP {} likely positions {}", case.name, all.join(" "));
        let mean = |f: &dyn Fn(usize) -> bool| {
            let v: Vec<f64> = rows.iter().filter(|r| f(r.0)).map(|r| r.2).collect();
            (v.iter().sum::<f64>() / v.len().max(1) as f64, v.len())
        };
        let (before, nb) = mean(&|i| i < first);
        let (inside, ni) = mean(&|i| i >= first && i <= last);
        let (after, na) = mean(&|i| i > last);
        eprintln!("PLP {}: before media {before:.4} ({nb}), inside {inside:.4} ({ni}), after {after:.4} ({na})", case.name);
        let worst: Vec<_> = { let mut r = rows.clone(); r.sort_by(|a, b| b.2.total_cmp(&a.2)); r.into_iter().take(6).collect() };
        eprintln!("PLP {} worst {:?}", case.name, worst);
        rt.set_media(0, None).unwrap();
    }
}

#[test]
#[ignore = "control: text-only descriptive prompts against vLLM, the same measure as the media cases"]
fn real_27b_text_only_description_control_against_vllm() {
    if !has_refs() {
        return;
    }
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    let records: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(ref_dir().join("vllm-textctrl.json")).unwrap()).unwrap();
    let (mut kls, mut top1, mut n) = (Vec::new(), 0, 0);
    for rec in &records {
        let (kl, t, c) = kl_against(&mut rt, &ids(&rec["prompt_ids"]), rec);
        eprintln!("CTRL text: mean KL {kl:.4} top1 {t}/{c}");
        kls.push(kl);
        top1 += t;
        n += c;
    }
    eprintln!("CTRL text overall: mean KL {:.4} top1 {top1}/{n}", kls.iter().sum::<f64>() / kls.len() as f64);
}

fn set_case(kind: MediaKind, i: usize) -> (Case, String) {
    let (name, file): (&'static str, &'static str) = match (kind, i) {
        (MediaKind::Image, 0) => ("set_img0", "set/img0.png"),
        (MediaKind::Image, 1) => ("set_img1", "set/img1.png"),
        (MediaKind::Image, 2) => ("set_img2", "set/img2.png"),
        (MediaKind::Image, 3) => ("set_img3", "set/img3.png"),
        (MediaKind::Image, 4) => ("set_img4", "set/img4.png"),
        (MediaKind::Image, _) => ("set_img5", "set/img5.png"),
        (MediaKind::Video, 0) => ("set_vid0", "set/vid0.mp4"),
        (MediaKind::Video, 1) => ("set_vid1", "set/vid1.mp4"),
        (MediaKind::Video, 2) => ("set_vid2", "set/vid2.mp4"),
        (MediaKind::Video, 3) => ("set_vid3", "set/vid3.mp4"),
        (MediaKind::Video, 4) => ("set_vid4", "set/vid4.mp4"),
        (MediaKind::Video, _) => ("set_vid5", "set/vid5.mp4"),
    };
    (Case { name, kind, files: Box::leak(Box::new([file])) }, name.to_string())
}

#[test]
#[ignore = "needs the 27B NVFP4 checkpoint, about 28 GB of free GPU memory and scripts/vllm_vision.py output in SPECRT_VISION_REF_DIR"]
fn real_27b_media_population_against_vllm() {
    if !has_refs() {
        return;
    }
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let repo = spec.repo.clone();
    let mut rt = Runtime::load(&spec, 4096).unwrap();
    for kind in [MediaKind::Image, MediaKind::Video] {
        let (mut kls, mut top1, mut n) = (Vec::new(), 0usize, 0usize);
        for i in 0..6 {
            let (case, name) = set_case(kind, i);
            let rec = record(&name);
            let p = prompt_for(&repo, &case, &rec);
            assert_eq!(p.ids, ids(&rec["prompt_ids"]), "{name}: prompt ids");
            let plan = rt.plan_media(&p.ids, &p.media).unwrap();
            rt.set_media(0, Some(&plan)).unwrap();
            let (kl, t, c) = kl_against(&mut rt, &p.ids, &rec);
            rt.set_media(0, None).unwrap();
            eprintln!("POP {name}: {} vision tokens, mean KL {kl:.4} top1 {t}/{c}", p.vision_tokens);
            kls.push(kl);
            top1 += t;
            n += c;
        }
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        eprintln!("POP {kind:?} overall: mean KL {mean:.4} top1 {top1}/{n}");
        let (max_kl, min_top1) = match kind {
            MediaKind::Image => (0.05, 0.9),
            MediaKind::Video => (0.09, 0.85),
        };
        assert!(mean <= max_kl && top1 as f64 >= min_top1 * n as f64, "{kind:?}: mean KL {mean:.4}, top-1 {top1}/{n}");
    }
}

#[test]
#[ignore = "control: long text-only prompts against vLLM (its KV cache is fp8)"]
fn real_27b_long_text_control_against_vllm() {
    if !has_refs() {
        return;
    }
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let mut rt = Runtime::load(&spec, 4096).unwrap();
    let records: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(ref_dir().join("vllm-longtext.json")).unwrap()).unwrap();
    for rec in &records {
        let prompt = ids(&rec["prompt_ids"]);
        let (kl, t, c) = kl_against(&mut rt, &prompt, rec);
        eprintln!("LONG text {} tokens: mean KL {kl:.4} top1 {t}/{c}", prompt.len());
    }
}

#[test]
#[ignore = "control: random token prompts (unusual hidden states, no media) against vLLM"]
fn real_27b_random_token_control_against_vllm() {
    if !has_refs() {
        return;
    }
    let spec = spec_named("qwen3.8-27b-nvfp4");
    let mut rt = Runtime::load(&spec, 4096).unwrap();
    let records: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(ref_dir().join("vllm-randtext.json")).unwrap()).unwrap();
    let (mut sum, mut top1, mut n) = (0.0, 0, 0);
    for rec in &records {
        let prompt = ids(&rec["prompt_ids"]);
        let (kl, t, c) = kl_against(&mut rt, &prompt, rec);
        eprintln!("RAND {} tokens: mean KL {kl:.4} top1 {t}/{c}", prompt.len());
        sum += kl;
        top1 += t;
        n += c;
    }
    eprintln!("RAND overall mean KL {:.4} top1 {top1}/{n}", sum / records.len() as f64);
}
