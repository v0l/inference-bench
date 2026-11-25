use serde_json::Value;
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const EMBED_BOUND: f32 = 1e-5;
const BACKBONE_BOUND: f32 = 3e-4;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/haru/tiny")
}

fn reference(key: &str) -> Vec<f32> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("reference.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir().join("reference").join(meta["files"][key]["file"].as_str().unwrap())).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn spec() -> Spec {
    let texts: Vec<String> = specrt::spec::EMBEDDED
        .iter()
        .map(|(_, text)| {
            let mut table: toml::Table = toml::from_str(text).unwrap();
            if table.get("name").and_then(|n| n.as_str()) == Some("haru") {
                table.remove("refuse");
            }
            table.to_string()
        })
        .collect();
    let registry = specrt::family::Registry::from_texts(texts.iter().map(String::as_str)).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(&registry, &std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = "tiny-haru".into();
    spec.repo = dir().display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn serving(slots: usize) -> Layout {
    let max_ctx = 256;
    let pages = max_ctx / specrt::codegen::PAGE;
    Layout { max_ctx, slots, kv_blocks: slots * pages, prefill_chunk: 64, kv_planes: 2, kv_headroom: 0, state_on_demand: false }
}

fn prompt_inputs(rt: &mut Runtime, n: usize) {
    let (model, user, model_mask, user_mask) = (reference("prompt.model_latents"), reference("prompt.user_latents"), reference("prompt.model_mask"), reference("prompt.user_mask"));
    rt.set_inputs(n, &[("model_latent", &model), ("user_latent", &user), ("model_mask", &model_mask), ("user_mask", &user_mask)]).unwrap();
}

fn frame_inputs(rt: &mut Runtime, model: &[f32], user: &[f32]) {
    let ones = vec![1.0f32; model.len() / 64];
    rt.set_inputs(ones.len(), &[("model_latent", model), ("user_latent", user), ("model_mask", &ones), ("user_mask", &ones)]).unwrap();
}

#[test]
#[ignore = "needs a CUDA GPU (sm_80 or newer)"]
fn the_tiny_haru_prompt_matches_the_reference_embeds_layers_hidden_and_logits() {
    let spec = spec();
    let (h, vocab) = (spec.dims.hidden, spec.dims.vocab);
    let tokens: Vec<u32> = reference("prompt.tokens").iter().map(|&t| t as u32).collect();
    let n = tokens.len();
    let mut rt = Runtime::load_layout(&spec, Layout { kv_planes: 2, prefill_chunk: 64, ..Layout::contiguous(256, 1) }).unwrap();
    prompt_inputs(&mut rt, n);
    let embeds = rt.hidden_states(&tokens, n, &[0]).unwrap();
    let e = max_rel(&embeds, &reference("prompt.embeds"));
    let want_layers = reference("prompt.layers");
    let trace = rt.trace_prefill(&tokens).unwrap();
    let layers: Vec<f32> = (0..spec.layers.count).map(|l| max_rel(&trace.layers[l * n * h..(l + 1) * n * h], &want_layers[(l + 1) * n * h..(l + 2) * n * h])).collect();
    let hidden = max_rel(&rt.normed_rows(n).unwrap(), &reference("prompt.hidden"));
    let want_logits = reference("prompt.logits");
    eprintln!("prompt: embeds {e:.2e} layers {layers:?} hidden {hidden:.2e} logits {:.2e}", max_rel(&trace.logits, &want_logits));
    assert!(e < EMBED_BOUND, "prompt.embeds {e}");
    assert!(layers.iter().all(|&r| r < BACKBONE_BOUND), "prompt.layers {layers:?}");
    assert!(hidden < BACKBONE_BOUND, "prompt.hidden {hidden}");
    for i in 0..n {
        assert_eq!(argmax(&trace.logits[i * vocab..(i + 1) * vocab]), argmax(&want_logits[i * vocab..(i + 1) * vocab]), "prompt.logits top 1 at {i}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU (sm_80 or newer)"]
fn the_tiny_haru_prompt_fed_one_embedding_row_at_a_time_matches_whole_prefill() {
    let spec = spec();
    let d = 64;
    let tokens: Vec<u32> = reference("prompt.tokens").iter().map(|&t| t as u32).collect();
    let (model, user, model_mask, user_mask, want) = (reference("prompt.model_latents"), reference("prompt.user_latents"), reference("prompt.model_mask"), reference("prompt.user_mask"), reference("prompt.hidden"));
    let mut rt = Runtime::load_layout(&spec, serving(1)).unwrap();
    let mut st = rt.stepper().unwrap();
    let set = |rt: &mut Runtime, i: usize| {
        rt.set_inputs(1, &[("model_latent", &model[i * d..(i + 1) * d]), ("user_latent", &user[i * d..(i + 1) * d]), ("model_mask", &model_mask[i..i + 1]), ("user_mask", &user_mask[i..i + 1])]).unwrap();
    };
    set(&mut rt, 0);
    rt.prefill_slot(&mut st, 0, &tokens[..1], 0, None).unwrap();
    let mut got = rt.normed_rows(1).unwrap();
    for i in 1..tokens.len() {
        set(&mut rt, i);
        rt.decode(&mut st, &[DecodeRow { slot: 0, token: tokens[i], pos: i, sampling: Sampling::GREEDY, step: i as u32 }]).unwrap();
        got.extend(rt.normed_rows(1).unwrap());
    }
    let r = max_rel(&got, &want);
    eprintln!("prompt one row at a time: hidden {r:.2e}");
    assert!(r < BACKBONE_BOUND, "prompt.hidden one row at a time {r}");
}

#[test]
#[ignore = "needs a CUDA GPU (sm_80 or newer)"]
fn the_tiny_haru_session_replays_frame_by_frame_through_batched_decode() {
    let spec = spec();
    let (h, d) = (spec.dims.hidden, 64);
    let tokens: Vec<u32> = reference("prompt.tokens").iter().map(|&t| t as u32).collect();
    let n = tokens.len();
    let (frame_tokens, latents, users, want_hidden, want_embeds) = (reference("session.token"), reference("session.latent"), reference("session.user_latent"), reference("session.hidden"), reference("session.embeds"));
    let frames = users.len() / d;
    let mut rt = Runtime::load_layout(&spec, serving(2)).unwrap();
    let mut st = rt.stepper().unwrap();
    prompt_inputs(&mut rt, n);
    let first = rt.prefill_slot(&mut st, 1, &tokens, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap();
    assert_eq!(first, frame_tokens[0] as u32, "first frame token");
    let first_hidden = max_rel(&rt.normed_rows(1).unwrap(), &want_hidden[..h]);
    eprintln!("session first frame hidden {first_hidden:.2e}");
    let mut worst = first_hidden;
    for i in 0..frames {
        frame_inputs(&mut rt, &latents[i * d..(i + 1) * d], &users[i * d..(i + 1) * d]);
        let row = DecodeRow { slot: 1, token: frame_tokens[i] as u32, pos: n + i, sampling: Sampling::GREEDY, step: i as u32 + 1 };
        let next = rt.decode(&mut st, &[row]).unwrap()[0];
        worst = worst.max(max_rel(&rt.normed_rows(1).unwrap(), &want_hidden[(i + 1) * h..(i + 2) * h]));
        assert_eq!(next, frame_tokens[i + 1] as u32, "greedy token of frame {}", i + 1);
    }
    let mut emb = Runtime::load_layout(&spec, Layout { kv_planes: 2, prefill_chunk: 64, ..Layout::contiguous(64, 1) }).unwrap();
    frame_inputs(&mut emb, &latents[..frames * d], &users);
    let ids: Vec<u32> = frame_tokens[..frames].iter().map(|&t| t as u32).collect();
    let e = max_rel(&emb.hidden_states(&ids, frames, &[0]).unwrap(), &want_embeds);
    eprintln!("session: embeds {e:.2e} hidden worst frame {worst:.2e}");
    assert!(e < EMBED_BOUND, "session.embeds {e}");
    assert!(worst < BACKBONE_BOUND, "session.hidden {worst}");
}
