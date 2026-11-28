use specrt::duplex::validate::{self, Trace};
use specrt::duplex::{Engine, EngineOptions};
use std::path::PathBuf;

fn model() -> PathBuf {
    std::env::var_os("HARU_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("models/Haru-4B"))
}

fn traces() -> PathBuf {
    std::env::var_os("HARU_TRACE_DIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp/haru1489"))
}

fn trace(name: &str) -> Trace {
    Trace::open(&traces().join(name)).unwrap_or_else(|e| panic!("{e:#}: record it with crates/specrt/scripts/haru_trace.py (docs/guides/haru-validation.md)"))
}

fn engine(layers: Vec<usize>) -> Engine {
    Engine::load(&model(), EngineOptions { slots: 1, max_ctx: 8192, trace_layers: layers, ..EngineOptions::default() }).unwrap_or_else(|e| panic!("{e:#}"))
}

const TOP1: f64 = 0.99;
const KL: f64 = 0.01;
const COSINE: f64 = 0.999;
const LATENT_MEAN: f64 = 2e-3;

fn gate(name: &str, r: &validate::Report) {
    let s = &r.summary;
    eprintln!("{name}: {}", serde_json::to_string(s).unwrap());
    assert!(s.top1_rate >= TOP1, "{name}: top 1 {}", s.top1_rate);
    assert!(s.kl_mean < KL && s.kl_max < KL, "{name}: KL mean {} max {}", s.kl_mean, s.kl_max);
    assert!(s.hidden_cos_min >= COSINE, "{name}: hidden cosine {}", s.hidden_cos_min);
    assert!(s.layer_cos_min.iter().all(|&c| c >= COSINE), "{name}: layer cosines {:?}", s.layer_cos_min);
    assert!(s.latent_rel_mean < LATENT_MEAN, "{name}: latent mean {}", s.latent_rel_mean);
}

#[test]
#[ignore = "needs the real Haru-4B, its recorded traces and GPU1"]
fn teacher_forced_against_the_reference_on_the_real_checkpoint() {
    let first = trace("ref-f32-greedy");
    let mut e = engine(first.layers());
    for name in ["ref-f32-greedy", "ref-f32-sampled", "ref-f32-speakfirst-greedy"] {
        let t = trace(name);
        let p = validate::prompt_check(&e, &t).unwrap();
        assert!(p.tokens_equal && p.masks_equal, "{name}: prompt {p:?}");
        assert!(p.reference_mean < 1e-4, "{name}: voice reference mean {}", p.reference_mean);
        gate(name, &validate::teacher_forced(&mut e, &t, None).unwrap());
    }
}

#[test]
#[ignore = "needs the real Haru-4B, its recorded traces and GPU1"]
fn greedy_free_running_matches_the_reference_token_for_token_including_silent_input() {
    let first = trace("ref-f32-greedy");
    let mut e = engine(first.layers());
    for name in ["ref-f32-greedy", "ref-f32-speakfirst-greedy", "ref-f32-silence-novoice-greedy", "ref-f32-silence-voice-greedy"] {
        let t = trace(name);
        let run = validate::free_running(&mut e, &t, 0).unwrap();
        eprintln!("{name}: first divergence {:?}, voiced {} vs {}: {}", run.first_divergence, run.voiced_frames, run.want_voiced_frames, run.text);
        assert_eq!(run.first_divergence, None, "{name}: greedy tokens");
        assert!(run.voiced_frames.abs_diff(run.want_voiced_frames) <= 1, "{name}: voiced frames {} against {}", run.voiced_frames, run.want_voiced_frames);
    }
}
