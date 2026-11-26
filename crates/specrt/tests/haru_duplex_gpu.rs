use serde_json::Value;
use specrt::duplex::prompt::Prompt;
use specrt::duplex::{Engine, EngineOptions, Mode, RecordedNoise, StartOptions};
use std::path::{Path, PathBuf};

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

const BACKBONE_BF16_TOLERANCE: f32 = 3e-4;
const USER_LATENT_BOUND: f32 = 1e-5;
const CODEC_GAIN: f32 = 25.0;

fn engine(slots: usize, graphs: bool) -> Engine {
    let opts = EngineOptions { slots, max_ctx: 256, graphs, ..EngineOptions::default() };
    Engine::load(&dir(), opts).unwrap_or_else(|e| panic!("{e:#}"))
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn each_frame_fed_the_reference_feedback_matches_the_reference_latent_user_stream_and_audio() {
    let mut e = engine(1, true);
    let user = reference("session.user_audio");
    let (want_tokens, want_latent, want_hidden, want_user) = (reference("session.token"), reference("session.latent"), reference("session.hidden"), reference("session.user_latent"));
    let want_audio: Vec<f32> = reference("session.audio").iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    let (id, first) = e.start_prompt(&prompt(), recorded()).unwrap();
    let (mut latents, mut audio) = (first.latent.clone(), first.audio.clone());
    let mut worst = (0f32, 0f32);
    for (i, f) in user.chunks(3200).enumerate() {
        e.set_feedback(id, want_tokens[i] as u32, &want_latent[i * 64..(i + 1) * 64]).unwrap();
        let o = e.step(&[(id, f.to_vec())]).unwrap().remove(0);
        assert_eq!(o.token, want_tokens[i + 1] as u32, "greedy token of frame {}", i + 1);
        let stream = e.rt.stream().clone();
        let u = stream.clone_dtoh(&e.rt.input_buffer("user_latent").unwrap().slice(..64)).unwrap();
        let h = max_rel(&e.rt.normed_rows(1).unwrap(), &want_hidden[(i + 1) * 64..(i + 2) * 64]);
        worst = (worst.0.max(max_rel(&u, &want_user[i * 64..(i + 1) * 64])), worst.1.max(h));
        latents.extend(&o.latent);
        audio.extend(&o.audio);
    }
    e.end(id).unwrap();
    let per: Vec<String> = latents.chunks(64).zip(want_latent.chunks(64)).map(|(a, b)| format!("{:.1e}", max_rel(a, b))).collect();
    let (l, a) = (max_rel(&latents, &want_latent), max_rel(&audio, &want_audio));
    eprintln!("teacher forced: user latent {:.1e} hidden {:.1e} latent {l:.1e} per frame {per:?} audio {a:.1e}", worst.0, worst.1);
    assert!(worst.0 < USER_LATENT_BOUND, "session.user_latent {}", worst.0);
    assert!(worst.1 < BACKBONE_BF16_TOLERANCE, "session.hidden {}", worst.1);
    assert!(l < BACKBONE_BF16_TOLERANCE, "session.latent {l}");
    assert!(a < CODEC_GAIN * l, "session.audio {a} for latents {l}");
    let mut s = e.codec.sessions(&e.codec.decoder, 1).unwrap();
    let own: Vec<f32> = latents.chunks(64).flat_map(|z| e.codec.run(&e.codec.decoder, z, &mut s).unwrap()).map(|v| v.clamp(-1.0, 1.0)).collect();
    assert_eq!(audio, own, "the frame step decodes exactly the latents it emits");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn free_running_drift_comes_from_the_backbone_floor_not_from_session_state() {
    let mut e = engine(1, false);
    let user = reference("session.user_audio");
    let (want_latent, want_hidden) = (reference("session.latent"), reference("session.hidden"));
    let (id, _) = e.start_prompt(&prompt(), recorded()).unwrap();
    let mut free = Vec::new();
    for (i, f) in user.chunks(3200).enumerate() {
        let o = e.step(&[(id, f.to_vec())]).unwrap().remove(0);
        free.push((max_rel(&e.rt.normed_rows(1).unwrap(), &want_hidden[(i + 1) * 64..(i + 2) * 64]), max_rel(&o.latent, &want_latent[(i + 1) * 64..(i + 2) * 64])));
    }
    eprintln!("free running hidden and latent per frame {free:?}");
    assert!(free[0].0 < 3.0 * BACKBONE_BF16_TOLERANCE, "the first fed back frame starts at the backbone floor");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_voice_reference_that_ends_mid_frame_encodes_like_the_reference_whole_clip() {
    let e = engine(1, false);
    let clip = reference("codec.clip");
    assert_ne!(clip.len() % 3200, 0, "the fixture clip ends mid frame");
    let got = e.encode_clip(&clip).unwrap();
    let want = reference("codec.clip_mean");
    let last = max_rel(&got[got.len() - 64..], &want[want.len() - 64..]);
    eprintln!("reference clip mean {:.2e}, last frame {last:.2e}", max_rel(&got, &want));
    assert!(max_rel(&got, &want) < 1e-4, "codec.clip_mean through the prompt encoder {}", max_rel(&got, &want));
}

fn prompt() -> Prompt {
    Prompt {
        tokens: reference("prompt.tokens").iter().map(|&t| t as u32).collect(),
        model_latents: reference("prompt.model_latents"),
        user_latents: reference("prompt.user_latents"),
        model_mask: reference("prompt.model_mask"),
        user_mask: reference("prompt.user_mask"),
    }
}

fn recorded() -> StartOptions {
    StartOptions {
        greedy: true,
        noise: Some(RecordedNoise { head: reference("session.head_noise"), user: reference("session.user_noise"), user_scale: reference("session.user_scale") }),
        ..StartOptions::default()
    }
}

struct Run {
    tokens: Vec<u32>,
    latents: Vec<f32>,
    audio: Vec<f32>,
}

fn replay(e: &mut Engine) -> Run {
    let user = reference("session.user_audio");
    let (id, first) = e.start_prompt(&prompt(), recorded()).unwrap();
    let mut run = Run { tokens: vec![first.token], latents: first.latent.clone(), audio: first.audio.clone() };
    for f in user.chunks(3200) {
        let out = e.step(&[(id, f.to_vec())]).unwrap().remove(0);
        run.tokens.push(out.token);
        run.latents.extend(&out.latent);
        run.audio.extend(&out.audio);
    }
    e.end(id).unwrap();
    run
}

fn check(what: &str, run: &Run) {
    let want_tokens: Vec<u32> = reference("session.token").iter().map(|&t| t as u32).collect();
    let (want_latent, want_audio) = (reference("session.latent"), reference("session.audio"));
    let per: Vec<String> = run.latents.chunks(64).zip(want_latent.chunks(64)).map(|(a, b)| format!("{:.1e}", max_rel(a, b))).collect();
    let a = max_rel(&run.audio, &want_audio.iter().map(|v| v.clamp(-1.0, 1.0)).collect::<Vec<_>>());
    eprintln!("{what}: tokens {:?} latent per frame {per:?} audio max rel {a:.2e} (abs {:.2e})", run.tokens, max_abs(&run.audio, &want_audio));
    assert_eq!(run.tokens, want_tokens, "{what}: session.token");
    let first = max_rel(&run.latents[..64], &want_latent[..64]);
    assert!(first < BACKBONE_BF16_TOLERANCE, "{what}: frame 0 latent from the prompt {first}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_six_frame_greedy_session_replays_every_token_on_graphs_and_eager() {
    for graphs in [false, true] {
        let mut e = engine(1, graphs);
        let run = replay(&mut e);
        check(if graphs { "graphs" } else { "eager" }, &run);
        let again = replay(&mut e);
        assert_eq!(again.tokens, run.tokens, "a second session on the same slot starts from clean state");
        assert!(max_rel(&again.audio, &run.audio) < 1e-6, "codec and recurrent state reset between sessions");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_sessions_batched_in_one_step_each_replay_the_reference() {
    let mut e = engine(3, true);
    let user = reference("session.user_audio");
    let (a, fa) = e.start_prompt(&prompt(), recorded()).unwrap();
    let (b, fb) = e.start_prompt(&prompt(), recorded()).unwrap();
    let mut runs = [Run { tokens: vec![fa.token], latents: fa.latent, audio: fa.audio }, Run { tokens: vec![fb.token], latents: fb.latent, audio: fb.audio }];
    for f in user.chunks(3200) {
        let out = e.step(&[(a, f.to_vec()), (b, f.to_vec())]).unwrap();
        for (r, o) in runs.iter_mut().zip(out) {
            r.tokens.push(o.token);
            r.latents.extend(&o.latent);
            r.audio.extend(&o.audio);
        }
    }
    check("batched session a", &runs[0]);
    check("batched session b", &runs[1]);
    let mut solo = engine(1, true);
    let alone = replay(&mut solo);
    for r in &runs {
        assert!(max_rel(&r.latents, &alone.latents) < 1e-5 && max_rel(&r.audio, &alone.audio) < 1e-5, "a batched session matches the same session alone");
    }
    assert_eq!(e.capture_secs.iter().map(|c| c.0).collect::<Vec<_>>(), vec![2], "one graph for the batch of two");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_snapshot_mid_session_restores_to_the_same_next_frames() {
    let user = reference("session.user_audio");
    let frames: Vec<Vec<f32>> = user.chunks(3200).map(<[f32]>::to_vec).collect();
    let mut e = engine(2, true);
    let (id, _) = e.start_prompt(&prompt(), recorded()).unwrap();
    for f in &frames[..3] {
        e.step(&[(id, f.clone())]).unwrap();
    }
    let snap = e.snapshot(id).unwrap();
    let store_dir = std::env::temp_dir().join(format!("haru-duplex-store-{}", std::process::id()));
    let store = specrt::sessions::SessionStore::open(&store_dir, e.rt.session_compat()).unwrap();
    let straight: Vec<_> = frames[3..].iter().map(|f| e.step(&[(id, f.clone())]).unwrap().remove(0)).collect();
    let key = store.write(&snap, 1).unwrap().id;
    e.end(id).unwrap();

    let mut fresh = engine(2, true);
    let back = fresh.resume(&store, &key, 2).unwrap();
    let resumed: Vec<_> = frames[3..].iter().map(|f| fresh.step(&[(back, f.clone())]).unwrap().remove(0)).collect();
    for (s, r) in straight.iter().zip(&resumed) {
        assert_eq!((s.token, s.index), (r.token, r.index), "token and frame index after restore");
        assert!(max_rel(&r.latent, &s.latent) < 1e-6 && max_rel(&r.audio, &s.audio) < 1e-6, "frame {} after restore", s.index);
    }
    let want: Vec<u32> = reference("session.token").iter().map(|&t| t as u32).collect();
    assert_eq!(resumed.iter().map(|f| f.token).collect::<Vec<_>>(), want[4..].to_vec());
    let _ = std::fs::remove_dir_all(store_dir);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn relay_mode_and_the_say_queue_constrain_the_text_and_interrupt_reports_what_was_said() {
    let mut e = engine(1, true);
    let user = reference("session.user_audio");
    let (pad, eop) = (e.duplex.pad, e.duplex.eop);
    let (id, _) = e.start_prompt(&prompt(), StartOptions { mode: Mode::Relay, seed: 7, ..StartOptions::default() }).unwrap();
    for f in user.chunks(3200).take(3) {
        let t = e.step(&[(id, f.to_vec())]).unwrap()[0].token;
        assert!(t == pad || t == eop, "relay mode spoke token {t}");
    }
    e.say_tokens(id, vec![5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
    e.duplex.sampling.eop_bias = 0.0;
    let mut said = Vec::new();
    for f in user.chunks(3200).cycle().take(30) {
        let t = e.step(&[(id, f.to_vec())]).unwrap()[0].token;
        assert!(t == pad || t == eop || (5..=12).contains(&t), "token {t} outside the queue");
        if t != pad && t != eop {
            said.push(t);
        }
        if said.len() == 3 {
            break;
        }
    }
    assert_eq!(said, vec![5, 6, 7][..said.len()].to_vec(), "queued words come out in order");
    let (r, _) = e.interrupt(id).unwrap();
    assert_eq!(r.said, said);
    assert_eq!([r.said.clone(), r.unsaid.clone()].concat(), vec![5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(e.live(id).unwrap().text.queued(), 0);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn max_frames_ends_the_session_and_the_window_keeps_it_running_past_the_context() {
    let user = reference("session.user_audio");
    let mut e = Engine::load(&dir(), EngineOptions { slots: 1, max_ctx: 64, max_frames: Some(5), ..EngineOptions::default() }).unwrap();
    let (id, _) = e.start_prompt(&prompt(), StartOptions { seed: 3, ..StartOptions::default() }).unwrap();
    let done: Vec<bool> = user.chunks(3200).take(4).map(|f| e.step(&[(id, f.to_vec())]).unwrap()[0].done).collect();
    assert_eq!(done, vec![false, false, false, true], "frame 0 plus four steps reach max_frames 5");
    assert!(e.step(&[(id, user[..3200].to_vec())]).is_err());

    let mut w = Engine::load(&dir(), EngineOptions { slots: 1, max_ctx: 64, window_frames: Some(32), ..EngineOptions::default() }).unwrap();
    let (id, _) = w.start_prompt(&prompt(), StartOptions { seed: 3, ..StartOptions::default() }).unwrap();
    let mut last = None;
    for i in 0..120 {
        let f = &user[(i % 6) * 3200..(i % 6 + 1) * 3200];
        let o = w.step(&[(id, f.to_vec())]).unwrap().remove(0);
        assert!(!o.done && o.audio.iter().all(|v| v.is_finite()), "frame {i}");
        last = Some(o.index);
    }
    let l = w.live(id).unwrap();
    assert_eq!(last, Some(120));
    assert!(l.kept < 64 && l.kept >= l.prompt_len + 32 - 16, "kept {} positions", l.kept);
}
