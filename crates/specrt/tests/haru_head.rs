use serde_json::Value;
use specrt::flow_head::{FlowHead, HostHead, Level};
use specrt::plan::{DitOp, Op};
use std::path::{Path, PathBuf};

const TOLERANCE: f32 = 1e-4;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/haru")
}

fn config(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join(name)).unwrap()).unwrap()
}

fn reference(key: &str) -> Vec<f32> {
    let meta = config("tiny/reference.json");
    let bytes = std::fs::read(dir().join("tiny/reference").join(meta["files"][key]["file"].as_str().unwrap_or_else(|| panic!("{key}")))).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

fn host(steps: Option<usize>) -> HostHead {
    HostHead::load(FlowHead::from_dir(&dir().join("tiny"), steps).unwrap(), &dir().join("tiny")).unwrap()
}

fn labels(head: &FlowHead, level: Level) -> Vec<String> {
    let levels = head.levels().unwrap();
    head.plan.head.iter().zip(&levels).filter(|(_, l)| **l == level).map(|(o, _)| o.label()).collect()
}

fn run(head: &HostHead, guidance: f32, frame_by_frame: bool) -> Vec<f32> {
    let hidden = reference("session.hidden");
    let noise = reference("session.head_noise");
    let (h, d) = (head.head.spec.hidden(), head.head.spec.size("latent_dim").unwrap());
    let frames = hidden.len() / h;
    if frame_by_frame {
        let mut s = head.session(&[guidance]).unwrap();
        (0..frames).flat_map(|f| head.frame(&mut s, &hidden[f * h..(f + 1) * h], &noise[f * d..(f + 1) * d]).unwrap()).collect()
    } else {
        let mut s = head.session(&vec![guidance; frames]).unwrap();
        head.frame(&mut s, &hidden, &noise).unwrap()
    }
}

#[test]
fn the_head_is_a_plan_from_the_family_and_the_graph() {
    let head = FlowHead::from_config(&config("haru-4b.config.json"), None).unwrap();
    assert_eq!((head.graph.name.as_str(), head.root.as_str(), head.steps), ("meanflow", "head", 1));
    assert_eq!((head.spec.hidden(), head.spec.size("mlp").unwrap(), head.spec.size("latent_dim").unwrap(), head.layers().unwrap()), (2560, 7680, 64, vec![4]));
    assert_eq!((head.spec.get("guidance_min").unwrap(), head.spec.get("guidance_max").unwrap(), head.spec.get("interval_embed").unwrap()), (1.0, 2.0, 7.0));
    let ops = &head.plan.blocks[0].ops;
    let modulate = ops.iter().find_map(|o| if let Op::Dit(d @ DitOp::Modulate { .. }) = o { Some(*d) } else { None }).unwrap();
    assert!(matches!(modulate, DitOp::Modulate { rms: true, weight: Some("head.layers.{i}.norm"), .. }), "{modulate:?}");
    assert!(ops.iter().any(|o| matches!(o, Op::Dit(DitOp::Glu { .. }))) && ops.iter().any(|o| matches!(o, Op::Dit(DitOp::GatedAdd { .. }))));
    assert!(head.plan.tail.iter().any(|o| matches!(o, Op::Dit(DitOp::Map { expr: "z + h_step * u", .. }))), "the step rule is an expression");
    assert!(head.scalar_values(0, &[2.5]).unwrap_err().to_string().contains("outside the model's range"));
    let mut header: Vec<String> = serde_json::from_str::<Vec<(String, String, Vec<usize>)>>(&std::fs::read_to_string(dir().join("haru-4b.header.json")).unwrap()).unwrap().into_iter().filter(|r| r.0.starts_with("head.")).map(|r| r.0).collect();
    header.sort();
    let mut planned: Vec<String> = Vec::new();
    let mut add = |ops: &[Op], blocks: Option<usize>| {
        for i in blocks.map_or(vec![None], |n| (0..n).map(Some).collect()) {
            for op in ops {
                let names = match op {
                    Op::Dit(DitOp::Linear { weight, .. }) => specrt::denoiser::weight_names(weight, i).into_iter().map(|(n, _)| n).collect(),
                    Op::Dit(DitOp::Modulate { weight: Some(w), .. }) => vec![format!("{}.weight", w.replace("{i}", &i.unwrap().to_string()))],
                    _ => Vec::new(),
                };
                planned.extend(names);
            }
        }
    };
    add(&head.plan.head, None);
    add(&head.plan.blocks[0].ops, Some(4));
    add(&head.plan.tail, None);
    planned.sort();
    planned.dedup();
    assert_eq!(planned, header, "the plan reads every head tensor of the real checkpoint");
}

#[test]
fn the_four_scalar_embeddings_are_hoisted_to_once_per_session() {
    let head = FlowHead::from_config(&config("haru-4b.config.json"), None).unwrap();
    let session = labels(&head, Level::Session);
    for w in ["t_embed.fc1", "t_embed.fc2", "dt_embed.fc2", "h_embed.fc2", "omega_embed.fc1", "omega_embed.fc2"] {
        assert!(session.iter().any(|l| l.starts_with(&format!("head.{w} "))), "{w} runs once per session: {session:?}");
    }
    assert!(session.iter().any(|l| l == "seqs map into emb"), "the sum of the four embeddings is hoisted: {session:?}");
    let frame = labels(&head, Level::Frame);
    assert!(frame.iter().any(|l| l.starts_with("head.cond_proj")) && frame.iter().any(|l| l == "seqs map into c"), "{frame:?}");
    assert!(labels(&head, Level::Step).iter().all(|l| l.starts_with("split z") || l.starts_with("head.latent_proj")), "{:?}", labels(&head, Level::Step));
    let r = head.report(1, &kir::Target::sm_120()).unwrap();
    let (per_session, _, _, _) = r.per(Level::Session, false);
    assert!(per_session > 52_000_000, "the hoisted embeddings read their 4 x 7.2 M bf16 weights once: {per_session}");
    let two = FlowHead::from_config(&config("haru-4b.config.json"), Some(2)).unwrap();
    let moving = labels(&two, Level::Step);
    assert!(moving.iter().any(|l| l.starts_with("head.t_embed.fc2")), "with two steps t moves, so its embedding runs every step: {moving:?}");
    assert!(labels(&two, Level::Session).iter().any(|l| l.starts_with("head.omega_embed.fc2")), "omega stays per session");
}

#[test]
fn the_rows_gemv_fuses_the_gated_residual_into_the_down_projection() {
    let head = FlowHead::from_config(&config("haru-4b.config.json"), None).unwrap();
    let fused = head.gated_epilogues(8, &kir::Target::sm_120()).unwrap();
    let ops = &head.plan.blocks[0].ops;
    let down = ops.iter().position(|o| matches!(o, Op::Dit(DitOp::Linear { weight, .. }) if weight.ends_with("down_proj"))).unwrap();
    for f in head.block_fusion(8, &kir::Target::sm_120()).unwrap() {
        for g in &f.groups {
            eprintln!("{}: [{}]", f.name, g.units.join(" + "));
        }
    }
    assert_eq!(fused, vec![std::collections::BTreeSet::from([down])], "the gated add joins the down projection epilogue");
}

#[test]
fn the_host_head_matches_the_session_latents_and_the_head_references() {
    for (key, guidance, steps) in [("session.latent", 1.5, None), ("head.guidance1.0", 1.0, None), ("head.guidance2.0", 2.0, None), ("head.steps2", 1.5, Some(2))] {
        let head = host(steps);
        let want = reference(key);
        let batched = run(&head, guidance, false);
        let alone = run(&head, guidance, true);
        let (e, b) = (max_rel(&batched, &want), max_rel(&alone, &batched));
        eprintln!("{key}: max rel {e:e}, frame by frame against batched {b:e}");
        assert!(e < TOLERANCE, "{key}: {e}");
        assert!(b < 1e-6, "{key}: {b}");
    }
}

#[test]
fn a_guidance_change_reruns_the_hoisted_embeddings_and_a_frame_does_not() {
    let head = host(None);
    let hidden = reference("session.hidden");
    let noise = reference("session.head_noise");
    let mut s = head.session(&[1.0]).unwrap();
    let session_ops = s.ran.clone();
    assert!(session_ops.iter().any(|l| l.starts_with("head.omega_embed.fc2")), "{session_ops:?}");
    head.frame(&mut s, &hidden[..64], &noise[..64]).unwrap();
    assert!(!s.ran.iter().any(|l| l.contains("_embed.")), "a frame runs no scalar embedding: {:?}", s.ran);
    head.set_guidance(&mut s, &[2.0]).unwrap();
    assert_eq!(s.ran, session_ops);
    let second = head.frame(&mut s, &hidden[..64], &noise[..64]).unwrap();
    let want = reference("head.guidance2.0");
    assert!(max_rel(&second, &want[..64]) < TOLERANCE);
}

#[test]
fn the_engine_noise_is_the_documented_generator() {
    let a = specrt::flow_head::frame_noise(7, 3, 64);
    assert_eq!(a, specrt::flow_head::frame_noise(7, 3, 64));
    assert_ne!(a, specrt::flow_head::frame_noise(7, 4, 64));
    let mean = a.iter().sum::<f32>() / 64.0;
    let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 64.0;
    assert!(mean.abs() < 0.5 && (0.4..2.0).contains(&var), "{mean} {var}");
    let head = host(None);
    let mut s = head.session(&[1.5]).unwrap();
    let out = head.frame(&mut s, &reference("session.hidden")[..64], &a).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn the_plan_report_counts_bytes_and_launches_for_the_real_config() {
    let head = FlowHead::from_config(&config("haru-4b.config.json"), None).unwrap();
    let r = head.report(1, &kir::Target::sm_120()).unwrap();
    let text = r.render();
    eprintln!("{text}");
    let (read, _, _, launches) = r.per(Level::Frame, false);
    let (_, _, _, fused) = r.per(Level::Frame, true);
    assert!((600_000_000..700_000_000).contains(&read), "per frame the head reads about 0.67 GB at bf16: {read}");
    assert!(fused < launches, "{fused} {launches}");
    assert!(text.contains("once per session") && text.contains("per frame: read"), "{text}");
}
