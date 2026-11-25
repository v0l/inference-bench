use specrt::sampler::{Pipeline, noise};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference/tiny-flux2-klein.json")).unwrap()).unwrap()
}

fn floats(key: &str) -> Vec<f32> {
    let file = meta()["files"][key]["file"].as_str().unwrap_or_else(|| panic!("no reference {key}")).to_string();
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn recorded_prompt(key: &str) -> String {
    let ids = floats(&format!("{key}.ids"));
    let mask = floats(&format!("{key}.mask"));
    ids.iter().zip(&mask).filter(|(_, m)| **m == 1.0).map(|(i, _)| format!("w{}", *i as u32)).collect::<Vec<_>>().join(" ")
}

#[test]
fn the_fixture_tokenizer_turns_a_text_prompt_into_the_recorded_ids() {
    let p = pipeline();
    let encoder = specrt::sampler::PromptEncoder::open(&fixture().join("tokenizer")).unwrap();
    let tokens = meta()["text_tokens"].as_u64().unwrap() as usize;
    assert_eq!(encoder.max_length(&p), tokens);
    for key in ["text", "negative"] {
        let got = encoder.encode(&p, &recorded_prompt(key), encoder.max_length(&p)).unwrap();
        let ids: Vec<u32> = floats(&format!("{key}.ids")).iter().map(|v| *v as u32).collect();
        let mask: Vec<u8> = floats(&format!("{key}.mask")).iter().map(|v| *v as u8).collect();
        assert_eq!((got.ids, got.mask), (ids, mask), "{key}");
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn pipeline() -> Pipeline {
    Pipeline::from_dir(&fixture()).unwrap()
}

#[test]
fn the_schedule_matches_the_diffusers_sigmas() {
    let m = meta();
    let p = pipeline();
    let sched = p.schedule(m["steps"].as_u64().unwrap() as usize, m["image_tokens"].as_u64().unwrap() as usize).unwrap();
    let want: Vec<f32> = m["sigmas"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap() as f32).collect();
    eprintln!("sigmas {:?} want {want:?}", sched.sigmas);
    assert_eq!(sched.sigmas.len(), want.len());
    assert!(max_diff(&sched.sigmas, &want) < 1e-6, "{:?} against {want:?}", sched.sigmas);
    let timesteps: Vec<f32> = m["timesteps"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap() as f32).collect();
    for (i, t) in timesteps.iter().enumerate() {
        assert!((sched.model_timestep(i) * 1000.0 - t).abs() < 1e-3, "step {i}");
    }
}

#[test]
fn the_euler_rule_from_the_pipeline_reproduces_the_recorded_latents() {
    let m = meta();
    let p = pipeline();
    let sched = p.schedule(3, 64).unwrap();
    let mut x = floats("noise");
    for i in 0..3 {
        let v = floats(&format!("step{i}.velocity"));
        p.step_host(&mut x, &v, None, &sched, i, 1.0).unwrap();
        let d = max_diff(&x, &floats(&format!("step{i}.latent")));
        eprintln!("step {i}: euler max diff {d:e}");
        assert!(d < 1e-5, "step {i}: {d}");
    }
    assert_eq!(m["guidance"].as_f64(), Some(1.0));
}

#[test]
fn a_wrong_update_rule_fails_the_recorded_latents() {
    let mut p = pipeline();
    p.update = kir::expr::Expr::parse("x + (s0 - s1) * v").unwrap();
    let sched = p.schedule(3, 64).unwrap();
    let mut x = floats("noise");
    p.step_host(&mut x, &floats("step0.velocity"), None, &sched, 0, 1.0).unwrap();
    assert!(max_diff(&x, &floats("step0.latent")) > 1e-2);
}

#[test]
fn the_cfg_combine_is_the_guided_pipeline_formula() {
    let p = pipeline();
    let sched = p.schedule(3, 64).unwrap();
    let c = [1.0f32, 2.0, -1.0];
    let u = [0.5f32, -2.0, 3.0];
    let mut x = [0.25f32, 0.0, 1.0];
    p.step_host(&mut x, &c, Some(&u), &sched, 0, 4.0).unwrap();
    let dt = sched.sigmas[1] - sched.sigmas[0];
    for i in 0..3 {
        let v = u[i] + 4.0 * (c[i] - u[i]);
        let want = [0.25f32, 0.0, 1.0][i] + dt * v;
        assert!((x[i] - want).abs() < 1e-6, "{i}: {} against {want}", x[i]);
    }
}

#[test]
fn the_host_noise_is_the_noise_the_reference_ran_from() {
    let m = meta();
    assert_eq!(m["noise"].as_str(), Some("splitmix64 counter, Box-Muller pairs"));
    let grid = noise(m["seed"].as_u64().unwrap(), 16 * 8 * 8);
    let tokens: Vec<f32> = (0..64).flat_map(|t| (0..16).map(move |c| (t, c))).map(|(t, c)| grid[c * 64 + t]).collect();
    let d = max_diff(&tokens, &floats("noise"));
    eprintln!("noise max diff {d:e}");
    assert!(d <= 1e-6, "{d}");
    assert_ne!(noise(1, 16), noise(2, 16));
    let n = noise(7, 1 << 16);
    let mean = n.iter().map(|v| *v as f64).sum::<f64>() / n.len() as f64;
    let var = n.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n.len() as f64;
    assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.02, "mean {mean} var {var}");
}

#[test]
fn the_prompt_is_templated_truncated_and_padded_like_transformers() {
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt-ids/qwen3.json")).unwrap()).unwrap();
    let repo = cases["repo"].as_str().unwrap();
    let Ok(json) = specrt::weights::find_in_snapshot(repo, "tokenizer.json") else {
        eprintln!("{repo} is not in the hub cache, nothing to compare");
        return;
    };
    let encoder = specrt::sampler::PromptEncoder::open(json.parent().unwrap()).unwrap();
    let p = pipeline();
    assert!(p.chat_prompt && p.masked_pad_keys);
    for c in cases["cases"].as_array().unwrap() {
        let prompt = c["prompt"].as_str().unwrap();
        let max = c["max_length"].as_u64().unwrap() as usize;
        assert_eq!(encoder.text(&p, prompt).unwrap(), c["text"].as_str().unwrap(), "{prompt:?}");
        let got = encoder.encode(&p, prompt, max).unwrap();
        let ids: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let mask: Vec<u8> = c["mask"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
        assert_eq!(got.ids, ids, "{prompt:?} at {max}");
        assert_eq!(got.mask, mask, "{prompt:?} at {max}");
    }
}

#[test]
fn the_pipeline_section_binds_the_scheduler_and_vae_configs() {
    let p = pipeline();
    assert_eq!(p.text_layers, vec![1, 2, 3]);
    assert_eq!(p.max_length, 512);
    assert_eq!(p.steps, 4);
    assert!(!p.options.offload_vae);
    assert!(p.options.offload_text_encoder);
    let x = [1.0f32, -2.0];
    let mut y = x;
    p.denormalise(&mut y, &[0.5, 0.25], &[0.1, -0.2]).unwrap();
    let eps = p.values["bn_eps"] as f32;
    assert!((y[0] - (1.0 * (0.5f32 + eps).sqrt() + 0.1)).abs() < 1e-6);
    assert!((y[1] - (-2.0 * (0.25f32 + eps).sqrt() - 0.2)).abs() < 1e-6);
    assert_eq!(p.postprocess(&[-3.0, 0.0, 3.0]).unwrap(), vec![0.0, 0.5, 1.0]);
}
