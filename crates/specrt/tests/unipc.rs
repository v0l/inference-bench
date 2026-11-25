use specrt::sampler::{LatentLayout, Pipeline, noise};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-wan21")
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference/tiny-wan21.json")).unwrap()).unwrap()
}

fn floats(key: &str) -> Vec<f32> {
    let file = meta()["files"][key]["file"].as_str().unwrap_or_else(|| panic!("no reference {key}")).to_string();
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn pipeline() -> Pipeline {
    Pipeline::from_dir(&fixture()).unwrap()
}

#[test]
fn the_wan_pipeline_binds_the_unipc_scheduler_as_data() {
    let p = pipeline();
    assert_eq!(p.name, "unipc");
    assert_eq!(p.latent, LatentLayout::Grid);
    assert_eq!(p.multistep.as_ref().unwrap().state, ["last", "m1", "m2"]);
    assert!(p.guided(5.0).unwrap() && !p.guided(1.0).unwrap());
    assert_eq!((p.values["vae_scale"], p.values["pixel_step"], p.values["temporal_scale"]), (4.0, 8.0, 2.0));
    assert!(p.step_rule(false).is_err(), "a multistep rule is not a one line update");
}

#[test]
fn the_unipc_schedule_matches_the_diffusers_sigmas_and_timesteps() {
    let m = meta();
    let sched = pipeline().schedule(3, 48).unwrap();
    let want: Vec<f32> = m["sigmas"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap() as f32).collect();
    assert!(max_diff(&sched.sigmas, &want) < 1e-6, "{:?} against {want:?}", sched.sigmas);
    let ts: Vec<f32> = m["timesteps"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap() as f32).collect();
    assert_eq!((0..3).map(|i| sched.model_timestep(i)).collect::<Vec<_>>(), ts);
}

#[test]
fn the_engine_noise_is_the_noise_the_wan_reference_ran_from() {
    let m = meta();
    assert_eq!(m["noise"].as_str(), Some("splitmix64 counter, Box-Muller pairs"));
    assert_eq!(noise(m["seed"].as_u64().unwrap(), 4 * 3 * 8 * 8), floats("noise"));
}

fn guided_run(p: &Pipeline) -> Vec<Vec<f32>> {
    let sched = p.schedule(3, 48).unwrap();
    let mut x = floats("noise");
    let mut state = p.new_state(x.len());
    let g = meta()["guidance"].as_f64().unwrap() as f32;
    (0..3)
        .map(|i| {
            let (c, u) = (floats(&format!("step{i}.velocity.cond")), floats(&format!("step{i}.velocity.uncond")));
            p.step_host_state(&mut x, &mut state, &c, Some(&u), &sched, i, g).unwrap();
            x.clone()
        })
        .collect()
}

#[test]
fn unipc_bh2_from_the_pipeline_reproduces_the_recorded_guided_latents() {
    for (i, x) in guided_run(&pipeline()).iter().enumerate() {
        let d = max_diff(x, &floats(&format!("step{i}.latent")));
        eprintln!("step {i}: unipc max diff {d:e}");
        assert!(d < 1e-5, "step {i}: {d}");
    }
}

#[test]
fn dropping_the_corrector_fails_the_recorded_latents() {
    let mut p = pipeline();
    let m = p.multistep.as_mut().unwrap();
    for (n, e) in m.assign.iter_mut() {
        if n == "y" {
            *e = kir::expr::Expr::parse("x").unwrap();
        }
    }
    let run = guided_run(&p);
    assert!(max_diff(&run[0], &floats("step0.latent")) < 1e-5, "step 0 has no corrector");
    assert!(max_diff(&run[1], &floats("step1.latent")) > 1e-3);
}

#[test]
fn the_latent_denormalises_per_channel_from_the_vae_config() {
    let p = pipeline();
    let mut x = floats("step2.latent");
    p.denormalise_channels(&mut x, 3 * 8 * 8).unwrap();
    let d = max_diff(&x, &floats("vae.latent"));
    assert!(d < 1e-6, "{d}");
}
