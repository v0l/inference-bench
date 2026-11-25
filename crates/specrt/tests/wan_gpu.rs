use cudarc::driver::CudaContext;
use specrt::denoiser::Shapes;
use specrt::denoiser_cuda::Denoiser;
use specrt::jit::Jit;
use specrt::sampler::Pipeline;
use std::path::{Path, PathBuf};

const VELOCITY_TOLERANCE: f32 = 1e-4;
const F32_BOUND: f32 = 2e-2;

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

fn shapes() -> Shapes {
    Shapes { image: 48, text: 16, patch_h: 4, patch_w: 4, frames: 3 }
}

fn gpu() -> (std::sync::Arc<CudaContext>, Jit) {
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-wan-{}", std::process::id()))).unwrap();
    (ctx, jit)
}

fn timestep(i: usize) -> f32 {
    meta()["timesteps"][i].as_f64().unwrap() as f32
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test wan_gpu -- --ignored --test-threads=1"]
fn the_engine_wan_denoiser_matches_diffusers_cond_and_uncond_on_every_step() {
    let (ctx, jit) = gpu();
    let p = Pipeline::from_dir(&fixture()).unwrap();
    let d = Denoiser::load_pipeline(&ctx.default_stream(), &jit, &fixture(), shapes(), &p, true).unwrap();
    let mut a = d.arena_for(2).unwrap();
    d.prepare(&mut a, &[&floats("text.embeds"), &floats("negative.embeds")]).unwrap();
    for i in 0..3 {
        let x = if i == 0 { floats("noise") } else { floats(&format!("bf16.step{}.latent", i - 1)) };
        let v = d.forward(&mut a, &x, timestep(i), None).unwrap();
        let half = v.len() / 2;
        for (part, which) in [(&v[..half], "cond"), (&v[half..], "uncond")] {
            let bf = max_diff(part, &floats(&format!("bf16.step{i}.velocity.{which}")));
            eprintln!("step {i} {which}: engine velocity {bf:e} to the bf16 weight run");
            assert!(bf < VELOCITY_TOLERANCE, "step {i} {which}: {bf}");
        }
    }
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test wan_gpu -- --ignored --test-threads=1"]
fn the_unipc_kernel_steps_the_latent_like_the_recorded_run() {
    let (ctx, jit) = gpu();
    let p = Pipeline::from_dir(&fixture()).unwrap();
    let d = Denoiser::load_pipeline(&ctx.default_stream(), &jit, &fixture(), shapes(), &p, true).unwrap();
    let mut a = d.arena_for(2).unwrap();
    d.prepare(&mut a, &[&floats("text.embeds"), &floats("negative.embeds")]).unwrap();
    let sched = p.schedule(3, 48).unwrap();
    let g = meta()["guidance"].as_f64().unwrap() as f32;
    d.set_latent(&mut a, &floats("noise")).unwrap();
    d.reset_state(&mut a).unwrap();
    for i in 0..3 {
        let scalars = p.multistep_scalars(&sched, i).unwrap();
        d.multistep(&mut a, sched.model_timestep(i), &scalars, g).unwrap();
        let x = d.latent(&a).unwrap();
        let bf = max_diff(&x, &floats(&format!("bf16.step{i}.latent")));
        let f = max_diff(&x, &floats(&format!("step{i}.latent")));
        eprintln!("step {i}: latent {bf:e} to the bf16 weight run, {f:e} to f32");
        assert!(bf < VELOCITY_TOLERANCE, "step {i}: {bf}");
        assert!(f < F32_BOUND, "step {i}: {f}");
    }
}
