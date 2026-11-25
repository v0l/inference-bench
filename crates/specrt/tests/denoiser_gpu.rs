use cudarc::driver::CudaContext;
use specrt::denoiser::{DenoiserGraph, Shapes};
use specrt::denoiser_cuda::{Denoiser, Trace};
use specrt::denoiser_host::{patchify, unpatchify};
use specrt::jit::Jit;
use std::path::{Path, PathBuf};

const VELOCITY_TOLERANCE: f32 = 2e-3;
const HIDDEN_TOLERANCE: f32 = 2e-3;
const IDENTITY_TOLERANCE: f32 = 2e-4;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
}

fn json(file: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(file)).unwrap()).unwrap()
}

fn floats_in(index: &serde_json::Value, key: &str) -> Vec<f32> {
    let file = index["files"][key]["file"].as_str().unwrap_or_else(|| panic!("no reference {key}"));
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn pipeline(key: &str) -> Vec<f32> {
    floats_in(&json("tiny-flux2-klein.json"), key)
}

fn reference(key: &str) -> Vec<f32> {
    floats_in(&json("tiny-flux2-klein.denoiser.json"), key)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn sigma(step: usize) -> f32 {
    json("tiny-flux2-klein.json")["sigmas"][step].as_f64().unwrap() as f32
}

fn step_input(step: usize) -> Vec<f32> {
    if step == 0 { pipeline("noise") } else { pipeline(&format!("step{}.latent", step - 1)) }
}

fn denoiser() -> Denoiser {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-denoiser-{}", std::process::id()));
    let jit = Jit::new(ctx, dir).unwrap();
    Denoiser::load(&stream, &jit, &fixture(), Shapes { image: 64, text: 16, patch_h: 8, patch_w: 8 }).unwrap()
}

fn joint_rows(x: &[f32], h: usize) -> Vec<f32> {
    x[..80 * h].to_vec()
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test denoiser_gpu -- --ignored"]
fn the_engine_denoiser_matches_diffusers_on_every_recorded_step() {
    let d = denoiser();
    let mut arena = d.arena().unwrap();
    eprintln!("arena {} bytes", arena.bytes);
    d.prepare(&mut arena, &pipeline("text.embeds")).unwrap();
    let h = d.spec.hidden();
    for step in 0..3 {
        let mut trace = Trace::default();
        let grid = d.forward(&mut arena, &unpatchify(&step_input(step), 4, 2, 8, 8), sigma(step), Some(&mut trace)).unwrap();
        let v = patchify(&grid, 4, 2, 8, 8);
        let to_f32 = max_diff(&v, &pipeline(&format!("step{step}.velocity")));
        let to_bf16 = max_diff(&v, &reference(&format!("bf16.step{step}.velocity")));
        eprintln!("step {step}: velocity max diff {to_f32:e} to the f32 reference, {to_bf16:e} to the bf16 weight reference");
        assert!(to_f32 < VELOCITY_TOLERANCE, "step {step}: velocity max diff {to_f32}");
        for (name, x) in &trace.blocks {
            let d = max_diff(&joint_rows(x, h), &reference(&format!("step{step}.{name}")));
            eprintln!("step {step} {name}: hidden max diff {d:e}");
            assert!(d < HIDDEN_TOLERANCE, "step {step} {name}: hidden max diff {d}");
        }
    }
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test denoiser_gpu -- --ignored"]
fn hoisting_the_text_projection_out_of_the_step_changes_nothing() {
    let d = denoiser();
    let latent = unpatchify(&step_input(1), 4, 2, 8, 8);
    let mut once = d.arena().unwrap();
    d.prepare(&mut once, &pipeline("text.embeds")).unwrap();
    let first = d.forward(&mut once, &latent, sigma(0), None).unwrap();
    let again = d.forward(&mut once, &latent, sigma(1), None).unwrap();
    let mut fresh = d.arena().unwrap();
    d.prepare(&mut fresh, &pipeline("text.embeds")).unwrap();
    let alone = d.forward(&mut fresh, &latent, sigma(1), None).unwrap();
    assert_eq!(again, alone, "a step after another step must equal the same step on a fresh arena");
    assert_ne!(first, again);
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test denoiser_gpu -- --ignored"]
fn the_engine_agrees_with_the_host_plan() {
    let d = denoiser();
    let graph = DenoiserGraph::embedded();
    let ck = specrt::weights::Checkpoint::open(&fixture().join("transformer")).unwrap();
    let host = specrt::denoiser_host::HostDenoiser::load(graph, &d.spec, &d.plan, &ck, true).unwrap();
    let ids = specrt::denoiser_host::grid_ids(16, 8, 8, d.spec.axes.len());
    let latent = unpatchify(&step_input(0), 4, 2, 8, 8);
    let text = pipeline("text.embeds");
    let want = host.forward(&specrt::denoiser_host::Inputs { latent: &latent, text: &text, timestep: sigma(0), ids: &ids, image: 64, text_rows: 16, patch_h: 8, patch_w: 8 }, &mut specrt::denoiser_host::Trace::default()).unwrap();
    let mut arena = d.arena().unwrap();
    d.prepare(&mut arena, &text).unwrap();
    let got = d.forward(&mut arena, &latent, sigma(0), None).unwrap();
    let diff = max_diff(&got, &want);
    eprintln!("engine against the bf16 host plan: {diff:e}");
    assert!(diff < IDENTITY_TOLERANCE, "{diff}");
}
