use cudarc::driver::CudaContext;
use specrt::denoiser::{DenoiserGraph, Shapes};
use specrt::denoiser_cuda::{Denoiser, Trace};
use specrt::jit::Jit;
use std::path::{Path, PathBuf};

const VELOCITY_TOLERANCE: f32 = 1e-4;
const HIDDEN_TOLERANCE: f32 = 1e-4;
const BF16_WEIGHT_BOUND: f32 = 5e-3;
const IDENTITY_TOLERANCE: f32 = 1e-4;

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
    d.prepare(&mut arena, &[&pipeline("text.embeds")]).unwrap();
    let h = d.spec.hidden();
    for step in 0..3 {
        let mut trace = Trace::default();
        let v = d.forward(&mut arena, &step_input(step), sigma(step), Some(&mut trace)).unwrap();
        let to_f32 = max_diff(&v, &pipeline(&format!("step{step}.velocity")));
        let to_bf16 = max_diff(&v, &reference(&format!("bf16.step{step}.velocity")));
        eprintln!("step {step}: velocity max diff {to_f32:e} to the f32 reference, {to_bf16:e} to the bf16 weight reference");
        assert!(to_bf16 < VELOCITY_TOLERANCE, "step {step}: velocity max diff {to_bf16} to the bf16 weight reference");
        assert!(to_f32 < BF16_WEIGHT_BOUND, "step {step}: velocity max diff {to_f32} to the f32 reference");
        let moved = max_diff(&joint_rows(&trace.blocks[0].1, h), &joint_rows(&trace.blocks[1].1, h));
        let moved_ref = max_diff(&reference(&format!("bf16.step{step}.double0")), &reference(&format!("bf16.step{step}.single0")));
        eprintln!("step {step}: the single block moves x by {moved:e} (reference {moved_ref:e})");
        assert!((moved - moved_ref).abs() < HIDDEN_TOLERANCE, "step {step}: the engine single block moves x by {moved}, the reference by {moved_ref}");
        for (name, x) in &trace.blocks {
            let want = reference(&format!("bf16.step{step}.{name}"));
            let got = joint_rows(x, h);
            let d = max_diff(&got, &want);
            let at = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).enumerate().fold((0, 0f32), |m, (i, v)| if v > m.1 { (i, v) } else { m }).0;
            eprintln!("step {step} {name}: hidden max diff {d:e} at row {} col {}", at / h, at % h);
            assert!(d < HIDDEN_TOLERANCE, "step {step} {name}: hidden max diff {d}");
        }
    }
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test denoiser_gpu -- --ignored"]
fn hoisting_the_text_projection_out_of_the_step_changes_nothing() {
    let d = denoiser();
    let latent = step_input(1);
    let mut once = d.arena().unwrap();
    d.prepare(&mut once, &[&pipeline("text.embeds")]).unwrap();
    let first = d.forward(&mut once, &latent, sigma(0), None).unwrap();
    let again = d.forward(&mut once, &latent, sigma(1), None).unwrap();
    let mut fresh = d.arena().unwrap();
    d.prepare(&mut fresh, &[&pipeline("text.embeds")]).unwrap();
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
    let network = graph.plan().unwrap();
    let host = specrt::denoiser_host::HostDenoiser::load(graph, &d.spec, &network, &ck, true).unwrap();
    let ids = specrt::denoiser_host::grid_ids(16, 8, 8, d.spec.axes.len());
    let latent = step_input(0);
    let text = pipeline("text.embeds");
    let want = host.forward(&specrt::denoiser_host::Inputs { latent: &latent, text: &text, timestep: sigma(0), ids: &ids, image: 64, text_rows: 16, patch_h: 8, patch_w: 8 }, &mut specrt::denoiser_host::Trace::default()).unwrap();
    let mut arena = d.arena().unwrap();
    d.prepare(&mut arena, &[&text]).unwrap();
    let got = d.forward(&mut arena, &latent, sigma(0), None).unwrap();
    let diff = max_diff(&got, &want);
    eprintln!("engine against the bf16 host plan: {diff:e}");
    assert!(diff < IDENTITY_TOLERANCE, "{diff}");
}
