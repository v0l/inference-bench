use specrt::denoiser::{self, DenoiserGraph, DenoiserSpec};
use specrt::denoiser_host::{HostDenoiser, Inputs, Trace, grid_ids, timestep_embedding};
use specrt::plan::{DitOp, Op};
use specrt::weights::Checkpoint;
use std::path::{Path, PathBuf};

const VELOCITY_TOLERANCE: f32 = 1e-4;
const HIDDEN_TOLERANCE: f32 = 1e-4;

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

fn spec() -> DenoiserSpec {
    DenoiserSpec::from_pipeline(&fixture()).unwrap()
}

fn step_input(step: usize) -> Vec<f32> {
    if step == 0 { pipeline("noise") } else { pipeline(&format!("step{}.latent", step - 1)) }
}

fn sigma(step: usize) -> f32 {
    json("tiny-flux2-klein.json")["sigmas"][step].as_f64().unwrap() as f32
}

fn ids(spec: &DenoiserSpec) -> Vec<i32> {
    let meta = json("tiny-flux2-klein.json");
    let side = (meta["image_tokens"].as_u64().unwrap() as f64).sqrt() as usize;
    grid_ids(meta["text_tokens"].as_u64().unwrap() as usize, side, side, spec.axes.len())
}

fn run_host(bf16: bool, step: usize, plan: &specrt::plan::Plan) -> (Vec<f32>, Trace) {
    let graph = DenoiserGraph::embedded();
    let spec = spec();
    let ck = Checkpoint::open(&fixture().join("transformer")).unwrap();
    let host = HostDenoiser::load(graph, &spec, plan, &ck, bf16).unwrap();
    let latent = step_input(step);
    let text = pipeline("text.embeds");
    let ids = ids(&spec);
    let mut trace = Trace::default();
    let v = host.forward(&Inputs { latent: &latent, text: &text, timestep: sigma(step), ids: &ids, image: 64, text_rows: 16 }, &mut trace).unwrap();
    (v, trace)
}

#[test]
fn the_tiny_flux2_config_binds_to_the_generic_denoiser_graph() {
    let s = spec();
    assert_eq!(s.hidden(), 32);
    assert_eq!(s.size("fused").unwrap(), 288);
    assert_eq!(s.size("cat").unwrap(), 128);
    assert_eq!(s.size("mod2").unwrap(), 192);
    assert_eq!(s.size("double").unwrap(), 1);
    assert_eq!(s.size("single").unwrap(), 1);
    assert_eq!(s.axes, vec![4, 4, 4, 4]);
    assert_eq!(s.get("time_scale").unwrap(), 1000.0);
}

#[test]
fn the_denoiser_plan_validates_from_its_signatures() {
    let graph = DenoiserGraph::embedded();
    let plan = graph.plan().unwrap();
    assert_eq!(plan.phase, specrt::plan::Phase::Forward);
    assert_eq!(plan.blocks.iter().map(|b| b.layers[0].as_str()).collect::<Vec<_>>(), ["double", "single"]);
    denoiser::validate(graph, &spec(), &plan).unwrap();
}

#[test]
fn a_plan_that_reads_the_attention_output_before_attention_is_refused() {
    let graph = DenoiserGraph::embedded();
    let mut plan = graph.plan().unwrap();
    let ops = &mut plan.blocks[0].ops;
    let at = ops.iter().position(|o| matches!(o, Op::Dit(DitOp::JointAttention { .. }))).unwrap();
    let attention = ops.remove(at);
    ops.push(attention);
    let e = denoiser::validate(graph, &spec(), &plan).unwrap_err().to_string();
    assert!(e.contains("before any node writes it") || e.contains("shape mismatch"), "{e}");
}

#[test]
fn the_timestep_embedding_matches_numpy() {
    let s = spec();
    for step in 0..3 {
        let t = sigma(step) * s.get("time_scale").unwrap() as f32;
        let got = timestep_embedding(t, s.size("freq").unwrap());
        let want = reference(&format!("step{step}.timestep"));
        let d = max_diff(&got, &want);
        assert!(d < 2e-4, "step {step}: {d}");
    }
}

#[test]
fn the_host_plan_matches_diffusers_in_f32() {
    let plan = DenoiserGraph::embedded().plan().unwrap();
    for step in 0..3 {
        let (v, trace) = run_host(false, step, &plan);
        let d = max_diff(&v, &pipeline(&format!("step{step}.velocity")));
        eprintln!("step {step}: velocity max diff {d:e}");
        assert!(d < VELOCITY_TOLERANCE, "step {step}: velocity max diff {d}");
        for (name, x) in &trace.blocks {
            let want = reference(&format!("step{step}.{name}"));
            let d = max_diff(x, &want);
            assert!(d < HIDDEN_TOLERANCE, "step {step} {name}: hidden max diff {d}");
        }
    }
}

#[test]
fn a_wrong_modulation_order_fails_the_reference() {
    let mut plan = DenoiserGraph::embedded().plan().unwrap();
    for op in plan.blocks.iter_mut().flat_map(|b| b.ops.iter_mut()).chain(plan.tail.iter_mut()) {
        if let Op::Dit(DitOp::Modulate { shift, scale, .. }) = op {
            std::mem::swap(shift, scale);
        }
    }
    let (v, trace) = run_host(false, 0, &plan);
    let d = max_diff(&v, &pipeline("step0.velocity"));
    let hidden = max_diff(&trace.blocks[0].1, &reference("step0.double0"));
    eprintln!("swapped shift and scale: velocity {d:e}, double0 hidden {hidden:e}");
    assert!(d > VELOCITY_TOLERANCE, "swapping shift and scale must fail the velocity check, max diff {d}");
    assert!(hidden > HIDDEN_TOLERANCE, "swapping shift and scale must fail the hidden check, max diff {hidden}");
}

#[test]
fn the_step_invariant_ops_are_the_text_projection_only() {
    let graph = DenoiserGraph::embedded();
    let s = spec();
    let plan = graph.plan().unwrap();
    let inv = denoiser::invariant_ops(graph, &s, &plan.head, &["latent", "timestep"]).unwrap();
    let hoisted: Vec<String> = plan.head.iter().zip(&inv).filter(|(_, i)| **i).map(|(o, _)| o.label()).collect();
    assert_eq!(hoisted, ["split text", "context_embedder (text)", "copy context"], "{hoisted:?}");
}

#[test]
fn the_plan_report_counts_bytes_flops_and_launches() {
    let graph = DenoiserGraph::embedded();
    let s = spec();
    let plan = graph.plan().unwrap();
    let r = denoiser::report(graph, &s, &plan, denoiser::count(64, 16)).unwrap();
    let (_, _, flops, launches) = r.total();
    let ops: usize = plan.head.len() + plan.blocks.iter().map(|b| b.ops.len()).sum::<usize>() + plan.tail.len();
    assert_eq!(launches, ops);
    let attention = 2 * 4 * 80 * 80 * 32;
    assert!(flops > attention as u64, "{flops}");
    let text = r.render();
    assert!(text.contains("step invariant"), "{text}");
}
