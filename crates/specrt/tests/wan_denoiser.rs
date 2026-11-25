use specrt::denoiser::{self, DenoiserGraph, DenoiserSpec};
use specrt::denoiser_host::{HostDenoiser, Inputs, Trace, video_ids};
use specrt::plan::{DitOp, Op};
use specrt::weights::Checkpoint;
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

fn step_input(i: usize) -> Vec<f32> {
    step_input_of("", i)
}

fn step_input_of(prefix: &str, i: usize) -> Vec<f32> {
    if i == 0 { floats("noise") } else { floats(&format!("{prefix}step{}.latent", i - 1)) }
}

fn timestep(i: usize) -> f32 {
    meta()["timesteps"][i].as_f64().unwrap() as f32
}

fn host_velocity(bf16: bool, i: usize, text: &str, plan: &specrt::plan::Plan) -> Vec<f32> {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    let spec = DenoiserSpec::from_pipeline(&fixture()).unwrap();
    let ck = Checkpoint::open(&fixture().join("transformer")).unwrap();
    let host = HostDenoiser::load(graph, &spec, plan, &ck, bf16).unwrap();
    let ids = video_ids(3, 4, 4);
    let latent = if bf16 { step_input_of("bf16.", i) } else { step_input(i) };
    let text = floats(text);
    host.forward(&Inputs { latent: &latent, text: &text, timestep: timestep(i), ids: &ids, image: 48, text_rows: 16, patch_h: 4, patch_w: 4, frames: 3 }, &mut Trace::default()).unwrap()
}

#[test]
fn the_wan_transformer_picks_the_cross_attention_graph_and_binds_its_config() {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    assert_eq!(graph.name, "cross-dit");
    let spec = DenoiserSpec::from_pipeline(&fixture()).unwrap();
    assert_eq!((spec.hidden(), spec.size("in_channels").unwrap(), spec.size("ffn").unwrap()), (48, 16, 96));
    assert_eq!(spec.axes, vec![8, 8, 8], "the 3 axis rope splits head_dim 24 as t, h, w");
    assert_eq!(DenoiserGraph::for_class("Flux2Transformer2DModel").unwrap().name, "denoiser");
    assert!(DenoiserGraph::for_class("SomeOtherTransformer").is_err());
}

#[test]
fn the_wan_plan_validates_from_its_signatures() {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    let spec = DenoiserSpec::from_pipeline(&fixture()).unwrap();
    let plan = graph.plan().unwrap();
    denoiser::validate(graph, &spec, &plan).unwrap();
    let mut broken = plan.clone();
    broken.blocks[0].ops.retain(|o| !matches!(o, Op::Dit(DitOp::Pack { into, .. }) if *into == "cross_kv"));
    let e = denoiser::validate(graph, &spec, &broken).unwrap_err().to_string();
    assert!(e.contains("cross_kv"), "{e}");
}

#[test]
fn the_host_wan_plan_matches_diffusers_velocities_in_f32() {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    let plan = graph.plan().unwrap();
    for i in 0..3 {
        for (text, which) in [("text.embeds", "cond"), ("negative.embeds", "uncond")] {
            let v = host_velocity(false, i, text, &plan);
            let d = max_diff(&v, &floats(&format!("step{i}.velocity.{which}")));
            eprintln!("step {i} {which}: host f32 velocity {d:e}");
            assert!(d < 1e-4, "step {i} {which}: {d}");
        }
    }
}

#[test]
fn the_host_wan_plan_with_bf16_weights_matches_the_bf16_weight_reference() {
    let plan = DenoiserGraph::for_pipeline(&fixture()).unwrap().plan().unwrap();
    for i in 0..3 {
        let v = host_velocity(true, i, "text.embeds", &plan);
        let d = max_diff(&v, &floats(&format!("bf16.step{i}.velocity.cond")));
        eprintln!("step {i}: host bf16 weight velocity {d:e}");
        assert!(d < 1e-4, "step {i}: {d}");
    }
}

#[test]
fn a_rope_on_the_wrong_axes_fails_the_reference() {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    let mut plan = graph.plan().unwrap();
    for op in plan.blocks[0].ops.iter_mut() {
        if let Op::Dit(DitOp::Pack { rope, .. }) = op {
            *rope = false;
        }
    }
    let v = host_velocity(false, 0, "text.embeds", &plan);
    assert!(max_diff(&v, &floats("step0.velocity.cond")) > 1e-3);
}

#[test]
fn every_wan_kernel_builds_from_a_description_and_compiles_with_nvrtc() {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    let spec = DenoiserSpec::from_pipeline(&fixture()).unwrap();
    let p = specrt::sampler::Pipeline::from_dir(&fixture()).unwrap();
    let shapes = denoiser::Shapes { image: 48, text: 16, patch_h: 4, patch_w: 4, frames: 3 };
    let t = kir::Target::sm_120();
    for guided in [false, true] {
        let plan = graph.multistep_plan(p.multistep.as_ref().unwrap(), &p.combine, guided).unwrap();
        denoiser::validate(graph, &spec, &plan).unwrap();
        let kernels = denoiser::described_kernels(graph, &spec, &plan, &shapes, &t).unwrap();
        let names: Vec<&str> = kernels.iter().map(|k| k.name).collect();
        let step = if guided { "dit_multistep_cfg" } else { "dit_multistep" };
        for want in ["dit_layernorm", "dit_modulate", "dit_unpatchify_last", "dit_table_add_288_288", "dit_table_add_48_96", step] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        let src = kir::cuda::emit_module(&kernels).unwrap();
        specrt::jit::cubin_for(&src, "sm_120", 1).unwrap_or_else(|e| panic!("compose: {e:#}\n{src}"));
    }
    let vk = kir::vision::VisionKernel::new(&t, spec.size("head_dim").unwrap(), true).unwrap().joint();
    let vision = kir::cuda::emit_vision(&vk).unwrap();
    assert!(vision.contains("vis_qk_pack(") && vision.contains("vis_flash_kv("));
    specrt::jit::cubin_for(&vision, "sm_120", 1).unwrap();
}

#[test]
fn the_wan_text_keys_and_values_are_step_invariant() {
    let graph = DenoiserGraph::for_pipeline(&fixture()).unwrap();
    let spec = DenoiserSpec::from_pipeline(&fixture()).unwrap();
    let plan = graph.plan().unwrap();
    let inv = denoiser::hoisted(graph, &spec, &plan).unwrap();
    let names: Vec<String> = plan.head.iter().zip(&inv).filter(|(_, h)| **h).map(|(o, _)| o.label()).collect();
    assert_eq!(names, ["split text", "condition_embedder.text_embedder.linear_1 (text)", "condition_embedder.text_embedder.linear_2 (text)"].map(String::from), "{names:?}");
    let blocks = denoiser::block_invariant(graph, &spec, &plan).unwrap();
    let labels: Vec<String> = plan.blocks[0].ops.iter().zip(&blocks[0]).filter(|(_, h)| **h).map(|(o, _)| o.label()).collect();
    assert_eq!(labels, ["attn2.to_k+attn2.to_v (text)".to_string(), "text norm and pack into cross_kv".to_string()], "{labels:?}");
}
