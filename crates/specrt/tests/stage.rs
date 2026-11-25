use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::spec::Spec;
use specrt::stage::{RuntimeStage, generate};
use specrt::wire::{Chain, StageServer};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const KEY: &[u8] = b"stage-test-key";

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 1024).collect()
}

struct Cluster {
    servers: Vec<StageServer>,
    chain: Chain,
}

fn cluster(spec: &Spec, cuts: &[usize], max_ctx: usize, slots: usize) -> Cluster {
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(spec.layers.count);
    let mut servers = Vec::new();
    let mut addrs = Vec::new();
    for pair in bounds.windows(2) {
        let stage = Arc::new(RuntimeStage::load(spec, [pair[0], pair[1]], max_ctx, slots, 0).unwrap());
        let server = StageServer::start("127.0.0.1:0", KEY, stage).unwrap();
        addrs.push(server.addr().to_string());
        servers.push(server);
    }
    let chain = Chain::connect(&addrs, KEY, Duration::from_secs(60)).unwrap();
    Cluster { servers, chain }
}

fn single_reference(spec: &Spec, prompt: &[u32], new_tokens: usize, max_ctx: usize) -> Vec<u32> {
    if spec.gated_residual.is_some() {
        return Runtime::load_batched(spec, max_ctx, 1).unwrap().generate_batch(std::slice::from_ref(&prompt.to_vec()), new_tokens).unwrap().tokens.remove(0);
    }
    let mut rt = Runtime::load_layout(spec, Layout { kv_planes: 1, ..Layout::contiguous(max_ctx, 1) }).unwrap();
    let pages = max_ctx.div_ceil(specrt::codegen::PAGE) as u32;
    rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
    rt.generate(prompt, new_tokens, Sampling::GREEDY).unwrap().tokens
}

fn pipeline_matches_single(dir: &str, cuts: &[usize]) {
    let spec = spec_for(dir);
    let p: Vec<u32> = prompt(300).into_iter().map(|t| t % spec.dims.vocab as u32).collect();
    let want = single_reference(&spec, &p, 64, 512);
    let mut c = cluster(&spec, cuts, 512, 2);
    let got = generate(&mut c.chain, 0, &p, 64, 128).unwrap();
    assert_eq!(got.tokens, want, "{dir} cut at {cuts:?}");
    assert_eq!(got.decode_steps.len(), 63);
    assert_eq!(got.decode_steps[0].stage.len(), cuts.len() + 1);
    let again = generate(&mut c.chain, 1, &p, 64, 100).unwrap();
    assert_eq!(again.tokens, want, "second slot, other chunking");
    drop(c.servers);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn qwen4_exp_two_stage_pipeline_matches_the_single_runtime() {
    pipeline_matches_single("tiny-qwen4-exp", &[2]);
    pipeline_matches_single("tiny-qwen4-exp", &[3]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_stage_hybrid_pipeline_over_tcp_matches_the_single_runtime() {
    pipeline_matches_single("tiny-qwen3.5", &[1]);
    pipeline_matches_single("tiny-qwen3.5", &[2]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn three_stage_hybrid_pipeline_over_tcp_matches_the_single_runtime() {
    pipeline_matches_single("tiny-qwen3.5", &[1, 2]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fp8_hybrid_pipeline_over_tcp_matches_the_single_runtime() {
    pipeline_matches_single("tiny-qwen3.5-fp8", &[2]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_pipeline_over_tcp_matches_the_single_runtime() {
    pipeline_matches_single("tiny-qwen3.5-hd256", &[1]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_dead_stage_fails_the_request_and_the_rest_keep_serving() {
    let spec = spec_for("tiny-qwen3.5");
    let p = prompt(40);
    let mut c = cluster(&spec, &[1], 512, 1);
    generate(&mut c.chain, 0, &p, 4, 64).unwrap();
    let last = c.servers.pop().unwrap();
    last.shutdown();
    std::thread::sleep(Duration::from_millis(600));
    let err = generate(&mut c.chain, 0, &p, 4, 64).err().unwrap();
    assert!(!format!("{err:#}").is_empty());
}

#[test]
fn a_stage_after_the_first_cannot_start_before_a_prelude_layer() {
    let spec = spec_for("tiny-qwen4-exp");
    let ple = spec.prelude.as_ref().unwrap().layers[0];
    assert!(ple >= 1);
    let cut_before = specrt::runtime::StageRange { device: 0, start: ple, end: spec.layers.count };
    let err = cut_before.check(&spec).unwrap_err().to_string();
    assert!(err.contains("prelude"), "{err}");
    specrt::runtime::StageRange { device: 0, start: ple + 1, end: spec.layers.count }.check(&spec).unwrap();
    specrt::runtime::StageRange { device: 0, start: 0, end: ple + 1 }.check(&spec).unwrap();
}

#[test]
fn the_planner_never_cuts_in_front_of_a_prelude_layer() {
    let spec = spec_for("tiny-qwen4-exp");
    let ple = spec.prelude.as_ref().unwrap().layers[0];
    let model = specrt::placement::CostModel::new(&spec, 512, 1);
    for start in 1..=ple {
        assert!(model.stage_fit_error(start, spec.layers.count).is_some_and(|e| e.contains("prelude")), "start {start}");
    }
    assert!(model.stage_fit_error(ple + 1, spec.layers.count).is_none());
}

#[test]
fn moe_stage_specs_keep_the_output_gate_and_drop_unused_attention_nodes() {
    let spec = spec_for("tiny-qwen4-exp");
    assert!(spec.attention.output_gate);
    assert!(spec.moe.is_some());
    let linear_only = spec.stage(0, 1);
    assert!(!linear_only.block(0).is_attention());
    assert!(linear_only.attention.node.is_empty());
    specrt::runtime::StageRange { device: 0, start: spec.layers.count - 1, end: spec.layers.count }.check(&spec).unwrap();
    let tail = spec.stage(spec.layers.count - 1, spec.layers.count);
    assert_eq!(tail.attention.node.is_empty(), !tail.block(0).is_attention());
}

#[test]
fn stage_specs_renumber_dense_moe_layers_and_nope_layers() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5.3-flash.config.json")).unwrap();
    let spec = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!(spec.moe.as_ref().unwrap().dense_layers, vec![0, 1, 2]);
    let head = spec.stage(0, 23);
    let tail = spec.stage(23, 45);
    assert_eq!(head.moe.as_ref().unwrap().dense_layers, vec![0, 1, 2]);
    assert!(tail.moe.as_ref().unwrap().dense_layers.is_empty());
    assert!((0..22).all(|l| tail.moe.as_ref().unwrap().is_sparse(l)));
    assert_eq!(tail.rope.nope_layers, (0..22).collect::<Vec<_>>());
    let middle = spec.stage(2, 5);
    assert_eq!(middle.moe.as_ref().unwrap().dense_layers, vec![0]);
    for (start, end) in [(0, 23), (23, 45), (2, 5)] {
        specrt::runtime::StageRange { device: 0, start, end }.check(&spec).unwrap();
    }
}

#[test]
fn stage_specs_cut_the_pattern_and_the_quant_boundary() {
    let spec = Spec::load(&fixture_dir("qwen3.8-27b-nvfp4.toml")).unwrap();
    let tail = spec.stage(40, 64);
    assert_eq!(tail.layers.count, 24);
    assert_eq!(tail.mlp.quant_until, Some(16));
    assert_eq!(tail.block(3), spec.block(43));
    let late = spec.stage(58, 64);
    assert_eq!(late.mlp_quant(0), spec.mlp_quant(58));
    let head = spec.stage(0, 40);
    assert!(head.mtp.is_none());
    assert_eq!(head.mlp_quant(39), spec.mlp_quant(39));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_first_stage_reports_its_gpu_time_for_prefill_not_just_the_launch() {
    let spec = spec_for("tiny-qwen3.5");
    let p = prompt(500);
    let mut c = cluster(&spec, &[1], 1024, 1);
    generate(&mut c.chain, 0, &p, 2, 500).unwrap();
    let got = generate(&mut c.chain, 0, &p, 2, 500).unwrap();
    let step = &got.prefill_steps[0];
    let (compute, round) = (step.compute[0].as_secs_f64(), step.stage[0].as_secs_f64());
    assert!(compute >= 0.5 * round, "stage 0 reported {compute} s of compute in a {round} s round trip");
}
