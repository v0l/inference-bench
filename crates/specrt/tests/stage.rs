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
        let stage = Arc::new(RuntimeStage::load(spec, [pair[0], pair[1]], max_ctx, slots).unwrap());
        let server = StageServer::start("127.0.0.1:0", KEY, stage).unwrap();
        addrs.push(server.addr().to_string());
        servers.push(server);
    }
    let chain = Chain::connect(&addrs, KEY, Duration::from_secs(60)).unwrap();
    Cluster { servers, chain }
}

fn single_reference(spec: &Spec, prompt: &[u32], new_tokens: usize, max_ctx: usize) -> Vec<u32> {
    let mut rt = Runtime::load_layout(spec, Layout { kv_planes: 1, ..Layout::contiguous(max_ctx, 1) }).unwrap();
    let pages = max_ctx.div_ceil(specrt::codegen::PAGE) as u32;
    rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
    rt.generate(prompt, new_tokens, Sampling::GREEDY).unwrap().tokens
}

fn pipeline_matches_single(dir: &str, cuts: &[usize]) {
    let spec = spec_for(dir);
    let p = prompt(300);
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
