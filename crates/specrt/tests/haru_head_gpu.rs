use cudarc::driver::CudaContext;
use serde_json::Value;
use specrt::flow_head::FlowHead;
use specrt::flow_head_cuda::{CudaHead, kernels};
use specrt::jit::Jit;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/haru")
}

fn reference(key: &str) -> Vec<f32> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("tiny/reference.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir().join("tiny/reference").join(meta["files"][key]["file"].as_str().unwrap())).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

#[test]
#[ignore = "needs nvrtc, no GPU: compiles every head kernel of the real config for sm_120"]
fn every_head_kernel_of_the_real_config_compiles_with_nvrtc() {
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("haru-4b.config.json")).unwrap()).unwrap();
    let t = kir::Target::sm_120();
    for steps in [1, 2] {
        let head = FlowHead::from_config(&config, Some(steps)).unwrap();
        let ks = kernels(&head, &t, &head.gated_epilogues(1, &t).unwrap()).unwrap();
        let src = ks.source().unwrap();
        let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src.as_str(), opts).unwrap_or_else(|e| panic!("{e:?}\n{}", src.as_str())).to_src();
        for n in ks.names() {
            assert!(ptx.contains(&format!(".entry {n}(")), "{n} missing from the PTX");
        }
        assert!(ks.names().iter().any(|n| n.ends_with("_gated")), "the down projection carries the gated residual");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_head_matches_the_session_latents_and_the_head_references() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-head-jit"));
    let jit = Jit::new(ctx.clone(), cache).unwrap();
    let tiny = dir().join("tiny");
    let hidden = reference("session.hidden");
    let noise = reference("session.head_noise");
    for fuse in [true, false] {
        for (key, guidance, steps) in [("session.latent", 1.5f32, None), ("head.guidance1.0", 1.0, None), ("head.guidance2.0", 2.0, None), ("head.steps2", 1.5, Some(2))] {
            let head = CudaHead::load(&stream, &jit, FlowHead::from_dir(&tiny, steps).unwrap(), &tiny, fuse).unwrap();
            let (h, d) = (head.head.spec.hidden(), head.head.spec.size("latent_dim").unwrap());
            let frames = hidden.len() / h;
            let mut batch = head.session(&vec![guidance; frames]).unwrap();
            let batched = head.frame(&mut batch, &hidden, &noise).unwrap();
            let launches = batch.launches;
            let mut one = head.session(&[guidance]).unwrap();
            let alone: Vec<f32> = (0..frames).flat_map(|f| head.frame(&mut one, &hidden[f * h..(f + 1) * h], &noise[f * d..(f + 1) * d]).unwrap()).collect();
            let want = reference(key);
            let (e, b) = (max_rel(&batched, &want), max_rel(&alone, &batched));
            eprintln!("{key} fuse {fuse}: max rel {e:e}, one session at a time against {frames} batched {b:e}, {launches} launches per frame");
            assert!(e < 1e-4, "{key}: {e}");
            assert!(b < 1e-5, "{key}: {b}");
        }
    }
    let head = CudaHead::load(&stream, &jit, FlowHead::from_dir(&tiny, None).unwrap(), &tiny, true).unwrap();
    let mut s = head.session(&[1.0]).unwrap();
    head.set_guidance(&mut s, &[2.0]).unwrap();
    let got = head.frame(&mut s, &hidden[..64], &noise[..64]).unwrap();
    assert!(max_rel(&got, &reference("head.guidance2.0")[..64]) < 1e-4, "a guidance change reruns the hoisted embeddings");
}
