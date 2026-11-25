use cudarc::driver::CudaContext;
use serde_json::Value;
use specrt::codec::HostCodec;
use specrt::codec::cuda::{CudaCodec, module};
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

fn check(what: &str, got: &[f32], want: &[f32], tol: f32) {
    let e = max_rel(got, want);
    eprintln!("{what}: max rel {e:e}");
    assert!(e < tol, "{what} {e}");
}

fn interleave(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

#[test]
#[ignore = "needs nvrtc, no GPU: compiles every codec kernel of the real config for sm_120"]
fn every_codec_kernel_of_the_real_config_compiles_with_nvrtc() {
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("haru-4b.config.json")).unwrap()).unwrap();
    let plan = specrt::codec::graph::plan(&config).unwrap();
    let target = kir::Target::sm_120();
    let mut all = Vec::new();
    for (steps, batch) in [(1, 1), (1, 8), (16, 1)] {
        all.extend(specrt::codec::compile::plan(&plan.encoder, steps * plan.frame_samples, batch).unwrap().kernels(&target).unwrap());
        all.extend(specrt::codec::compile::plan(&plan.decoder, steps, batch).unwrap().kernels(&target).unwrap());
    }
    all.sort_by_key(|k| k.name());
    all.dedup_by_key(|k| k.name());
    let m = module(&all).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(m.source.as_str(), opts).unwrap().to_src();
    for name in &m.kernels {
        assert!(ptx.contains(&format!(".entry {name}(")), "{name} missing from the PTX");
    }
    eprintln!("{} kernels, {} conv variants", m.kernels.len(), all.len());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_codec_streams_the_tiny_fixture_like_the_reference() {
    let host = HostCodec::open(&dir().join("tiny")).unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let cache = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-codec-jit"));
    let jit = Jit::new(ctx.clone(), cache).unwrap();
    let (frame, d) = (host.plan.frame_samples, host.plan.latent_dim);
    let clip = reference("codec.clip");
    let gpu = CudaCodec::load(&ctx, &jit, &host, &[(1, 1)]).unwrap();

    let mut s = gpu.sessions(&gpu.encoder, 1).unwrap();
    assert_eq!(s.floats_per_session(), host.plan.encoder.state_floats());
    let clip_mean = reference("codec.clip_mean");
    let whole = gpu.run(&gpu.encoder, &clip, &mut s).unwrap();
    check("codec.clip_mean", &whole, &clip_mean, 1e-4);

    let audio = reference("session.user_audio");
    let user_mean = reference("session.user_mean");
    let mut s = gpu.sessions(&gpu.encoder, 1).unwrap();
    let streamed: Vec<f32> = audio.chunks(frame).flat_map(|f| gpu.run(&gpu.encoder, f, &mut s).unwrap()).collect();
    check("session.user_mean", &streamed, &user_mean, 1e-4);
    let whole = gpu.run(&gpu.encoder, &audio, &mut gpu.sessions(&gpu.encoder, 1).unwrap()).unwrap();
    check("encode frame by frame against whole clip", &streamed, &whole, 1e-5);

    let clip_audio = reference("codec.clip_audio");
    let mut s = gpu.sessions(&gpu.decoder, 1).unwrap();
    assert_eq!(s.floats_per_session(), host.plan.decoder.state_floats());
    check("codec.clip_audio", &gpu.run(&gpu.decoder, &clip_mean, &mut s).unwrap(), &clip_audio, 1e-4);

    let latents = reference("session.latent");
    let session_audio = reference("session.audio");
    let mut s = gpu.sessions(&gpu.decoder, 1).unwrap();
    let streamed: Vec<f32> = latents.chunks(d).flat_map(|z| gpu.run(&gpu.decoder, z, &mut s).unwrap()).collect();
    check("session.audio", &streamed, &session_audio, 1e-4);
    let whole = gpu.run(&gpu.decoder, &latents, &mut gpu.sessions(&gpu.decoder, 1).unwrap()).unwrap();
    check("decode frame by frame against whole clip", &streamed, &whole, 1e-5);

    let silence = gpu.run(&gpu.encoder, &vec![0.0; 16 * frame], &mut gpu.sessions(&gpu.encoder, 1).unwrap()).unwrap();
    check("codec.silence", &silence[15 * d..], &reference("codec.silence"), 1e-4);

    let mut pair = gpu.sessions(&gpu.encoder, 2).unwrap();
    let reversed: Vec<Vec<f32>> = audio.chunks(frame).rev().map(<[f32]>::to_vec).collect();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, f) in audio.chunks(frame).enumerate() {
        let both = gpu.run(&gpu.encoder, &interleave(&[f.to_vec(), reversed[i].clone()]), &mut pair).unwrap();
        a.extend_from_slice(&both[..d]);
        b.extend_from_slice(&both[d..]);
    }
    check("batched session 0 against user_mean", &a, &user_mean, 1e-4);
    let mut solo = gpu.sessions(&gpu.encoder, 1).unwrap();
    let alone: Vec<f32> = reversed.iter().flat_map(|f| gpu.run(&gpu.encoder, f, &mut solo).unwrap()).collect();
    check("batched session 1 against the same audio alone", &b, &alone, 1e-6);

    let mut pair = gpu.sessions(&gpu.decoder, 2).unwrap();
    let mut out0 = Vec::new();
    for z in latents.chunks(d) {
        let y = gpu.run(&gpu.decoder, &interleave(&[z.to_vec(), z.to_vec()]), &mut pair).unwrap();
        assert_eq!(y[..frame], y[frame..], "both sessions decode the same latent");
        out0.extend_from_slice(&y[..frame]);
    }
    check("batched decode against session.audio", &out0, &session_audio, 1e-4);
}
