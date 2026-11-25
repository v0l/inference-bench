#[path = "common/conv_cases.rs"]
mod conv_cases;

use conv_cases::{case, cases, expected, max_abs};
use cudarc::driver::CudaContext;
use specrt::jit::Jit;
use specrt::vae::Grid;
use specrt::vae::compile::{Dims, compile};
use specrt::vae::cuda::{CudaVae, conv_once, kernels, module};
use specrt::vae::graph::load_plan;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn reference(name: &str, key: &str) -> (Vec<f32>, Vec<usize>) {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture(name).join("reference").join(format!("{name}.json"))).unwrap()).unwrap();
    let entry = &meta["files"][key];
    let bytes = std::fs::read(fixture(name).join("reference").join(entry["file"].as_str().unwrap())).unwrap();
    let shape = entry["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
    (bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(), shape)
}

fn latent(name: &str) -> Grid {
    let (v, s) = reference(name, "vae.latent");
    match s.as_slice() {
        [1, c, h, w] => Grid::from_channels_first(&v, *c, 1, *h, *w),
        [1, c, t, h, w] => Grid::from_channels_first(&v, *c, *t, *h, *w),
        other => panic!("latent shape {other:?}"),
    }
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (4.0 / mse.max(1e-30)).log10()
}

fn gpu() -> (Arc<CudaContext>, Jit) {
    let ctx = CudaContext::new(0).unwrap();
    let dir = std::env::var_os("SPECRT_CACHE").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("specrt-vae-jit"));
    let jit = Jit::new(ctx.clone(), dir).unwrap();
    (ctx, jit)
}

#[test]
#[ignore = "needs nvrtc, no GPU: compiles every VAE kernel for sm_120"]
fn every_vae_kernel_compiles_with_nvrtc() {
    let target = kir::Target::sm_120();
    let mut all = Vec::new();
    for name in ["tiny-flux2-klein", "tiny-wan21"] {
        let p = load_plan(&fixture(&format!("{name}/vae"))).unwrap();
        let l = latent(name);
        for fuse in [false, true] {
            for first in [true, false] {
                let c = compile(&p, Dims { t: 1, h: l.h, w: l.w }, &target, fuse, first).unwrap();
                all.extend(kernels(&c, &target).unwrap().into_values());
            }
        }
    }
    for c in cases() {
        let k = case(&c);
        let shape = kir::conv::ConvShape { cin: k.conv.cin, cout: k.conv.cout, k: k.conv.k, kt: k.conv.kt, upsample: k.upsample, interleave: k.interleave };
        all.push(kir::conv::ConvKernel::new(&target, shape, kir::conv::Prologue::None, kir::conv::Epilogue { residual: false, stats: None }).unwrap());
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
fn every_vae_conv_shape_matches_numpy_on_the_gpu() {
    let (ctx, jit) = gpu();
    for c in cases() {
        let k = case(&c);
        let got = conv_once(&ctx, &jit, &k.input, &k.conv, k.upsample, k.interleave).unwrap().to_channels_first();
        let want = expected(&c);
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = max_abs(&got, &want);
        let fan = (k.conv.cin * k.conv.k * k.conv.k * k.conv.kt) as f32;
        eprintln!("{:<16} max abs {err:e} peak {peak:.3}", c["name"].as_str().unwrap());
        assert!(err <= 1e-6 * fan.sqrt() * peak.max(1.0), "{}: max abs {err}", c["name"]);
    }
}

fn decode(name: &str, fuse: bool) -> (Vec<f32>, f32, f64) {
    let (ctx, jit) = gpu();
    let l = latent(name);
    let shapes = [(Dims { t: 1, h: l.h, w: l.w }, true), (Dims { t: 1, h: l.h, w: l.w }, false)];
    let vae = CudaVae::load(&ctx, &jit, &fixture(&format!("{name}/vae")), &shapes, fuse).unwrap();
    let got = vae.decode(&l).unwrap().to_channels_first();
    let (want, _) = reference(name, "vae.decoded");
    let (err, db) = (max_abs(&got, &want), psnr(&got, &want));
    (got, err, db)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_vae_decodes_both_tiny_fixtures_like_diffusers() {
    for name in ["tiny-flux2-klein", "tiny-wan21"] {
        let (fused, err, db) = decode(name, true);
        let (plain, err_plain, _) = decode(name, false);
        let drift = max_abs(&fused, &plain);
        let peak = plain.iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!("{name}: fused max abs {err:e} PSNR {db:.1} dB, unfused max abs {err_plain:e}, fused against unfused {drift:e} over a peak of {peak}");
        assert!(err < 1e-4 && db > 60.0, "{name}");
        assert!(err_plain < 1e-4, "{name}");
        assert!(drift <= 1e-5 * peak.max(1.0), "{name}: fused and unfused differ by {drift}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_gpu_tiled_decode_matches_the_host_tiled_decode() {
    use specrt::vae::tiled::{Tiling, decode_tiled};
    let (ctx, jit) = gpu();
    let l = latent("tiny-flux2-klein");
    let tiling = Tiling { tile: 12, overlap: 8 };
    let vae = CudaVae::load(&ctx, &jit, &fixture("tiny-flux2-klein/vae"), &[(Dims { t: 1, h: 12, w: 12 }, true)], true).unwrap();
    let gpu_tiled = decode_tiled(&l, 2, tiling, &mut |x| vae.decode(x)).unwrap().to_channels_first();
    let host = specrt::vae::host::HostVae::open(&fixture("tiny-flux2-klein/vae")).unwrap();
    let host_tiled = decode_tiled(&l, 2, tiling, &mut |x| host.decode(x)).unwrap().to_channels_first();
    let err = max_abs(&gpu_tiled, &host_tiled);
    eprintln!("gpu tiled against host tiled: max abs {err:e}");
    assert!(err < 1e-4);
    assert!(vae.arena_of(Dims { t: 1, h: 12, w: 12 }, true).unwrap() < vae.arena_of(Dims { t: 1, h: 16, w: 16 }, true).unwrap());
}
