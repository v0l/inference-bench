#[path = "common/conv_cases.rs"]
mod conv_cases;

use conv_cases::{case, cases, expected, host_output, max_abs};
use specrt::vae::graph::{load_plan, plan};
use specrt::vae::host::HostVae;
use specrt::vae::{Grid, VaeOp};
use std::path::{Path, PathBuf};

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

#[test]
fn the_host_conv_matches_numpy_on_every_vae_conv_shape() {
    for c in cases() {
        let name = c["name"].as_str().unwrap();
        if name.starts_with("big") {
            continue;
        }
        let got = host_output(&case(&c));
        let want = expected(&c);
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = max_abs(&got, &want);
        assert!(err <= 2e-6 * scale.max(1.0), "{name}: max abs {err} over a peak of {scale}");
    }
}

#[test]
fn the_flux2_vae_plan_comes_from_the_graph_and_its_tensors_exist() {
    let p = load_plan(&fixture("tiny-flux2-klein/vae")).unwrap();
    assert_eq!((p.graph.as_str(), p.latent_channels, p.out_channels, p.temporal, p.spatial_scale()), ("kl", 4, 3, false, 2));
    let ck = specrt::weights::Checkpoint::open(&fixture("tiny-flux2-klein/vae")).unwrap();
    p.check_tensors(&|n| ck.shape_of(n)).unwrap();
    let decoder = ck.names_under("decoder.").len() + ck.names_under("post_quant_conv").len();
    assert_eq!(p.tensors().len(), decoder, "every decoder tensor is read once");
    let convs = p.ops.iter().filter(|o| matches!(o, VaeOp::Conv { .. })).count();
    assert_eq!(convs, 1 + 1 + 4 + 2 + 4 + 1 + 4 + 1 + 1);
}

#[test]
fn the_wan_vae_plan_is_causal_with_one_temporal_upsample() {
    let p = load_plan(&fixture("tiny-wan21/vae")).unwrap();
    assert_eq!((p.graph.as_str(), p.latent_channels, p.out_channels, p.temporal, p.spatial_scale(), p.time_ups()), ("causal3d", 4, 3, true, 4, 1));
    assert_eq!(p.output_frames(3), 5);
    let ck = specrt::weights::Checkpoint::open(&fixture("tiny-wan21/vae")).unwrap();
    p.check_tensors(&|n| ck.shape_of(n)).unwrap();
    let decoder = ck.names_under("decoder.").len() + ck.names_under("post_quant_conv").len();
    assert_eq!(p.tensors().len(), decoder);
}

#[test]
fn a_vae_class_without_a_graph_is_refused_by_name() {
    let err = plan(&serde_json::json!({"_class_name": "AutoencoderKLMochi"})).unwrap_err();
    assert!(err.to_string().contains("AutoencoderKLMochi"), "{err}");
}

#[test]
fn the_host_vae_decodes_the_tiny_flux2_latent_like_diffusers() {
    let vae = HostVae::open(&fixture("tiny-flux2-klein/vae")).unwrap();
    let out = vae.decode(&latent("tiny-flux2-klein")).unwrap();
    let (want, shape) = reference("tiny-flux2-klein", "vae.decoded");
    assert_eq!(vec![1, out.c, out.h, out.w], shape);
    let got = out.to_channels_first();
    let (err, db) = (max_abs(&got, &want), psnr(&got, &want));
    eprintln!("tiny-flux2-klein host decode: max abs {err:e}, PSNR {db:.1} dB");
    assert!(err < 1e-4 && db > 60.0);
}

#[test]
fn the_host_vae_decodes_the_tiny_wan_latent_frame_by_frame_like_diffusers() {
    let vae = HostVae::open(&fixture("tiny-wan21/vae")).unwrap();
    let out = vae.decode(&latent("tiny-wan21")).unwrap();
    let (want, shape) = reference("tiny-wan21", "vae.decoded");
    assert_eq!(vec![1, out.c, out.t, out.h, out.w], shape);
    let got = out.to_channels_first();
    let (err, db) = (max_abs(&got, &want), psnr(&got, &want));
    eprintln!("tiny-wan21 host decode: max abs {err:e}, PSNR {db:.1} dB");
    assert!(err < 1e-4 && db > 60.0);
}
