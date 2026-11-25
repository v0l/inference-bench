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

fn compiled(name: &str, fuse: bool, first: bool) -> specrt::vae::compile::Compiled {
    let p = load_plan(&fixture(&format!("{name}/vae"))).unwrap();
    let l = latent(name);
    specrt::vae::compile::compile(&p, specrt::vae::compile::Dims { t: 1, h: l.h, w: l.w }, &kir::Target::sm_120(), fuse, first).unwrap()
}

#[test]
fn the_compiler_rediscovers_the_conv_norm_act_fusions_of_the_kl_decoder() {
    let c = compiled("tiny-flux2-klein", true, true);
    let s = &c.fusion;
    let grouped = |units: &[&str]| s.groups.iter().any(|g| g.units.iter().map(String::as_str).eq(units.iter().copied()));
    let want: [&[&str]; 4] = [
        &["mid_block.resnets.0.norm1 apply", "conv mid_block.resnets.0.conv1", "mid_block.resnets.0.norm2 stats"],
        &["mid_block.resnets.0.norm2 apply", "conv mid_block.resnets.0.conv2", "add 6", "mid_block.attentions.0.group_norm stats"],
        &["upsample 27", "conv up_blocks.0.upsamplers.0.conv", "up_blocks.1.resnets.0.norm1 stats"],
        &["conv_norm_out apply", "conv conv_out"],
    ];
    let report = specrt::vae::compile::fusion_report(s, c.lowered.count, &kir::Target::sm_120()).render();
    for units in want {
        assert!(grouped(units), "no group {units:?}\n{report}");
    }
    assert_eq!((s.launches_before, s.launches_after), (56, 21), "{report}");
    let stats_apart = s.rejected.iter().find(|d| d.producer == ["mid_block.resnets.0.norm1 stats"] && d.consumers.iter().any(|c| c.iter().any(|u| u == "mid_block.resnets.0.norm1 apply")));
    assert!(stats_apart.is_some(), "{report}");
}

#[test]
fn the_compiler_folds_residuals_upsamples_and_the_clamp_into_the_causal_convs() {
    let c = compiled("tiny-wan21", true, false);
    let s = &c.fusion;
    let grouped = |units: &[&str]| s.groups.iter().any(|g| g.units.iter().map(String::as_str).eq(units.iter().copied()));
    for units in [&["conv mid_block.resnets.0.conv2", "add 6"][..], &["upsample 28", "conv up_blocks.0.upsamplers.0.resample.1"], &["conv conv_out", "clamp 55"]] {
        assert!(grouped(units), "no group {units:?}");
    }
    assert_eq!((s.launches_before, s.launches_after), (56, 44));
}

#[test]
fn a_grid_reduction_never_fuses_with_its_reader() {
    for name in ["tiny-flux2-klein"] {
        let c = compiled(name, true, true);
        for g in &c.fusion.groups {
            let stats: Vec<&String> = g.units.iter().filter(|u| u.ends_with(" stats")).collect();
            for st in stats {
                let apply = st.replace(" stats", " apply");
                assert!(!g.units.contains(&apply), "{st} and {apply} share a launch");
            }
        }
    }
}

fn run_schedule(name: &str, fuse: bool) -> Vec<f32> {
    use specrt::vae::run::Runner;
    let vae = HostVae::open(&fixture(&format!("{name}/vae"))).unwrap();
    let (first, steady) = (compiled(name, fuse, true), compiled(name, fuse, false));
    let mut frame = 0;
    let out = vae
        .decode_with(&latent(name), &mut |x, s| {
            let c = if frame == 0 { &first } else { &steady };
            frame += 1;
            Runner { vae: &vae, compiled: c }.run(x, s)
        })
        .unwrap();
    out.to_channels_first()
}

#[test]
fn the_fused_schedule_decodes_exactly_like_the_unfused_one() {
    for name in ["tiny-flux2-klein", "tiny-wan21"] {
        let fused = run_schedule(name, true);
        let plain = run_schedule(name, false);
        assert_eq!(fused, plain, "{name}");
        let (want, _) = reference(name, "vae.decoded");
        assert!(max_abs(&fused, &want) < 1e-4, "{name}");
    }
}

#[test]
fn the_arena_is_the_peak_of_live_values_and_offsets_never_overlap() {
    use specrt::vae::compile::lifetimes;
    for (name, first) in [("tiny-flux2-klein", true), ("tiny-wan21", true), ("tiny-wan21", false)] {
        for fuse in [false, true] {
            let c = compiled(name, fuse, first);
            let a = &c.arena;
            assert!(a.size >= a.peak && a.peak > 0, "{name}");
            let life = lifetimes(&c.schedule);
            for (v, (o, s)) in &a.offsets {
                for (w, (p, t)) in &a.offsets {
                    let (x, y) = (life[v], life[w]);
                    if v < w && x.0 <= y.1 && y.0 <= x.1 {
                        assert!(o + s <= *p || p + t <= *o, "{name}: values {v} and {w} overlap in the arena");
                    }
                }
            }
        }
        assert!(compiled(name, true, first).arena.peak <= compiled(name, false, first).arena.peak, "fusion never grows the arena");
    }
    let c = compiled("tiny-flux2-klein", true, true);
    let top = 32 * 32 * 16 * 4;
    assert!(c.arena.peak >= 2 * top as u64 && c.arena.peak < 8 * top as u64, "{}", c.arena.peak);
}

#[test]
fn the_tile_size_is_the_largest_whose_arena_fits_the_budget() {
    use specrt::vae::tiled::{arena_bytes, choose};
    let p = load_plan(&fixture("tiny-flux2-klein/vae")).unwrap();
    let t = kir::Target::sm_120();
    let full = arena_bytes(&p, &t, 128, 128).unwrap();
    assert_eq!(choose(&p, &t, 128, 128, full, 8, 0.25).unwrap(), None);
    let budget = arena_bytes(&p, &t, 40, 40).unwrap();
    let tiling = choose(&p, &t, 128, 128, budget, 8, 0.25).unwrap().unwrap();
    assert_eq!(tiling.tile, 40);
    assert!(arena_bytes(&p, &t, 48, 48).unwrap() > budget);
    assert_eq!(tiling.overlap, 10);
    assert!(choose(&p, &t, 128, 128, 1024, 8, 0.25).is_err());
    let w = load_plan(&fixture("tiny-wan21/vae")).unwrap();
    assert!(arena_bytes(&w, &t, 60, 104).unwrap() > arena_bytes(&w, &t, 30, 52).unwrap());
}

const DIFFUSERS_TILED_PSNR: f64 = 18.1;

#[test]
fn tiled_and_untiled_decodes_agree_within_the_seam_tolerance() {
    use specrt::vae::tiled::{Tiling, decode_tiled};
    let vae = HostVae::open(&fixture("tiny-flux2-klein/vae")).unwrap();
    let l = latent("tiny-flux2-klein");
    let whole = vae.decode(&l).unwrap();
    let tiling = Tiling { tile: 12, overlap: 8 };
    let mut tiles = 0;
    let tiled = decode_tiled(&l, 2, tiling, &mut |x| {
        tiles += 1;
        vae.decode(x)
    })
    .unwrap();
    assert_eq!(tiles, 4);
    let (a, b) = (whole.to_channels_first(), tiled.to_channels_first());
    let (err, db) = (max_abs(&a, &b), psnr(&a, &b));
    eprintln!("tiled 12 overlap 8 against untiled: max abs {err:e}, PSNR {db:.1} dB");
    assert!(db > DIFFUSERS_TILED_PSNR + 2.0 && err < 0.75, "seams: max abs {err}, PSNR {db}");
    let one = decode_tiled(&l, 2, Tiling { tile: 16, overlap: 4 }, &mut |x| vae.decode(x)).unwrap();
    assert_eq!(one.to_channels_first(), a, "a single tile is the untiled decode");
}
