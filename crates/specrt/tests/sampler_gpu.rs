use cudarc::driver::CudaContext;
use specrt::denoiser::Shapes;
use specrt::denoiser_cuda::Denoiser;
use specrt::generate::{Generator, Request};
use specrt::jit::Jit;
use specrt::sampler::{Pipeline, Prompt};
use std::path::{Path, PathBuf};

const LATENT_TOLERANCE: f32 = 1e-4;
const IMAGE_TOLERANCE: f32 = 1e-3;
const BF16_WEIGHT_BOUND: f32 = 2e-2;
const PSNR_BOUND: f64 = 60.0;
const ARENA_BOUND: u64 = 1 << 20;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference/tiny-flux2-klein.json")).unwrap()).unwrap()
}

fn floats(key: &str) -> Vec<f32> {
    let file = meta()["files"][key]["file"].as_str().unwrap_or_else(|| panic!("no reference {key}")).to_string();
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (1.0 / mse.max(1e-30)).log10()
}

fn prompt(key: &str) -> Prompt {
    let ids = floats(&format!("{key}.ids")).iter().map(|v| *v as u32).collect();
    let mask = floats(&format!("{key}.mask")).iter().map(|v| *v as u8).collect();
    Prompt { ids, mask }
}

fn gpu() -> (std::sync::Arc<CudaContext>, Jit) {
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-sampler-{}", std::process::id()))).unwrap();
    (ctx, jit)
}

fn request(guidance: f32) -> Request {
    let m = meta();
    Request { prompt: prompt("text"), negative: Some(prompt("negative")), height: 32, width: 32, steps: Some(3), guidance: Some(guidance), seed: m["seed"].as_u64().unwrap() }
}

fn check_run(guidance: f32, prefix: &str) {
    let (ctx, jit) = gpu();
    let mut g = Generator::open(ctx, jit, &fixture(), 64).unwrap();
    if prefix.is_empty() {
        g.pipeline.values.insert("distilled".into(), 1.0);
    } else {
        g.pipeline.values.insert("distilled".into(), 0.0);
    }
    let img = g.generate(&request(guidance)).unwrap();
    assert_eq!(img.guided, !prefix.is_empty());
    for (i, l) in img.latents.iter().enumerate() {
        let bf = max_diff(l, &floats(&format!("bf16.{prefix}step{i}.latent")));
        let f = max_diff(l, &floats(&format!("{prefix}step{i}.latent")));
        eprintln!("{prefix}step {i}: latent {bf:e} to the bf16 weight run, {f:e} to the f32 run");
        assert!(bf < LATENT_TOLERANCE, "{prefix}step {i}: {bf}");
        assert!(f < BF16_WEIGHT_BOUND, "{prefix}step {i}: {f}");
    }
    let image_key = if prefix.is_empty() { "image".to_string() } else { "cfg.image".to_string() };
    let bf = max_diff(&img.pixels, &floats(&format!("bf16.{image_key}")));
    let (f, db) = (max_diff(&img.pixels, &floats(&image_key)), psnr(&img.pixels, &floats(&format!("bf16.{image_key}"))));
    eprintln!("{image_key}: {bf:e} to the bf16 weight run (PSNR {db:.1} dB), {f:e} to the f32 run");
    assert!(bf < IMAGE_TOLERANCE && db > PSNR_BOUND, "{image_key}: {bf}, {db} dB");
    assert!(f < BF16_WEIGHT_BOUND, "{image_key}: {f}");
    let m = &img.memory;
    eprintln!("memory: {m:?}");
    assert!(!m.encoder_resident_in_loop, "offload_text_encoder releases the encoder before the loop");
    assert!(!g.encoder_loaded());
    assert!(m.arena < ARENA_BOUND, "arena {} over {ARENA_BOUND}", m.arena);
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test sampler_gpu -- --ignored --test-threads=1"]
fn three_unguided_steps_from_the_seed_give_the_reference_image() {
    check_run(1.0, "");
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test sampler_gpu -- --ignored --test-threads=1"]
fn three_guided_steps_batch_both_prompts_and_give_the_reference_image() {
    check_run(meta()["cfg_guidance"].as_f64().unwrap() as f32, "cfg.");
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test sampler_gpu -- --ignored --test-threads=1"]
fn the_offload_options_decide_what_stays_resident_and_change_nothing_else() {
    let (ctx, jit) = gpu();
    let mut g = Generator::open(ctx, jit, &fixture(), 64).unwrap();
    g.pipeline.values.insert("distilled".into(), 1.0);
    g.pipeline.options.offload_text_encoder = false;
    g.pipeline.options.offload_vae = false;
    let first = g.generate(&request(1.0)).unwrap();
    assert!(g.encoder_loaded() && g.vae_loaded());
    let kept = g.generate(&request(1.0)).unwrap();
    assert!(kept.memory.encoder_resident_in_loop && kept.memory.vae_resident_in_loop, "{:?}", kept.memory);
    g.pipeline.options.offload_text_encoder = true;
    g.pipeline.options.offload_vae = true;
    let freed = g.generate(&request(1.0)).unwrap();
    assert!(!freed.memory.encoder_resident_in_loop && !freed.memory.vae_resident_in_loop, "{:?}", freed.memory);
    assert!(!g.encoder_loaded() && !g.vae_loaded());
    assert_eq!(first.pixels, kept.pixels);
    assert_eq!(kept.pixels, freed.pixels);
    let mut other = request(1.0);
    other.seed += 1;
    assert_ne!(g.generate(&other).unwrap().pixels, freed.pixels, "the seed picks the noise");
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test sampler_gpu -- --ignored --test-threads=1"]
fn the_euler_epilogue_on_proj_out_equals_the_unfused_step() {
    let (ctx, jit) = gpu();
    let stream = ctx.default_stream();
    let p = Pipeline::from_dir(&fixture()).unwrap();
    let rule = p.step_rule(false).unwrap();
    let shapes = Shapes { image: 64, text: 16, patch_h: 8, patch_w: 8 };
    let d = Denoiser::load_with(&stream, &jit, &fixture(), shapes, &rule, false).unwrap();
    assert!(d.fused_step);
    let sched = p.schedule(3, 64).unwrap();
    let text = floats("text.embeds");
    let run = |fused: bool| {
        let mut a = d.arena().unwrap();
        d.prepare(&mut a, &[&text]).unwrap();
        d.set_latent(&mut a, &floats("noise")).unwrap();
        (0..3)
            .map(|i| {
                if fused { d.step(&mut a, sched.model_timestep(i), sched.sigmas[i + 1], 1.0) } else { d.step_unfused(&mut a, sched.model_timestep(i), sched.sigmas[i + 1], 1.0) }.unwrap();
                d.latent(&a).unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (fused, plain) = (run(true), run(false));
    for i in 0..3 {
        let diff = max_diff(&fused[i], &plain[i]);
        eprintln!("step {i}: fused Euler epilogue against the unfused step {diff:e}");
        assert_eq!(fused[i], plain[i], "step {i}: the epilogue computes x + (s1 - s0) v in the same order");
    }
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test sampler_gpu -- --ignored --test-threads=1"]
fn two_sequences_in_one_forward_equal_two_single_forwards() {
    let (ctx, jit) = gpu();
    let stream = ctx.default_stream();
    let shapes = Shapes { image: 64, text: 16, patch_h: 8, patch_w: 8 };
    let rule = Pipeline::from_dir(&fixture()).unwrap().step_rule(true).unwrap();
    let d = Denoiser::load_with(&stream, &jit, &fixture(), shapes.clone(), &rule, true).unwrap();
    let (cond, uncond) = (floats("text.embeds"), floats("negative.embeds"));
    let x = floats("noise");
    let mut both = d.arena_for(2).unwrap();
    d.prepare(&mut both, &[&cond, &uncond]).unwrap();
    let v = d.forward(&mut both, &x, 1.0, None).unwrap();
    let single = |t: &[f32]| {
        let mut a = d.arena_for(1).unwrap();
        d.prepare(&mut a, &[t]).unwrap();
        d.forward(&mut a, &x, 1.0, None).unwrap()
    };
    let (c, u) = (single(&cond), single(&uncond));
    let half = v.len() / 2;
    let (dc, du) = (max_diff(&v[..half], &c), max_diff(&v[half..], &u));
    eprintln!("batched against single: cond {dc:e}, uncond {du:e}");
    assert!(dc < 1e-6 && du < 1e-6, "{dc} {du}");
    assert!(max_diff(&c, &u) > 1e-3, "the two prompts give different velocities");
    let one = d.arena_bytes(1).unwrap();
    let two = d.arena_bytes(2).unwrap();
    eprintln!("arena: {one} bytes without guidance, {two} with");
    assert!(two > one && two < 2 * one + 4096, "{one} {two}");
}
