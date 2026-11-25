use cudarc::driver::CudaContext;
use specrt::denoiser::Shapes;
use specrt::denoiser_cuda::Denoiser;
use specrt::jit::Jit;
use specrt::sampler::Pipeline;
use std::path::{Path, PathBuf};

const VELOCITY_TOLERANCE: f32 = 1e-4;
const F32_BOUND: f32 = 2e-2;

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

fn shapes() -> Shapes {
    Shapes { image: 48, text: 16, patch_h: 4, patch_w: 4, frames: 3 }
}

fn gpu() -> (std::sync::Arc<CudaContext>, Jit) {
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-wan-{}", std::process::id()))).unwrap();
    (ctx, jit)
}

fn timestep(i: usize) -> f32 {
    meta()["timesteps"][i].as_f64().unwrap() as f32
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test wan_gpu -- --ignored --test-threads=1"]
fn the_engine_wan_denoiser_matches_diffusers_cond_and_uncond_on_every_step() {
    let (ctx, jit) = gpu();
    let p = Pipeline::from_dir(&fixture()).unwrap();
    let d = Denoiser::load_pipeline(&ctx.default_stream(), &jit, &fixture(), shapes(), &p, true).unwrap();
    let mut a = d.arena_for(2).unwrap();
    d.prepare(&mut a, &[&floats("text.embeds"), &floats("negative.embeds")]).unwrap();
    for i in 0..3 {
        let x = if i == 0 { floats("noise") } else { floats(&format!("bf16.step{}.latent", i - 1)) };
        let v = d.forward(&mut a, &x, timestep(i), None).unwrap();
        let half = v.len() / 2;
        for (part, which) in [(&v[..half], "cond"), (&v[half..], "uncond")] {
            let bf = max_diff(part, &floats(&format!("bf16.step{i}.velocity.{which}")));
            eprintln!("step {i} {which}: engine velocity {bf:e} to the bf16 weight run");
            assert!(bf < VELOCITY_TOLERANCE, "step {i} {which}: {bf}");
        }
    }
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test wan_gpu -- --ignored --test-threads=1"]
fn three_guided_steps_and_the_causal_vae_give_the_reference_video() {
    use specrt::generate::{Generator, Plan, Request, VideoHooks, VideoRequest};
    let (ctx, jit) = gpu();
    let mut g = Generator::open(ctx, jit, &fixture(), 64).unwrap();
    let prompt = |key: &str| specrt::sampler::Prompt { ids: floats(&format!("{key}.ids")).iter().map(|v| *v as u32).collect(), mask: floats(&format!("{key}.mask")).iter().map(|v| *v as u8).collect() };
    let m = meta();
    let request = Request { prompt: prompt("text"), negative: Some(prompt("negative")), height: 32, width: 32, steps: Some(3), guidance: Some(m["guidance"].as_f64().unwrap() as f32), seed: m["seed"].as_u64().unwrap() };
    let req = VideoRequest { request, frames: 5 };
    let mut steps = Vec::new();
    let mut decoded = Vec::new();
    let run = |g: &mut Generator, plan: Plan, steps: &mut Vec<(usize, usize)>, decoded: &mut Vec<(usize, usize)>| g.generate_video(&req, plan, &mut VideoHooks { on_step: &mut |i, n, _| { steps.push((i, n)); true }, on_frames: &mut |f, n| { decoded.push((f, n)); true } }).unwrap();
    let v = run(&mut g, Plan::default(), &mut steps, &mut decoded);
    assert_eq!(steps, vec![(1, 3), (2, 3), (3, 3)]);
    assert_eq!(decoded, vec![(1, 5), (3, 5), (5, 5)], "one latent frame decodes to 1 then 2 output frames through the causal cache");
    assert_eq!((v.frames.len(), v.height, v.width), (5, 32, 32));
    for (i, l) in v.latents.iter().enumerate() {
        let bf = max_diff(l, &floats(&format!("bf16.step{i}.latent")));
        eprintln!("step {i}: latent {bf:e} to the bf16 weight run");
        assert!(bf < VELOCITY_TOLERANCE, "step {i}: {bf}");
    }
    let got: Vec<f32> = v.frames.concat();
    let want = floats("bf16.video");
    let (d, db) = (max_diff(&got, &want), psnr(&got, &want));
    eprintln!("video: {d:e} to the bf16 weight run, PSNR {db:.1} dB, f32 run {:e}; memory {:?}", max_diff(&got, &floats("video")), v.memory);
    assert!(d < 1e-3 && db > 60.0, "{d} {db}");
    assert!(!v.memory.encoder_resident_in_loop);
    let seq = run(&mut g, Plan { sequential_cfg: true, vae_tile: None }, &mut Vec::new(), &mut Vec::new());
    let ds = max_diff(&seq.frames.concat(), &got);
    eprintln!("sequential guidance against batched: {ds:e}, arena {} against {}", seq.memory.arena, v.memory.arena);
    assert!(ds < 1e-3 && seq.memory.arena < v.memory.arena);
    let tiling = specrt::vae::tiled::Tiling { tile: 6, overlap: 4 };
    let mut tiled_frames = Vec::new();
    let t0 = std::time::Instant::now();
    let tiled = run(&mut g, Plan { sequential_cfg: false, vae_tile: Some(tiling) }, &mut Vec::new(), &mut tiled_frames);
    let tiled_secs = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let whole = run(&mut g, Plan::default(), &mut Vec::new(), &mut Vec::new());
    let whole_secs = t1.elapsed().as_secs_f64();
    let db = psnr(&tiled.frames.concat(), &whole.frames.concat());
    eprintln!("spatial VAE tiles on the video (tile 6, overlap 4 latent pixels): PSNR {db:.1} dB against whole frames, vae arena {} against {}, run {tiled_secs:.3} s against {whole_secs:.3} s", tiled.memory.vae_arena, whole.memory.vae_arena);
    assert_eq!(tiled.frames.len(), 5);
    assert!(db > 20.0, "{db}");
    assert!(tiled.memory.vae_arena < whole.memory.vae_arena);
    assert_eq!(tiled_frames.last(), Some(&(5, 5)));
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (1.0 / mse.max(1e-30)).log10()
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test --test wan_gpu -- --ignored --test-threads=1"]
fn the_unipc_kernel_steps_the_latent_like_the_recorded_run() {
    let (ctx, jit) = gpu();
    let p = Pipeline::from_dir(&fixture()).unwrap();
    let d = Denoiser::load_pipeline(&ctx.default_stream(), &jit, &fixture(), shapes(), &p, true).unwrap();
    let mut a = d.arena_for(2).unwrap();
    d.prepare(&mut a, &[&floats("text.embeds"), &floats("negative.embeds")]).unwrap();
    let sched = p.schedule(3, 48).unwrap();
    let g = meta()["guidance"].as_f64().unwrap() as f32;
    d.set_latent(&mut a, &floats("noise")).unwrap();
    d.reset_state(&mut a).unwrap();
    for i in 0..3 {
        let scalars = p.multistep_scalars(&sched, i).unwrap();
        d.multistep(&mut a, sched.model_timestep(i), &scalars, g).unwrap();
        let x = d.latent(&a).unwrap();
        let bf = max_diff(&x, &floats(&format!("bf16.step{i}.latent")));
        let f = max_diff(&x, &floats(&format!("step{i}.latent")));
        eprintln!("step {i}: latent {bf:e} to the bf16 weight run, {f:e} to f32");
        assert!(bf < VELOCITY_TOLERANCE, "step {i}: {bf}");
        assert!(f < F32_BOUND, "step {i}: {f}");
    }
}
