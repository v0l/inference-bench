use crate::family::{Family, FamilyPipeline};
use crate::generate::{Generator, Hooks, Plan, Request};
use crate::sampler::{Pipeline, PromptEncoder};
use crate::vae::compile::Dims;
use crate::vae::tiled::Tiling;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const VAE_TILE_ALIGN: usize = 8;
pub const VAE_TILE_OVERLAP: f64 = 0.25;
pub const ADMISSION_HEADROOM: u64 = 256 << 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    #[default]
    Text,
    Image,
    Video,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::Text => "text",
            ModelKind::Image => "image",
            ModelKind::Video => "video",
        }
    }

    pub fn parse(s: &str) -> Option<ModelKind> {
        match s {
            "text" => Some(ModelKind::Text),
            "image" => Some(ModelKind::Image),
            "video" => Some(ModelKind::Video),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MediaJob {
    pub kind: ModelKind,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub n: usize,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub sampler: Option<String>,
    #[serde(default)]
    pub frames: Option<usize>,
    #[serde(default)]
    pub preview_every: Option<usize>,
}

impl MediaJob {
    pub fn image(prompt: &str, width: usize, height: usize, seed: u64) -> MediaJob {
        MediaJob { kind: ModelKind::Image, prompt: prompt.into(), negative_prompt: String::new(), n: 1, width, height, seed, steps: None, guidance: None, sampler: None, frames: None, preview_every: None }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub seed: u64,
}

impl Picture {
    pub fn png(&self) -> Result<Vec<u8>> {
        png(&self.rgb, self.width, self.height)
    }
}

pub fn png(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let img = image::RgbImage::from_raw(width as u32, height as u32, rgb.to_vec()).context("pixel buffer size")?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaOutput {
    pub images: Vec<Picture>,
    pub frames: Option<Vec<Vec<Picture>>>,
    pub steps: usize,
    pub step_secs: f64,
    pub total_secs: f64,
    pub admission: Admission,
}

impl MediaOutput {
    pub fn steps_per_sec(&self) -> f64 {
        if self.step_secs > 0.0 { self.steps as f64 / self.step_secs } else { 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaEvent {
    Progress { image: usize, step: usize, steps: usize },
    Preview { image: usize, step: usize, picture: Picture },
}

#[derive(Debug)]
pub struct InvalidJob(pub String);

impl std::fmt::Display for InvalidJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidJob {}

fn invalid(msg: String) -> anyhow::Error {
    InvalidJob(msg).into()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MediaInfo {
    pub kind: ModelKind,
    pub family: String,
    pub pixel_step: usize,
    pub sizes: Vec<String>,
    pub default_size: String,
    pub default_steps: usize,
    pub default_guidance: f32,
    pub samplers: Vec<String>,
    pub max_images: usize,
    pub components: Vec<Component>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub name: String,
    pub bytes: u64,
    pub resident: bool,
}

pub fn parse_size(text: &str) -> Option<(usize, usize)> {
    let (w, h) = text.trim().split_once(['x', 'X'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

fn nearest(v: usize, step: usize) -> usize {
    (((v as f64) / step as f64).round() as usize).max(1) * step
}

impl MediaInfo {
    pub fn check(&self, job: &MediaJob) -> Result<()> {
        if job.kind != self.kind {
            return Err(invalid(format!("this is a {} model; a {} job does not run on it", self.kind.label(), job.kind.label())));
        }
        if job.prompt.trim().is_empty() {
            return Err(invalid("prompt is empty".into()));
        }
        if job.n == 0 || job.n > self.max_images {
            return Err(invalid(format!("n must be 1 to {}, not {}", self.max_images, job.n)));
        }
        let s = self.pixel_step;
        if job.width == 0 || job.height == 0 || !job.width.is_multiple_of(s) || !job.height.is_multiple_of(s) {
            return Err(invalid(format!("size {}x{} is not a multiple of {s}; the nearest valid size is {}x{}", job.width, job.height, nearest(job.width, s), nearest(job.height, s))));
        }
        if job.width > 4096 || job.height > 4096 {
            return Err(invalid(format!("size {}x{} is over 4096 on a side", job.width, job.height)));
        }
        if let Some(steps) = job.steps
            && !(1..=200).contains(&steps)
        {
            return Err(invalid(format!("steps must be 1 to 200, not {steps}")));
        }
        if let Some(g) = job.guidance
            && !(0.0..=30.0).contains(&g)
        {
            return Err(invalid(format!("guidance must be 0 to 30, not {g}")));
        }
        if let Some(name) = &job.sampler
            && !self.samplers.iter().any(|s| s == name)
        {
            return Err(invalid(format!("sampler {name} is not one of {}", self.samplers.join(", "))));
        }
        if job.frames.is_some() && self.kind != ModelKind::Video {
            return Err(invalid("frames is for video models".into()));
        }
        Ok(())
    }
}

pub fn is_pipeline(dir: &Path) -> bool {
    dir.join("model_index.json").is_file()
}

pub fn pipeline_class(dir: &Path) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model_index.json"))?)?;
    v["_class_name"].as_str().map(String::from).context("model_index.json has no _class_name")
}

pub fn family_for(dir: &Path) -> Result<Family> {
    let class = pipeline_class(dir)?;
    crate::family::active().for_pipeline(&class).cloned().with_context(|| format!("pipeline {class} is not supported: no family file matches it"))
}

pub fn resolve_dir(model: &str) -> Option<PathBuf> {
    let p = Path::new(model);
    if is_pipeline(p) {
        return Some(p.to_path_buf());
    }
    crate::weights::find_in_snapshot(model, "model_index.json").ok().and_then(|f| f.parent().map(Path::to_path_buf))
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|it| it.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).filter_map(|e| std::fs::metadata(e.path()).ok()).map(|m| m.len()).sum()).unwrap_or(0)
}

pub fn info(dir: &Path) -> Result<MediaInfo> {
    let family = family_for(dir)?;
    let fp: &FamilyPipeline = &family.pipeline;
    let pipeline = Pipeline::from_dir(dir)?;
    ensure!(pipeline.name == fp.graph, "family {} runs the {} pipeline graph, the folder reads as {}", family.name, fp.graph, pipeline.name);
    let kind = ModelKind::parse(&fp.kind).with_context(|| format!("family {} has pipeline kind {:?}", family.name, fp.kind))?;
    let components = fp.components.iter().map(|c| Component { name: c.clone(), bytes: dir_bytes(&dir.join(c)), resident: fp.resident.contains(c) }).collect();
    Ok(MediaInfo {
        kind,
        family: family.name.clone(),
        pixel_step: pipeline.values.get("pixel_step").copied().unwrap_or(16.0) as usize,
        sizes: fp.sizes.clone(),
        default_size: fp.default_size.clone(),
        default_steps: pipeline.steps,
        default_guidance: pipeline.guidance,
        samplers: fp.samplers.clone(),
        max_images: fp.max_images.max(1),
        components,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Admission {
    pub weights: u64,
    pub denoiser_arena: u64,
    pub vae_arena: u64,
    pub free: u64,
    pub sequential_cfg: bool,
    pub vae_tile: Option<usize>,
}

pub struct Budget<'a> {
    pub denoiser_arena: &'a dyn Fn(usize) -> Result<u64>,
    pub vae_arena: &'a dyn Fn(usize, usize) -> Result<u64>,
}

pub fn admit(free: u64, resident: u64, guided: bool, latent: (usize, usize), b: &Budget) -> Result<(Plan, Admission)> {
    let room = free.saturating_sub(ADMISSION_HEADROOM);
    let mut a = Admission { weights: resident, free, ..Admission::default() };
    let both = (b.denoiser_arena)(if guided { 2 } else { 1 })?;
    let one = (b.denoiser_arena)(1)?;
    let (den, sequential) = if resident + both <= room {
        (both, false)
    } else if guided && resident + one <= room {
        (one, true)
    } else {
        return Err(invalid(format!(
            "the denoiser needs {} MB of activations{} on top of {} MB of weights, and the GPU has {} MB free: lower the size or free GPU memory",
            (if guided { one } else { both }) >> 20,
            if guided { " (guidance run as two sequential forwards)" } else { "" },
            resident >> 20,
            free >> 20
        )));
    };
    a.denoiser_arena = den;
    a.sequential_cfg = sequential;
    let vae_room = room.saturating_sub(resident);
    let (lh, lw) = latent;
    let full = (b.vae_arena)(lh, lw)?;
    let tile = if full <= vae_room {
        a.vae_arena = full;
        None
    } else {
        let mut best = None;
        let mut t = VAE_TILE_ALIGN;
        while t < lh.max(lw) {
            let bytes = (b.vae_arena)(t.min(lh), t.min(lw))?;
            if bytes > vae_room {
                break;
            }
            best = Some((t, bytes));
            t += VAE_TILE_ALIGN;
        }
        let Some((t, bytes)) = best else {
            return Err(invalid(format!("the VAE needs {} MB for its smallest tile of {VAE_TILE_ALIGN} latent pixels, and {} MB is free after the weights", (b.vae_arena)(VAE_TILE_ALIGN.min(lh), VAE_TILE_ALIGN.min(lw))? >> 20, vae_room >> 20)));
        };
        a.vae_arena = bytes;
        a.vae_tile = Some(t);
        Some(Tiling { tile: t, overlap: ((t as f64 * VAE_TILE_OVERLAP).round() as usize).min(t - 1) })
    };
    Ok((Plan { sequential_cfg: sequential, vae_tile: tile }, a))
}

pub fn to_rgb(pixels: &[f32], width: usize, height: usize) -> Vec<u8> {
    let plane = width * height;
    let mut out = vec![0u8; plane * 3];
    for i in 0..plane {
        for c in 0..3 {
            out[i * 3 + c] = (pixels[c * plane + i].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

pub struct ImageModel {
    pub info: MediaInfo,
    pub dir: PathBuf,
    generator: Generator,
    prompts: PromptEncoder,
}

impl ImageModel {
    pub fn load(model: &str, runtime: Option<&crate::spec::RuntimeOptions>) -> Result<ImageModel> {
        let dir = resolve_dir(model).with_context(|| format!("{model} is not a diffusers pipeline folder"))?;
        let info = info(&dir)?;
        let cache = runtime.and_then(|r| r.cache_dir.clone()).unwrap_or_else(crate::jit::default_cache_dir);
        crate::hardware::ensure_nvrtc(&cache)?;
        let ctx = cudarc::driver::CudaContext::new(0).context("cuda init")?;
        let jit = crate::jit::Jit::new(ctx.clone(), cache)?;
        let probe = Pipeline::from_dir(&dir)?;
        let max_tokens = PromptEncoder::open(&dir.join("tokenizer")).map(|p| p.max_length(&probe)).unwrap_or(probe.max_length).max(16).next_multiple_of(64);
        let mut generator = Generator::open(ctx, jit, &dir, max_tokens)?;
        generator.keep_denoiser = true;
        let prompts = generator.prompt_encoder()?;
        crate::progress::report(crate::progress::Stage::Weights, 1, 1);
        Ok(ImageModel { info, dir, generator, prompts })
    }

    pub fn set_offload_text_encoder(&mut self, on: bool) {
        self.generator.pipeline.options.offload_text_encoder = on;
    }

    pub fn plan(&self, job: &MediaJob) -> Result<(Plan, Admission)> {
        let p = &self.generator.pipeline;
        let guidance = job.guidance.unwrap_or(p.guidance);
        let guided = p.guided(guidance)?;
        let scale = p.values["vae_scale"] as usize;
        let patch = p.values["latent_patch"] as usize;
        let d = Dims { t: 1, h: job.height / scale, w: job.width / scale };
        let shapes = crate::denoiser::Shapes { image: (d.h / patch) * (d.w / patch), text: self.prompts.max_length(p), patch_h: d.h / patch, patch_w: d.w / patch };
        let spec = crate::denoiser::DenoiserSpec::from_pipeline(&self.dir)?;
        let graph = crate::denoiser::DenoiserGraph::embedded();
        let weights = crate::denoiser_cuda::weight_bytes(&self.dir)?;
        let resident = if self.generator.denoiser_loaded() { 0 } else { weights };
        let vae_plan = crate::vae::graph::load_plan(&self.dir.join("vae"))?;
        let target = crate::codegen::cuda_target(self.generator.context())?;
        let free = self.generator.free_bytes()?;
        let budget = Budget {
            denoiser_arena: &|seqs| Ok(crate::denoiser_cuda::arena_bytes(graph, &spec, &shapes, seqs)? as u64),
            vae_arena: &|h, w| crate::vae::tiled::arena_bytes(&vae_plan, &target, h, w),
        };
        let (plan, mut a) = admit(free, resident, guided, (d.h, d.w), &budget)?;
        a.weights = weights;
        Ok((plan, a))
    }

    pub fn run(&mut self, job: &MediaJob, on_event: &mut dyn FnMut(MediaEvent) -> bool) -> Result<MediaOutput> {
        self.info.check(job)?;
        let started = std::time::Instant::now();
        let (plan, admission) = self.plan(job)?;
        let p = self.generator.pipeline.clone();
        let max = self.prompts.max_length(&p);
        let prompt = self.prompts.encode(&p, &job.prompt, max)?;
        let negative = Some(self.prompts.encode(&p, &job.negative_prompt, max)?);
        let mut out = MediaOutput { admission, ..MediaOutput::default() };
        let mut step_secs = 0.0;
        for i in 0..job.n {
            let seed = job.seed.wrapping_add(i as u64);
            let req = Request { prompt: prompt.clone(), negative: negative.clone(), height: job.height, width: job.width, steps: job.steps, guidance: job.guidance, seed };
            let mut last = std::time::Instant::now();
            let mut on_step = |step: usize, steps: usize, _latent: &[f32]| {
                step_secs += last.elapsed().as_secs_f64();
                last = std::time::Instant::now();
                on_event(MediaEvent::Progress { image: i, step, steps })
            };
            let img = match self.generator.generate_with(&req, plan, &mut Hooks { on_step: &mut on_step }) {
                Ok(img) => img,
                Err(e) if e.downcast_ref::<crate::generate::Cancelled>().is_some() => return Err(e),
                Err(e) => bail!(e),
            };
            out.steps += img.latents.len();
            out.images.push(Picture { width: img.width, height: img.height, rgb: to_rgb(&img.pixels, img.width, img.height), seed });
        }
        out.step_secs = step_secs;
        out.total_secs = started.elapsed().as_secs_f64();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
    }

    #[test]
    fn the_family_file_describes_the_tiny_pipeline() {
        let i = info(&tiny()).unwrap();
        assert_eq!((i.kind, i.family.as_str(), i.pixel_step, i.default_steps), (ModelKind::Image, "flux2_klein", 4, 4));
        assert_eq!(i.components.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["text_encoder", "transformer", "vae"]);
        assert!(i.components.iter().all(|c| c.bytes > 0));
        assert!(!i.components[0].resident && i.components[1].resident);
    }

    #[test]
    fn a_job_is_checked_before_any_launch() {
        let i = info(&tiny()).unwrap();
        let ok = MediaJob::image("a fox", 32, 32, 1);
        i.check(&ok).unwrap();
        let bad = i.check(&MediaJob { width: 30, ..ok.clone() }).unwrap_err().to_string();
        assert!(bad.contains("not a multiple of 4") && bad.contains("nearest valid size is 32x32"), "{bad}");
        assert!(i.check(&MediaJob { n: 9, ..ok.clone() }).is_err());
        assert!(i.check(&MediaJob { prompt: " ".into(), ..ok.clone() }).is_err());
        assert!(i.check(&MediaJob { sampler: Some("ddim".into()), ..ok.clone() }).unwrap_err().to_string().contains("euler"));
        assert!(i.check(&MediaJob { kind: ModelKind::Video, ..ok.clone() }).is_err());
        assert!(i.check(&MediaJob { frames: Some(5), ..ok }).is_err());
    }

    #[test]
    fn sizes_parse_as_width_by_height() {
        assert_eq!(parse_size("1024x768"), Some((1024, 768)));
        assert_eq!(parse_size("7"), None);
    }

    fn budget<'a>(den: &'a dyn Fn(usize) -> Result<u64>, vae: &'a dyn Fn(usize, usize) -> Result<u64>) -> Budget<'a> {
        Budget { denoiser_arena: den, vae_arena: vae }
    }

    #[test]
    fn admission_tiles_the_vae_then_runs_guidance_sequentially_then_refuses_naming_bytes() {
        let mb = 1u64 << 20;
        let den = |seqs: usize| Ok(100 * mb * seqs as u64);
        let vae = |h: usize, w: usize| Ok((h * w) as u64 * mb / 4);
        let head = ADMISSION_HEADROOM;
        let (p, a) = admit(head + 2000 * mb, 1000 * mb, true, (32, 32), &budget(&den, &vae)).unwrap();
        assert_eq!((p.sequential_cfg, p.vae_tile), (false, None));
        assert_eq!(a.vae_arena, 256 * mb);
        let (p, a) = admit(head + 1100 * mb, 1000 * mb, false, (64, 64), &budget(&den, &vae)).unwrap();
        assert_eq!(p.vae_tile.map(|t| t.tile), Some(16));
        assert_eq!(a.vae_tile, Some(16));
        let two = |seqs: usize| Ok(if seqs == 2 { 300 * mb } else { 100 * mb });
        let (p, _) = admit(head + 1250 * mb, 1000 * mb, true, (8, 8), &budget(&two, &vae)).unwrap();
        assert!(p.sequential_cfg);
        let e = admit(head + 1050 * mb, 1000 * mb, true, (8, 8), &budget(&two, &vae)).unwrap_err();
        assert!(e.downcast_ref::<InvalidJob>().is_some());
        let msg = e.to_string();
        assert!(msg.contains("100 MB of activations") && msg.contains("1000 MB of weights"), "{msg}");
    }

    #[test]
    fn pixels_become_rgb_bytes_and_a_png() {
        let rgb = to_rgb(&[0.0, 1.0, 0.5, 0.5, 2.0, -1.0], 2, 1);
        assert_eq!(rgb, [0, 128, 255, 255, 128, 0]);
        let png = png(&rgb, 2, 1).unwrap();
        let back = image::load_from_memory(&png).unwrap().to_rgb8();
        assert_eq!((back.width(), back.height(), back.into_raw()), (2, 1, rgb));
    }
}
