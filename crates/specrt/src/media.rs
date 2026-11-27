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
    Duplex,
    Transcription,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::Text => "text",
            ModelKind::Image => "image",
            ModelKind::Video => "video",
            ModelKind::Duplex => "duplex",
            ModelKind::Transcription => "transcription",
        }
    }

    pub fn parse(s: &str) -> Option<ModelKind> {
        match s {
            "text" => Some(ModelKind::Text),
            "image" => Some(ModelKind::Image),
            "video" => Some(ModelKind::Video),
            "duplex" => Some(ModelKind::Duplex),
            "transcription" => Some(ModelKind::Transcription),
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
    #[serde(default)]
    pub offload_text_encoder: Option<bool>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub sequential_cfg: Option<bool>,
    #[serde(default)]
    pub tiled_vae: Option<bool>,
}

impl MediaJob {
    pub fn image(prompt: &str, width: usize, height: usize, seed: u64) -> MediaJob {
        MediaJob { kind: ModelKind::Image, prompt: prompt.into(), negative_prompt: String::new(), n: 1, width, height, seed, steps: None, guidance: None, sampler: None, frames: None, preview_every: None, offload_text_encoder: None, fps: None, sequential_cfg: None, tiled_vae: None }
    }

    pub fn video(prompt: &str, width: usize, height: usize, frames: usize, seed: u64) -> MediaJob {
        MediaJob { kind: ModelKind::Video, frames: Some(frames), ..MediaJob::image(prompt, width, height, seed) }
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
    pub mp4: Option<Vec<u8>>,
    pub fps: u32,
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
    Frames { done: usize, total: usize },
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
    #[serde(default)]
    pub frames: Vec<usize>,
    #[serde(default)]
    pub default_frames: usize,
    #[serde(default)]
    pub fps: u32,
    #[serde(default)]
    pub frame_step: usize,
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
    pub fn tokens(&self, width: usize, height: usize) -> usize {
        let s = self.pixel_step.max(1);
        (width / s) * (height / s)
    }

    pub fn latent_frames(&self, frames: Option<usize>) -> usize {
        match (self.kind, frames) {
            (ModelKind::Video, Some(f)) if self.frame_step > 0 => (f.saturating_sub(1)) / self.frame_step + 1,
            _ => 1,
        }
    }

    pub fn fits(&self, size: &str) -> bool {
        parse_size(size).is_some_and(|(w, h)| self.tokens(w, h) <= crate::denoiser_cuda::MAX_ROWS)
    }

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
        let tokens = self.tokens(job.width, job.height) * self.latent_frames(job.frames);
        if tokens > crate::denoiser_cuda::MAX_ROWS {
            return Err(invalid(format!("size {}x{}{} is {tokens} latent tokens, over the {} one denoiser launch takes; pick a smaller size or fewer frames", job.width, job.height, job.frames.map_or(String::new(), |f| format!(" at {f} frames")), crate::denoiser_cuda::MAX_ROWS)));
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
        if self.kind == ModelKind::Video {
            let f = job.frames.unwrap_or(self.default_frames);
            let step = self.frame_step.max(1);
            if f == 0 || f > 1025 || !(f - 1).is_multiple_of(step) {
                let near = (f.saturating_sub(1) + step / 2) / step * step + 1;
                return Err(invalid(format!("frames must be 1 plus a multiple of {step}, at most 1025, not {f}; the nearest valid count is {near}")));
            }
            if let Some(fps) = job.fps
                && !(1..=120).contains(&fps)
            {
                return Err(invalid(format!("fps must be 1 to 120, not {fps}")));
            }
            if job.n != 1 {
                return Err(invalid("a video job makes one video".into()));
            }
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
    let video = kind == ModelKind::Video;
    let mut i = MediaInfo {
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
        frames: if video { fp.frames.clone() } else { Vec::new() },
        default_frames: if video { fp.default_frames.max(1) } else { 0 },
        fps: if video { fp.fps.max(1) } else { 0 },
        frame_step: if video { pipeline.values.get("temporal_scale").copied().unwrap_or(1.0) as usize } else { 0 },
    };
    let sizes: Vec<String> = i.sizes.iter().filter(|s| i.fits(s)).cloned().collect();
    ensure!(!sizes.is_empty(), "family {} offers no size this engine runs at pixel step {}", family.name, i.pixel_step);
    if !sizes.contains(&i.default_size) {
        i.default_size = sizes.iter().max_by_key(|s| parse_size(s).map_or(0, |(w, h)| w * h)).cloned().unwrap_or_default();
    }
    i.sizes = sizes;
    Ok(i)
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
            return Err(invalid(format!("the VAE does not fit for its smallest tile of {VAE_TILE_ALIGN} latent pixels with {} MB free after the weights", vae_room >> 20)));
        };
        a.vae_arena = bytes;
        a.vae_tile = Some(t);
        Some(Tiling { tile: t, overlap: ((t as f64 * VAE_TILE_OVERLAP).round() as usize).min(t - 1) })
    };
    Ok((Plan { sequential_cfg: sequential, vae_tile: tile }, a))
}

pub fn latent_preview(tokens: &[f32], count: usize) -> Vec<u8> {
    let channels = tokens.len().checked_div(count).unwrap_or(0);
    let mut out = vec![0u8; count * 3];
    if channels == 0 {
        return out;
    }
    for c in 0..3 {
        let vals: Vec<f32> = (0..count).map(|t| tokens[t * channels + c % channels]).collect();
        let (lo, hi) = vals.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        let span = (hi - lo).max(1e-6);
        for (t, v) in vals.iter().enumerate() {
            out[t * 3 + c] = (((v - lo) / span) * 255.0).round() as u8;
        }
    }
    out
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

pub fn crop_rgb(rgb: &[u8], width: usize, w: usize, h: usize) -> Vec<u8> {
    (0..h).flat_map(|y| rgb[y * width * 3..(y * width + w) * 3].iter().copied()).collect()
}

pub struct ImageModel {
    pub info: MediaInfo,
    pub dir: PathBuf,
    pub device: usize,
    generator: Generator,
    prompts: PromptEncoder,
}

thread_local! {
    static DEVICE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

pub fn with_device<T>(device: Option<usize>, f: impl FnOnce() -> T) -> T {
    let before = DEVICE.with(|d| d.replace(device));
    let out = f();
    DEVICE.with(|d| d.set(before));
    out
}

pub const DEVICE_KEY: &str = "media.device";

impl ImageModel {
    pub fn load(model: &str, runtime: Option<&crate::spec::RuntimeOptions>) -> Result<ImageModel> {
        let dir = resolve_dir(model).with_context(|| format!("{model} is not a diffusers pipeline folder"))?;
        let info = info(&dir)?;
        let cache = runtime.and_then(|r| r.cache_dir.clone()).unwrap_or_else(crate::jit::default_cache_dir);
        crate::hardware::ensure_nvrtc(&cache)?;
        let device = DEVICE.with(|d| d.get()).unwrap_or(0);
        let ctx = cudarc::driver::CudaContext::new(device).with_context(|| format!("cuda init on device {device}"))?;
        let jit = crate::jit::Jit::new(ctx.clone(), cache)?;
        let probe = Pipeline::from_dir(&dir)?;
        let max_tokens = PromptEncoder::open(&dir.join("tokenizer")).map(|p| p.max_length(&probe)).unwrap_or(probe.max_length).max(16).next_multiple_of(64);
        let mut generator = Generator::open(ctx, jit, &dir, max_tokens)?;
        generator.keep_denoiser = true;
        let prompts = generator.prompt_encoder()?;
        crate::progress::report(crate::progress::Stage::Weights, 1, 1);
        Ok(ImageModel { info, dir, device, generator, prompts })
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
        let t = self.info.latent_frames(job.frames.or((self.info.kind == ModelKind::Video).then_some(self.info.default_frames)));
        let d = Dims { t, h: job.height / scale, w: job.width / scale };
        let (ph, pw) = (d.h / patch, d.w / patch);
        let shapes = crate::denoiser::Shapes { image: t * ph * pw, text: self.prompts.max_length(p), patch_h: ph, patch_w: pw, frames: t };
        let spec = crate::denoiser::DenoiserSpec::from_pipeline(&self.dir)?;
        let graph = crate::denoiser::DenoiserGraph::for_pipeline(&self.dir)?;
        let weights = crate::denoiser_cuda::weight_bytes(&self.dir)?;
        let resident = if self.generator.denoiser_loaded() { 0 } else { weights };
        let vae_plan = crate::vae::graph::load_plan(&self.dir.join("vae"))?;
        let target = crate::codegen::cuda_target(self.generator.context())?;
        let free = self.generator.free_bytes()?;
        let budget = Budget {
            denoiser_arena: &|seqs| Ok(crate::denoiser_cuda::arena_bytes(graph, &spec, &shapes, seqs)? as u64),
            vae_arena: &|h, w| match kir::conv::attention_fits(&target, h * w) {
                true => crate::vae::tiled::arena_bytes(&vae_plan, &target, h, w),
                false => Ok(u64::MAX),
            },
        };
        let (plan, mut a) = admit(free, resident, guided, (d.h, d.w), &budget)?;
        a.weights = weights;
        let o = p.options;
        let mut plan = plan;
        if job.sequential_cfg.unwrap_or(o.sequential_cfg) && guided && !plan.sequential_cfg {
            plan.sequential_cfg = true;
            a.sequential_cfg = true;
            a.denoiser_arena = (budget.denoiser_arena)(1)?;
        }
        let tile = match job.tiled_vae {
            Some(true) => Some(o.vae_tile.filter(|t| *t > 0).unwrap_or_else(|| (d.h.max(d.w) / 2).next_multiple_of(VAE_TILE_ALIGN).max(VAE_TILE_ALIGN))),
            Some(false) => None,
            None => o.vae_tile,
        };
        if let Some(t) = tile.filter(|t| *t > 0 && plan.vae_tile.is_none_or(|p| p.tile > *t)) {
            plan.vae_tile = Some(Tiling { tile: t, overlap: ((t as f64 * VAE_TILE_OVERLAP).round() as usize).min(t - 1) });
            a.vae_tile = Some(t);
            a.vae_arena = (budget.vae_arena)(t.min(d.h), t.min(d.w))?;
        }
        Ok((plan, a))
    }

    pub fn run(&mut self, job: &MediaJob, on_event: &mut dyn FnMut(MediaEvent) -> bool) -> Result<MediaOutput> {
        self.info.check(job)?;
        let started = std::time::Instant::now();
        let default_offload = Pipeline::from_dir(&self.dir)?.options.offload_text_encoder;
        self.set_offload_text_encoder(job.offload_text_encoder.unwrap_or(default_offload));
        let (plan, admission) = self.plan(job)?;
        let p = self.generator.pipeline.clone();
        let max = self.prompts.max_length(&p);
        let prompt = self.prompts.encode(&p, &job.prompt, max)?;
        let negative = Some(self.prompts.encode(&p, &job.negative_prompt, max)?);
        let mut out = MediaOutput { admission, ..MediaOutput::default() };
        if self.info.kind == ModelKind::Video {
            let frames = job.frames.unwrap_or(self.info.default_frames);
            let fps = job.fps.unwrap_or(self.info.fps);
            let req = crate::generate::VideoRequest { request: Request { prompt, negative, height: job.height, width: job.width, steps: job.steps, guidance: job.guidance, seed: job.seed }, frames };
            let mut last = std::time::Instant::now();
            let mut step_secs = 0.0;
            let video = {
                let events = std::cell::RefCell::new(&mut *on_event);
                let mut on_step = |step: usize, steps: usize, _: &[f32]| {
                    step_secs += last.elapsed().as_secs_f64();
                    last = std::time::Instant::now();
                    (events.borrow_mut())(MediaEvent::Progress { image: 0, step, steps })
                };
                let mut on_frames = |done: usize, total: usize| (events.borrow_mut())(MediaEvent::Frames { done, total });
                self.generator.generate_video(&req, plan, &mut crate::generate::VideoHooks { on_step: &mut on_step, on_frames: &mut on_frames })?
            };
            let pictures: Vec<Picture> = video.frames.iter().map(|f| Picture { width: video.width, height: video.height, rgb: to_rgb(f, video.width, video.height), seed: job.seed }).collect();
            let (w, h) = (crate::mp4::even(video.width), crate::mp4::even(video.height));
            let rgb: Vec<Vec<u8>> = pictures.iter().map(|p| crop_rgb(&p.rgb, p.width, w, h)).collect();
            out.mp4 = Some(crate::mp4::encode(&rgb, w, h, fps)?.mp4());
            out.fps = fps;
            out.steps = video.latents.len();
            out.images = pictures.first().cloned().into_iter().collect();
            out.frames = Some(vec![pictures]);
            out.step_secs = step_secs;
            out.total_secs = started.elapsed().as_secs_f64();
            return Ok(out);
        }
        let mut step_secs = 0.0;
        for i in 0..job.n {
            let seed = job.seed.wrapping_add(i as u64);
            let req = Request { prompt: prompt.clone(), negative: negative.clone(), height: job.height, width: job.width, steps: job.steps, guidance: job.guidance, seed };
            let mut last = std::time::Instant::now();
            let (pw, ph) = (job.width / self.info.pixel_step.max(1), job.height / self.info.pixel_step.max(1));
            let every = job.preview_every.filter(|k| *k > 0);
            let mut on_step = |step: usize, steps: usize, latent: &[f32]| {
                step_secs += last.elapsed().as_secs_f64();
                last = std::time::Instant::now();
                let go = on_event(MediaEvent::Progress { image: i, step, steps });
                if go && every.is_some_and(|k| step.is_multiple_of(k)) && pw * ph > 0 {
                    let picture = Picture { width: pw, height: ph, rgb: latent_preview(latent, pw * ph), seed };
                    return on_event(MediaEvent::Preview { image: i, step, picture });
                }
                go
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

    #[test]
    fn every_kind_label_parses_back_and_the_old_ones_keep_their_names() {
        let kinds = [ModelKind::Text, ModelKind::Image, ModelKind::Video, ModelKind::Duplex, ModelKind::Transcription];
        assert_eq!(kinds.map(|k| k.label()), ["text", "image", "video", "duplex", "transcription"]);
        for k in kinds {
            assert_eq!(ModelKind::parse(k.label()), Some(k));
            assert_eq!(serde_json::to_value(k).unwrap(), serde_json::json!(k.label()));
        }
    }

    fn tiny() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-flux2-klein")
    }

    #[test]
    fn the_family_file_describes_the_tiny_pipeline() {
        let i = info(&tiny()).unwrap();
        assert_eq!((i.kind, i.family.as_str(), i.pixel_step, i.default_steps), (ModelKind::Image, "flux2_klein", 4, 4));
        assert_eq!(i.default_size, "720x1280", "the largest offered size the tiny model's pixel step allows");
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
        let big = i.check(&MediaJob { width: 1024, height: 1024, ..ok.clone() }).unwrap_err().to_string();
        assert!(big.contains("65536 latent tokens"), "{big}");
        assert!(!i.sizes.contains(&"1024x1024".to_string()) && i.fits(&i.default_size), "{:?} {}", i.sizes, i.default_size);
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
