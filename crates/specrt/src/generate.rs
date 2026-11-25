use crate::denoiser::Shapes;
use crate::denoiser_cuda::Denoiser;
use crate::jit::Jit;
use crate::runtime::Runtime;
use crate::sampler::{Pipeline, Prompt, PromptEncoder, noise, stack_layers};
use crate::spec::Spec;
use crate::vae::Grid;
use crate::vae::compile::Dims;
use crate::vae::cuda::CudaVae;
use crate::vae::host::Weights;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub prompt: Prompt,
    pub negative: Option<Prompt>,
    pub height: usize,
    pub width: usize,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub pixels: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub latents: Vec<Vec<f32>>,
    pub guided: bool,
    pub memory: Memory,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Memory {
    pub text_encoder: u64,
    pub denoiser_weights: u64,
    pub arena: u64,
    pub vae_arena: u64,
    pub encoder_resident_in_loop: bool,
    pub vae_resident_in_loop: bool,
}

pub struct Generator {
    pub dir: PathBuf,
    pub pipeline: Pipeline,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    jit: Jit,
    encoder: Option<Runtime>,
    vae: Option<CudaVae>,
    max_tokens: usize,
}

fn read_tensor(ck: &crate::weights::Checkpoint, name: &str) -> Result<Vec<f32>> {
    ck.floats(name)
}

impl Generator {
    pub fn open(ctx: Arc<CudaContext>, jit: Jit, dir: &Path, max_tokens: usize) -> Result<Generator> {
        let pipeline = Pipeline::from_dir(dir)?;
        let stream = ctx.default_stream();
        Ok(Generator { dir: dir.to_path_buf(), pipeline, ctx, stream, jit, encoder: None, vae: None, max_tokens })
    }

    pub fn encoder_spec(&self) -> Result<Spec> {
        let te = self.dir.join("text_encoder");
        let mut spec = Spec::from_hf_config(&std::fs::read_to_string(te.join("config.json"))?).context("the text encoder config")?;
        spec.repo = te.display().to_string();
        spec.runtime.split_activations = Some(true);
        Ok(spec)
    }

    pub fn prompt_encoder(&self) -> Result<PromptEncoder> {
        PromptEncoder::open(&self.dir.join("tokenizer"))
    }

    fn load_encoder(&mut self) -> Result<&mut Runtime> {
        if self.encoder.is_none() {
            self.encoder = Some(Runtime::load(&self.encoder_spec()?, self.max_tokens)?);
        }
        Ok(self.encoder.as_mut().unwrap())
    }

    pub fn encoder_loaded(&self) -> bool {
        self.encoder.is_some()
    }

    pub fn vae_loaded(&self) -> bool {
        self.vae.is_some()
    }

    pub fn encode(&mut self, prompt: &Prompt) -> Result<Vec<f32>> {
        let layers = self.pipeline.text_layers.clone();
        let masked = self.pipeline.masked_pad_keys;
        let rt = self.load_encoder()?;
        let h = rt.spec().dims.hidden;
        let n = prompt.ids.len();
        let real = if masked { prompt.real() } else { n };
        let mut states = rt.hidden_states(&prompt.ids, real.max(1), &layers)?;
        if !masked {
            for l in 0..layers.len() {
                states[(l * n + prompt.real()) * h..(l + 1) * n * h].fill(0.0);
            }
        }
        Ok(stack_layers(&states, layers.len(), n, h))
    }

    fn release_encoder(&mut self) {
        self.encoder = None;
    }

    fn vae_dims(&self, height: usize, width: usize) -> Result<Dims> {
        let step = self.pipeline.values["pixel_step"] as usize;
        let scale = self.pipeline.values["vae_scale"] as usize;
        ensure!(height > 0 && width > 0 && height % step == 0 && width % step == 0, "{height}x{width} is not a multiple of {step}");
        Ok(Dims { t: 1, h: height / scale, w: width / scale })
    }

    fn load_vae(&mut self, d: Dims) -> Result<&CudaVae> {
        if self.vae.is_none() {
            self.vae = Some(CudaVae::load(&self.ctx, &self.jit, &self.dir.join("vae"), &[(d, true)], true)?);
        }
        Ok(self.vae.as_ref().unwrap())
    }

    fn decode(&mut self, tokens: &[f32], d: Dims) -> Result<Vec<f32>> {
        let p = self.pipeline.clone();
        let patch = p.values["latent_patch"] as usize;
        let ck = crate::weights::Checkpoint::open(&self.dir.join("vae"))?;
        let var = read_tensor(&ck, &p.tensors["bn_var"])?;
        let mean = read_tensor(&ck, &p.tensors["bn_mean"])?;
        let mut x = tokens.to_vec();
        p.denormalise(&mut x, &var, &mean)?;
        let channels = p.values["latent_channels"] as usize;
        let (ph, pw) = (d.h / patch, d.w / patch);
        let grid_cf = crate::denoiser_host::unpatchify(&x, channels, patch, ph, pw);
        let latent = Grid::from_channels_first(&grid_cf, channels, 1, d.h, d.w);
        let out = self.load_vae(d)?.decode(&latent)?.to_channels_first();
        p.postprocess(&out)
    }

    pub fn generate(&mut self, req: &Request) -> Result<Image> {
        let p = self.pipeline.clone();
        let opts = p.options;
        let steps = req.steps.unwrap_or(p.steps);
        let guidance = req.guidance.unwrap_or(p.guidance);
        let guided = p.guided(guidance)?;
        let d = self.vae_dims(req.height, req.width)?;
        let patch = p.values["latent_patch"] as usize;
        let (ph, pw) = (d.h / patch, d.w / patch);
        let image = ph * pw;
        let text = req.prompt.ids.len();
        let mut memory = Memory::default();
        let cond = self.encode(&req.prompt)?;
        let uncond = if guided {
            let neg = req.negative.as_ref().context("a guided run needs a negative prompt (the empty prompt when none is given)")?;
            ensure!(neg.ids.len() == text, "the negative prompt is padded to {} tokens, the prompt to {text}", neg.ids.len());
            Some(self.encode(neg)?)
        } else {
            None
        };
        memory.text_encoder = self.encoder.as_ref().map_or(0, |r| r.weight_bytes);
        if opts.offload_text_encoder {
            self.release_encoder();
        }
        if opts.offload_vae {
            self.vae = None;
        }
        let shapes = Shapes { image, text, patch_h: ph, patch_w: pw };
        let rule = p.step_rule(guided)?;
        let den = Denoiser::load_with(&self.stream, &self.jit, &self.dir, shapes, &rule, guided)?;
        let mut arena = den.arena_for(if guided { 2 } else { 1 })?;
        memory.denoiser_weights = den.weight_bytes as u64;
        memory.arena = arena.bytes as u64;
        memory.encoder_resident_in_loop = self.encoder_loaded();
        memory.vae_resident_in_loop = self.vae_loaded();
        let texts: Vec<&[f32]> = std::iter::once(cond.as_slice()).chain(uncond.as_deref()).collect();
        den.prepare(&mut arena, &texts)?;
        let channels = p.values["token_channels"] as usize;
        let grid = noise(req.seed, image * channels);
        let tokens: Vec<f32> = (0..image).flat_map(|t| (0..channels).map(move |c| (t, c))).map(|(t, c)| grid[c * image + t]).collect();
        den.set_latent(&mut arena, &tokens)?;
        let sched = p.schedule(steps, image)?;
        let mut latents = Vec::with_capacity(steps);
        for i in 0..steps {
            den.step(&mut arena, sched.model_timestep(i), sched.sigmas[i + 1], guidance)?;
            latents.push(den.latent(&arena)?);
        }
        let last = latents.last().cloned().context("no steps ran")?;
        drop(arena);
        drop(den);
        let pixels = self.decode(&last, d)?;
        memory.vae_arena = self.vae.as_ref().map_or(Ok(0), |v| v.arena_of(d, true))?;
        if opts.offload_vae {
            self.vae = None;
        }
        Ok(Image { pixels, height: req.height, width: req.width, latents, guided, memory })
    }
}
