# Image and video generation

Diffusion pipelines run on the engine as data: a family file per pipeline class, graph TOML for the denoiser, VAE and scheduler, and the existing text encoder families. No Rust names an architecture. Served on `/v1/images/generations` and `/v1/videos`.

A pipeline has three parts and no token loop: a text encoder run once per prompt (hidden states out), a denoiser run N times over a latent and driven by a scheduler, and a VAE decoder from latent to pixels or frames.

## Supported pipelines

| Family | Match | Text encoder | Denoiser graph | Scheduler entry | VAE graph |
|---|---|---|---|---|---|
| `specs/family-flux2_klein.toml` (image) | `Flux2KleinPipeline` | Qwen3 (`family-qwen3.toml`), hidden states of layers `L/4`, `L/2`, `3L/4` stacked | `graphs/denoiser.toml` (FLUX.2 double and single stream blocks) | `flow-euler` | `kl` |
| `specs/family-wan.toml` (video) | Wan 2.1 T2V | UMT5 encoder, `graphs/encoder.toml`, on the host (`specrt::encoder::HostEncoder`; the GPU lowering is #1406) | `graphs/denoiser-cross.toml` (picked by `WanTransformer3DModel`) | `unipc` | `causal3d` |

`specrt::media::info` reads a diffusers folder into `MediaInfo` (kind, family, pixel step, sizes, defaults, component bytes); an unmatched class is refused by name. The family `[pipeline]` table names the kind, pipeline entry, components and which stay resident, sizes and default, samplers, `max_images`, frames and fps for video.

## Text encoder

`Runtime::hidden_states(ids, real, layers)` runs the decoder graph's prefill plan (no head, no final norm, KV not kept), copies `x` after each chosen layer and stops after the last. Pad keys are masked with a per run key limit in `flash_mma`, so padded rows are computed as diffusers computes them. `sampler::PromptEncoder` applies the tokenizer's chat template (`enable_thinking` false), truncates to `max_length` (512) and pads.

## Denoiser

`graphs/denoiser.toml`: `[fields]` bind the transformer, VAE and pipeline configs, `[symbols]` derive widths, `[[buffer]]` declares activations with their rows (`image`, `text`, `joint`, `seqs`), and `[[head]]`, `[[block]]` (`double`, `single`), `[[tail]]` and `[[step]]` lower to a `Plan` in `Phase::Forward` with one `Op::Dit` per node. Ops: `Patchify`, `Unpatchify`, `TimestepEmbed`, `Split`, `Copy`, `Linear` (per stream, joined or sliced weights), `Modulate`, `GatedAdd`, `Glu`, `AxisRope`, `JointAttention`, and in the cross graph `table_add`, `layer_norm`, `pack`, `attention`, `cross_attention`, `MultiStep`. `denoiser::Signer` signs them and `denoiser::validate` checks every read was written.

Kernels: compose descriptions in `crates/specrt/specs/dit_*.toml`, the tile GEMM `vis_mma` (and `vis_mma_step` with the scheduler update in its store), `vis_flash` and `vis_flash_kv` (non causal, per sequence key ranges, a second key and value source for cross attention), `vis_axis_rope`, `vis_qk_pack`.

Passes ([kir-compiler.md](kir-compiler.md)): the cost model gains a compute ridge (`Costs::with_compute`), so every norm prologue onto a compute bound tile GEMM is refused (`Reason::ComputeBound`) unless `compute_prologue`. Epilogues fuse: gated residuals into `to_out`, `to_add_out` and `linear_out`, SwiGLU into `linear_in`, both halves of the single block's `to_out`, and the scheduler step into `proj_out` when there is no CFG. H1 hoists step invariant work (`denoiser::hoisted`: the text split and `context_embedder`; `denoiser::block_invariant`: each Wan block's text K and V into `cross_kv`, 0.19 GB for 30 blocks).

Runtime: `denoiser_cuda::Denoiser` (`load`, `arena`, `prepare`, `forward`) and `denoiser_host::HostDenoiser` (the same plan in f32 on the CPU). CFG runs the conditional and unconditional prompts as two sequences of one forward; the guided step reads both.

## Scheduler

`graphs/pipeline.toml`, one entry per scheduler: `[fields]` bind the scheduler config, `text_layers`, `max_length`, default steps and guidance, the `cfg` condition, and the schedule as expressions (`mu`, `sigma` with the exponential or static shift, the model timestep, `update = x + (s1 - s0) * v`, `combine = u + g * (c - u)`, `latent_in`, `output`). `unipc` (`bh2`, order 2, flow sigmas) adds `[pipeline.multistep]`: state (`last`, `m1`, `m2`), per step host scalars and the update as assignments, lowered to one `MultiStep` compose kernel. `require` and `refuse` guard unsupported scheduler configs. Noise is `sampler::noise(seed, n)` (splitmix64 then Box-Muller), not torch's generator; `scripts/tiny_diffusion.py` feeds the same noise to diffusers. Image i of n uses `seed + i`.

`[pipeline.options]`: `offload_text_encoder` (on: the encoder is dropped after encoding), `offload_vae` (off), `sequential_cfg` (two passes in one single sequence arena), `vae_tile`.

## VAE

`graphs/vae.toml` holds a decoder graph per family as macros over the diffusers config (`kl` for `AutoencoderKL` and `AutoencoderKLFlux2`, `causal3d` for `AutoencoderKLWan` and `AutoencoderKLQwenImage`). `specrt::vae::graph::plan` expands it into `VaeOp`s (`conv`, `group_norm`, `rms_norm`, `upsample`, `time_up`, `attention`, `add`, `clamp`) and checks every tensor shape. Group norm lowers to a `GridReduce` stats unit and an apply unit; the fusion pass puts the apply (with SiLU) in the next conv's prologue, the next statistics in the conv epilogue, residual adds into `conv2`, the nearest upsample into the next conv's load and the clamp into `conv_out` (`Reason::GridBarrier` and `Reason::Layout` keep the rest apart).

`kir::conv` is an implicit GEMM over NHWC (8 x 16 output patch, `bn = min(128, round_up(Cout, 32))`, `bk = min(32, Cin)` per tap, stages from shared memory), a direct family under 8 channels, a causal 3D variant with a per conv frame cache, and the `Rows` family for few outputs over a long reduction. Mid attention is one block per query row with scores in shared memory, any head dim. `vae::compile::arena` walks the fused schedule and assigns offsets by interval colouring; `vae::tiled::choose` picks the largest tile whose arena fits and `decode_tiled` blends tiles with linear ramps. The causal VAE decodes one latent frame at a time.

## Admission and serving

`media::admit` adds the denoiser weights (unless resident) and `denoiser_cuda::arena_bytes` for 1 or 2 sequences plus `vae::tiled::arena_bytes`, against free memory less 256 MB. If the VAE does not fit it tiles; if guided CFG does not fit it runs sequentially; otherwise it refuses naming the MB. `MediaInfo::check` refuses bad sizes (naming the nearest valid one), over 65,535 latent tokens, steps, guidance or sampler out of range, before any launch. The chosen plan comes back in `usage.admission`.

- `POST /v1/images/generations` (`crates/server/src/images.rs`): `model`, `prompt`, `n`, `size`, `response_format` (`b64_json` default, `url` served from `GET /v1/images/files/{id}.png` for `images.file_ttl_secs`), `seed`, `negative_prompt`, `steps`, `guidance`, `sampler`, `stream` (SSE `image_generation.progress`, `partial_image`, `completed`). A closed connection cancels between steps.
- `POST /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content` (MP4: H.264 from `openh264`, boxes written by `specrt::mp4`), `DELETE /v1/videos/{id}` (`crates/server/src/videos.rs`). `seconds` becomes the nearest frame count the VAE takes at `fps`.
- Both share the media gate: `images.jobs_per_gpu` running plus `images.queue_depth` waiting, 429 with `Retry-After` past that. `/v1/models` carries `type`.

The app opens an image model on its own load page (size, steps, guidance, release the text encoder, the GPU, the components and the fit) and replaces chat with a Generate page ([app.md](app.md)).

## Measured state

| Case | Value | Source |
|---|---|---|
| tiny FLUX.2 klein, GPU1, against diffusers with bf16 transformer weights | latents 6e-6 unguided, 1.7e-5 guided after 3 steps; image PSNR 124 and 114 dB | #1284 |
| tiny Wan 2.1, GPU1 | velocities 8e-6 per step, latents 2.6e-5 after 3 UniPC steps, video PSNR 150.8 dB | #1286 |
| klein 4B plan at 1024x1024 | 227 launches a step after fusion (276 before), 34.8 TFLOP, 48.7 GB read; VAE 40 launches (117), arena 2.15 GB | #1285 |
| Wan 2.1 1.3B plan, 5 s of 480p | 32,760 tokens, 283 TFLOP a forward, 3.67 GB of activations a sequence | #1286 |

The real klein 4B and Wan 2.1 1.3B have not been run end to end; Euler and UniPC are the only samplers.

## Tests

`crates/specrt/tests/diffusion.rs` (fixtures `tiny-flux2-klein` and `tiny-wan21` from `scripts/tiny_diffusion.py` keep the real classes and tensor names, references load), `denoiser.rs` and `denoiser_gpu.rs` (against `scripts/denoiser_reference.py`), `vae.rs` and `vae_gpu.rs` (conv cases against numpy from `scripts/conv_reference.py`), `sampler.rs` and `sampler_gpu.rs`, `text_encoder_gpu.rs`, `wan_denoiser.rs` and `wan_gpu.rs`, `crates/server/tests/images.rs`, `images_gpu.rs`, `videos.rs`, `videos_gpu.rs`, and `crates/ui/src/tests_images.rs`. Plans: `inference-bench kernels --spec <diffusers folder> --plan --graph denoiser|vae --size WxH [--frames N] [--compiled] [--fusion]`.
