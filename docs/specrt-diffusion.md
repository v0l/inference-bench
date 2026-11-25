# Text to image and text to video (#1281)

Analysis only: no engine code. Hub configs were read on 24 Nov from each repo's `model_index.json` and component `config.json` (unauthenticated: gated repos were read through public mirrors with the same files, named below). Pipeline code read from `diffusers` 0.43.0 in a private venv (below). Parameter counts come from the safetensors headers where they were summed, else from the file sizes and dtypes the hub API lists.

A diffusion pipeline has three parts and no token loop:

1. A text encoder: an LLM or a T5 or CLIP encoder run once per prompt, hidden states out, no sampling, no KV cache kept.
2. A denoiser (DiT or MMDiT transformer) run N times over a latent, driven by a scheduler (pure arithmetic on latents between the runs).
3. A VAE decoder (convolutions, norms, upsampling; 3D and causal in time for video) that turns the final latent into pixels or frames.

## 1. Candidates

### Image

| Model (repo) | Text encoder | Denoiser | VAE | Scheduler, steps | Licence | Quantised copies |
|---|---|---|---|---|---|---|
| **FLUX.2 klein 4B** (`black-forest-labs/FLUX.2-klein-4B`, `Flux2KleinPipeline`) | `Qwen3ForCausalLM` (Qwen3-4B: 36 layers, hidden 2560, 4.02B, bf16 8.0 GB). Hidden states of layers 9, 18, 27 stacked to 7680 wide, 512 tokens padded | `Flux2Transformer2DModel` 3.88B: 5 double stream (MMDiT) and 20 single stream (parallel attention and MLP) blocks, 24 heads x 128, SwiGLU mlp 3.0, qk RMS norm, 4 axis rope (32 each, theta 2000), shared adaLN modulation, no guidance embedding, `in_channels` 128 | `AutoencoderKLFlux2` 84M: 2D, `block_out_channels` 128 to 512, GroupNorm 32, SiLU, mid attention, 32 latent channels, 2x2 patchify plus BatchNorm running stats | `FlowMatchEulerDiscreteScheduler`, dynamic exponential shift with an empirical `mu(seq_len, steps)`; distilled: 4 steps, guidance 1 (no CFG). `FLUX.2-klein-base-4B`: same weights format, 50 steps, CFG 4 | Apache 2.0, not gated | BFL `FLUX.2-klein-4b-fp8` (4.07 GB), BFL `FLUX.2-klein-4b-nvfp4`, `unsloth/FLUX.2-klein-4B-GGUF` (Q2_K to BF16), `leejet/FLUX.2-klein-4B-GGUF`, mflux q4 |
| FLUX.2 klein 9B (`black-forest-labs/FLUX.2-klein-9B`) | same pipeline | 9B | same VAE | same | gated, non-commercial | BFL fp8, nvfp4, GGUF |
| FLUX.2 dev (`black-forest-labs/FLUX.2-dev`, read via `unsloth/FLUX.2-dev`) | `Mistral3ForConditionalGeneration` (Mistral Small 3 24B, 40 layers, hidden 5120, 48 GB bf16) | `Flux2Transformer2DModel` 32B: 8 double, 48 single, 48 heads x 128, joint dim 15360 | `AutoencoderKLFlux2` | flow Euler, dynamic shift, 50 steps | FLUX non-commercial, gated | BFL `FLUX.2-dev-NVFP4`, `unsloth/FLUX.2-dev-FP8`, GGUF (unsloth, city96) |
| FLUX.1 schnell (`black-forest-labs/FLUX.1-schnell`, read via `unsloth/FLUX.1-schnell`) | `CLIPTextModel` (123M, pooled) and `T5EncoderModel` (T5-XXL encoder, 24 layers, 4.76B) | `FluxTransformer2DModel` 11.9B: 19 double, 38 single, 24 x 128, pooled text into the modulation | `AutoencoderKL` 84M, 16 channels, GroupNorm 32 | flow Euler, shift 1, 4 steps | Apache 2.0 but gated (dev: non-commercial) | GGUF (city96, unsloth), BFL `FLUX.1-dev-NVFP4` and fp8 (dev) |
| Qwen-Image (`Qwen/Qwen-Image`, `-2512`) | `Qwen2_5_VLForConditionalGeneration` text model (`qwen2_5_vl_text`, 28 layers, hidden 3584, 8.3B) | `QwenImageTransformer2DModel` 20.4B: 60 MMDiT double stream layers, 24 x 128, 3 axis rope (16, 56, 56) | `AutoencoderKLQwenImage`: the Wan 2.1 3D causal VAE config (base 96, z 16) | flow Euler, dynamic shift, 50 steps, true CFG | Apache 2.0 | `unsloth/Qwen-Image-FP8`, GGUF (unsloth, QuantStack), community NVFP4 |
| Qwen-Image 2.1 (`Qwen/Qwen-Image-2.1`, Sept 2026) | `Qwen3VLForConditionalGeneration` text (`qwen3_vl_text`, 36 layers, hidden 4096, interleaved mrope) | `QwenImage21Transformer2DModel` 7.1B: 32 layers, 32 x 128, mlp 3, patch 1, 64 latent channels, `causal_condition` | `AutoencoderKLQwenImage21` (z 64, 16x spatial, 8x temporal) | flow Euler dynamic shift; Turbo: 8 fixed `sample_sigmas` in `model_index.json` | `qwen-research` licence (not Apache) | `unsloth/Qwen-Image-2.1-FP8`, GGUF (unsloth), community NVFP4 |
| HunyuanImage 2.1 (`hunyuanvideo-community/HunyuanImage-2.1-Diffusers`) | Qwen2.5-VL-7B text and a ByT5 glyph encoder (`T5EncoderModel`, 219M) | 17.4B: 20 double, 40 single, 2 refiner, 28 x 128 | `AutoencoderKLHunyuanImage`, 32x spatial, 64 channels | flow Euler shift 5, `AdaptiveProjectedMixGuidance` | Tencent Hunyuan community licence: does not apply in the EU, UK or South Korea | Comfy repack, unsloth fp8 |
| CogView4 6B (`zai-org/CogView4-6B`) | `GlmModel` (GLM-4-9B, 40 layers, hidden 4096, 8.8B) | `CogView4Transformer2DModel` 6.4B: 28 layers, 32 x 128 | `AutoencoderKL` 16 channels, `block_out_channels` 128 to 1024 | flow Euler, linear dynamic shift | Apache 2.0 | almost none (one quanto int8) |
| SD 3.5 medium (`stabilityai/stable-diffusion-3.5-medium`, read via `adamo1139/stable-diffusion-3.5-medium-ungated`) | two `CLIPTextModelWithProjection` (L and G) and T5-XXL | `SD3Transformer2DModel` 2.5B MMDiT-X: 24 layers, 24 x 64, dual attention in layers 0 to 12 | `AutoencoderKL` 16 channels | flow Euler shift 3 | Stability community licence, gated | GGUF (city96, second-state) |

### Video

| Model (repo) | Text encoder | Denoiser | VAE | Scheduler | Licence | Quantised |
|---|---|---|---|---|---|---|
| **Wan 2.1 T2V 1.3B** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 401k downloads) | `UMT5EncoderModel` (umt5-xxl encoder: 24 layers, d 4096, gated GELU, relative position buckets, 5.68B, f32 on the hub), 512 tokens, rows past the prompt zeroed | `WanTransformer3DModel` 1.42B: 30 blocks of self attention, cross attention to text, GELU-tanh FFN 8960; 12 heads x 128; qk RMS norm across heads; 3D rope split 44 / 42 / 42 of the head; adaLN from a per block `scale_shift_table` plus a shared `time_proj` (6 vectors); Conv3d patch (1, 2, 2) | `AutoencoderKLWan` 127M: causal Conv3d (time padding 2 in front, a 2 frame cache per conv), per pixel channel RMS norm, nearest upsample plus a causal time conv for temporal upsampling, single head mid attention; 8x spatial, 4x temporal, z 16 | `UniPCMultistepScheduler` (flow prediction, `bh2`, order 2, flow shift 3), 50 steps, CFG 5 | Apache 2.0 | GGUF (samuelchristlie, off-grid-ai); a 4 bit NVFP4 weight-only test copy |
| Wan 2.2 TI2V 5B (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | same umt5-xxl (bf16, 11.4 GB) | 5.0B: 30 blocks, 24 x 128, FFN 14336, 48 latent channels, per token timesteps (`expand_timesteps`) | Wan 2.2 VAE: 16x spatial, 4x temporal, z 48, patch 2, residual blocks (0.7B) | UniPC, flow shift 5 | Apache 2.0 | `QuantStack/Wan2.2-TI2V-5B-GGUF` (560k), unsloth GGUF and FP8, community NVFP4 |
| Wan 2.2 T2V A14B (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) | same | two 14B experts (high noise and low noise, `boundary_ratio` 0.875): 40 blocks, 40 x 128 | Wan 2.1 VAE | UniPC | Apache 2.0 | Intel AutoRound NVFP4, GGUF |
| HunyuanVideo (`hunyuanvideo-community/HunyuanVideo`) | `LlamaModel` (8B, hidden 4096) and CLIP-L | 13B: 20 double, 40 single, 2 refiner, guidance embedding | `AutoencoderKLHunyuanVideo`, causal 3D | flow Euler shift 7 | Tencent community licence, excludes the EU, UK, South Korea | city96 GGUF |
| HunyuanVideo 1.5 (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`) | Qwen2.5-VL-7B text and ByT5 | 8.3B: 54 layers, 16 x 128 | 16x spatial, 4x temporal, 32 channels | flow Euler shift 5, CFG | same territory exclusion | jayn7 GGUF |
| CogVideoX 2B (`zai-org/CogVideoX-2b`) | T5-XXL | 1.7B: 30 layers, 30 x 64, sinusoidal positions | `AutoencoderKLCogVideoX`, 3D | `CogVideoXDDIMScheduler`, v prediction | Apache 2.0 (the 5B is not) | few |
| LTX-Video 2B (`Lightricks/LTX-Video`) | T5-XXL | `LTXVideoTransformer3DModel` 1.9B: 28 layers, 32 x 64, cross attention | 419M, 32x spatial, 8x temporal, 128 latent channels | flow Euler, dynamic shift | LTX open weights licence | city96 GGUF |

### Recommendation

**Image: FLUX.2 klein 4B first.** It needs the smallest set of new pieces and those pieces serve the most models:

- Its text encoder is a Qwen3, a family specrt already runs. Only the output changes: three intermediate residual streams instead of logits. `tests/diffusion.rs` proves the tiny copy's encoder config is accepted by `Spec::from_hf_config` today. FLUX.1 and SD 3.5 need T5 and CLIP (two new families); Qwen-Image needs the Qwen2.5-VL text model.
- Its denoiser is the MMDiT double stream block plus the single stream parallel block. That pair is FLUX.1, FLUX.2 dev and klein 9B, HunyuanImage 2.1 and HunyuanVideo (plus refiner); Qwen-Image and SD 3.5 are the double stream block alone.
- Its VAE is the plain 2D `AutoencoderKL` decoder (3x3 and 1x1 conv, GroupNorm, SiLU, nearest upsample, one mid attention) that FLUX.1, SD 3.5 and CogView4 also use, plus a BatchNorm denormalise and a 2x2 unpatchify (elementwise and layout).
- Apache 2.0 and not gated: the real weights can be downloaded by tests and run by users.
- Distilled to 4 steps with no CFG, so the first end to end run is short; the base variant on the same code adds CFG.
- BFL ships FP8 and NVFP4 copies (the NVFP4 tile GEMM exists) and GGUF copies exist (the ggml path exists). bf16 is 8 GB plus 7.8 GB, about 13 GB in use per the model card.
- The user already makes concept images with FLUX (the 3d-assets skill drives FLUX.1-dev through ComfyUI); klein is the local replacement.

**Video: Wan 2.1 T2V 1.3B first, then Wan 2.2 TI2V 5B.** Alibaba, Apache 2.0, the most downloaded diffusers video model, 1.4B denoiser. Its VAE config is the Qwen-Image VAE, so this slice also unlocks the Qwen-Image decoder. It is a different block family from MMDiT (self attention plus cross attention to text, adaLN from a learned table), so it got its own fixture. Its text encoder (UMT5) is a T5, the one new encoder family that also serves FLUX.1, SD 3.5, CogVideoX, LTX-Video and the ByT5 glyph encoder. HunyuanVideo and HunyuanImage are out: their licence does not apply in the EU, where the user is. CogVideoX uses an older DDIM v-prediction scheduler and LTX-Video a non-Apache licence.

## 2. diffusers

`diffusers` is not in `~/.venv-vllm` and not in a pip or uv cache. Installed into a private venv, never the shared one:

```bash
uv venv /tmp/diffusers-1281 -p 3.13
VIRTUAL_ENV=/tmp/diffusers-1281 uv pip install --index-url https://download.pytorch.org/whl/cpu torch
VIRTUAL_ENV=/tmp/diffusers-1281 uv pip install diffusers transformers safetensors accelerate sentencepiece protobuf numpy
```

Result: diffusers 0.43.0, torch 2.15.0+cpu, transformers 5.24.0. It has every pipeline above: `Flux2KleinPipeline`, `Flux2Pipeline`, `FluxPipeline`, `QwenImagePipeline`, `QwenImage21Pipeline`, `WanPipeline`, `HunyuanVideoPipeline`, `HunyuanVideo15Pipeline`, `HunyuanImagePipeline`, `CogView4Pipeline`, `CogVideoXPipeline`, `StableDiffusion3Pipeline`, `LTXPipeline`.

Pipeline facts the engine has to reproduce (from `pipelines/flux2/pipeline_flux2_klein.py` and `pipelines/wan/pipeline_wan.py`):

- FLUX.2 klein prompt: Qwen chat template (`enable_thinking=False`), padded to 512 with an attention mask. `hidden_states[k]` is the residual stream after k layers, before the final norm (checked on the fixture: equal to the layer output, 0.0 difference). The padded rows are kept and fed to the denoiser unmasked, and they depend on the mask (0.17 max change with the mask removed), so the encoder must run padded rows with pad keys masked.
- Sigmas: `linspace(1, 1/N, N)`, then `s' = e^mu / (e^mu + 1/s - 1)` with `mu` from `compute_empirical_mu(image_seq_len, N)` (two linear fits in the pipeline code, not in the config). Checked against the fixture: 1.0, 0.93401, 0.77966.
- Timestep into the model is `sigma`, scaled by 1000 inside. The update is `x + (s1 - s0) * v`; `tests/diffusion.rs` checks it reproduces the recorded latents to 1e-5.
- Final latent: unpack tokens to the 2x2 patched grid, `x * sqrt(var + eps) + mean` with the VAE BatchNorm stats, unpatchify, decode, `(x / 2 + 0.5).clamp(0, 1)`.
- Wan: prompt embeddings zeroed past the prompt length, CFG always two forwards, UniPC multistep keeps the last model outputs, latents denormalised with `latents_mean` and `latents_std`, VAE decodes one latent frame at a time with a per conv cache of 2 frames.

## 3. Fixtures

`crates/specrt/scripts/tiny_diffusion.py` builds both on CPU in 3 seconds, deterministic from a clean tree (reference md5s equal across runs):

- `tests/fixtures/tiny-flux2-klein` (1.27 MB): a real `Flux2KleinPipeline` folder (`model_index.json`, `scheduler/`, `text_encoder/` Qwen3 4 layers hidden 32, `transformer/` one double and one single block, hidden 32, 2 heads x 16, rope axes 4 x 4, `vae/` 2 levels 16 and 32 channels, 4 latent channels, GroupNorm 8). 32 x 32 image, 16 x 16 latent, 8 x 8 x 16 packed tokens (64 image tokens), 16 text tokens.
- `tests/fixtures/tiny-wan21` (1.30 MB): a real `WanPipeline` folder (UMT5 2 layers d 32, Wan DiT 2 blocks of 2 heads x 24, VAE base 8 with one temporal upsample), 5 frames of 32 x 32, latent 3 x 8 x 8.

References in `reference/` (f32 little endian with a JSON index of shapes): token ids and masks, text hidden states and embeddings for the prompt and a negative prompt, the initial noise, the velocity and the latent after each of 3 steps, the VAE input latent, the decoded image or frames, and the pipeline output. FLUX.2 also has a guided run (the base variant, CFG 4) and Wan records the conditional and unconditional velocities per step.

`tests/diffusion.rs` pins the component classes, configs, real tensor names and shapes, the reference shapes and finiteness, the Euler update between steps, the decode post-processing, size under 20 MB, and that every component config is still refused by `from_hf_config` and `from_hf_config_serving`. Diffusers folders have no top level `config.json`, so the fixture walkers in `golden.rs`, `plan_check.rs`, `compile_plan.rs`, `source_dump.rs` and `ui/src/detail.rs` skip them.

## 4. Gap list

Marks: **specrt** (exists in the engine: kernel or family named), **kir** (a kir primitive or description family exists, the kernel does not), **new**.

### Text encoder

| Piece | Needed for | Status |
|---|---|---|
| Qwen3 decoder run to selected layers, no lm_head, no sampling | FLUX.2 klein | **specrt**: `family-qwen3.toml`, the prefill plan. Missing: a plan output of the residual `x` after layers 9, 18, 27 (a `Copy` of `x` into a 3 slot output after those layers, then concat), prefill only, KV not kept |
| Padded prompt with masked pad keys | FLUX.2, Wan | **new** as a request shape: prefill with a key mask (pad rows still computed). The causal flash kernel already bounds keys per query; a per sequence key limit (`seg_hi`) is enough |
| Mistral 3 text model (FLUX.2 dev) | later | **specrt**: `family-llama.toml` matches `mistral` |
| Qwen2.5-VL and Qwen3-VL text (Qwen-Image, HunyuanImage, Qwen-Image 2.1) | later | **new** family files over existing pieces (mrope exists for qwen3.5) |
| T5 / UMT5 encoder: bidirectional attention with relative position buckets, no rope, gated GELU, RMS norm without mean | Wan, FLUX.1, SD 3.5, LTX, CogVideoX | **new** family; bidirectional exists (`AttentionShape.bidirectional`, `kir/src/attention.rs:55`), relative bias buckets are new (an additive bias per head from a small table: a `Compose` gather into the scores) |
| CLIP text tower with pooled output | FLUX.1, SD 3.5, HunyuanVideo | **new** family; LayerNorm and attention exist in the vision tower (`vis_layernorm`, `vis_flash`) |

### Denoiser (FLUX.2 MMDiT first, Wan DiT second)

| Piece | FLUX.2 | Wan | Status |
|---|---|---|---|
| Patchify / unpatchify | 2x2 in the pipeline (latent 32 ch to 128) and a `Linear` `x_embedder` | `Conv3d` (1, 2, 2) stride = kernel, which is a linear over the patch | **new** op, pure index map; the linear is the existing tile GEMM |
| Timestep embedding: sinusoid, then 2 linears with SiLU | `time_guidance_embed` (no guidance embed in klein) | `condition_embedder.time_embedder` then `time_proj` to 6 x hidden | **new** sinusoid op; the linears are decode-shaped (1 or 2 rows): **specrt** GEMV |
| Text projection | `context_embedder` linear | `text_embedder` linear, GELU-tanh, linear | **specrt** tile GEMM, GELU-tanh map epilogue exists in `expr.rs:82` |
| adaLN modulation | shared `Flux2Modulation` per stream: SiLU, linear to 3 or 6 vectors (shift, scale, gate) | per block `scale_shift_table` plus the shared `time_proj` output | **new** op: LayerNorm without affine, then `(1 + scale) x + shift`. LayerNorm exists in the vision family (`vis_layernorm`) |
| Gated residual add `x + gate * y` | both streams, attention and MLP | self attention and FFN | **new** epilogue stage list (Residual is `acc + r` today) |
| Joint attention over text and image tokens | double stream: separate qkv weights per stream, one attention over the concatenation; single stream: one fused `to_qkv_mlp_proj` | self attention over video tokens | **kir**: non-causal flash with segment bounds exists as `vis_flash` (`kir/src/cuda/vision/kernels.rs:175`) |
| Cross attention to text | none | `attn2`, K and V from 512 text rows | **new** variant: q and kv from different buffers |
| qk norm | per head RMS norm (`norm_q`, `norm_k`, `norm_added_q`, `norm_added_k`) | RMS norm across all heads (`rms_norm_across_heads`) | **specrt**: per head in `qk_norm_rope`, full width in `rmsnorm_segment` (#638) |
| Rope | 4 axes (t, h, w, l) of 32, theta 2000, interleaved real pairs, text ids on the l axis | 3 axes 44 / 42 / 42 of 128 | **kir**: rope with per segment angles exists (`span_rope`, mrope sections); the position ids per axis are a new input |
| Parallel block (single stream) | fused qkv plus SwiGLU input projection, concat attention and MLP into one `to_out` | none | **specrt** tile GEMM plus a column split epilogue (**new** rule, see the compiler plan) |
| MLP | SwiGLU 3.0 | GELU-tanh | **specrt**: Act and Map epilogues |
| Output | `AdaLayerNormContinuous` then `proj_out` | table modulation then `proj_out` | same pieces as above |

### Scheduler as data

A `[pipeline]` section in the family TOML, read from the diffusers scheduler config through `[fields]` aliases like any other family field, with the update rule as a kir `Expr` so it compiles to an elementwise kernel (and can fuse into the last GEMM's epilogue):

```toml
[pipeline]
steps = 4
guidance = 1.0
cfg = "none"
sigmas = "linspace(1, 1 / steps, steps)"
shift = "exp(mu) / (exp(mu) + 1 / s - 1)"
mu = "empirical(image_tokens, steps)"
timestep = "s"
update = "x + (s1 - s0) * v"
latent_in = "x * sqrt(bn_var + bn_eps) + bn_mean"

[pipeline.cfg_rule]
combine = "u + g * (c - u)"
```

| Item | FLUX.2 klein | Wan 2.1 | Status |
|---|---|---|---|
| Sigma schedule and shift | linspace, exponential shift with empirical mu | `use_flow_sigmas`, static flow shift 3 | **new**: host side, a few expressions over the step index |
| Update rule | flow Euler, one model output | UniPC `bh2`, order 2, predictor plus corrector, keeps the last 2 model outputs | Euler: **kir** (`Compose` with an `Epilogue` stage). UniPC: **new** state of 2 latents plus per step coefficients computed on the host |
| CFG | none (distilled); base: 2 forwards, `u + g (c - u)` | 2 forwards, guidance 5 | **new**: batch cond and uncond as 2 sequences of one forward |
| Seeded noise | `torch.randn` on CPU with a seeded generator | same | **new**: the engine cannot reproduce torch's CPU generator. Generate on the host from a documented generator (a counter based Philox or the existing sampler RNG) and have the reference script load the exported noise; the fixtures already record the noise they used |

### VAE

| Piece | FLUX.2 (2D) | Wan (3D causal) | Status |
|---|---|---|---|
| Conv 3x3, stride 1, same padding | every resnet, up block | Conv3d 3x3x3 with 2 frames of causal padding | **new** heavy family (implicit GEMM) |
| Conv 1x1 | `post_quant_conv`, `conv_shortcut` | same, 3D | **specrt**: it is the tile GEMM over pixels |
| GroupNorm 32 | every resnet | none | **kir**: `group_norm` exists as a row grouped norm (`kir/src/recur.rs:269`, used by `mamba2_gated_norm`), it normalises a group within a row; VAE GroupNorm reduces a channel group over all pixels: **new** grid reduction |
| Channel RMS norm per pixel | none | `WanRMS_norm` (`F.normalize` over channels, times `sqrt(C) * gamma`) | **specrt**-like: in NHWC it is a row RMS norm with a scale |
| SiLU | after each norm | same | **specrt** |
| Upsample nearest x2 | `Upsample2D` then conv | spatial nearest then conv; temporal: causal time conv to 2C then interleave | **new** op, folds into the next conv's input addressing |
| Mid attention, single head | head_dim 512 over 16384 tokens at 1024 px | head_dim 384, per frame | **new** shape: above `vision::MAX_HEAD_DIM` 128 |
| BatchNorm denormalise and 2x2 unpatchify | yes | latent mean and std instead | **new** elementwise and index map |
| Tiled decode | spatial tiles with overlap and blend | temporal chunks with the causal cache, then spatial tiles | **new** runtime loop over the plan |

## 5. Memory per stage

FLUX.2 klein 4B at 1024 x 1024: latent 128 x 128 x 32, packed 64 x 64 x 128, so 4096 image tokens plus 512 text tokens = 4608 joint tokens, hidden 3072.

| Stage | Weights | Activations | After the stage |
|---|---|---|---|
| Text encoder | 8.0 GB bf16 (NVFP4 Qwen3-4B about 2.6 GB) | 512 tokens: about 75 MB of KV for the single prefill, under 50 MB of activations; output 512 x 7680 bf16 = 7.9 MB | weights evicted to host or dropped; the embeddings stay |
| Denoiser, per step | 7.75 GB bf16, 4.07 GB FP8, about 2.3 GB NVFP4 | single block peak: the `to_qkv_mlp_proj` output 4608 x 27648 bf16 = 255 MB, residual f32 57 MB, the attention and MLP concat 113 MB, about 0.5 GB live; CFG (base variant) doubles it. About 35 TFLOP per step | resident for all steps |
| VAE decode | 0.17 GB | the 256 channel tensors at 1024 x 1024 are 537 MB each in bf16, about 3 live in a resnet: about 1.6 GB (3.2 GB if run in f32, `force_upcast`); about 10 TFLOP | freed; 512 x 512 tiles cut it 4x |

Peak with everything resident: about 18 GB. With the text encoder out of the way after encoding: about 10 GB.

Wan 2.1 1.3B, 5 seconds of 480p (832 x 480, 81 frames at 16 fps): latent 21 x 60 x 104 x 16, patch (1, 2, 2) gives 21 x 30 x 52 = 32760 video tokens, hidden 1536.

| Stage | Weights | Activations | After the stage |
|---|---|---|---|
| Text encoder (umt5-xxl) | 11.4 GB bf16, the largest piece | 512 tokens, small; output 512 x 4096 | evict; text K and V of every cross attention layer are step invariant, precompute once (30 x 512 x 2 x 1536 bf16 = 94 MB) |
| Denoiser, per forward | 2.8 GB bf16 | residual f32 201 MB, qkv 302 MB, FFN 587 MB: about 1.2 GB live per block, 2.4 GB with CFG batched. 283 TFLOP per forward, 71% of it self attention (the S squared term); 50 steps x 2 = 28 PFLOP | resident |
| VAE decode | 0.25 GB | whole clip at once: 6.2 GB per full resolution 96 channel tensor, not feasible. Per latent frame chunk (4 output frames): 307 MB per tensor plus the causal caches (2 frames per conv, about 1.2 GB at full resolution): about 2 to 3 GB | freed |

Offloading per stage: the text encoder runs once per request, so its weights can live on the host (or NVMe) and be streamed layer by layer with the existing offload path (`docs/specrt-offload.md`) at little cost. The denoiser's weights are read every step: streaming them costs the weight bytes per step (7.75 GB over PCIe 5 at about 50 GB/s is 155 ms per step against about 90 ms of compute), so it is the fallback for models bigger than VRAM (Qwen-Image 41 GB, Wan 2.2 A14B), overlapped block by block. The VAE is activation bound, not weight bound: tile it in space (image) and chunk it in time (video).

## 6. Compiler plan

Diffusion is built on the kir compiler (`docs/specrt-kir-compiler.md`), not beside it. The compiler was written for decode, which is memory bound (bytes plus launches). A DiT step and a VAE are compute bound at these sizes: a 4608 x 3072 x 3072 GEMM is 87 GFLOP over about 75 MB, about 1160 flops per byte, far over the ridge of any of our GPUs. What carries over and what is new:

### 6.1 Plan IR

A denoiser forward becomes a `Plan` (`crates/specrt/src/plan.rs:487`) with the same sections and no KV or token loop:

- `head`: patchify, timestep sinusoid, the time and modulation linears (1 row), `x_embedder`, `context_embedder`, rope tables.
- `blocks`: one `PlanBlock` per block kind (`double`, `single` for FLUX.2; `dit` for Wan), keyed by name like the layer kinds of a hybrid decoder (`Plan::layer_of`, `plan.rs:506`).
- `tail`: final modulation, `proj_out`, unpatchify, the scheduler update.
- `mtp`, `mtp_head`: empty.

It comes from a generic `graphs/denoiser.toml` (and `graphs/vae.toml`) through the same `Graph` lowering as `graphs/decoder.toml`, with a new phase `Phase::Forward` (no cache rows). `Count` (`signature.rs:73`) gains `image`, `text` and per VAE level `grid` sizes with zero defaults, so every LLM count is unchanged. New `Rows`: `Image`, `Text`, `Joint` (image plus text, one buffer with the text rows first so concat and split are aliases, as `gemv_layout` already does for `Split`), `Grid { level }`.

New `Op` kinds and their rows in the signature table (`Signer::signature`, `signature.rs:394`):

| Op | Class | Reads | Writes |
|---|---|---|---|
| `Patchify { p }` | Gather | `Input("latent")` Grid(0) x C | `Local("tokens")` Image x C p p |
| `Unpatchify { p }` | Gather | `Buf(Out)` Image x C p p | `Local("latent")` Grid(0) x C |
| `TimestepEmbed` | Elementwise | `Input("timestep")` Seqs x 1 | `Local("tproj")` Seqs x freq |
| `Modulate { stream, set }` (adaLN) | RowReduce | `Buf(X)` Image or Text x h, `Local("mod")` Seqs x 3h | `Buf(XnPlanes)` same rows x h |
| `GatedAdd { from, set }` | Elementwise | `Buf(from)` rows x h, `Local("mod")` gate slice, `Buf(X)` | `Buf(X)` |
| `AxisRope { axes }` (with qk norm, no KV write) | Elementwise | `Buf(Qkv)` Joint x 3h, `Input("ids")` Joint x axes | `Buf(Qkv)` |
| `JointAttention` | Attention | `Buf(Qkv)` Joint x 3h | `Buf(AttnPlanes)` Joint x h |
| `CrossAttention` | Attention | `Buf(Qkv)` Image x h, `Local("text kv")` Text x 2h | `Buf(AttnPlanes)` Image x h |
| `SchedulerStep { rule }` | Elementwise | `Local("latent")` Image x C, the velocity (2 row ranges with CFG), `Input("sigmas")` | `Local("latent")` |
| `Conv { taps, stride, causal }` | Gemm | Grid(l) x Cin, `Weight` Cout x taps Cin, with `causal` a `State("conv frames")` of 2 frames | Grid(l) x Cout |
| `GroupNormStats { groups }` | GridReduce (**new class**) | Grid(l) x C | `Local("group stats")` groups x 2 |
| `GroupNormApply { act }` | Elementwise | Grid(l) x C, the stats, weight, bias | Grid(l) x C |
| `Upsample { factor, temporal }` | Gather | Grid(l) x C | Grid(l + 1) x C |

`GridReduce` is new because a VAE GroupNorm reduces a channel group over every pixel, not one row: no single block owns the input. The fusion engine may attach its partial sums to a producer epilogue and its apply to a consumer prologue, never fuse it whole.

`validate_plan` (`signature.rs:543`) walks head, blocks and tail unchanged once the new ops have signatures; it must run on the denoiser plan and the VAE plan, reporting each like the decoder phases, and `inference-bench kernels` (`crates/cli/src/main.rs:284`, `kernels_command` at `:435`) gains `--plan --graph denoiser|vae --size 1024x1024 [--frames N]` to print the same per section bytes, launches and (new) flops.

### 6.2 Passes that apply unchanged

| Pass | Applies to diffusion as | Existing legality and cost code |
|---|---|---|
| G1 canonicalise | fold the `1 +` of `(1 + scale)` into the modulation linear's bias at load (as `unit_offset` is folded for norms); drop the absent guidance embedder; dead rows: the text rows of the last single block are discarded (`hidden_states[:, num_txt_tokens:]`), so its `to_out` runs on image rows only | needs one rule: dead row narrowing (shrink an op's rows when only a row range of its output is read) |
| G2 legalise | `Op::kernels` (`plan.rs:411`) gets entries for the new ops; WGSL refuses `Conv` and the attention variants by name until emitted | accepts as is |
| G3 layout assign | the joint buffer layout (text rows then image rows), NHWC grids for the conv family (K is taps x Cin contiguous), planes or single bf16 for GEMM inputs | accepts; the NHWC choice is a new layout rule |
| G4 epilogue attach | gated residual `x + gate * acc` into `to_out` and `linear_out`; SwiGLU into `linear_in` (Act epilogue exists); GELU-tanh into Wan's FFN up (Map epilogue exists); bias, residual and the GroupNorm partial sums into a conv; the Euler update into `proj_out` when there is no CFG | accepts the elementwise ones: the consumer reads only the producer's output plus a per sequence vector, and `Owner::Tile` (`fusion.rs:58`) gives every element to its owner. Needs: a spelled stage list for the gated residual (`kir/src/heavy.rs:161`, `spelled`), a column split rule for the single block's fused projection (legal when the tile width divides the 3h boundary), and a `BlockSum` stage in a conv epilogue for the group partial sums |
| G6 light group | LayerNorm plus `(1 + scale) x + shift` plus plane split into one `Compose`; qk norm plus axis rope (row 6 of the hand list without the KV write) | accepts: rows match and a row kernel owns the whole row |

### 6.3 Passes that do not apply unchanged

G5 prologue attach stays off for compute bound heavy ops. It was proven on decode GEMVs, where the norm prologue saves a memory pass in a memory bound kernel. On a tile GEMM the norm needs the row statistics before the first k tile, so every column block re-reads its rows and normalises into shared memory on the critical load path. The saving is the standalone modulate kernel (about 35 to 50 us at 4608 x 3072) against a GEMM of about 0.2 ms; whether the GEMM slows by less than that is unmeasured. `prologue_legal` (`fusion.rs:683`) gets a refusal `Reason::ComputeBound` unless `[runtime] compute_prologue = true`, which a later measurement may turn on. Folding the per step modulation into the weights (`W diag(1 + s)`) is rejected by the same cost model: it rewrites 56 MB of weights per layer per step to save a 28 MB activation pass.

Cost model. `Costs` (`fusion.rs:19`, `for_target` at `:34`) prices a group in bytes. Add a compute term:

```
time(g) = max(bytes(g) / bandwidth, flops(g) / peak) + launches(g) * launch
saving(P, C) = time(P) + time(C) - time(P ∪ C)
```

`flops` comes from the signatures (GEMM `2 m n k`, attention `4 Sq Sk d` per head set, conv `2 pixels Cout Cin taps`), `peak` is a per target tensor core rate measured once by a probe next to the bandwidth probe (`kir::bandwidth`) and cached in `[runtime]`. Per group the max picks the term: below the ridge `peak / bandwidth` flops per byte the formula is exactly today's bytes plus launches, above it the flops dominate and a fusion is worth the time of the light kernel it removes plus its launch. The text encoder run is the decoder graph's prefill plan, compiled with today's `Costs::for_target` (no `peak`), so its decisions are bit for bit today's; only the denoiser and VAE plans are compiled with `Costs::for_target(t).with_compute(peak)`. The proof is that `fusion_plan.rs` and `compile_plan.rs` decisions do not change.

A new pass outside the decode list: **H1 step invariant hoist**. Ops whose inputs do not depend on the latent or the timestep run once per request, not once per step: `context_embedder` (FLUX), the text K and V of every Wan cross attention, the rope tables. Signature reads already say this (no read of `Local("latent")` or `Input("timestep")` on any path).

### 6.4 Heavy kernel IR

`Heavy { family, tile, prologue, epilogue }` (`kir/src/heavy.rs:49`, `Family` at `:23`) gains:

- **Attention**: reuse the vision family (`kir/src/vision.rs`, `vis_flash`), not the decoder attention family, which reads a paged KV cache. `vis_flash` is already non-causal with per query key bounds (`seg_lo`, `seg_hi`); joint attention is one segment `[0, S)`. Budget at `AT_BQ` 128 query rows (8 warps x 16), `AT_BK` 64 keys: shared memory `2 * 128 * (hd + 8) * 2` (`vision.rs:58`) is 69,632 B at head_dim 128 and 36,864 B at 64, under the 99 KB sm_120 block limit; registers per thread are the O accumulator `16 hd / 32` (64 at 128), the score tile 32 and the Q fragments 32, about 130 to 170 at head_dim 128, under 255, one 256 thread block per SM. Sequence length (4096 to 16384, 32760 for 5 s of Wan) only sets the key loop count: K and V stream in 64 key tiles, so shared memory does not grow; a 4608 token step is 36 x 24 = 864 blocks. Needed: a second key and value source for cross attention, and a head_dim 384 and 512 shape for the VAE mid attention (`MAX_HEAD_DIM` is 128): at 512, `BQ` 16 and `BK` 32 with K and V sharing one buffer and O in shared memory is 16 KB Q, 32 KB K or V, 32 KB O f32 = 80 KB.
- **Tile GEMM**: reuse `Family::Tile`, `Fp8Act`, `Fp4Act` as is for every linear (rows are tokens; BFL ships FP8 and NVFP4).
- **Conv** (new): implicit GEMM over NHWC with M = output pixels (x frames), N = Cout, K = taps x Cin. Weights permuted at load to `[Cout][kt][kh][kw][Cin]`. The A tile is gathered by a `Layout` (`kir/src/layout.rs`) composing the grid layout with the tap offsets and zero fill at the border (`cp_async16_zfill`, as `vis_flash` loads); nearest upsample folds into that layout as a halving of the input coordinate. Tile choice formula, no tables: 1x1 is the plain tile GEMM; 3x3 takes `bm` = 128 output pixels as an 8 x 16 patch (the halo stays in L1), `bn = min(128, round_up(Cout, 32))`, `bk` = 32 input channels per tap, stages from the shared memory budget as K1 does. 3D causal adds `kt = 3` over frames t - 2, t - 1, t with the two earlier frames read from a `State` cache per conv layer, like the delta-net conv state; Wan's `time_conv` (3, 1, 1) is the same with a 1x1 spatial kernel. The 4 channel input and 3 channel output convs are memory bound: the roofline sends them to a direct row kernel.

No kernel text is written by hand: each family is emitted by kir and `every_kernel_comes_from_kir` (`crates/specrt/tests/codegen.rs:302`) and `pass_and_cost_files_name_no_architecture` (`tests/no_arch_names.rs:115`) apply to every file added.

### 6.5 Proof plan per slice

- Rediscovery: given the unfused plan, the passes must find what a hand design would. Denoiser: modulate as one light group; gated residual into `to_out` and `linear_out`; SwiGLU into `linear_in`; qk norm with axis rope; the Euler update into `proj_out` without CFG. VAE: conv, then GroupNorm partial sums in its epilogue, then normalise and SiLU in the next conv's prologue (the conv, norm, act fusion of the TensorRT style plugins); residual add into `conv2`; nearest upsample into the next conv's load. Each as a `fusion_plan.rs` style decision test naming the groups.
- Identity: each pass against the unfused plan on `tiny-flux2-klein` and `tiny-wan21` on GPU1. Bit identical in f32 where the arithmetic order is unchanged (every elementwise epilogue: `x + g * acc` is computed in the same order); stated tolerance where it is not (group statistics from tile partial sums change the summation order: 1e-5 relative on the decoded image).
- Counts: `inference-bench kernels --plan --graph denoiser --compiled --fusion` and `--graph vae` launch, bytes and flops totals before and after each pass, pasted in the slice report.

### 6.6 Memory planning

The arena of a denoiser forward or a VAE tile is a plan query, not a constant. Liveness is already computed from the signatures for fusion (`fusion.rs:485`, `values` with `live`). Add `signature::arena(plan, spec, count) -> Arena { peak, at, buffers }`: walk the fused plan in schedule order, a tensor is live from its writer to its last reader, the peak is the largest sum of live bytes, and a `memory_plan` pass after `fuse` assigns offsets by interval colouring so the runtime allocates one arena. The server calls it before accepting a request: weights resident plus `arena(count(request))` against free memory. If the VAE does not fit, pick the largest tile whose arena fits (the arena grows with tile area, so a search over tile sizes in steps of 8 latent pixels); if the denoiser does not fit, run CFG as two sequential forwards (half the arena) and otherwise refuse with the bytes named, before anything is launched.

### Passes to queue, in order

1. Diffusion signatures, `Rows` and `Count` fields, `graphs/denoiser.toml` and `graphs/vae.toml` lowering, `validate_plan` on both, `kernels --plan --graph`. CPU only. Done when both fixture plans validate and the report prints.
2. Compute aware cost term with the peak probe. Done when every decoder decision in `fusion_plan.rs` and `compile_plan.rs` is unchanged and the denoiser dry run picks the groups of 6.5.
3. Gated residual and column split epilogue rules, dead row narrowing (G1), step invariant hoist (H1). Done by rediscovery and launch counts.
4. Conv heavy family with the upsample layout fold, GroupNorm stats epilogue and apply prologue. Done by rediscovery, identity on the tiny VAE, exact conv tests against numpy.
5. Vision attention family: separate key and value source, head_dim 384 and 512 shape. Done by identity against the diffusers attention outputs.
6. `arena` plan query and `memory_plan` offsets. Done when the arena reported for the tiny fixtures equals the runtime's allocation and a request over the budget is refused or tiled before launch.

### Denoiser on the engine (#1282)

What landed, in the order the compiler sees it:

- `graphs/denoiser.toml`: the generic denoiser graph. `[fields]` bind the diffusers `transformer`, `vae` and `pipeline` configs, `[symbols]` derive the widths (`hidden`, `mlp`, `fused`, `cat`, the modulation set widths), `[[buffer]]` declares every activation with its rows (`image`, `text`, `joint`, `seqs`) and form. `[[head]]`, `[[block]]` (`double`, `single`, with their `count` field and tensor `prefix`) and `[[tail]]` lower to a `Plan` with `Phase::Forward` and one `Op::Dit(DitOp)` per node. Nothing in Rust names the architecture; `DenoiserSpec::from_pipeline` reads a diffusers folder.
- Ops: `Patchify`, `Unpatchify`, `TimestepEmbed`, `Split` (with SiLU), `Copy`, `Linear` (rows per stream, `+` joins weights, `[a:b]` slices weight columns, `add` accumulates), `Modulate { set, shift, scale }`, `GatedAdd { set, gate }`, `Glu { offset }`, `AxisRope`, `JointAttention`. `denoiser::Signer` signs them over `Rows::Image | Text | Joint` and `Count { image, text }` (zero for every decoder count). `denoiser::validate` walks the plan and checks every read against what was written, per row stream.
- Kernels: compose descriptions in `specs/` (`dit_modulate` is LayerNorm plus `(1 + scale) x + shift` plus the plane split in one kernel with two block sums; `dit_gated_add`, `dit_glu`, `dit_split`, `dit_silu_split`, `dit_copy`, `dit_timestep`, `dit_patchify`, `dit_unpatchify`). Compose gained several block sums, an `index` stage, a bf16 `split_store`, row strides and offsets, nested layouts with symbol sizes, and `sin`/`cos`. The tile GEMM is `vis_mma`, attention is `vis_flash` with one segment `[0, S)`, and qk norm plus the 4 axis table rope is `vis_axis_rope` in the vision family (`VisionKernel::joint`).
- Runtime: `denoiser_cuda::Denoiser` with `load`, `arena` (its own buffers, no KV state), `prepare` (runs the step invariant head ops once) and `forward(latent, sigma)`. `denoiser_host::HostDenoiser` runs the same plan on the CPU in f32, which is how the graph, the spec binding, the modulation order and the rope were proven before any kernel ran.
- H1: `denoiser::hoisted` marks head ops whose reads do not depend on the latent or the timestep and whose writes no later section rewrites: `split text` and `context_embedder`. The copy of the context into `x` stays in the step because the blocks rewrite `x`.
- Cost model: `Costs::with_compute(tflops, gbs)` adds the ridge; `compute_bound` is false for every decoder unit (their flops are 0), so `fusion_plan.rs` and `compile_plan.rs` are unchanged. A prologue onto a compute bound GEMM is refused with `Reason::ComputeBound` unless `compute_prologue`.

Tolerances, from `scripts/denoiser_reference.py` (diffusers CPU f32, and the same model with its weights rounded to bf16): the host plan matches the f32 velocity to 5e-7 and every block output to 1e-4. With bf16 weights the reference itself moves 3e-3 from f32, so the engine (bf16 weights, f32 accumulation, split bf16 activations) is held to 1e-4 against the bf16 weight reference and 5e-3 against f32.

Counts for FLUX.2 klein 4B at 1024 x 1024 (4096 image tokens, 512 text), from `inference-bench kernels --spec crates/specrt/tests/fixtures/tiny-flux2-klein/klein-4b-transformer.json --plan --graph denoiser --fusion`:

| Section | Launches before | After fusion | What fused |
|---|---|---|---|
| head (x1) | 14 | 14 (12 per step after the hoist) | none |
| double (x5) | 20 | 14 | gated residual into `to_out.0`, `to_add_out`, `linear_out` (both streams); SwiGLU into `linear_in` (both streams) |
| single (x20) | 8 | 7 | both halves of `to_out` and the gated residual into one epilogue |
| tail (x1) | 3 | 3 | none |
| step | 277 | 227 | |

34.8 TFLOP and 48.7 GB read per step. Every modulate prologue (11 per step) is refused as compute bound. The fused groups are the dry run of the fusion engine; the runtime still launches the unfused plan, so the identity of each fused kernel against the unfused plan is the next step (queue item 3).

## 7. Slices

Filed as #1282 to #1286 by the supervisor; the order and the done test of each:

1. **#1281** (this): analysis, fixtures, `tests/diffusion.rs`. Done: the test passes on CPU.
2. **#1282 denoiser** (FLUX.2 double and single blocks on the engine). Compiler passes 1 to 3 and 5 (joint attention). Done: the velocity of each of the 3 recorded steps of `tiny-flux2-klein` from `step{i}.latent` inputs within 1e-4 of the reference on GPU1 under `gpu-run 1`, the rediscovery tests, and the launch counts pasted.
3. **#1283 VAE decoder** (2D first, then the 3D causal path for Wan). Compiler pass 4 and the head_dim 512 attention of pass 5, `arena` (pass 6) for tiling. Done: `vae.decoded` of both fixtures within 1e-4 max abs (PSNR over 60 dB), tiled against untiled within a stated seam tolerance, exact conv tests against numpy.
4. **#1284 text encoder and sampler loop**: Qwen3 hidden states of selected layers with masked pad keys, `[pipeline]` with flow Euler, CFG batched, host noise from a documented generator. Pass H1 and the Euler update epilogue. Done: `text.hidden` and `text.embeds` within 1e-4, then the whole 3 step run from the recorded noise to `image` within 1e-3, guided and unguided.
5. **#1285 serving**: family file for FLUX.2 klein, the routes below, the load and Generate pages, `arena` driving admission. Done: the tiny fixture served end to end through `POST /v1/images/generations` (byte compare of the PNG decoded pixels against `image` within one 8 bit level), the real klein 4B loaded through the load dialog in headless sway with `nvidia-smi` showing it on the chosen GPU and the screenshots at 1280 x 800 and 390 x 844 read.
6. **#1286 video**: UMT5 family, Wan DiT (cross attention, table modulation, 3 axis rope), UniPC as `[pipeline]` state, 3D causal VAE with the frame cache and temporal chunks, the video route. Done: `tiny-wan21` velocities, latents and `video` within the same tolerances, then Wan 2.1 1.3B at 480p on GPU1.

## 7a. VAE decoder (#1283, landed)

The VAE decoder is a plan, compiled by the same fusion engine as the decoder graph:

- `graphs/vae.toml` holds one decoder graph per family (`kl` for `AutoencoderKL` and `AutoencoderKLFlux2`, `causal3d` for `AutoencoderKLWan` and `AutoencoderKLQwenImage`), written as macros over the diffusers config fields (`block_out_channels`, `layers_per_block`, `dim_mult`, `temperal_downsample`). `specrt::vae::graph::plan` expands it into `VaeOp`s (`conv`, `group_norm`, `rms_norm`, `upsample`, `time_up`, `attention`, `add`, `clamp`) and checks every tensor shape against the checkpoint. A new VAE family is a graph entry, no Rust.
- Plan IR: `Rows::Grid(level)` and `Count::grids` size each resolution level; group norm lowers to a `GridReduce` stats unit and an elementwise apply unit; the fusion engine refuses to fuse a grid reduction with its reader (`Reason::GridBarrier`) and folds the nearest upsample only into an implicit GEMM load (`Reason::Layout`).
- Rediscovered by the pass on the plain plan, no hand arrangement: group norm apply (with SiLU) as the next conv's prologue, the next group norm's statistics in the conv epilogue, residual adds into `conv2` and `to_out`, the upsample into the next conv's load, the clamp into `conv_out`. Tiny KL VAE: 56 launches to 21. Tiny Wan VAE: 56 to 44 (its channel RMS norm is a row reduction over channels that a conv tile does not own).
- kir `conv`: `ConvKernel` is an implicit GEMM over NHWC (M output pixels, N Cout, K taps x Cin) with the tile from a formula: an 8 x 16 output patch (1 x 128 for 1x1), `bn = min(128, round_up(Cout, 32))`, `bk = min(32, Cin)` per tap, stages from shared memory (at most 2). Convs with under 8 input or output channels go to the direct family. The causal 3D variant reads the earlier frames from a per conv frame cache; `time_up` is the (3, 1, 1) causal conv to 2C with the interleave in the store.
- Mid attention is one head over the pixels of a frame, any head dim (384 and 512 included): one block per query row with the scores in shared memory, so the key count is bounded by shared memory, not the head dim.
- Memory: `vae::compile::arena` walks the fused schedule, takes the peak of live bytes and assigns offsets by interval colouring; `vae::tiled::choose` picks the largest tile (steps of 8 latent pixels) whose arena fits the budget, and `decode_tiled` blends overlapping tiles with linear ramps. Tiled against untiled on the tiny fixture (tile 12, overlap 8): 25.1 dB PSNR, against 18.1 dB for diffusers' own tiled decode at its tile 24.
- Proof: host decode against `vae.decoded` 2.8e-6 (KL) and 2.2e-7 (Wan) max abs; fused and unfused schedules bit identical on the host; every conv shape against numpy (`fixtures/conv-cases`). GPU tests in `tests/vae_gpu.rs`.
- Counts: `inference-bench kernels --plan --graph vae --spec <diffusers folder> --size 1024x1024 [--compiled] [--fusion]`.

## 7b. Text encoder and sampler loop (#1284, landed)

From a prompt to an image on the engine, with no per-architecture Rust:

- Pipeline as data: `graphs/pipeline.toml` has one entry per pipeline class (`flow-euler` for `Flux2KleinPipeline` with `FlowMatchEulerDiscreteScheduler`). `[fields]` bind the scheduler, VAE, text encoder and `model_index.json` configs; `text_layers` (`layers / 4, layers / 2, 3 layers / 4`, which is 9, 18, 27 for Qwen3-4B and 1, 2, 3 for the tiny copy), `max_length` 512, the default steps and guidance (distilled 4 and 1, base 50 and 4), the `cfg` condition, and the schedule as expressions: `mu` (the empirical fit from the pipeline code), the per step `sigma` (linspace with the exponential or static shift), the model timestep, and the step itself as kir `Expr`s: `update = x + (s1 - s0) * v`, `combine = u + g * (c - u)`, `latent_in` (BatchNorm denormalise) and `output` (`(x / 2 + 0.5)` clamped). `sampler::Pipeline` reads it; adding DDIM or UniPC is a new entry plus, for multistep rules, state inputs.
- Runtime options: `[pipeline.options] offload_text_encoder` (default on: the encoder runtime is dropped after the prompt is encoded, before the denoiser loads) and `offload_vae` (default off: the VAE stays loaded between requests). The denoiser is resident for every step. Both change no pixel (`sampler_gpu.rs`).
- Prompt: `sampler::PromptEncoder` renders the tokenizer's chat template (`enable_thinking` false), tokenises, truncates to `max_length` and pads with the config's `pad_token`, with the mask. Checked against transformers on the Qwen3 tokenizer.
- Text encoder: the existing Qwen3 family on the engine. `Runtime::hidden_states(ids, real, layers)` runs the decoder graph's prefill plan (today's compile, no lm_head, no final norm, KV not kept), copies `x` after each chosen layer and stops after the last one. Pad keys are masked with a per run key limit in `flash_mma` (`key_limit`, every other caller passes no limit), so padded rows are computed as diffusers computes them. `sampler::stack_layers` gives the `[tokens, layers x hidden]` embedding. The tiny fixture's encoder is 2 heads of 32 (the engine's flash kernel takes head dims that are multiples of 16), with bf16-exact weights.
- Step in the plan: `graphs/denoiser.toml` now runs on the pipeline's packed latent tokens (`latent` rows: one copy shared by both CFG sequences) and ends with a `[[step]]` node. `DenoiserGraph::sampling_plan(rule, guided)` lowers it to `DitOp::Step` with the rule, and its kernel is a compose description generated from the rule (`denoiser::step_description`). Without CFG the fusion pass attaches the step to `proj_out`'s epilogue (`tail`: 3 launches to 2); with CFG it refuses (`Reason::Rows`: the step reads both sequences' rows). The CUDA denoiser runs the fused form as `vis_mma_step`, the tile GEMM with the rule lowered into its store (`kir::cuda::emit_vision_step`); it is bit identical to the unfused step.
- CFG: two sequences of one forward. `Count::seqs` scales the image, text and joint rows; the arena holds both sequences, the attention is two segments, the axis rope tables repeat per sequence, and the shared latent is split once and copied. The guided step reads the conditional rows then the unconditional rows. Batched velocities equal single forwards (0 and 7e-7).
- Noise: the engine cannot reproduce torch's CPU generator, so the initial latent comes from `sampler::noise(seed, n)`: a splitmix64 counter (the VAE tests' generator) giving uniform pairs, then Box-Muller in f64, in channels first order. `scripts/tiny_diffusion.py` (`engine_noise`) generates the same numbers and feeds them to diffusers as `latents`; `sampler.rs` checks they match to 1e-6.
- H1: `context_embedder` and the text split stay hoisted out of the step loop; the rope tables are built once per arena.

Tolerances (tiny fixture, GPU1): text hidden states 7e-6, embeddings 9e-6 (bound 1e-4). Against a diffusers run with the transformer weights rounded to bf16 (the engine's weights): latents 6e-6 unguided and 1.7e-5 guided after 3 steps (bound 1e-4), decoded image 3e-6 and 1e-5 (bound 1e-3), PSNR 124 and 114 dB (bound 60). Against the f32 diffusers run the bf16 weights alone move the image 2e-3 (bound 2e-2). Arena: 327 KB unguided, 650 KB guided (bound 1 MB), VAE arena 263 KB.

## 7c. Serving (#1285, landed)

- Family file: `specs/family-flux2_klein.toml` matches `[match] pipelines = ["Flux2KleinPipeline"]` (spec format 18 added `pipelines` and the family `[pipeline]` table). `[pipeline]` names the kind (`image`), the pipeline graph entry (`flow-euler`), the components and which stay resident, the offered sizes and default, the samplers and `max_images`, and the text encoder family (`qwen3`). It composes the finished pieces only: `sampler::Pipeline`, `DenoiserSpec::from_pipeline` with the `graphs/denoiser.toml` sampling plan, `vae::graph::load_plan`, and the Qwen3 encoder from `Spec::from_hf_config`. `specrt::media::info` reads a folder into `MediaInfo` (kind, family, pixel step, sizes, defaults, component bytes); a folder whose class no family matches is refused naming the class.
- One job type for images and later video: `media::MediaJob { kind, prompt, negative_prompt, n, width, height, seed, steps, guidance, sampler, frames, preview_every, offload_text_encoder }` and `MediaOutput { images, frames, steps, step_secs, total_secs, admission }`. `MediaInfo::check` refuses a wrong kind, an empty prompt, `n` over `max_images`, a size off the pixel step (naming the nearest valid size), a size over 65535 latent tokens (one denoiser row launch; the real klein at 1024x1024 is 4096, the tiny fixture's pixel step of 4 makes 1024x1024 65536, so its offered sizes stop below that), steps, guidance or sampler out of range, before any launch. Image i of n uses `seed + i`.
- Admission (`media::admit`, 6.6): the denoiser weights (unless already resident) plus `denoiser_cuda::arena_bytes` for 1 or 2 sequences, and `vae::tiled::arena_bytes` for the latent, against free memory less 256 MB. If the VAE does not fit it takes the largest tile in steps of 8 latent pixels whose arena fits (overlap a quarter of the tile), and a tile whose mid attention would need more shared memory than the card has counts as not fitting (the tiny fixture at 720x1280 decodes in tiles of 152); if the guided denoiser does not fit it runs the two passes one after the other in one single-sequence arena (half the bytes; within 3.3e-6 of the batched image on the tiny fixture); otherwise the job is refused naming the MB of activations, weights and free memory. The chosen plan and bytes come back in `usage.admission`.
- Runtime: `generate::Generator::generate_with(req, plan, hooks)` reports each step to a hook that can stop the job (`generate::Cancelled`), keeps the compiled denoiser between requests of the same shape (`keep_denoiser`), and decodes tiled when the plan says so. `media::ImageModel` is what `native::specrt_factory` loads for a pipeline folder; it runs on the engine thread like a text model, and `Target::generate_media` / `Handle::generate_media` carry progress and the cancel flag across it.
- Server: `POST /v1/images/generations` (crates/server/src/images.rs) with the chat route's auth, request id, access log and metrics; its own gate of `images.jobs_per_gpu` running plus `images.queue_depth` waiting, a 429 with `Retry-After` past that; `stream: true` for `image_generation.progress`, `image_generation.partial_image`, `image_generation.completed`; a closed connection cancels between steps; `response_format: url` writes the PNG to `images.file_dir` and serves it from `GET /v1/images/files/{id}.png` for `images.file_ttl_secs`. `/v1/models` and `/v1/capabilities` carry `type` (serde default `text`, additive, no API bump).
- UI: an image model opens its own load page (size, steps, guidance and "release the text encoder after encoding" in the open, the GPU to load on (the selected card with the most free memory by default, sent as the `media.device` engine override and checked by the landing notice), the three components with their bytes and where each lives, the fit against that card's free memory). Once loaded the first rail tab reads `generate` and shows the Generate page: prompt, collapsed negative prompt, size presets, seed with dice and lock, steps, guidance, n, the step progress bar and stop, a result grid with save (to `<data dir>/images`), copy (clipboard image), reuse and click to enlarge, and the request table with steps per second and total time. At phone width the controls stack above a one column grid. Screenshots: `docs/shots/images/`.
- Coverage: `popular_fetch.py` reads a repo with `model_index.json` as a pipeline (component configs committed, size from the component safetensors); `popular_models.rs` classifies it through `media::family_for`. `black-forest-labs/FLUX.2-klein-4B` reads LOADS: 16.0 GB of weights (text encoder 8.0, denoiser and VAE 7.9), peak 16.0 GB on 1 card with the encoder released after encoding, 0 GB host RAM. FP8 and NVFP4 copies read UNKNOWN-QUANT until the denoiser reads those weights.

Plans for the real klein 4B at 1024x1024 (`inference-bench kernels --spec <klein 4B configs> --plan --graph denoiser --size 1024x1024 --fusion`, and `--graph vae --compiled`):

| Plan | Launches before | After fusion | Per pass |
|---|---|---|---|
| denoiser step (head x1, double x5, single x20, tail x1) | 276 | 227 (head 14, double 14, single 7, tail 2) | 34.8 TFLOP, 48.7 GB read, 29.0 GB written |
| VAE decode (kl) | 117 | 40 | 10.5 TFLOP, 32.2 GB read, arena peak 2.15 GB |

Every modulate and SwiGLU prologue onto a tile GEMM is refused as compute bound; the gated residuals fuse into `to_out`, `to_add_out` and `linear_out`, SwiGLU into `linear_in`, and both halves of the single block's `to_out` share one epilogue.

Real UI walk (headless sway, `inference-bench-ui` from this branch, the tiny fixture under a model folder, GPU1 lock held): Models, search `klein`, Load opens the image load page with GPU 1 picked (95.5 GB free), load model, the rail tab turns to `generate`, prompt typed, generate image: a 720x1280 image, 4 steps at 12.1 steps per second, 3.8 s with the compile, the notice reads "running on GPU 1 (558 MB)". `nvidia-smi --query-compute-apps` showed the app holding 786 MiB on GPU1. The app process also holds 233 MiB on GPU0 with no model loaded at all (same figure on a run that only started the app), which is its startup device probing, not the image model. Screenshots: `docs/shots/images/real-ui/`.

Not done here: the real klein 4B has not been loaded (it needs the 16 GB download and a supervisor GPU window); the Euler sampler is the only one. Partial images are a min-max view of the packed latent tokens (one pixel per latent token, a pixel step square of the image), not a VAE decode.

## 8. API shape

Images, OpenAI compatible, all additions optional with defaults:

```
POST /v1/images/generations
{
  "model": "flux2-klein-4b",
  "prompt": "a red fox, studio light",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json",
  "output_format": "png",
  "seed": 42,
  "negative_prompt": "",
  "steps": 4,
  "guidance": 1.0,
  "sampler": "euler",
  "stream": false
}
→ { "created": 1795525506, "data": [ { "b64_json": "...", "seed": 42 } ] }
```

- `response_format` `url` returns `/v1/images/files/{id}.png`, served from a short lived local folder (expiry from the server config) behind the same auth. `b64_json` is the default for a local server.
- `size` must be a multiple of the model's pixel step (16 for klein); others are refused with the nearest valid size in the message. Defaults come from `[pipeline]` in the family file.
- `stream: true` sends server sent events with OpenAI's names, `image_generation.partial_image` (optional low resolution preview every k steps, `partial_images` sets k as in OpenAI's API) and `image_generation.completed`, plus `image_generation.progress` `{ "step": i, "steps": n }`.
- One generation job per GPU with a bounded queue; a full queue answers 429 with `Retry-After`. A dropped client cancels the job between steps.
- `/v1/models` gains `"type": "text" | "image" | "video"` (serde default `text`, additive); `/v1/capabilities` lists the new routes and their fields. No API version bump: nothing existing changes.

Video, after OpenAI's asynchronous Videos API (jobs take minutes):

```
POST /v1/videos  { "model", "prompt", "seconds": "5", "size": "832x480",
                   "seed", "negative_prompt", "steps", "guidance", "fps", "sampler" }
→ { "id": "video_...", "object": "video", "status": "queued", "progress": 0 }
GET    /v1/videos/{id}            status and progress (0 to 100)
GET    /v1/videos/{id}/content    MP4
DELETE /v1/videos/{id}            cancel or delete
```

Both routes run on one generic generation job type in the server (queue, progress, cancel, metrics), not on the chat request path.

UI:

- Load page: an image or video model is its own kind. Its common options sit outside Advanced: default size, steps, guidance, "keep the text encoder on the host after encoding", "tiled VAE decode". The placement shows the three components separately (text encoder, denoiser, VAE) with where each goes.
- A Generate page instead of chat for these models: prompt box, negative prompt (collapsed), size presets, seed with a dice button and a lock, steps, guidance, n; a step progress bar with the optional preview; a result grid (click to enlarge, save, copy seed, reuse settings) and a history strip. At phone width the controls stack above the grid and the grid drops to one column.
- Server page and the landing page: the model row carries an `image` or `video` badge and a curl example for its route.
