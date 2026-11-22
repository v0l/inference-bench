# Image and video input in specrt

Issue #484. The Qwen3.5 checkpoint we serve has a vision tower. This document is the plan and the
state of each slice.

## Slices

1. Preprocessing (done): `crates/specrt/src/vision.rs`.
2. Vision encoder on the JIT path, built from `vision_config`.
3. LM integration: embedding splice, M-RoPE position ids, mixed prefill, media hashes in prefix keys.
4. Server and API: `image_url` and `video_url` parts, limits, URL fetch policy, usage, 400 without vision.
5. App: chat attachments, model detail vision tower, "vision: yes" on the Models page.
6. Bench: image and video workloads in `bench` and `prefixbench`.

## Preprocessing

Everything comes from `preprocessor_config.json` (images) and `video_preprocessor_config.json`
(video): `size.shortest_edge` and `size.longest_edge` are pixel bounds, `patch_size`,
`merge_size`, `temporal_patch_size`, `image_mean`, `image_std`, and for video optionally `fps`,
`min_frames`, `max_frames` (defaults 2, 4, 768 as in transformers).

- Resize target: `smart_resize`, multiples of `patch_size * merge_size`, kept inside the pixel bounds.
  Rounding is half to even, like Python.
- Resampling: bicubic with antialiasing on uint8, reimplemented to match the integer fixed point
  kernel of torchvision's CPU path (PIL's algorithm), so patches are bit-identical to transformers.
- Normalise `(v - mean * 255) / (std * 255)` in f32, then patchify to
  `(t * h * w, 3 * temporal_patch_size * patch_size^2)` in merge-block order. A still image repeats
  its frame `temporal_patch_size` times; grid is `(1, h, w)`.
- Images are decoded with the `image` crate (png, jpeg, webp, gif, bmp).
- Video is decoded by an `ffmpeg` and `ffprobe` subprocess. A Rust binding would link libav and make
  every build and every packaged binary depend on its dev headers; the subprocess needs only the
  `ffmpeg` binary at run time, and only for video. Only the sampled frames are decoded, through a
  `select` filter, as raw rgb24 over a pipe.
- Frame sampling: `int(total / fps * cfg.fps)` clamped to `[min_frames, max_frames]` and `total`,
  indices `linspace(0, total - 1, n).round()`.
- Prompt expansion: the template emits one `<|vision_start|><|image_pad|><|vision_end|>` per image
  and the same with `<|video_pad|>` per video. `expand_placeholders` replaces image slots by
  `grid_t * grid_h * grid_w / merge_size^2` pads, and a video slot by one
  `<T seconds><|vision_start|>pads<|vision_end|>` group per temporal patch, inside the template's own `<|vision_start|>` and `<|vision_end|>` (only the pad token is replaced, as in transformers and vLLM), with T the mean
  timestamp of the frames in the patch (transformers' Qwen3-VL prompt layout).

## Reference and tests

`crates/specrt/scripts/vision_reference.py <dir>` runs transformers' `Qwen2VLImageProcessor` and
`Qwen3VLVideoProcessor` on generated images and a generated ffmpeg video and saves the patch
tensors (f32 little endian) and grids. The outputs are committed in
`crates/specrt/tests/fixtures/vision`. `cargo test -p inference-bench-specrt --test vision` checks
bit equality for an exact-size image, a resized image, a large image with the real config bounds,
and two videos, plus the frame indices and the placeholder expansion.

## Security

URL fetching for `image_url` and `video_url` will be off by default (data URLs only), because
fetching a user-supplied URL from the server is an SSRF risk. It is enabled by a server config key.

## Server and API (slice 4, done)

- `crates/core/src/media.rs` extracts `image_url` and `video_url` content parts (vLLM shape: a
  `{url}` object or a bare string) into `ChatRequest::media`, each item with a SHA-256 content hash
  that the prefix cache will fold into block keys.
- Limits and policy come from the `[media]` table of the server config: `max_images` (8),
  `max_videos` (1), `max_image_bytes` (20 MiB), `max_video_bytes` (200 MiB), `allow_url_fetch`
  (false), `fetch_timeout_secs` (10).
- Only `data:...;base64,` URLs are accepted by default. Fetching `http(s)` URLs is an SSRF risk
  (the server would reach internal hosts on a client's behalf), so it is off until
  `media.allow_url_fetch = true`. Other schemes (`file:` and so on) are always rejected.
- A target reports `Capabilities::vision`. A request with media to a model whose target has no vision
  gets a 400 saying the config has no `vision_config`.
- Not yet: usage counting of vision tokens, and a `max_pixels` limit (both need the preprocessing
  to run in the server path, which lands with slice 3).

## Remaining

Slices 2, 3, 5 and 6 are tracked in follow-up issues. `specrt::vision::Vision::from_hf_config`
already reads `vision_config` and the four token ids from `config.json`; the encoder and the LM
splice build on it.

## Vision encoder (slice 2, done)

`crates/specrt/src/vision_encoder.rs` and `templates/vision.cu`, built from `vision_config` of the
checkpoint and JIT compiled like the other kernels. `vision_encoder::plan(cfg)` lists the ops
(patch embed with interpolated position embeddings, per block norm, qkv, 2D rope, per frame
attention, proj, norm, mlp up and down, then the merger and a deepstack merger after each block
named in `deepstack_visual_indexes`), so the engine tab can draw the tower from the same list.

- Weights are the checkpoint's `visual.*` tensors in bf16 (converted to f16 on the device in fast
  mode). The patch embed Conv3d is a GEMM over the flattened patch; position interpolation is
  align_corners bilinear, as in transformers.
- Attention is limited to one frame (`h * w` patches), which is how transformers builds `cu_seqlens`
  for images and videos.
- Kernels (`templates/vision.cu`): every linear is a tensor-core GEMM (`vis_mma`, 128x128 tiles,
  three stage cp.async pipeline, bias, gelu and the residual add in the epilogue). Layernorms write the
  next GEMM's input planes directly. Rope is applied while packing q, k and v per head into
  `vis_flash`, a flash attention with mma.sync, online softmax and a per patch frame segment, so
  several images and video frames go through one call without attending across each other. The
  position table interpolation (`vis_pos`) runs on the device from four taps per patch.
- Precision, `[runtime.vision] precision`:
  - `fast` (default): activations and weights in f16, f32 accumulate.
  - `exact`: each activation is split into a high and a low half (bf16 for the GEMMs against the
    bf16 weights, f16 for attention), which keeps the f32 transformers bound.
- Batching: `VisionEncoder::encode_batch` takes several images in one pass. The buffers for 4,800
  patches (one 960x1280 photo) are allocated at load, before the KV pool takes the free memory, so a
  normal request never allocates. A larger batch allocates its own.
- Bounds, `tests/vision_encoder.rs` (ignored, needs the 27B FP8 snapshot and a GPU), merged
  embeddings of four fixtures against `transformers` fp32 from `scripts/vision_encoder_reference.py`:

  | mode | rel rms | max abs / largest value | asserted |
  |---|---|---|---|
  | exact | 2.8e-5 to 4.2e-5 | at most 4.7e-5 | rms < 1e-4, max < 5e-5 |
  | fast | 4.6e-4 to 2.4e-3 | at most 4.4e-3 | rms < 5e-3, max < 1e-2 |

  The fast error is below the 1 to 6 percent embedding noise that left the LM's KL unchanged (see
  the table further down), and below the transformers bf16 tower (rel rms 1e-2 to 3e-2).
- Speed, 960x1280 photo (grid `[1, 80, 60]`, 4,800 patches), best of 5:

  | GPU | master (fp32 GEMM, scalar attention) | exact | fast |
  |---|---|---|---|
  | RTX PRO 6000 Max-Q (quiet GPU) | 4,516 ms | 96.8 ms | 54.0 ms |
  | GB10 (Spark, shared with its vLLM) | 53,456 ms | 503 ms | 442 ms |

## Serving (#498)

- Encoding runs off the engine thread. `serve::run` hands every media request to a
  `specrt-vision` thread that owns the `MediaEncoder`, which takes all the requests queued at that
  moment, plans them together and encodes their images in batches of up to `max_batch` images and
  4,800 patches. The engine keeps stepping decode and prefill of running sequences meanwhile, and
  submits a request with its finished plan. The encoder runs on its own CUDA stream.
- Cache: encoder outputs are kept in host memory by the SHA-256 of the media bytes, least recently
  used first out, bounded by `[runtime.vision] cache_mb` (1024 by default, about 40 classify photos; 0 turns it off). A retry
  of the same image is planned without touching the GPU. A 960x1280 photo is 1,200 rows of 5,120
  floats, 24.6 MB.
- A request cancelled while it waits for the encoder is dropped before or after encoding, never
  admitted.
- Tests: `serve::watchdog_tests::decode_keeps_stepping_while_an_image_encodes` (a fake encoder held
  until a 40 token text request has finished; with the old in-submit encoding it times out),
  `media_encode::tests` (cache hit, LRU bound, one encoder call for a burst, a bad request failing
  alone).
- Classify replay (#606 set: 29 property photos, mostly 960x1280, 16 output tokens, 8 concurrent
  clients, streamed; orcarouter/Qwen3.8-27B-Uncensored-NVFP4, 8 slots, 8 GB KV, max batched tokens
  2048; quiet RTX PRO 6000, `/mnt/data/tmp/inference-bench-498/replay.sh`):

  | run | req/min | TTFT p50 | TTFT p95 | over 90 s | encoder calls |
  |---|---|---|---|---|---|
  | master, first pass | 10.7 | 12.3 s | 35.3 s | 0 | 29 (265 s of encoding) |
  | master, same images again | 10.7 | 12.2 s | 36.7 s | 0 | 29 |
  | branch, first pass | 58.3 | 3.07 s | 6.14 s | 0 | 23 (about 1.6 s) |
  | branch, same images again | 59.6 | 2.82 s | 5.94 s | 0 | 0 |

  What is left of TTFT is LM prefill: each request is 956 to 1,826 prompt tokens, about 1,200 of
  them vision tokens, and the server prefills at about 1,500 tokens/s whether the images come
  from the cache or not, so 8 concurrent requests queue behind about 6 s of prefill. That is the
  prefill phase of the parity plan, not the encoder. Greedy answers on four classify photos are
  byte-identical to master with `precision = "fast"`.
- Against vLLM (`tests/vision_lm.rs`, references from `scripts/vllm_vision.py` on a vLLM started
  with `docs/qwen27-run/launch.sh kvauto` on port 8011, bf16 KV, no MTP), mean KL over 32 teacher
  forced tokens. `SPECRT_VISION_PRECISION=fast|exact` picks the encoder for the test:

  | case | fast | exact |
  |---|---|---|
  | image (house) | 0.0140, top-1 32/32 | 0.0139, 32/32 |
  | images3 | 0.0361, 32/32 | 0.0364, 32/32 |
  | video (6 frames) | 0.0432, 29/32 | 0.0432, 29/32 |
  | video10 | 0.0577, 30/32 | 0.0570, 30/32 |
  | 6 image population | 0.0455, 174/192 | 0.0454, 175/192 |
  | 6 video population | 0.0613, 170/192 | 0.0613, 171/192 |
  | text control (no media) | 0.0116, 188/192 | 0.0116, 188/192 |

  The fast encoder moves no case by more than 0.0007.

## LM integration (slice 3, done)

- `Spec.rope.mrope` (`section`, `interleaved`) is read from `rope_parameters` of the checkpoint
  config, and `Spec.vision` from `vision_config` plus the four token ids. Specs written as TOML
  (like `qwen3.8-27b.toml`) are completed from the snapshot's `config.json` at load
  (`engine::complete_spec`), so no spec file needs a vision section. A text model has neither and
  runs the old rope path (the kernel constants `MROPE*` compile it out).
- Position ids: `vision::plan_media` walks the templated ids. Text advances all three axes
  together; each run of image or video pad tokens gets `(t, h, w)` from its grid, offset by the
  text position before it, and the position after a frame advances by `max(h, w) / merge`, as in
  transformers' `get_rope_index`. Positions are a per slot device table. Tokens past the prompt
  (decode) use `index + delta`, with `delta = next_position - prompt_len`.
- Rope kernels (`qk_norm_rope`, `kv_norm_rope`, `flash_mma`) pick the axis for every frequency pair
  from `mrope_section` and `mrope_interleaved`.
- Splice: `Runtime::plan_media` encodes each item with the vision encoder; the `embed` op is followed
  by a `splice_rows` kernel that overwrites the placeholder rows of the chunk being prefilled, so
  mixed text and vision chunks work with chunked prefill and fused steps.
- Prefix cache: each placeholder token is keyed by a tag derived from the SHA-256 of the media
  bytes and the row inside the item (`Seq.keys`), so the same image hits the cache and a different
  image with the same size does not.
- Usage: `prompt_tokens_details.vision_tokens` counts the placeholder tokens. `[media] max_pixels`
  and `max_video_secs` limit the preprocessing; over-limit and undecodable media are 400s.

Tests (`tests/vision_lm.rs`, ignored, need the 27B and the vLLM output of
`scripts/vllm_vision.py` made through port 8001):

- Image: prompt ids equal vLLM's, greedy prefix 32/32, teacher forced top-1 32/32, mean KL 0.0058
  against vLLM's logprobs (NVFP4 spec, the checkpoint vLLM serves). Bound: top-1 at least 90%, KL
  at most 0.05, as for text.
- Video (3 s, 8 fps, 6 sampled frames, 240 vision tokens): prompt ids equal vLLM's, top-1 28/32,
  mean KL 0.062. Looser bound (top-1 80%, KL 0.1); the cause of the gap to the image case is
  open, see the follow-up issue.
- Greedy "what is in this image" on a drawn house with the FP8 spec: "a house in a field with a
  sun in the sky".

## App (slice 5, done)

- Chat: attach with the `attach` key (a file dialog through `rfd`), drag and drop onto the window,
  or paste an image from the clipboard. Pending attachments show a thumbnail (videos show their
  name) with a remove key; the key is disabled with a note on models without vision
  (`Loaded.vision`, from the engine's `Capabilities::vision`). Turns keep their attachments as
  base64 in the saved conversation and are sent as OpenAI `image_url` / `video_url` parts through
  the same path the server uses (`Engine::chat_media`). Screenshot: `docs/gui/chat-attachment.png`.
- Models page: a "vision: yes / no" chip from `vision_config` in the checkpoint config.
- Model detail: a "vision tower" section (blocks, hidden, heads, merge, parameters per part, and the
  op plan from `vision_encoder::plan`). The parameter count is asserted equal to the sum of the
  checkpoint's `visual.*` tensors. Screenshot: `docs/gui/detail-vision-tower.png`.

## Bench (slice 6, done)

`inference-bench loadtest --media images:N|video:FILE` sends streaming chat requests with data URL
media (noise images of 448x448 so every request is distinct, or the given video) to any
OpenAI-compatible server, and reports TTFT, inter token gaps and vision tokens
(`prompt_tokens_details.vision_tokens` from specrt, `multimodal_tokens` from vLLM).
`prefixbench <spec> media on 64 2 [video.mp4]` runs the scheduler directly: cold and warm passes
per case, so the warm pass shows the prefix-cache hit on the same media.

Measured on the 27B NVFP4 spec, 3 requests per case, one client, 64 output tokens, one at a time,
vLLM on port 8001 (read only) against specrt behind `inference-bench serve`:

| workload | vision tokens/req | vLLM TTFT p50 | specrt TTFT p50 |
|---|---|---|---|
| 1 image 448x448 | 196 | 0.106 s | 0.469 s |
| 3 images | 588 | 0.156 s | 1.046 s |
| 10 s video, 10 fps 320x240 | 800 (specrt) / 880 (vLLM) | 0.165 s | 1.221 s |

specrt TTFT includes the preprocessing and the vision encoder (the first f32 kernel set with a naive
GEMM, before #498), vLLM's includes its encoder cache. `prefixbench` (scheduler only, no HTTP): image_1 cold 815 ms, warm 436 ms with 208 of
211 tokens cached; images_3 2306 ms and 1153 ms; 3 s video 837 ms and 287 ms.
The frame count differs from vLLM on the 10 s clip (see the follow-up issue).

## Video against vLLM: what was checked (#496, #497)

Frame sampling (#497) is fixed in the sense that it was never different: `sample_indices` equals the
transformers and vLLM sampler (checked against `vllm.multimodal.video` and `Qwen3VLVideoProcessor`).
The 880 vs 800 gap was vLLM's `multimodal_tokens.video`, which counts the 10 timestamp text tokens
and 20 `vision_start`/`vision_end` tokens as video tokens; its prompt has exactly 800 video pad tokens
in 10 frames of 80, like ours (900 prompt tokens both sides, ids identical). The 10 s, 100 frame clip
is now a fixture (`clip10`) with reference indices, timestamps, grid and patches from
`vision_reference.py`, asserted in `tests/vision.rs`.

For the video logit gap every stage before the language model was compared with the reference and
is identical or within the encoder bound:

| stage | result |
|---|---|
| frames decoded by ffmpeg vs opencv (vLLM) | bit equal |
| frame indices, timestamps, grid, patch tensor vs transformers fed by vLLM's loader | bit equal |
| prompt ids vs vLLM | equal for every case |
| M-RoPE positions and delta vs transformers `get_rope_index` and vLLM `_get_mrope_input_positions` | equal, 0 of 284 and 900 positions differ |
| vision encoder vs transformers fp32 | rel rms 2e-6 to 8e-6 |
| LM with the transformers fp32 or bf16 embeddings instead of ours | same KL (0.0620, 0.0616) |
| LM with t axis variants (per frame, equal to h, plain positions) | KL 0.062 to 0.081, so positions are not the cause |
| noise of 1 to 6 percent rms added to the embeddings | KL unchanged |

Controls against the same vLLM server, 32 teacher forced tokens each: descriptive text prompts mean
KL 0.014, top-1 96%; long text (100 to 1700 tokens) KL 0.006 to 0.012; random token prompts (no media,
unusual hidden states) mean KL 0.042, top-1 94%, with single prompts at 0.0016 and 0.16. The media
populations (6 distinct images, 6 distinct videos, `real_27b_media_population_against_vllm`): images mean
KL 0.040, top-1 94%; videos mean KL 0.074, top-1 90%. vLLM here serves with an fp8 KV cache and
speculative MTP while specrt keeps bf16 planes, which is the one remaining difference that is not
in the media path, so the gap on inputs with unusual or repetitive hidden states is not attributable to
vision. Images keep the text gate (mean KL 0.05, top-1 90%); videos are gated at KL 0.09 and top-1 85%,
a bound set from this measurement and not the text bound.

Run against a vLLM with an unquantised KV cache and no MTP (`--kv-cache-dtype auto`, bf16 planes like
specrt, `docs/qwen27-run/launch.sh kvauto`, log in `docs/qwen27-run/vision-lm-kvauto.log`): the
image population falls to mean KL 0.033, top-1 93% (178/192) and the video population to mean KL
0.060, top-1 94% (181/192), from 0.074 and 90%. Single clips: 3 s video 0.108, 10 s video 0.048, 2
frames 0.057, 4 frames 0.066, 3 images 0.109. So the fp8 KV cache and MTP explain about a fifth of
the video gap, and the rest stays: video KL is still above the 0.05 text bound while the image
population is inside it. The video gate therefore stays at KL 0.09 and top-1 85%, and #496 stays
open for the remaining 0.01 to 0.06 on video. The text-only controls, the prompt logprob files and the position references are written by
`vllm_vision.py` and `vision_positions.py` (see below).

## Per layer comparison against transformers (#496)

Same ids, same embeddings, same M-RoPE positions (transformers' `get_rope_index`), 27B FP8 spec. Ours:
`real_27b_dump_layer_states` (`SPECRT_LAYER_DUMP=/tmp/x`, writes the residual stream after each layer).
Transformers: `scripts/vision_layers.py <snapshot> /tmp/x video image` (bf16 text model loaded straight to
the GPU, FP8 weights dequantised, one process after the other). Relative rms of the difference per layer,
for all tokens (media-only and text-only columns in the script output):

| layer | video (284 tok) | image (160 tok) | video, bf16 noise 0.5% on the embeddings, transformers vs itself |
|---|---|---|---|
| 0 | 0.0053 | 0.0047 | 0.0086 |
| 10 | 0.0072 | 0.0069 | |
| 20 | 0.0103 | 0.0104 | 0.0087 |
| 35 | 0.0184 | 0.0131 | 0.0200 |
| 41 | 0.0372 | 0.0223 | 0.0772 |
| 42 | 0.0619 | 0.0604 | 0.1036 |
| 54 | 0.166 | 0.056 | 0.215 |
| 63 | 0.253 | 0.063 | 0.198 |

An M-RoPE or layout error would show as a jump in the first layers. There is none: layer 0 is 0.005 for
both, the same as the bf16 rounding of the embeddings. The error grows smoothly with depth for both
media, then both jump at layer 42 (0.02 to 0.06), where a few channels reach activations of 50 to
500 (max abs in the dump column). That is the massive-activation layer, where a bf16 rounding difference
in the residual is amplified for every input. Image and video are the same up to layer 41; video then
grows from 0.06 to 0.25 while image stays flat at 0.06. Perturbing the embeddings of the same video
by 0.5% noise and running transformers against itself reproduces the video curve (0.10 at layer 42, 0.2
at 54 to 63), and the same noise on the image reaches 0.27 at layer 54 and 0.16 at 63, more than our own
image difference. So both prompts amplify small input differences strongly in layers 42 to 63; the
video drift is bf16 rounding sensitivity, not a position or layout bug, and the image's flat 0.06 is a
softer draw of the same effect. Nothing here points
at a fixable cause, so the video gate (KL 0.09, top-1 85%) stays.

## Running the tests

Non-27B suite (the 27B tests are all named `real_27b_*`, including every test in `tests/vision_lm.rs`):

```
SPECRT_REF_DIR=/tmp/ref387 cargo test --release --workspace -- --include-ignored --skip real_27b
```

The 27B vision tests need a GPU with about 36 GB free, one at a time under `flock /tmp/gpu27.lock`,
the vLLM server on port 8001 and its reference output:

```
~/.venv-vllm/bin/python crates/specrt/scripts/vllm_vision.py /tmp/vis484
CUDA_VISIBLE_DEVICES= ~/.venv-vllm/bin/python crates/specrt/scripts/vision_positions.py /tmp/vis484
SPECRT_VISION_REF_DIR=/tmp/vis484 flock /tmp/gpu27.lock cargo test --release -p inference-bench-specrt \
  --test vision_lm --test vision_encoder -- --include-ignored --nocapture --test-threads=1 \
  --skip bisect --skip noise
```

`vllm_vision.py` also writes `vllm-textctrl/longtext/randtext.json` and `plp-*.json` (prompt logprobs);
`vision_positions.py` (needs the transformers and vLLM packages, no server) writes `vllm-pos.json` and
`hf-pos-video.json` from the prompt ids.

Without `SPECRT_VISION_REF_DIR` the `vision_lm` tests print `skipped` and pass. The fixtures behind
`vision.rs` come from `scripts/vision_reference.py` and `scripts/vision_encoder_reference.py`
(`~/.venv-vllm/bin/python`, CPU only).

## Classify replay TTFT (#612)

`orcarouter/Qwen3.8-27B-Uncensored-NVFP4` is served from its HF config. The checkpoint's compressed-tensors
`input_activations` (FP8 per token on the projections, FP4 group 16 on the MLPs) now select `act_quant = "fp8"`
and `[mlp] act_quant = "fp4"` in `Spec::from_hf_config_serving`, which `resolve_spec` uses. `Spec::from_hf_config`
stays weight-only, so the exact-reference tests are unchanged. Before this, the app served the checkpoint without
the act-quant path and prefilled 1,505 tokens in 940 to 1,000 ms; now the step is 300 to 360 ms.

Replay (`/mnt/data/tmp/inference-bench-498/replay.sh`, 29 photos, 8 concurrent, 8 slots, 32k context, 8 GB KV,
max batched tokens 2048, vision cache on, beside the user's other GPU work), on the #498 branch plus this change:

| run | TTFT p50 | TTFT p95 | max | requests over 90 s |
|---|---|---|---|---|
| master (before) | 12.3 s | 35.3 s | 36.3 s | 0 |
| #498 branch, weight-only spec | 3.47 s | 6.18 s | 7.19 s | 0 |
| #498 branch + act-quant from config, pass 1 | 1.02 s | 2.38 s | 2.67 s | 0 |
| same, pass 2 (cache warm) | 0.96 s | 2.03 s | 2.44 s | 0 |
