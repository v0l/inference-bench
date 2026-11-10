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

- Weights are the checkpoint's `visual.*` tensors in bf16, kept bf16 on the device. Activations are
  f32. The patch embed Conv3d is a GEMM over the flattened patch, the bilinear position table
  interpolation (align_corners) and the rope tables are built on the host.
- Attention is limited to one frame (`h * w` patches), which is how transformers builds `cu_seqlens`
  for images and videos. Images in one request are encoded one at a time, so no cross image mask is
  needed.
- Bound: `tests/vision_encoder.rs` (ignored, needs the 27B FP8 snapshot and a GPU) compares the merged
  embeddings of three fixtures with `transformers` run by `scripts/vision_encoder_reference.py`.
  Measured relative rms 2e-6 to 8e-6 and max abs error at most 2.2e-3 on values up to 94. The test
  asserts rms below 1e-4 and max below 5e-5 of the largest value.

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

specrt TTFT includes the preprocessing and the vision encoder (a first, unoptimised f32 kernel set
with a naive GEMM), vLLM's includes its encoder cache; the encoder is the target for a later
speed pass. `prefixbench` (scheduler only, no HTTP): image_1 cold 815 ms, warm 436 ms with 208 of
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
