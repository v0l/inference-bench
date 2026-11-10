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
  `<T seconds><|vision_start|>pads<|vision_end|>` group per temporal patch, with T the mean
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
