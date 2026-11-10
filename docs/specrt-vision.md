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
