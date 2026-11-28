# Image and video input

Vision towers of Qwen3.5 and 3.8 (Qwen3-VL layout), Gemma 3 (SigLIP with pan and scan, see [../models/gemma3.md](../models/gemma3.md)) and Llama 4, from the checkpoint's `vision_config` and the family `[vision]` table.

## Pieces

| Piece | Where |
|---|---|
| Preprocessing, prompt expansion, M-RoPE positions (`plan_media`) | `crates/specrt/src/vision.rs` |
| Encoder plan and runner | `crates/specrt/src/vision_encoder.rs`, kernels in `crates/kir/src/cuda/vision/` |
| Encoder thread and cache | `crates/specrt/src/media_encode.rs` |
| Request parsing and limits | `crates/core/src/media.rs`, the server's `[media]` table |

## Preprocessing

Everything comes from `preprocessor_config.json` and `video_preprocessor_config.json`: pixel bounds, `patch_size`, `merge_size`, `temporal_patch_size`, mean and std, and for video `fps`, `min_frames`, `max_frames` (defaults 2, 4, 768).

- `smart_resize` to multiples of `patch_size * merge_size` inside the bounds (half to even rounding), bicubic with antialias on uint8 matching torchvision's integer kernel, normalise, patchify in merge block order. Patches are bit identical to transformers.
- Images decode with the `image` crate (png, jpeg, webp, gif, bmp). Video decodes with an `ffmpeg` and `ffprobe` subprocess (only the sampled frames, raw rgb24), so builds need no libav; the `ffmpeg` binary is needed only for video.
- Frame sampling `int(total / fps * cfg.fps)` clamped to the frame bounds, indices `linspace(0, total - 1, n).round()`, equal to transformers and vLLM.
- `expand_placeholders` replaces each image pad by `t * h * w / merge^2` pads and each video slot by one `<T seconds><|vision_start|>pads<|vision_end|>` group per temporal patch.

## Encoder

`vision_encoder::plan(cfg)` lists the ops (patch embed with interpolated position embeddings, per block norm, qkv, 2D rope, per frame attention, proj, MLP, the merger and the deepstack mergers), which the app's model detail draws. Kernels: `vis_mma` (tensor core GEMM, 128x128 tiles, 3 stage cp.async, bias, GELU and residual in the epilogue), `vis_flash` (flash attention with per patch frame segments, so several images and frames go through one call without attending across each other), `vis_pos` (position table interpolation). Buffers for 4,800 patches (one 960x1280 photo) are allocated at load, before the KV pool.

`[runtime.vision]`:

| Key | Default | Meaning |
|---|---|---|
| `precision` | `fast` | `fast`: f16 activations and weights, f32 accumulate; `exact`: hi/lo split, keeps the f32 transformers bound |
| `cache_mb` | 1024 | host cache of encoder outputs by SHA-256 of the media bytes, LRU; 0 is off |
| `max_batch` | 8 | images per encoder pass |
| `pan_and_scan` | false | tiled image models also read crops |

Encoding runs on a `specrt-vision` thread on its own stream: it takes every queued media request, encodes their images in batches, and the engine keeps stepping running sequences meanwhile. A request cancelled while waiting is never admitted.

## LM integration

- `Spec.rope.mrope` (`section`, `interleaved`) and `Spec.vision` come from the config; TOML specs are completed from the snapshot's `config.json` at load (`engine::complete_spec`). Text advances all three axes; a pad run gets `(t, h, w)` from its grid as in transformers' `get_rope_index`; decode positions are `index + delta`.
- The `embed` op is followed by `splice_rows`, which overwrites the placeholder rows of the chunk being prefilled, so chunked prefill and fused steps work with media.
- Prefix cache keys tag each placeholder token with the media hash and row, so the same image hits and a different image of the same size does not.
- Usage: `prompt_tokens_details.vision_tokens`.

## Server

`image_url` and `video_url` content parts (a `{url}` object or a string). `[media]`: `max_images` 8, `max_videos` 1, `max_image_bytes` 20 MiB, `max_video_bytes` 200 MiB, `max_pixels`, `max_video_secs`, `allow_url_fetch` false, `fetch_timeout_secs` 10. Only `data:` URLs by default: fetching user URLs from the server is an SSRF risk; other schemes are always refused. A model without vision answers 400. The app attaches images by key, drag and drop or paste and sends them as the same parts.

## Measured state

27B NVFP4 against vLLM (`crates/specrt/tests/vision_lm.rs`, 32 teacher forced tokens, fast encoder):

| Case | Mean KL | Top-1 | Gate |
|---|---|---|---|
| image | 0.0140 | 32/32 | KL 0.05, top-1 90% |
| 6 image population | 0.0455 | 174/192 | KL 0.05, top-1 90% |
| 6 video population | 0.0613 | 170/192 | KL 0.09, top-1 85% |
| text control | 0.0116 | 188/192 | |

The video gate is looser on purpose: frames, indices, patches, prompt ids, positions and the encoder (rel rms 2e-6 to 8e-6 against fp32) all match the reference, and a per layer comparison shows the gap growing in the massive activation layers 42 to 63, where transformers against itself with 0.5 percent embedding noise drifts the same way (#496).

Encoder, 960x1280 photo, quiet RTX PRO 6000: 54.0 ms fast, 96.8 ms exact. Classify replay (29 photos, 8 concurrent, 27B NVFP4 with act quant from the config): TTFT p50 1.02 s, p95 2.38 s, about 2 s at p95 with the cache warm (#612).

## Tests

`crates/specrt/tests/vision.rs` (CPU, bit equal patches and indices against `tests/fixtures/vision` from `scripts/vision_reference.py`), `vision_encoder.rs` (GPU, against `scripts/vision_encoder_reference.py`: exact rms under 1e-4, fast under 5e-3), `vision_lm.rs` (GPU, `SPECRT_VISION_REF_DIR` from `scripts/vllm_vision.py` against a vLLM started with `crates/specrt/scripts/vllm27b_serve.sh kvauto`, and `scripts/vision_positions.py`), `vision_multi_gpu.rs`, `gemma3_vision.rs`, `gemma3_tiny_vision.rs`, `llama4_vision.rs`, `llama4_preprocess.rs`, `real_vision_tiled.rs`, the `media_encode::tests` and `serve::watchdog_tests` unit tests. Without `SPECRT_VISION_REF_DIR` the `vision_lm` tests print `skipped`.
