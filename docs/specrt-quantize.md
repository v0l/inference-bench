# Quantise at load (#1417)

`[runtime] quantize` in the model spec or the server config turns a bf16 checkpoint into a lower bit one while it streams in, so no quantised copy is stored and the bf16 model never sits in host memory whole.

```toml
[runtime]
quantize = "nvfp4_g16"   # or "fp8_channel", "auto" (nvfp4_g16), "off" (default)
```

## What gets quantised

- `nvfp4_g16`: every attention `qkv` and `o`, delta-net `in_proj` and `out_proj`, dense `gate_up` and `down`, and the routed and shared expert stacks. A linear is quantised when its input width is a multiple of 128 and its output a multiple of 16; any other linear stays bf16. Dense MLPs also need the shapes `dense_fp4_fits` asks for.
- `fp8_channel`: the whole checkpoint is served as `quant = "fp8_channel"` with one scale per output row. Mixture-of-experts is refused for now (use nvfp4_g16).
- The keep list stays bf16. It is data in `specs/family-base.toml` `[quantization.keep]`: the first and last two layers, `embed_tokens`, `lm_head`, norms, MoE routers and gates, and any tensor a family lists under `sensitive`. The delta-net `b` and `a` gates and every vector stay bf16 as well.
- A pipeline stage counts the keep list in whole model layers.
- The option needs the tensor core path (`gemm_fp32 = false`) and is not on wgpu yet.

## How the scales are chosen

Round to nearest, with the scale chosen by a small search instead of taken from the absolute maximum (`crates/specrt/src/quantize.rs`):

- nvfp4_g16: per group of 16, the e4m3 scale code from the max and the codes 1 to 4 steps below and 1 above; the one with the lowest squared error wins.
- fp8_channel: per row, the max over 448 times 1.0, 0.969, 0.938, 0.906 and 0.875.

The CPU search is bit exact against `scripts/quant_search_reference.py` (`tests/quant_search.rs`). The same search as kir kernels (`kir::weight_quant::ScaleSearch`) is bit exact against the CPU one (`tests/quant_search_gpu.rs`, GPU).

## What a load reports

`runtime loaded` logs `quantize`, `quantize_secs` (time spent in the search, summed over loader threads), `quantize_tensors`, `quantize_bf16_bytes` and `bytes_read` next to `load_secs`. `Runtime::quantized` and `Runtime::bytes_read` carry the same numbers.

## Measured so far

Tiny fixtures on GPU1, all layers quantised (keep list edges set to 0), mean KL of the prompt logits against the bf16 load over 4 prompts of 96 tokens (`tests/quantize_gpu.rs`):

| fixture | format | mean KL | weights on device |
|---|---|---|---|
| tiny-qwen3.5 | fp8_channel | 0.0018 | 4.4 MB against 7.4 MB |
| tiny-qwen3.5 | nvfp4_g16 | 0.020 | 2.9 MB against 7.4 MB |
| tiny-qwen3.5-moe | nvfp4_g16 | 0.035 | 5.5 MB against 16.7 MB |

The tiny fixtures have random weights, so these figures only show the path works. The bound per format on real weights (Qwen3-0.6B) is slice 1d.

## Trade off

Quantising on the fly reads the full bf16 model from disk on every start, against a stored NVFP4 copy that reads a quarter of that. It saves disk and costs load time. The planned `quantize_cache` option (step 3) stores the result next to the model for the second start.
