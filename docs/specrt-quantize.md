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

The tiny fixtures have random weights, so these figures only show the path works.

### Qwen3-0.6B, the bound per format

`tests/quantize_qwen3.rs` (GPU1, ignored by default) loads the real Qwen3-0.6B with each format and with the default keep list (first and last two layers, embedding and the tied head stay bf16), runs the fixed prompt set `crates/specrt/quantcheck/prompts.toml` (4 prompts, 13 to 189 tokens, 353 positions; the same set `quantcheck` will use) and compares every position's logits against the bf16 load.

| prompt | tokens | fp8_channel mean / max KL, top-1 | nvfp4_g16 mean / max KL, top-1 |
|---|---|---|---|
| short | 13 | 0.0071 / 0.047, 100% | 0.070 / 0.36, 76.9% |
| code | 49 | 0.0040 / 0.052, 98.0% | 0.051 / 0.50, 87.8% |
| science | 102 | 0.0255 / 0.150, 93.1% | 0.143 / 0.77, 74.5% |
| story | 189 | 0.0124 / 0.088, 94.7% | 0.163 / 6.19, 79.4% |
| all | 353 | 0.0148 / 0.150, 94.9% | 0.138 / 6.19, 79.0% |

The bounds are named constants in `crates/specrt/src/quantize.rs` and the test asserts them:

| format | `*_MEAN_KL_BOUND` | `*_TOP1_FLOOR` |
|---|---|---|
| fp8_channel | 0.02 (measured 0.0148) | 0.92 (measured 0.949) |
| nvfp4_g16 | 0.155 (measured 0.138) | 0.72 (measured 0.790) |

Plain round to nearest from the max (no scale search) gives 0.0149 mean KL for fp8_channel, so the search makes no measurable difference there, and 0.166 for nvfp4_g16, which is over the nvfp4 bound: the search is worth about 17 percent of the nvfp4 KL and the test fails without it. A 0.6B model is the hard case for weight only 4 bit (small matrices, little redundancy), so nvfp4_g16 on every layer is not a safe blanket default here; that is what the per layer `quantcheck` recipe is for.

Load on GPU1 (warm page cache, one run after a warm up load):

| quantize | load_secs | search secs (summed over loader threads) | tensors | bytes read | weights on device |
|---|---|---|---|---|---|
| off | 0.27 | 0 | 0 | 1.19 GB | 1.50 GB |
| fp8_channel | 1.40 | 4.61 | 112 | 1.19 GB | 1.08 GB |
| nvfp4_g16 | 1.16 | 3.69 | 96 | 1.19 GB | 0.96 GB |

The bytes read are the same with the option on and off: the bf16 checkpoint is read once either way. The device bytes include the bf16 embedding table (0.31 GB) and the four kept layers. On a cold JIT cache the first quantised load took 4.5 s.

## Trade off

Quantising on the fly reads the full bf16 model from disk on every start, against a stored NVFP4 copy that reads a quarter of that. It saves disk and costs load time. The planned `quantize_cache` option (step 3) stores the result next to the model for the second start.
