# Quantisation

Weight formats, activation quantisation, the KV cache dtype, requantise and quantise at load, and quantcheck recipes. The kernels that run each format are in [gemm.md](gemm.md); GGUF quants are in [gguf.md](gguf.md).

## Weight formats

`Quant` (`crates/specrt/src/spec.rs`): `bf16`, `q8`, `fp8_block128`, `fp8_channel`, `nvfp4_g16`, `gguf`. The spec sets `quant` for the projections and `[mlp] quant` with `quant_until` for the MLPs, so `Spec::mlp_quant(layer)` can differ per layer. `Spec::from_hf_config` derives this from a compressed-tensors `quantization_config.config_groups` (format, type, bits, strategy, group size, targets mapped to layer and module) and refuses anything else; ModelOpt mixed precision is mapped per module too (see [../models/glm5.md](../models/glm5.md)).

| Format | Stored as | Value |
|---|---|---|
| `fp8_channel` | e4m3 weights, one bf16 scale per output row | `w = e4m3(q) * scale[r]` |
| `fp8_block128` | e4m3 weights, `weight_scale_inv` on a 128 x 128 grid | per block scale |
| `nvfp4_g16` | packed e2m1 (byte b of a row holds columns 2b low nibble, 2b+1 high), e4m3 scale per 16 columns, f32 `weight_global_scale` and `input_global_scale` | `w = e2m1(q) * e4m3(s[r, c/16]) / weight_global_scale` |

`crates/specrt/src/nvfp4.rs` holds the NVFP4 decode, round to nearest, gate/up interleave and CPU dequant. On the device an `Fp4` scale buffer is a 16 byte header (`1 / weight_global_scale`, `input_global_scale`) then the raw group scales. `e2m1 * e4m3` is exact in bf16, so the weight only path is exact to the dequantised weights. `crates/specrt/tests/nvfp4.rs` matches the real 27B dequant bit for bit against compressed-tensors (`reference.py nvfp4-dequant`).

## Activation quantisation

Spec options, each a declared numerics change (see the pass contract in [kir-compiler.md](kir-compiler.md)):

- `act_quant = "fp8"`: FP8 projection inputs quantised per token (`quant_rows_fp8_token`), or per 128 columns for block-128 checkpoints, into the `Fp8Act` GEMM.
- `[mlp] act_quant = "fp4"`: NVFP4 MLP inputs quantised per 16 columns like vLLM's `scaled_fp4_quant` (`sx = e4m3(input_global_scale * amax / 6)`), into the `Fp4Act` GEMM with `alpha = 1 / (input_global_scale * weight_global_scale)`.

Both apply only above 8 rows (`kir::gemm::act_rows`). The shipped 27B specs set both for serving; the `*-exact.toml` specs are weight only and back the exact reference tests.

## Output head

`head_quant` (`bf16` default, `fp8_channel`, `nvfp4_g16`) quantises the bf16 head at load. `fp8_channel` needs hidden a multiple of 128, `nvfp4_g16` a multiple of 64 and the tensor core path. A tied head quantises only its output copy. On the 27B NVFP4 `fp8_channel` stays inside the vLLM gate (teacher forced top-1 96.6 percent, KL 0.0095) and `nvfp4_g16` does not (95.8 percent, 0.0136), so the default is `bf16` (#363).

The MTP draft head (`speculative.head_quant`, default `nvfp4_g16`, `bf16` under `gemm_fp32`) is a separate copy used only for drafts: on the 27B it costs 0.014 accepted tokens per step and gains 8.5 percent at batch 1 (#415). See [mtp.md](mtp.md).

## KV cache dtype

`[runtime] kv_dtype`: `f16` (default) or `fp8`.

- fp8 layout: per K or V plane, e4m3 rows then one f32 scale per head per token row (`amax / 448`, quantised as the token is appended in `norm_rope_head`, never rewritten). `kv_planes` is 1. All four attention families dequantise in registers; the decode kernel and `flash_mma` load raw bytes and scales with cp.async and convert into the mma fragments.
- `kv_f16_layers`: the first N attention layers stay f16. Unset, `Spec::for_kv_context` keeps half the attention layers in f16 up to `FP8_KV_VERIFIED_CONTEXT` (32,768 tokens) and three quarters above it; `kv_context_note` warns when an explicit value is lower. With the long context default, KV per token is 0.86x of f16.
- A prefix cache entry or snapshot written with one dtype never matches another.

Why f16 stays the default: on the 27B NVFP4 the loss sits in the early full attention layers, both K and V, and comes from the e4m3 mantissa, not from the kernels (emulated fp8 in the f16 store gives the same KL). With 8 of 16 layers f16 the drift reaches max KL 14 at 64k on agentic text; 12 of 16 stays under 0.031 to 160k (#763). int8 with a per row scale drifts the same way and was not merged (#796, branch `issue-796-int8-kv-code`).

| Measured, 27B NVFP4, GPU1, 18 to 22 Nov 2026 | f16 | fp8 (default f16 layers) |
|---|---|---|
| KV per token | 1.00x | 0.86x above 32k, 0.75x below |
| batch 8 decode at 8k | 299.5 to 299.9 tok/s | 298.9 to 299.1 |
| batch 8 decode at 32k | 223.0 to 223.2 | 220.8 to 220.9 |
| KL against f16 at 160k, 12 f16 layers | | mean 0.00083, max 0.029 |

Gate command: `crates/specrt/scripts/kvgate27b.sh <out> <bin dir> [kl|needle|speed|mtp|all]` (`src/bin/kvgate.rs`, which also has `kvgate loop` for the 150k loop check).

## Requantise at load

`[runtime.requant]` reformats a tensor class of an FP8 checkpoint as it is read; the checkpoint format is the default. Classes `lin_in`, `lin_out`, `qkv`, `o`, `mlp_tail` take `nvfp4_g16` or `fp8_channel`; `lm_head` takes `bf16`, `fp8_channel` or `nvfp4_g16`. A requantised matrix has no input scale, so it runs the weight only NVFP4 kernels, never the FP4 activation path. The CPU quantiser feeds on dequantised FP8 rows (`tests/requant.rs` checks bytes).

On the 27B NVFP4 only `mlp_tail = nvfp4_g16` (KL 0.0030, top-1 97.5 percent, 1.0 GB less) and `lm_head = fp8_channel` (KL 0.00056, 1.2 GB less) pass the gate (KL under 0.01, top-1 at least 97 percent); the low memory profile sets `mlp_tail`. Together they are flat at batch 1, +3 percent at batch 4 and -4 percent at batch 8, so max performance sets neither (#605).

## Quantise at load

`[runtime] quantize`: `off` (default), `auto` (nvfp4_g16), `nvfp4_g16`, `fp8_channel`, `recipe`. A bf16 checkpoint is quantised as it streams in (`crates/specrt/src/quantize.rs`); nothing is stored unless `quantize_cache` is on.

- `nvfp4_g16` covers attention `qkv` and `o`, delta-net `in_proj` and `out_proj`, dense `gate_up` and `down` and the expert stacks, for linears whose input width is a multiple of 128 and output a multiple of 16.
- `fp8_channel` serves the whole checkpoint as `fp8_channel`; MoE is refused.
- The keep list (`crates/specrt/specs/family-base.toml` `[quantization.keep]`) stays bf16: the first and last two layers, `embed_tokens`, `lm_head`, norms, routers and gates, a family's `sensitive` tensors, the delta-net `a` and `b` gates and every vector. A stage counts it in whole model layers.
- Scales come from a small search instead of the max: per 16 group the e4m3 code from the max and the codes 4 below and 1 above, lowest squared error wins; per FP8 row the max over 448 times 1.0 to 0.875. The CPU search is bit exact against `scripts/quant_search_reference.py`, the kir kernels (`kir::weight_quant::ScaleSearch`) against the CPU.
- Needs the tensor core path; not on wgpu.

Bounds asserted on Qwen3-0.6B (`crates/specrt/tests/quantize_qwen3.rs`, prompt set `crates/specrt/quantcheck/prompts.toml`, 353 positions):

| Format | Mean KL bound (measured) | Top-1 floor (measured) |
|---|---|---|
| fp8_channel | 0.02 (0.0148) | 0.92 (0.949) |
| nvfp4_g16 | 0.155 (0.138) | 0.72 (0.790) |

Plain rounding gives 0.166 on nvfp4_g16, over the bound, so the search is required.

## quantcheck recipes

```bash
inference-bench quantcheck <model> [--formats nvfp4,fp8] [--kl 0.02[,0.05,...]] [--budget-gb N] [--out PATH] [--max-ctx 512]
```

It loads the bf16 model once, then for each layer, format and rounding swaps that layer in place (`Runtime::requantize_layer`, bit identical to a fresh load) and measures KL and top-1 against bf16. Per layer it keeps the rounding with the lower `mean KL + 0.01 * max KL`; greedy then raises the most damaging layer one format (nvfp4, fp8, bf16) until the summed KL fits the budget, and checks the result with one whole model run, rescaling up to 4 rounds. Edge layers are measured like any other; named and sensitive tensors stay bf16. A model must fit one GPU.

The recipe (`<model>.quantrecipe.toml`, a few KB) carries the model, engine commit, `FORMAT_VERSION`, prompt hash and per layer format, rounding, KL and bytes. Load it with:

```toml
[runtime]
quantize = "recipe"
quantize_recipe = "Qwen3-0.6B.quantrecipe.toml"
```

A relative path is read beside the spec. A different model, `format_version` or layer count is refused; a different engine commit or prompt set loads with a warning. The recipe becomes the spec's `quant_plan` (per layer format and `search`, plus `kernels` to plan); fp8 layers beside bf16 ones get `_f8w*` and `_f8b*` kernel names.

## Quantised weight cache

`[runtime] quantize_cache` (off) writes the converted matrices on first load and reads them on the next: under `<hub>/models--<org>--<name>/specrt-quant/<key>/` for a hub model, `<cache_dir>/quant/<model>/<key>/` otherwise. The key hashes the cache version, `FORMAT_VERSION`, the search candidates, the spec without serving options and each source file's blob hash or size and mtime. Writes go to a `.partial` folder renamed at the end. Not used by pipeline stages and tensor parallel ranks.

## Measured state

Qwen3-0.6B, desktop GPU1, Nov 2026 (#1417):

| Weights | Device bytes | Mean KL | Top-1 | Decode b1 tok/s |
|---|---|---|---|---|
| bf16 | 1.503 GB | 0 | 100% | 388 |
| fp8_channel, edges bf16 | 1.139 GB | 0.0107 | 96.0% | 387 |
| nvfp4_g16, edges bf16 | 0.961 GB | 0.1384 | 79.0% | 393 |
| recipe KL 0.1 | 0.930 GB | 0.0912 | 85.6% | 396 |

With `quantize_cache` a cold start of a mixed plan drops from 1.28 s to 0.46 s for 0.32 GB of disk.

## Tests

`crates/specrt/tests/quant_search.rs`, `quantize_load.rs`, `quantcheck.rs`, `quantize_cache.rs`, `kv_dtype.rs`, `requant.rs` (CPU); `quant_search_gpu.rs`, `quantize_gpu.rs`, `quantcheck_gpu.rs`, `quantize_qwen3.rs`, `head_quant.rs`, `nvfp4.rs`, `nvfp4_quant.rs`, `requant27b.rs` (GPU).
