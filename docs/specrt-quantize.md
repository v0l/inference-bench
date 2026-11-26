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

## quantcheck: a per layer recipe

```bash
inference-bench quantcheck <model> [--formats nvfp4,fp8] [--kl 0.02[,0.05,...]] [--budget-gb N] [--out PATH] [--max-ctx 512]
```

It loads the bf16 model once on one GPU with the kernels for every format it tries planned in (`quant_plan.kernels`), runs the shipped prompt set (`quantcheck/prompts.toml`) once for the reference logits, then for each layer and each format, with search and with plain rounding, swaps that one layer in place (`Runtime::requantize_layer`, which re-reads the layer from the checkpoint) and measures mean KL, max KL and top-1 of every position against bf16. A swapped layer gives the same logits and bytes as a fresh load of that plan, bit for bit (`tests/quantcheck_gpu.rs`).

Choosing:

- Per layer and format it keeps the rounding with the lower `mean KL + 0.01 * max KL` (`MAX_KL_WEIGHT`). This is what finding 2 of slice 1d asked for: the scale search lowers squared weight error, and on 6 of 28 layers of Qwen3-0.6B at KL 0.1 the measured logits preferred plain rounding (layer 0 nvfp4: search 0.0184 mean and 0.70 max, plain 0.0153 and 0.34).
- Greedy: every layer starts at its lowest bit format; the layer with the largest score moves up one format (nvfp4, fp8, bf16) until the summed per layer mean KL is inside the budget, or the next move would pass `--budget-gb`.
- The per layer KLs do not add up exactly, so the chosen recipe is checked by one whole model run against bf16. If that is over the budget the target is scaled down and greedy runs again (up to 4 rounds). On Qwen3-0.6B the sum was slightly pessimistic every time (estimated 0.0977, measured 0.0912 at budget 0.1), so one round was enough.
- The keep list's edge layers (first and last two) are not forced to bf16 under a recipe: quantcheck measures them like any other layer. The named and sensitive tensors (embeddings, head, norms, routers, `sensitive`) still stay bf16.

The recipe, `<model>.quantrecipe.toml`, is about 4 KB for 28 layers: recipe version, model, engine commit, `FORMAT_VERSION`, prompt set hash, positions, budgets, total and bf16 bytes, the estimated and checked KL, and per layer the format, search or plain, its own mean and max KL and its bytes. `Recipe::plan()` turns it into a `quant_plan`, the spec field the loader reads per layer. A model of `n` layers with `f` formats costs `2fn` probes; each probe is one layer re-read and one pass over 353 positions.

For a model that does not fit one card, quantcheck still needs every layer resident: run it on a stage, or wait for the streaming variant; `requantize_layer` refuses offloaded or tensor parallel runtimes.

Spec changes: `quant_plan` with `layers` (format and `search` per layer) and `kernels` (formats to plan kernels for even where no layer uses them yet). A plan needs a bf16 checkpoint and `quantize = "off"`. fp8_channel layers beside bf16 layers get their own kernel names (`<linear>_f8w*` tiles and `_f8b*` batch kernels) so they do not collide with the bf16 `_mma` tiles; the single stream gemv path refuses a mixed fp8 layer and the plan path runs it.

### Qwen3-0.6B

`inference-bench quantcheck crates/specrt/specs/qwen3-0.6b.toml --kl 0.01,0.02,0.05,0.1` on GPU1, 36 s for 112 probes. Weights are the device bytes of all weights, including the 0.31 GB bf16 embedding (tied head); bf16 is 1.503 GB.

| choice | layers | weights | mean KL | max KL | top-1 |
|---|---|---|---|---|---|
| recipe KL 0.01 | 5 bf16, 23 fp8 | 1.154 GB | 0.0092 | 0.079 | 95.8% |
| recipe KL 0.02 | 26 fp8, 2 nvfp4 | 1.063 GB | 0.0174 | 0.121 | 94.1% |
| recipe KL 0.05 | 17 fp8, 11 nvfp4 | 0.996 GB | 0.0475 | 0.310 | 87.3% |
| recipe KL 0.1 | 8 fp8, 20 nvfp4 | 0.930 GB | 0.0912 | 0.857 | 85.6% |
| blanket fp8_channel, edges bf16 | 4 bf16, 24 fp8 | 1.139 GB | 0.0107 | 0.132 | 96.0% |
| blanket nvfp4_g16, edges bf16 | 4 bf16, 24 nvfp4 | 0.961 GB | 0.1384 | 6.19 | 79.0% |

The KL 0.1 recipe is 3% smaller than blanket nvfp4 with 6.6 points more top-1, a third less mean KL and a seventh of the worst position (`tests/quantize_qwen3.rs`, `a_quantcheck_recipe_beats_blanket_nvfp4_*`, which also checks a fresh load of the recipe reproduces the recorded KL exactly and batch decode equals single). The KL 0.02 recipe beats blanket fp8 on size (1.063 against 1.139 GB) at the same KL class. The damaging layers are the last one (nvfp4 max KL 2.1) and the first three; quantising the last layer in fp8 and the middle in nvfp4 is what the recipe buys over the fixed keep list.

Speed at this size does not move (single run each, GPU1, prompt 1024 for prefill, 128 plus 255 tokens for decode):

| weights | prefill tok/s | decode b1 | decode b8 |
|---|---|---|---|
| bf16 | 3275 | 388 | 3078 |
| fp8_channel | 3174 | 387 | 3065 |
| nvfp4_g16 | 3236 | 393 | 3118 |
| recipe KL 0.1 | 3249 | 396 | 3139 |

All inside 3%: a 0.6B model is launch bound, so the win here is memory only. The 27B class numbers are the next measurement (needs a supervisor window).

## Trade off

Quantising on the fly reads the full bf16 model from disk on every start, against a stored NVFP4 copy that reads a quarter of that. It saves disk and costs load time. `quantize_cache` stores the result next to the model for the second start.

## Loading a recipe

```toml
[runtime]
quantize = "recipe"
quantize_recipe = "Qwen3-0.6B.quantrecipe.toml"
```

The same two keys are server and app options (`runtime.quantize`, `runtime.quantize_recipe`). A relative path in a spec file is read beside the spec. At load (`Spec::resolve_recipe`, called from `Runtime::load_inner` and `apply_quantize`) the recipe becomes the spec's `quant_plan` and the loader quantises each layer as it streams in, so nothing quantised is stored and only the few KB recipe is kept. The recipe is checked against the model first:

- refused: a different model (`model`, compared by its last path part, case insensitive), a different `format_version` ("run quantcheck again"), a different layer count, or layers not numbered 0 to n-1. The error names the recipe file and both values.
- warned and loaded: a different engine commit or prompt set hash; the KL figures in the file may then be stale or not comparable, but the plan is still valid.

On Qwen3-0.6B a load from the KL 0.1 recipe file gives mean KL 0.0912 and top-1 85.6%, exactly the check quantcheck recorded, and 929,694,976 weight bytes, exactly its count (`a_quantcheck_recipe_beats_blanket_nvfp4_*`, which now loads through the file). The tiny fixture test does the same on the tiny model (`quantcheck_gpu.rs`), and `quantcheck.rs` covers the refusals and warnings on CPU.

## The quantised weight cache

`[runtime] quantize_cache = true` (off by default) writes what the load converted and reads it back on the next start, so the conversion and most of the bf16 read are skipped.

- Where: for a model in the HF cache, `<hub>/models--<org>--<name>/specrt-quant/<key>/`, next to `snapshots/` and `blobs/`, never inside a snapshot and never beside the user's own files. A checkpoint outside the hub caches under `<cache_dir>/quant/<model>/<key>/` (`[runtime] cache_dir`, default the user cache folder).
- Key: sha256 of the cache version, `FORMAT_VERSION`, the scale search candidates, the spec with serving options cleared (so the plan, keep list, quant and tensor names count but KV, graphs and batch do not) and each source file's blob hash (from the hub symlink) or its size and mtime. A new recipe, a new spec or new weights is a new folder.
- What: one safetensors file per load group with each quantised matrix (nvfp4 packed, scales and the two globals; fp8 values and row scales), and `quant.toml` with the key, model, bytes and the source tensors it covers. A hit drops those tensors from the read plan; norms, embeddings, the head and kept bf16 layers are still read from the checkpoint. It is written to a `.partial` folder and renamed when the load finishes, so a crash leaves nothing a later start would read. A failed write is a warning and the start goes on.
- Not covered yet: pipeline stages and tensor parallel ranks convert at load as before.
- Reported: `runtime loaded` logs `quantize_cache = (dir, bytes, read|written)`, and `Runtime::quant_cache` carries it.

Qwen3-0.6B, a plan of 10 fp8_channel and 18 nvfp4_g16 layers, GPU1 (`quantize_cache_load_time_on_qwen3_0_6b`; cold means the page cache of the source and cache files was dropped with `posix_fadvise` first):

| start | load | weights | conversion (thread seconds) | source bytes read | cache on disk |
|---|---|---|---|---|---|
| cold, no cache | 1.28 s | 0.89 s | 3.13 s | 1.19 GB | none |
| cold, first start writes the cache | 1.27 s | 0.88 s | 2.98 s | 1.19 GB | 0.32 GB written |
| cold, cache read | 0.46 s | 0.09 s | 0 | 0.31 GB | 0.32 GB |
| warm, no cache | 1.29 s | 0.91 s | 3.15 s | 1.19 GB | none |
| warm, cache read | 0.47 s | 0.10 s | 0 | 0.31 GB | 0.32 GB |

The cache costs 0.32 GB of disk here (about a quarter of the 1.19 GB checkpoint) and cuts the load by 0.8 s; writing it costs nothing measurable. The 0.31 GB still read is the bf16 embedding and norms. Logits from a cached start equal a converting start bit for bit (`quantize_cache.rs`: CPU on the tiny dense and MoE fixtures, matrix by matrix, plus the GPU test on logits).
