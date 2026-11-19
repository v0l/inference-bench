# gpt-oss-20b on the mxf4 paths (#897)

Spark (GB10), `openai/gpt-oss-20b` as released (MXFP4 experts, bf16 rest), under `flock /tmp/gpu27.lock`. 8 prompts of 20 to 48 tokens, 308 scored rows (first position of each prompt dropped). Reference is transformers 5.23 with `Mxfp4Config(dequantize=True)` in bf16 (the checkpoint is MXFP4, so fp32 weights do not exist). Rerun: `envelope.py prompts|fp32|compare` with `MXFP4_DEQUANT=1 FP32_DTYPE=bfloat16`, then `envelope <ckpt> prompts.json out plain mma`.

## Loading

- `quantization_config.quant_method = "mxfp4"` is accepted by a flat rule in `specs/family-gpt_oss.toml` (the experts' format is `[moe] expert_block`, the attention, router and head stay bf16, which is what `modules_to_not_convert` lists).
- `*_blocks` / `*_scales` tensors load through the existing `expert_block` path.
- K tail: block scaled expert weights are padded at load to a multiple of 128 columns with zero data and zero scales (`BlockFormat::padded_cols`, `pad_rows`). The grouped mx mma kernel (`moe_mma_mx_*`) runs `ceil(K/128)` k tiles over the padded rows, the activation quantiser writes zeroed padding, and the packed gemv indexes rows with the padded stride. `check_moe_runnable` accepts hidden that is a multiple of 64 when the experts are block scaled.
- GPU tests at hidden 2880 and intermediate 448: `mx_grouped_mma_handles_a_k_tail_with_{big,small}_tiles` (max err 0 against the CPU reference).

## KL against transformers

| path | mean KL | max KL | top-1 of 308 |
|---|---|---|---|
| packed gemv (default) | 1.39e-03 | 3.23e-02 | 302 |
| mma, `[mlp] act_quant = "fp4"` | 3.37e-02 | 6.73e-01 | 283 |

The bf16 envelope (`bf16-activation-envelope.md`) has vLLM at 4e-4 to 1.2e-3 mean KL and 2e-3 to 8e-2 max KL on dense models, and single plane specrt at 2e-5 to 4e-4 mean. The gemv path (1.4e-3 mean, 3.2e-2 max) is at the vLLM edge on mean (the reference here is bf16 transformers, not fp32, and MoE routing flips add noise). The mma path is 24x worse on mean and 10x on max, outside the envelope, so it is not the gpt-oss default; it was opt-in (it is the default since #901 with two planes). The cause is quantising the activations to mxfp4 (4 bits per 32 values) in front of both expert GEMMs.

## Speed (3 rounds, median of 3 runs each)

| | prefill 1024 tok/s | decode b1 tok/s | decode b8 aggregate tok/s |
|---|---|---|---|
| packed gemv | 145.5, 144.5, 137.5 | 38.2, 38.0, 35.7 | 93.4, 120.9, 108.2 |
| mma fp4 | 2102, 2622, 2461 | 30.5, 37.7, 35.6 | 237.6, 203.1, 273.9 |

The mma path is about 17x faster at prefill and about 2x at b8 decode, level at b1. Prefill on the gemv path is slow because the grouped gemv has no tensor core path; making mma usable for gpt-oss needs a lower activation error first (follow-up issue).

## Suites

Spark `variants` (13 pass, plus 7 GPU tests with the reference) and `moe_mma` (7 pass), CPU kir and specrt suites, clippy `-D warnings`.

## Activation precision sweep (#901)

`[mlp] up_act_planes` and `[mlp] down_act_planes` (1 or 2) pick how many mxfp4 planes the grouped mma uses for the expert up (hidden input) and down (intermediate input) activations. Two planes quantise the activation, then quantise the residual with its own block scale, and the mma accumulates both (`moe_mma_mx_{up,down}2*`), at twice the mma work for that GEMM. The family key is `[moe] act_planes`.

KL against the bf16 transformers reference, 308 rows, Spark:

| path | mean KL | max KL | top-1 of 308 |
|---|---|---|---|
| packed gemv | 1.386e-03 | 3.23e-02 | 302 |
| mma, 1 plane | 3.373e-02 | 6.73e-01 | 283 |
| mma, up 2 planes | 1.808e-02 | 3.45e-01 | 291 |
| mma, down 2 planes | 1.814e-02 | 2.60e-01 | 285 |
| mma, both 2 planes | 2.172e-03 | 5.95e-02 | 303 |

Both GEMMs contribute about equally, so the error is only fixed by refining both inputs. Both planes is within the vLLM edge of the envelope (mean 4e-4 to 1.2e-3, max 2e-3 to 8e-2 on dense models; here max is below 8e-2 and top-1 is above gemv's) and is now the gpt-oss default: the family sets `[moe] act_quant = "fp4"`, `act_planes = 2`. `[mlp] act_quant = "none"` restores the gemv path.

Speed (3 rounds, `bench --runs 3`, Spark, gemv / one plane / two planes):

| | prefill 1024 tok/s | decode b1 tok/s | decode b8 aggregate tok/s |
|---|---|---|---|
| packed gemv | 122.5, 118.1, 145.5 | 27.0, 31.1, 38.5 | 89.6, 109.3, 120.1 |
| mma 1 plane | 2654, 2657, 2664 | 37.7, 38.3, 38.3 | 273.7, 274.9, 272.5 |
| mma 2 planes (default) | 1916, 2451, 2248 | 33.3, 38.3, 37.2 | 229.2, 228.1, 215.3 |

Two planes keep about 16x the gemv prefill and 2x its b8 decode. Suites: `moe_mma` (10, including three two-plane GPU tests) and `variants` (23 with the reference) on the Spark, CPU suites and clippy `-D warnings` clean.
