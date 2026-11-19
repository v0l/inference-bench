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

The bf16 envelope (`bf16-activation-envelope.md`) has vLLM at 4e-4 to 1.2e-3 mean KL and 2e-3 to 8e-2 max KL on dense models, and single plane specrt at 2e-5 to 4e-4 mean. The gemv path (1.4e-3 mean, 3.2e-2 max) is at the vLLM edge on mean (the reference here is bf16 transformers, not fp32, and MoE routing flips add noise). The mma path is 24x worse on mean and 10x on max, outside the envelope, so it is not the gpt-oss default; it stays opt-in through `[mlp] act_quant = "fp4"`. The cause is quantising the activations to mxfp4 (4 bits per 32 values) in front of both expert GEMMs.

## Speed (3 rounds, median of 3 runs each)

| | prefill 1024 tok/s | decode b1 tok/s | decode b8 aggregate tok/s |
|---|---|---|---|
| packed gemv | 145.5, 144.5, 137.5 | 38.2, 38.0, 35.7 | 93.4, 120.9, 108.2 |
| mma fp4 | 2102, 2622, 2461 | 30.5, 37.7, 35.6 | 237.6, 203.1, 273.9 |

The mma path is about 17x faster at prefill and about 2x at b8 decode, level at b1. Prefill on the gemv path is slow because the grouped gemv has no tensor core path; making mma usable for gpt-oss needs a lower activation error first (follow-up issue).

## Suites

Spark `variants` (13 pass, plus 7 GPU tests with the reference) and `moe_mma` (7 pass), CPU kir and specrt suites, clippy `-D warnings`.
