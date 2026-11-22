# Requantise at load (#605)

`[runtime.requant]` in a spec (or `runtime.requant.<class>` in the load options) reformats a tensor class when the checkpoint is read, with the checkpoint format as the default. Classes: `lin_in`, `lin_out`, `qkv`, `o`, `mlp_tail`, `lm_head`. FP8 classes take `nvfp4_g16` (or `fp8_channel`); `lm_head` takes `bf16`, `fp8_channel` or `nvfp4_g16`.

```toml
[runtime.requant]
mlp_tail = "nvfp4_g16"
lm_head = "fp8_channel"
```

The quantiser is the existing CPU one (`quantize_nvfp4`, `quantize_fp8_channel`), fed with the dequantised FP8 rows. A requantised matrix carries no input scale, so it takes the weight-only NVFP4 kernels (`_f4w`, `_f4b*`, GEMV) and never the FP4 activation path. `tests/requant.rs` checks the tiny FP8 fixture byte for byte against the CPU quantiser. Requantised attention and delta-net classes need the scheduler decode path, not the single stream step.

## Byte table (27B NVFP4, device weights, GB)

| set | weights | saved |
|---|---|---|
| checkpoint format | 24.783 | |
| lin_in | 22.892 | 1.891 |
| lin_out | 24.074 | 0.709 |
| qkv | 24.231 | 0.552 |
| o | 24.547 | 0.236 |
| mlp_tail | 23.779 | 1.004 |
| lm_head nvfp4_g16 | 22.240 | 2.543 |
| lm_head fp8_channel | 23.552 | 1.231 |

## Quality table

20 prompts x 64 teacher-forced tokens (1,280 rows), `tests/requant27b.rs`, KL and top-1 against the unchanged checkpoint on the same tokens; gate is KL under 0.01 and top-1 at least 97%.

| set | mean KL | top-1 vs checkpoint | top-1 vs vLLM greedy | verdict |
|---|---|---|---|---|
| lin_in nvfp4 | 0.0185 | 94.1% | 93.6% | fail |
| lin_out nvfp4 | 0.0165 | 94.8% | 94.0% | fail |
| qkv nvfp4 | 0.0172 | 95.3% | 95.7% | fail |
| o nvfp4 | 0.0149 | 95.0% | 94.7% | fail |
| mlp_tail nvfp4 | 0.0030 | 97.5% | 95.0% | pass |
| lm_head nvfp4 | 0.0053 | 96.3% | 93.4% | fail (top-1) |
| lm_head fp8_channel | 0.00056 | 98.4% | 94.8% | pass |

Kept: `mlp_tail = nvfp4_g16`, `lm_head = fp8_channel`. The low memory profile sets `mlp_tail` (1 GB less weights). The max performance profile sets neither: the speed table below is flat at batch 1, up 3% at batch 4 and down 4% at batch 8, and 8 is the default slot count.

## Speed (GPU1, `bench --ab`, 5 rounds, 192 new tokens, MTP k=2, A = checkpoint format, B = kept set)

| batch | A tok/s | B tok/s | verdict |
|---|---|---|---|
| 1 | 105.8 | 107.7 | no difference outside the A/A band (+1.8%) |
| 4 | 320.6 | 330.7 | B faster (+3.1%) |
| 8 | 563.2 | 540.9 | B slower (-4.0%) |

The 20% single-stream target is not met: the kept classes cut 2.2 GB of 24.8 GB per token (9%), and the classes that would cut more (`lin_in`, `lin_out`, `qkv`, `o`) fail the quality gate. At batch 8 the weight-only NVFP4 batch kernels are slower than the FP8 batch kernels.

## MTP acceptance (27B, k=2, 20 prompts x 128 tokens, batch 1, GPU1)

| set | mean accepted per step | draft acceptance | position 1 / 2 | decode tok/s |
|---|---|---|---|---|
| checkpoint format | 2.408 | 0.704 | 0.815 / 0.592 | 97.9 |
| mlp_tail nvfp4 + lm_head fp8 | 2.414 | 0.707 | 0.816 / 0.599 | 99.1 |

Acceptance does not move (+0.3%), so the kept set costs nothing in drafts.

## Tiny fixture bound

`requantised_logits_on_random_tiny_weights...` (GPU, ignored by default) requires KL under 0.05 and top-1 above 0.5. The fixture has random weights and a flat output distribution, so a 4-bit requantisation of every class flips many near-ties: measured KL 0.025, top-1 0.667. The 0.5 floor only catches a broken kernel; the real guards are `requantised_weights_match_the_cpu_quantiser_bit_for_bit` (exact bytes) and the 27B gate above.

## GPU quantiser (#593)

`nvfp4::quantize_rows_on_device` takes a bf16 host slice and returns raw device buffers for the draft head. Requant starts from FP8 rows, which would first need a device dequant, and every requantised class goes through `Matrix::Fp4` upload and the CPU-reference test. Not a one-line change, so the CPU quantiser stays. Load time on the 27B with all classes set was 11 to 23 s against 10.6 s unchanged.

Commands: `scripts/requant27b.sh <class=quant>...` (quality, needs `REQUANT_WORK` and `B=<requant27b test binary>`), `scripts/requant_ab27b.sh` (speed).
