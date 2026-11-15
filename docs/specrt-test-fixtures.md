# Stand-ins for the 27B: tiny fixture and the 4B benchmark (#635)

Two cheaper tiers that cover the 27B code paths without stopping the user's app. `docs/testing.md` (#634) lists the
tiers; these are the entries to put in the "any GPU, seconds" and "mid-size, minutes" tiers.

## Tiny Qwen3.5 fixture, `tests/fixtures/tiny-qwen3.5-27b`

A 4 layer Qwen3.5ForConditionalGeneration with random weights (13 MB, checked in, references 3 MB beside it in
`reference/`). It has the 27B's shape in miniature:

- `layer_types` `[linear, linear, linear, full]` (the first group of the real config), gated delta net, output gate,
  qk norm, partial rotary (rotary dim 64), hidden 256, head dim 256.
- compressed-tensors `mixed-precision` config in the 27B's layout: FP8 per channel weights with FP8 per token dynamic
  activations on the projections (group_0), NVFP4 group 16 weights with local FP4 activations on the MLPs of layers
  0 to 2 (group_1), the last layer's MLP in FP8. The spec derives `act_quant = fp8` and `[mlp] act_quant = fp4`
  from it, the same as the real checkpoint (`Spec::from_hf_config_serving`).
- an MTP head (`mtp.safetensors`, `mtp_num_hidden_layers = 1`), and a 2 block vision tower (hidden 128, merge 2,
  `out_hidden_size` 256) with mrope sections `[11, 11, 10]` interleaved as in the real config.

Regenerate (CPU, 10 seconds, transformers 5.22, under `systemd-run -p MemoryMax=8G`):

```
~/.venv-vllm/bin/python crates/specrt/scripts/tiny27b.py all
```

`fixture` writes only the checkpoint, `reference` only the references. The script calibrates the NVFP4 input scales
on the weights, writes the shards, then runs transformers on the dequantised weights for the references:
weight-only logits and greedy tokens, the same with the activation quantisation hooks, chained MTP draft logits at
8 prefix lengths, and vision embeddings plus full multimodal logits for one image and two images.

Tests (`cargo test --release -p inference-bench-specrt --test tiny27b -- --include-ignored`, about 100 s of
which most is the first JIT, under `flock /tmp/gpu27.lock`):

| test | gate |
|---|---|
| `fixture_spec_is_the_27b_shape_in_miniature` (no GPU) | spec equals the real config's layer pattern, quant groups, mrope, MTP, vision |
| `weight_only_logits_kl_and_greedy_match_transformers` | KL 2e-10 (gate 1e-5), top-1 100%, 64 greedy tokens identical |
| `act_quantised_logits_stay_inside_the_reference_activation_envelope` | KL to weight-only 1.60e-2 against the reference's 1.54e-2 (gate 1.5x), top-1 70.5% against 69.6% (the `real27b.rs` gate shape) |
| `mtp_drafts_and_chained_logits_match_transformers` | 8 prefixes, chained draft logit error 3e-5 (gate 2e-3 of scale), both drafts identical |
| `speculative_greedy_equals_plain_greedy_and_oracle_drafts_accept` | k 1 to 3: MTP and oracle speculation equal plain greedy, oracle acceptance above k + 0.8 |
| `speculative_greedy_with_act_quant_equals_its_plain_greedy` | the act-quant graphs agree with their own plain decode |
| `vision_embeddings_and_mrope_logits_match_transformers` | embeddings relative error 6e-6, logits KL 6e-10, one and two images, image changes the logits |

The random MTP head accepts about nothing, so acceptance statistics come from the oracle test and from the 4B
below, not from the fixture.

## Qwen3.5-4B NVFP4, `scripts/bench4b.sh`

`scripts/quantize4b.py` quantises `Qwen/Qwen3.5-4B-Base` (HF cache) to the 27B's layout: MLPs of layers 0 to 27 in
NVFP4 group 16 with calibrated input scales (the prompts of `spec_prompts.txt`), projections and the last 4 MLPs in
FP8 per channel, vision tower, MTP head and bf16 embeddings kept as is. Our own quantiser, no llm-compressor. 4.9 GB on
disk, 7.1 GB on the device, about 45 seconds on one card. `bench4b.sh` makes it on first use into
`~/.cache/specrt-models/Qwen3.5-4B-NVFP4`, then runs the bench binary under `flock /tmp/gpu27.lock`.

Baseline, RTX PRO 6000 Max-Q, vLLM on the other card, 3 runs each, median (commit 1655db2):

| | weight-only spec | act-quant spec (as served) |
|---|---|---|
| prefill 128 / 1,024 / 4,096 tok/s | 5,709 / 9,556 / 9,397 | 11,947 / 23,086 / 21,470 |
| decode, single, 128 / 1,024 / 4,096 prompt tok/s | 255 / 253 / 241 | 253 / 250 / 240 |
| decode aggregate, batch 4 / 8 / 16 tok/s | | 870 / 1,586 / 2,509 |
| MTP k=2, 20 prompts, tok/s | | 356 (acceptance length 2.27, per position 0.81 / 0.46); 231 without MTP |

## What the 4B predicts, and what needs the 27B

Same family, same card, same kernels, 5x fewer weight bytes (5.9 against 28 GB), 8 against 16 attention layers,
hidden 2560 against 5120.

Reliable on the 4B:

- Act-quant against weight-only prefill (#612: 940 to 1,000 ms down to 300 to 360 ms for 1,505 tokens on the 27B,
  about 3x; the 4B shows 2.4x at 1,024). Direction and rough size carry over, so a prefill GEMM change can be
  judged here first.
- Single stream decode is bandwidth bound: 250 tok/s on 5.9 GB is 1.5 TB/s of the 1,640 GB/s read ceiling, and the
  27B without MTP (42 tok/s on about 28 GB, 1.2 TB/s) is in the same regime. Changes to the GEMV read path, the head
  and fusion show up in the same direction, with a different size.
- Batch scaling shape (#604): aggregate decode grows with batch while per sequence falls, 4B 220 to 157 tok/s per
  sequence from batch 4 to 16. A change that lowers batch 8 and 16 here lowers them on the 27B.
- MTP acceptance mechanics (greedy identity, per position rates) are exercised on real weights, though the rate
  itself is a 4B number.
- Correctness of every code path, through the tiny fixture.

Still needs the 27B:

- Kernel dispatch at a given row count (#604: the 17 to 32 row b32 GEMVs lost to the act-quant tile GEMM on the 27B
  at 24 rows). The 4B's GEMMs have half the K and about half the N, so tile and wave occupancy decisions
  (which tile, split count, GEMV against tile crossover) can flip. Confirm any crossover on the 27B.
- Long context decode attention (#617). The 4B has 32 KiB of KV per token against the 27B's 64 KiB, but only 5.9 GB
  of weights against 28 GB, so KV is 2.4x heavier relative to weights here. It shows whether an attention change
  helps and in what direction, but over-states the end-to-end gain, and the split count that fills 188 SMs depends
  on rows times kv_heads, not on the model size.
- Memory limits, KV pool sizing, preemption and prefix cache pressure: the 27B leaves 3 GB beside vLLM, the 4B
  leaves 80 GB.
- Vision speed (#612 replay): the 4B tower is 24 deep, 1,024 wide, against the 27B's 27 deep, 1,152 wide; use the
  27B for timing, the tiny fixture for correctness.
- Absolute tok/s, MTP acceptance rates and anything quoted to the user.
