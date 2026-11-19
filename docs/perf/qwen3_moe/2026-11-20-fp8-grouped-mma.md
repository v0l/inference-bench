# qwen3_moe FP8 prefill, grouped tensor core experts (#885)

Spark GB10, `Qwen/Qwen3-30B-A3B-FP8`, built from branch `issue-885`.

## Result

The grouped FP8 MMA expert path (`moe_quant_f8`, `moe_mma_up_f8`, `moe_mma_down_f8`, tokens sorted per expert, activations quantised to E4M3 with a 32 wide scale, weight block-128 scales applied per K block) is opt-in: `[mlp] act_quant = "fp8"` in the spec, or `bench --fp8-act`. It is not the serving default because it does not fit the accuracy envelope.

| path | prefill 1024 | prefill 8192 |
|---|---|---|
| default (per-expert GEMV, bf16-exact activations) | 253.7 | 283.5 |
| `act_quant = "fp8"` (grouped MMA) | 2982.3 | 2558.0 |
| vLLM (2026-11-19 pass) | 3959.2 | 6064.5 |

Paired A/B, `perfpass.sh --ab`, 3 rounds, master vs this branch with `--fp8-act`: prefill 1024 B/A 11.5x (rounds 2570.9, 3293.0, 3297.8 against 224.5, 317.7, 132.0), prefill 8192 B/A 8.7x (2379.3, 1267.9, 2703.1 against 274.6, 154.2, 264.0). The script labels both "refuse (noisy)" because the Spark carries live load (Frigate, vLLM) and the rounds vary 2x; the direction is not in doubt. Decode is level within that noise.

## Accuracy against fp32 transformers

`crates/specrt/scripts/envelope_moe_fp8.py` rebuilds the checkpoint layer by layer in fp32 (FP8 weights dequantised with their block scales) and writes the reference logits; `envelope.py compare` scores specrt. 8 real prompts, 310 rows (the first position of each prompt dropped).

| path | mean KL | max KL | top-1 of 310 |
|---|---|---|---|
| default (bf16 activations, GEMV experts) | 6.97e-07 | 1.70e-04 | 310 |
| `act_quant = "fp8"` | 2.94e-03 | 1.09e-01 | 306 |

The FP8 activation path is above vLLM's distance from fp32 on the dense families in `bf16-activation-envelope.md` (4e-4 to 1.2e-3 mean, 2e-3 to 7.7e-2 max), so it stays opt-in. A first version with a 128 wide activation scale was worse (5.2e-3 mean, 0.54 max, 302 top-1). vLLM was not run on this MoE checkpoint.

## Prefill profile 1024 with the opt-in path

`moe_mma_up_f8` 32.5%, `moe_mma_down_f8` 19.2%, `qkv_mma_big` 11.8%, `o_mma_big` 10.5%, `flash_mma` 9.6%, `moe_route_logits` 6.5% (424 us per layer, was 520 us), `moe_combine` 5.6%.

`moe_route_logits` now handles 4 tokens per block so each router row is read once per 4 tokens; it is still bandwidth shaped, a tensor core tile for the 129 x 2048 router GEMM is the next step.

## GPU suites on the Spark (references from transformers 5.23)

`linear` 34 passed, `moe_mma` 5 passed, `moe_mma_fp8` 3 passed, `variants` 19 passed, `tiny27b` 8 passed.
