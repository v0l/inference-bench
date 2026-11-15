# Mixture of experts in specrt

## What runs

Qwen3.5/Qwen3-Next style sparse MLPs: softmax router over `num_experts`, top-k, `norm_topk_prob`, one shared expert with a sigmoid gate, per-layer sparsity from `mlp_only_layers` and `decoder_sparse_step`. Experts are bf16 or NVFP4 (compressed-tensors, per-expert tensors). The spec carries it in `[moe]` (`experts`, `active`, `expert_intermediate`, `shared_intermediate`, `router`, `norm_topk`, `shared_gate`, `dense_layers`, `sparse_step`). `Spec::check_moe_runnable` is the single gate. Experts can be bf16, NVFP4, block FP8 (Qwen3.6-35B-A3B-FP8) or come from a GGUF `qwen3moe` file (`ffn_*_exps`, dequantised to bf16 at load). MTP works with a MoE layer: the draft layer's MLP is routed like any other layer (`scripts/moe_mtp_fixture.py` builds the tiny fixture).

## Plan and kernels

Sparse layers use `Op::MoeRoute`, `Op::MoeExperts`, `Op::MoeCombine` in place of the dense MLP ops (`Plan::attention_moe`, `Plan::linear_moe`, picked by `Plan::layer_of`).

| Kernel | Job |
| --- | --- |
| `moe_route_logits` | post-attention norm, router and shared-gate GEMV across blocks (one block row of 8 experts each) |
| `moe_route` | softmax, top-k, shared gate, one warp per token |
| `moe_up_*`, `moe_down_*` | one block per token and expert pair (CUDA cores; bf16, fp4, fp8) |
| `moe_permute` | counts, prefix sum, scatter of pairs by expert (batches of 17 tokens or more) |
| `moe_group_up_*`, `moe_group_down_*` | CUDA-core grouped kernels for non-NVFP4 experts |
| `moe_quant_f32`, `moe_tiles`, `moe_mma_up[_s]`, `moe_mma_down[_s]` | NVFP4 tensor-core path: activations quantised to fp4, one launch per GEMM over per-expert tiles of the permuted tokens, `mma.kind::mxf4nvf4.block_scale` m16n8k64, gate and up in one pass with SwiGLU in the epilogue, then down |
| `moe_combine` | weighted sum plus shared expert, added into the residual |

The shared expert is stacked as expert index `experts`, so it goes through the same grouped path. NVFP4 expert weights are stacked per layer with a per-expert inverse global scale. The mma path (`templates/moe_mma.cu`) runs when the spec has NVFP4 experts with `[mlp] act_quant = "fp4"` (read from the checkpoint's `input_activations` by `Spec::from_hf_config_serving`) on sm_120, for 8 or more tokens; 64-row tiles when experts get at least 32 tokens each, else 16-row tiles. Below 8 tokens the per-pair GEMV kernels are faster. It is a template next to `moe.cu`, not a kir family (#609 has not merged here).

## Correctness

Fixture `tests/fixtures/tiny-qwen3.5-moe` (3 layers, 8 experts, top-2, shared expert, random weights) and references from `scripts/reference.py tiny-moe <fixture> <dir>`. `tests/linear.rs` runs per-layer decode and prefill, logits plus greedy against transformers, and batch decode against single-sequence greedy (batches 2, 8, 16), all within the dense tolerances (1e-3).

## Correctness of the mma path

`tests/moe_mma.rs` compares the three kernels (quantise, up with SwiGLU, quantise, down) against a CPU model that applies the same fp4 activation rounding: 16 experts plus a shared one, top-4, 512 wide, at 1, 8 and 48 tokens. Max error 2.4e-4 of a 384 scale. `tests/linear.rs` (`moe_*`, `moe_fp8_*`) and `tests/mtp.rs` (`moe_*`) cover the tiny fixtures, `tests/legacy_attention.rs` loads `attention_split` and `flash_attention` at GQA 32/2 with head dim 256 and checks static plus dynamic shared memory against the device limit. Both used to overflow: `attention_split` staged all query groups (`GROUP x HEAD_DIM` floats per warp) and now merges one group at a time, and `flash_attention` moved its tiles from static to dynamic shared memory sized by `legacy_flash_smem`. They are still reachable only with `[runtime] gemm_fp32 = true`.

## Qwen3.5-122B-A10B NVFP4 on one RTX PRO 6000 (96 GB)

`tests/real122b.rs` (`SPECRT_122B_DIR`, `SPECRT_122B_PROMPT`, `SPECRT_122B_REF` for the vLLM comparison). Loads in about 25 s. Same card and quiet GPU window for both columns. vLLM from `~/.venv-vllm`, `moe_backend="marlin"` (the FlashInfer fused-MoE JIT build exceeds the 24 GB RAM cap), caches redirected to `/tmp`, nothing in `~/.cache/flashinfer` touched; `ignore_eos`, greedy, max_num_seqs 8, CUDA graphs on.

| Measure | specrt before (#632) | specrt now | vLLM |
| --- | --- | --- | --- |
| decode, batch 1 | about 68 tok/s | about 96 tok/s | 95.9 tok/s |
| decode, batch 8 (aggregate) | not measured | 346 tok/s | 430 tok/s |
| prefill, 4096 tokens | 10.3 s | 0.85 s (12x) | 0.22 s |

Decode b1 got there by splitting `moe_route` (one block read the whole router at 2 bytes a load, 3.6 ms a token) into a multi-block logits kernel and a one-warp select. Prefill went 10.3 s, to 2.98 s (mma), to 0.85 s after the old timing stopped copying 4096 x 248320 logits to the host.

Achieved bandwidth on active-expert bytes (NVFP4 weights plus group scales, 5.3 MB per expert, 3 matrices of 1024 x 3072, 9 experts per token per layer including the shared one, 48 layers):

| Case | Active expert bytes per step | Steps per s | Rate |
| --- | --- | --- | --- |
| decode b1 | 2.29 GB | 96 | about 220 GB/s |
| decode b8 | about 14.8 GB (about 58 distinct experts a layer) | 43 | about 640 GB/s |
| prefill 4096 | all 257 experts a layer, 12.5 GB per token pass | 1.2 passes/s | about 15 GB/s of weights, compute bound |

Batch 1 reaches about 12% of the card's roughly 1.8 TB/s: the expert kernels are about a third of the token time, the rest is the linear attention, attention projections and the head, and 9 x 3 small GEMVs per layer.

### Against vLLM logits

`real_122b_top1_and_kl_match_vllm` feeds our greedy-from-vLLM continuation (48 tokens after the capital-cities prompt) through a full prefill and compares the top-20 logprobs. With `[mlp] act_quant = "fp4"` (the checkpoint's setting) top-1 agrees on 43 of 48 positions, mean KL 0.29; with bf16 activations 41 of 48, mean KL 0.29. The mean KL of 0.29 is higher than the dense 27B comparisons and its cause is not isolated (the tiny MoE fixtures match transformers to 1e-3, so the kernels are right at that size; at 122B depth the candidates are bf16 versus fp32 residual handling and vLLM's Marlin weight-only path). The misses are 5 to 7 of 48 positions, all in repetitive text. The test currently asserts at most 6 top-1 misses and mean KL under 0.4.

## Not done

Nothing from the issue list. `SPECRT_122B_ACT=0` runs the 122B with bf16 activations for comparison. The 122B decode b8 gap to vLLM (346 vs 430) and the 4k prefill gap (0.85 vs 0.22 s) remain.
