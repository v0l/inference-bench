# Mixture of experts in specrt

## What runs

Qwen3.5/Qwen3-Next style sparse MLPs: softmax router over `num_experts`, top-k, `norm_topk_prob`, one shared expert with a sigmoid gate, per-layer sparsity from `mlp_only_layers` and `decoder_sparse_step`. Experts are bf16 or NVFP4 (compressed-tensors, per-expert tensors). The spec carries it in `[moe]` (`experts`, `active`, `expert_intermediate`, `shared_intermediate`, `router`, `norm_topk`, `shared_gate`, `dense_layers`, `sparse_step`). `Spec::check_moe_runnable` is the single gate; it refuses FP8 experts, GGUF and MoE with MTP for now.

## Plan and kernels

Sparse layers use `Op::MoeRoute`, `Op::MoeExperts`, `Op::MoeCombine` in place of the dense MLP ops (`Plan::attention_moe`, `Plan::linear_moe`, picked by `Plan::layer_of`).

| Kernel | Job |
| --- | --- |
| `moe_route` | post-attention norm, router GEMV, softmax, top-k, shared gate, per token |
| `moe_up_*`, `moe_down_*` | one block per token and expert pair (decode and small batches) |
| `moe_permute` | counts, prefix sum, scatter of pairs by expert (batches of 17 tokens or more) |
| `moe_group_up_*`, `moe_group_down_*` | one block column per expert, each weight row read once for two tokens |
| `moe_combine` | weighted sum plus shared expert, added into the residual |

The shared expert is stacked as expert index `experts`, so it goes through the same grouped path. NVFP4 expert weights are stacked per layer with a per-expert inverse global scale. These kernels use CUDA cores, not the fp4 mma.

## Correctness

Fixture `tests/fixtures/tiny-qwen3.5-moe` (3 layers, 8 experts, top-2, shared expert, random weights) and references from `scripts/reference.py tiny-moe <fixture> <dir>`. `tests/linear.rs` runs per-layer decode and prefill, logits plus greedy against transformers, and batch decode against single-sequence greedy (batches 2, 8, 16), all within the dense tolerances (1e-3).

## Qwen3.5-122B-A10B NVFP4 on one RTX PRO 6000 (96 GB)

`tests/real122b.rs` (`SPECRT_122B_DIR`, `SPECRT_122B_PROMPT`). Loads in about 25 s. Greedy continuation of "The capital of France is Paris. The capital of Germany is" is " Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. ..." (coherent).

| Measure | specrt now |
| --- | --- |
| decode, batch 1 | about 68 tok/s |
| prefill, 4096 tokens | 10.3 s (about 400 tok/s) |

Not yet measured against vLLM on the same card (vLLM was stopped for this window, no reference logits exist), and no achieved-bandwidth figure. The prefill number is far from vLLM: the grouped kernels do not use tensor cores.

## Not done

Tensor-core grouped GEMM (fp4 mma), FP8 experts, GGUF `ffn_*_exps`, MTP with a MoE layer, vLLM top-1/KL comparison, bandwidth table. These are tracked as issues.
