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

The shared expert is stacked as expert index `experts`, so it goes through the same grouped path. NVFP4 expert weights are stacked per layer with a per-expert inverse global scale. The mma path (`kir::cuda::moe::mma`) runs when the spec has NVFP4 experts with `[mlp] act_quant = "fp4"` (read from the checkpoint's `input_activations` by `Spec::from_hf_config_serving`) on sm_120, for 8 or more tokens; 64-row tiles when experts get at least 32 tokens each, else 16-row tiles. Below 8 tokens the per-pair GEMV kernels are faster. It is a kir family (`kir::moe::Family::GroupedMma`).

## Correctness

Fixture `tests/fixtures/tiny-qwen3.5-moe` (3 layers, 8 experts, top-2, shared expert, random weights) and references from `scripts/reference.py tiny-moe <fixture> <dir>`. `tests/linear.rs` runs per-layer decode and prefill, logits plus greedy against transformers, and batch decode against single-sequence greedy (batches 2, 8, 16), all within the dense tolerances (1e-3).

## Correctness of the mma path

`tests/moe_mma.rs` compares the three kernels (quantise, up with SwiGLU, quantise, down) against a CPU model that applies the same fp4 activation rounding: 16 experts plus a shared one, top-4, 512 wide, at 1, 8 and 48 tokens. Max error 2.4e-4 of a 384 scale. `tests/linear.rs` (`moe_*`, `moe_fp8_*`) and `tests/mtp.rs` (`moe_*`) cover the tiny fixtures, `tests/legacy_attention.rs` (the kernels now live in `crates/kir/src/cuda/attention/`, fixed there for every shape) loads `attention_split` and `flash_attention` at GQA 32/2 with head dim 256 and checks static plus dynamic shared memory against the device limit. Both used to overflow: `attention_split` staged all query groups (`GROUP x HEAD_DIM` floats per warp) and now merges one group at a time, and `flash_attention` moved its tiles from static to dynamic shared memory sized by `AttentionShape::legacy_flash_smem`. They are still reachable only with `[runtime] gemm_fp32 = true`.

## Qwen3.5-122B-A10B NVFP4 on one RTX PRO 6000 (96 GB)

`tests/real122b.rs` (`SPECRT_122B_DIR`, `SPECRT_122B_PROMPT`, `SPECRT_122B_REF` for the vLLM comparison). Loads in about 25 s. Same card and quiet GPU window for both columns. vLLM from `~/.venv-vllm`, `moe_backend="marlin"` (the FlashInfer fused-MoE JIT build exceeds the 24 GB RAM cap), caches redirected to `/tmp`, nothing in `~/.cache/flashinfer` touched; `ignore_eos`, greedy, max_num_seqs 8, CUDA graphs on.

| Measure | specrt before (#632) | specrt now | vLLM |
| --- | --- | --- | --- |
| decode, batch 1 | about 68 tok/s | about 100 tok/s | 99.9 tok/s |
| decode, batch 8 (aggregate) | not measured | 365 tok/s | 435 tok/s |
| prefill, 4096 tokens | 10.3 s | 0.79 s (13x) | 0.21 s |

Decode b1 got there by splitting `moe_route` (one block read the whole router at 2 bytes a load, 3.6 ms a token) into a multi-block logits kernel and a one-warp select. Prefill went 10.3 s, to 2.98 s (mma), to 0.79 s after the old timing stopped copying 4096 x 248320 logits to the host.

Achieved bandwidth on active-expert bytes (NVFP4 weights plus group scales, 5.3 MB per expert, 3 matrices of 1024 x 3072, 9 experts per token per layer including the shared one, 48 layers):

| Case | Active expert bytes per step | Steps per s | Rate |
| --- | --- | --- | --- |
| decode b1 | 2.29 GB | 100 | about 230 GB/s |
| decode b8 | about 14.8 GB (about 58 distinct experts a layer) | 46 | about 680 GB/s |
| prefill 4096 | all 257 experts a layer, 12.5 GB per token pass | 1.2 passes/s | about 15 GB/s of weights, compute bound |

Batch 1 reaches about 12% of the card's roughly 1.8 TB/s: the expert kernels are about a third of the token time, the rest is the linear attention, attention projections and the head, and 9 x 3 small GEMVs per layer.

### Against vLLM and transformers

`scripts/moe_trace.py` runs the 122B through transformers layer by layer on the free card (NVFP4 weights dequantised per layer, 38 s) and writes the hidden state after every layer. `real_122b_per_layer_trace_matches_the_transformers_oracle` compares specrt with it (bf16 activations on our side, since the oracle has none): the 90th percentile token error stays under 1e-2 of the layer's largest value across all 48 layers and layers 0 to 2 match to 1e-6.

`real_122b_top1_and_kl_match_vllm` feeds vLLM's greedy continuation (64 tokens after a 99 token prompt of prose, code and a question) through a full prefill and compares the top-20 logprobs. With the checkpoint's fp4 activations: top-1 64 of 64, mean KL 0.0106, max 0.24. It asserts the 27B thresholds (top-1 at least 90%, mean KL at most 0.05).

The first version of this comparison gave 43 of 48 and mean KL 0.29. The cause was not MoE: `Runtime::load_layout` left every slot's block table at zero, so with a contiguous layout every KV page aliased block 0. Prompts of 16 tokens or fewer fit in one page and were exact; from 17 tokens the first full-attention layer attended over a corrupted cache. The 27B tests never hit it because they set the table (`load_batched`, `set_block_table`). `load_layout` now gives each slot its own contiguous blocks, and `serving.rs` has a test for it. `tests/flash_prefill.rs` and `tests/gemm_shapes.rs` check the flash kernel and the attention GEMMs at the 122B shapes against a CPU reference, so those are known good.

## Not done

Nothing from the issue list. The 122B decode b8 gap to vLLM (365 vs 435) and the 4k prefill gap (0.79 vs 0.21 s) remain.
