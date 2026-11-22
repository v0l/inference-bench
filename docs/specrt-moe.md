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

The 122B decode b8 gap to vLLM (365 vs 435) and the 4k prefill gap remain (#652 below).

### #652 profile (nsys, GPU1, 122B, same card)

4k prefill, 0.58 to 0.62 s of kernels (one 4096 token pass, 48 layers, 36 of them linear attention and 12 full attention, every layer MoE):

| Kernel | ms | Share |
| --- | --- | --- |
| `lin_in_mma_big` (GDN input projections) | 98 | 16% |
| `moe_mma_up` (+ `_s`) | 83 + 18 | 17% |
| `flash_mma` | 73 | 12% |
| `moe_route_logits` (before) | 77 | 13% |
| `gdn_chunk_prep` + `gdn_chunk_scan` | 52 + 30 | 14% |
| `moe_mma_down` (+ `_s`) | 49 + 6 | 9% |
| `lin_out_mma_big`, `qkv_mma_big`, `o_mma_big` | 48 + 26 + 14 | 14% |

The expert GEMMs are 26% of prefill. Up does 4096 x 9 pairs x 2 x 1024 x 3072 x 2 FLOP x 48 layers = 11.1 TFLOP in 0.101 s of kernel time, about 110 TFLOP/s on the fp4 mma; down 5.6 TFLOP in 0.055 s, about 100 TFLOP/s. A 128-row tile (`BM=128`, 4 warps) was tried and was slower for up (99 ms against 88 ms), so it was dropped. The rest of the 0.8 s to 0.21 s gap is not the expert GEMM: the dense GDN and attention projections (`lin_in_mma_big`, `lin_out_mma_big`, `qkv_mma_big`, `o_mma_big`, 186 ms) and the GDN chunk kernels (82 ms) are the larger share and are separate kernels from `templates/moe_mma.cu` (that file is now `kir::cuda::moe::mma`).

Changes kept: `moe_route_logits` computes 4 experts per warp and loads the activation tile once per 4 experts (router reads were 13% of prefill, now 5%, 4k prefill 0.607 to 0.575 s), and `moe_tiles` builds the tile map with a block scan instead of one thread walking 257 experts (21 us to a few us, 96 launches a pass).

b8 decode (369 tok/s before and after): `moe_mma_up_s` 133 ms of 387 ms of kernel time (34%, 173 us a call, 48 layers x 16 steps), `moe_mma_down_s` 43 ms, `lin_in_mma_tiny` 47 ms, lm head 15 ms. Bytes: 14.8 GB of active experts per step over about 22 ms of up/down time per step is about 680 GB/s, 38% of the card's peak. Our 16-row tile has one CTA row per expert, so the grid is `inter/64 x active_experts` CTAs of 128 threads. Whether that is too few loads in flight is untested.

b8 decode fix kept: the 16-row tile kernels (`moe_mma_up_s`, `moe_mma_down_s`) use a 6 stage cp.async pipeline instead of 3 (`MMA_STAGES_SMALL`, 6 x 16.5 KB shared memory per CTA). More bytes in flight per CTA is what a 16-row tile lacks. Alternating runs of the same `real_122b_batch_8_decode_speed` binary pairs on GPU1 (3 pairs, base then new): base 362.0, 370.4, 371.6 tok/s, new 383.5, 383.2, 383.0 tok/s, about +3.5% (+5.9% on the first pair). Stage counts 8 and 10 gave 382.0 and no further gain, so 6 stays. vLLM is 435, so b8 is still 12% short of it; the 5% bar was not cleared on the median and `bench --ab` could not be used: it loads both arms into one process and two 122B copies (about 70 GB each) do not fit one 96 GB card, and it varies the spec, not the kernel source. The two-GPU command would be `bench --ab <122b spec A> <122b spec B>` with each arm pinned to its own card, which the bench does not do yet.

The top-1 flip at position 34 (ours 1691, vLLM 303, KL 0.357) is not from this issue: master gives 63/64, mean KL 0.01344, max 0.35707, and so do builds with only the route-logits change, only the tile-map change, both, and both plus the 6 stage kernels, all identical to five digits. The "64 of 64" in the table above predates the #644 merge fixes and is stale; the test thresholds (90%, 0.05) hold on master.

vLLM and FlashInfer, read-only from `~/.venv-vllm/lib/python3.*/site-packages/vllm/model_executor/layers/fused_moe/`:

| Item | vLLM | specrt |
| --- | --- | --- |
| Token tile `BLOCK_SIZE_M` | Marlin path: picks the first of 8, 16, 32, 48, 64 where `M * topk / E / block_m < 0.9` (`experts/marlin_moe.py:309`); Triton default 64 with config files per (E, N, device) in `configs/` (`fused_moe.py:1388`) | 16 when under 32 pairs per expert, else 64; the 8 to 48 steps are missing |
| N tile and pipeline | at M <= 8 the block-scaled default uses N=64 and 4 stages at M <= 4, 3 stages up to 8 (`fused_moe.py:1415`) | BN 64 always; 3 stages, now 6 for the 16-row tile |
| Token packing | `moe_align_block_size` sorts (token, expert) pairs by expert and pads each expert to a multiple of `BLOCK_SIZE_M`, returns `sorted_token_ids` and `expert_ids` per tile (`moe_align_block_size.py:55`); with expert parallelism `expert_map` drops invalid experts | `moe_permute` (one block) plus `moe_tiles` (block scan, no padding, tile map of (expert, m)) |
| Router top-k | separate `topk_softmax` op (`router/fused_topk_router.py:20`, `fused_topk_bias_router.py:30`), not fused with the align step | `moe_route_logits` (norm plus logits, multi-block) then `moe_route` (one warp per token) then `moe_permute`; also not fused |
| CUTLASS / FlashInfer | `experts/flashinfer_cutlass_moe.py:372` calls `flashinfer_cutlass_fused_moe` with `tp_size`/`ep_size`; the grouped GEMM and its persistent scheduler live in the FlashInfer JIT sources (not read here: the 24 GB build cap) | not applicable on sm_120 mma kernels |

No vLLM path fuses the router with the permutation, and none uses a persistent kernel in the Marlin or Triton route we compare against (the 0.21 s prefill uses Marlin). The difference that matters for b8 is the smaller M tiles (8) and deeper pipelining at tiny M, which the stage change partly matches.

Grouped GEMM as a kir family: it already is one (`kir::moe::Family::GroupedMma`, emitted by `kir::cuda::moe::mma`), so the port the issue asks for exists; #609's `select` would only choose among the tile variants, and the measured choice (16-row vs 64-row, stage count) is a constant here. Not done as a separate step because the profile shows no GEMM-shape gap that a different family would close.

Still open for #652: 4k prefill 0.575 s against 0.3 s needs the GDN and attention projection GEMMs (186 ms) and GDN chunk kernels (82 ms), which are outside the expert kernel; b8 is 383 against 435 tok/s.
