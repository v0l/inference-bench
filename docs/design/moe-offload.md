# Mixture of experts and offload

Sparse MLPs, the expert kernels, and the tiered weight store for models larger than the card (layer streaming, the expert cache, the NVMe tier). Code: `crates/specrt/src/runtime/moe/`, `crates/kir/src/moe.rs`, `crates/kir/src/cuda/moe/`, `crates/specrt/src/offload.rs`, `crates/specrt/src/runtime/stream.rs`, `crates/specrt/src/diskio.rs`.

## MoE

The spec's `[moe]` (from the family spec and `config.json`): `experts`, `active`, `expert_intermediate`, `shared_intermediate`, `router` (`softmax` or `sigmoid` with `score_bias`, groups and `topk_group`), `norm_topk`, `shared_gate`, `dense_layers`, `sparse_step`, `routed_scale`, `router_bias`, `expert_bias`, `hash_table`, `expert_block` with `gate_up_interleaved` (MXFP4), `parallel_dense` with `router_norm`, `router_norm_scale`, `input_norm`, `expert_scale` (Gemma 4), and `fused_experts` tensor names. `Spec::check_moe_runnable` is the single gate. Experts can be bf16, NVFP4, block FP8, MXFP4 or GGUF (`ffn_*_exps`, dequantised at load). MTP works with MoE layers.

Sparse layers plan `Op::MoeRoute`, `Op::MoeExperts`, `Op::MoeCombine`. The shared expert is stacked as index `experts` and goes through the same path.

| Kernel | Job |
|---|---|
| `moe_route_logits` | post attention norm, router and shared gate logits across blocks, 4 experts per warp |
| `moe_route` | softmax or sigmoid, grouped top k, shared gate, one warp per token |
| `moe_up_*`, `moe_down_*` | one block per (token, expert) pair, bf16, fp4, fp8, mx; below 8 tokens |
| `moe_permute`, `moe_tiles` | counts, prefix sum, scatter by expert, tile map by block scan |
| `moe_quant_f32`, `moe_mma_up[_s]`, `moe_mma_down[_s]` | NVFP4 tensor core path (`kir::moe::Family::GroupedMma`): fp4 activations, `mxf4nvf4.block_scale` m16n8k64, gate and up in one pass with SwiGLU in the epilogue; 64 row tiles at 32 or more tokens per expert, else 16 row tiles with 6 cp.async stages (`MMA_STAGES_SMALL`) |
| `moe_group_up_*`, `moe_group_down_*` | CUDA core grouped kernels for non NVFP4 experts |
| `moe_combine` | weighted sum plus shared expert into the residual |

The mma path runs with NVFP4 experts and `[mlp] act_quant = "fp4"` on sm_12x for 8 or more tokens. An 8 row tile (#990) and a 128 row tile were slower and dropped.

## Offload

`[runtime.offload]` (spec, server config, the load dialog's offload group):

| Key | Default | Meaning |
|---|---|---|
| `vram_budget` | 0 (everything resident) | bytes of card memory the weights may use |
| `host_budget` | 0 | cap on pinned host memory for pages; what does not fit goes to `nvme_path` |
| `nvme_path` | none | folder for page files (`<prefix>-<pid>-<random>.pages`, removed on drop and at start for dead pids) |
| `prefetch_depth` | 1 | layers copied ahead |
| `expert_slots` | 0 (derived from `vram_budget`) | VRAM slots for experts |
| `policy` | `score` | `lru`, `frequency`, `score` |
| `promote_margin`, `score_half_life`, `promote_per_step` | 1.15, 64, 16 | score policy |
| `pointer_table` | off | decode a fully resident layer straight from a slot table |
| `hotlist` | none | expert use counts from a calibration run (`--save-hotlist`, or `<spec>.hotlist`) |

Dense layers: the planner keeps embeddings, norms and the head resident, pins the first and last layer, fills VRAM, and streams the rest from pinned host memory through a ring of `prefetch_depth + 1` slots on a second stream, one layer ahead of compute. CUDA graphs are off for a streaming model. Decode runs at the link ceiling (streamed bytes over 57.8 GB/s); prefill is unchanged because compute hides the copy.

Experts: with `vram_budget > 0` on a MoE model routed experts live in pinned host memory (or NVMe pages) and the kernels read them from a slot pool through the router's index buffer, remapped to slots after routing. The score policy keeps a decaying use score per expert; a miss is copied into a scratch slot for that pass only and evicts nothing; after the last MoE layer up to `promote_per_step` high scoring experts are copied into the coldest resident slots on a low priority stream, and a slot is published only when its event completes (polled, never synchronised). Expert biases travel with their weights. With the pointer table, the table is rebuilt from the book each layer and a layer with a miss takes the staged path. NVMe reads go through the shared `diskio` O_DIRECT pool (4 threads, 4 MB chunks).

`OffloadStats` (printed by `bench`, shown in the serving panel): hits, `host_to_vram_bytes`, `nvme_reads`, `nvme_read_bytes`, `nvme_read_failures`, `nvme_wait_nanos`. `/metrics`: `inference_offload_hit_ratio`, `inference_offload_read_bytes_total`.

Why `score`: on the 122B cut to 8 layers it beat `lru` by 1.05x to 1.37x at batch 1 and 1.40x to 2.14x at batch 8, copying 22 to 55 percent fewer bytes (#1224). Below full residency the cost is the host round trip per MoE layer (router output back, plan, copy, fence), not the link: the compute stream is busy 16 to 19 percent of the span. A CPU tier for cold experts measured 2.7x the copy rate on bandwidth (AVX-512 VNNI GEMV at 156 GB/s) but would pay the same round trip, so it was not built (`crates/specrt/src/bin/cpubench.rs`).

## Measured state

Desktop GPU1 unless noted:

| Case | Value | Source |
|---|---|---|
| Qwen3.5-122B-A10B NVFP4 resident, decode b1 / b8 | about 100 / 383 tok/s (vLLM marlin 99.9 / 435) | #652 |
| same, 4k prefill | 0.575 s (vLLM 0.21 s; the gap is bf16 projections and GDN chunk kernels, at or above cuBLAS) | #652, #991 |
| same against vLLM, teacher forced | top-1 63 of 64, mean KL 0.0134 | #644 |
| 27B FP8 with `vram_budget` 12 GB | 2.9 tok/s decode (ceiling 2.9), 1.79k prefill | offload sweep |
| full 122B NVFP4, 90 GB slots, 4 GB host, NVMe pages | 2.0 tok/s decode, 104.6 tok/s prefill, 13 GB host RSS | #1022 |

## Tests

`crates/specrt/tests/moe_mma.rs` (kernels against a CPU model with the same fp4 rounding), `linear.rs` `moe_*`, `mtp.rs` `moe_*`, `moe_route.rs`, `real122b.rs` (per layer trace against transformers, top-1 and KL against vLLM), `offload.rs` (bit identical to resident with streamed layers, the expert cache, NVMe pages, score policy, pointer table, a promotion still copying), and `cargo test -p inference-bench-specrt --lib offload` (CPU). Fixtures: `tests/fixtures/tiny-qwen3.5-moe`, `tiny-glm4-moe`, `tiny-gpt-oss`, `tiny-gemma4-moe`.
