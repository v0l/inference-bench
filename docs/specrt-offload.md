# Offloading: tiered weight store

## Status

Step 1 is in `crates/specrt/src/offload.rs`: the policy and bookkeeping layer, tested on CPU with fake slow tiers. The CUDA data path (copy stream, pinned buffers, real NVMe reads) and the MoE expert cache are not wired yet.

| Piece | State |
| --- | --- |
| `OffloadSettings` (`vram_budget`, `host_budget`, `nvme_path`, `prefetch_depth`, `policy`), TOML parse with defaults | done |
| `TieredStore`: pages with tiers VRAM, host, NVMe; LRU or frequency eviction under per-tier budgets; pinned pages; prefetch with recorded order; no read of a resident page | done, tested |
| `plan_layers`: reserves resident tensors and staging buffers (`prefetch_depth` x largest layer), pins first and last layer, fills VRAM then host, rest NVMe | done, tested |
| `tokens_per_second_ceiling` and `OffloadStats::prometheus` (`inference_offload_hit_ratio`, `inference_offload_read_bytes_total`) | done, tested |
| Throttled fake NVMe test (read time follows bandwidth, second pass is free) | done |
| Layer streaming on a second CUDA stream, bit-identical logits with 1 layer resident | not done: needs GPU, see below |
| MoE expert cache, hotlist | blocked on routed-expert kernels being page-addressable; the `[moe]` block and kernels exist (`docs/specrt-moe.md`), the cache keyed by (layer, expert) is the same `TieredStore` with expert pages |
| CPU compute tier | not evaluated |
| UI readout | not done |

## Design

- Page = one layer's weights (dense) or one expert (MoE). Always-needed tensors (norms, embeddings, lm_head, attention for MoE) are resident and counted in `always_resident`.
- Tier 1 pinned memory comes from `hostpool::HostPool`; KV offload and session snapshots must share it and the same I/O thread pool.
- Decode of a dense model reads every streamed byte every token, so speed is bounded by link bandwidth. MoE reads only routed experts, so the hit ratio of the expert cache sets the speed.
- Prefetch: for dense, layer N+1 while N computes (`prefetch_depth` buffers). For MoE, the previous token's routing is the hotlist for the next layer.

## Ceilings (tokens/s upper bound, one stream, decode)

`ceiling = 1 / max(streamed bytes / link, all bytes / HBM read rate)`, computed by `OffloadPlan::tokens_per_second_ceiling`. Assumptions: PCIe 5 x16 about 50 GB/s host to GPU, NVMe 10 GB/s, HBM effective 1500 GB/s. Weight sizes: 27B at FP8 about 27 GB, NVFP4 about 15 GB, bf16 54 GB.

| Format | VRAM budget | Streamed per token | Ceiling from host RAM | Ceiling from NVMe |
| --- | --- | --- | --- | --- |
| FP8 27 GB | 48 GB | 0 | 55 | 55 |
| FP8 27 GB | 24 GB | about 5 GB | about 10 | about 2 |
| FP8 27 GB | 12 GB | about 17 GB | about 3 | about 0.6 |
| NVFP4 15 GB | 12 GB | about 5 GB | about 10 | about 2 |

Figures are arithmetic from the assumptions above, not measurements. This session found the desktop GPU 0 busy (57 GB used by the live app) and read link state at idle (gen1 x16, which downshifts at idle), so measured bandwidth is still to be taken.

## GPU handoff (not run here: the user's app holds GPU 0, and GPU work stays off while it serves)

1. Wire a copy stream and a spare VRAM staging buffer into the loader; with a tiny fixture and `vram_budget` forcing 1 resident layer, assert logits bit-identical to the fully resident run.
2. On GPU 1 under `~/.agents/bin/gpu-run 1 --`, run the 27B with `vram_budget` 12, 24, 48 GB and record tokens/s against the ceilings above.
3. Measure `cudaMemcpyAsync` from pinned memory and `dd iflag=direct` on the model drive for the real link figures.
