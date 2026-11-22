# Offloading: tiered weight store

## Status

| Piece | State |
| --- | --- |
| `[runtime.offload]` in the model spec and the server config (`vram_budget`, `host_budget`, `nvme_path`, `prefetch_depth`, `policy`), also in the options schema (group "offload") | done |
| `crates/specrt/src/offload.rs`: pages, tiers, LRU or frequency eviction, prefetch order, planner, ceiling, stats | done, 17 CPU tests |
| Dense layer streaming on a second CUDA stream: layers over `vram_budget` live in pinned host memory and copy into a ring of `prefetch_depth + 1` shared VRAM slots, one layer ahead of compute (`runtime/stream.rs`) | done, bit identical on GPU |
| `inference_offload_hit_ratio`, `inference_offload_read_bytes_total` on `/metrics` | done, rendered when a model streams |
| NVMe tier on the GPU path | planner knows it; the loader refuses a budget that needs it |
| MoE expert cache: expert pages (layer, expert) in pinned host memory, a shared pool of VRAM slots, missing experts copied after the router picks them, router output of the previous token as prefetch for the next MoE layer, `ExpertHotlist` counts saved to a file and loaded through `[runtime.offload] hotlist` | done, bit identical on the tiny MoE fixtures (bf16 and FP8) |
| CPU compute tier, Models page readout | separate issues |

Settings (all in the spec or server config, no env vars):

```toml
[runtime.offload]
vram_budget = 24000000000
host_budget = 0
prefetch_depth = 1
policy = "lru"
```

`vram_budget` is the bytes of card memory the weights may use. The planner keeps embeddings, norms and the head resident, pins the first and last layer, fills VRAM with layers, and puts the rest in pinned host memory. Streaming needs `prefetch_depth + 1` staging slots of the largest layer, which count against the budget. CUDA graphs are switched off for a streaming model.

## Measured links (this desktop, GPU 1, RTX PRO 6000)

| Link | Result |
| --- | --- |
| Pinned host to device, 1 GiB x 5 (`linkbench`) | 57.8 GB/s every run |
| Pinned device to host | 57.4 GB/s |
| Model drive, `dd iflag=direct bs=4M` on the Samsung 9100 PRO 4 TB (root, where the HF cache lives) | 10.0 to 10.3 GB/s |

The ceiling function takes these as inputs: `tokens_per_second_ceiling(host_gbs, nvme_gbs, resident_read_gbs)`.

## 27B FP8 (Qwen3.8-27B-FP8, 31.06 GB of weights), batch 1 decode and 2k prefill

`bench --runs 2 --vram-gb N` on GPU 1, 2048 token prompt, 128 new tokens, 3 alternating rounds, medians of 2 runs per round. Ceiling is `1 / max(streamed bytes / 57.8 GB/s, resident-speed time)`, resident speed taken from the 48.5 tok/s baseline.

| vram_budget | Layers streamed | Streamed per token | Ceiling tok/s | Decode tok/s (3 rounds) | Reached | Prefill tok/s (3 rounds) | Device memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fully resident | 0 | 0 | 48.5 | 48.7, 48.3, 48.4 | 100% | 1804, 1774, 1768 | 32.9 GB |
| 24 GB | 21 of 64 | 8.01 GB | 7.2 | 7.1, 7.1, 7.1 | 98% | 1815, 1786, 1786 | not remeasured after the trim |
| 12 GB | 52 of 64 | 19.83 GB | 2.9 | 2.9, 2.9, 2.9 | 100% | 1795, 1786, 1789 | 14.0 GB (rerun with the trim: 2.9 tok/s, 1782 prefill) |

Device memory with the 12 GB cap is 13.99 GB with a 2048 token prompt (weights, staging, KV, scratch), measured on the merged head after the allocator trim. The sweep rows above ran before the trim, so their `peak_device` read 31.4 GB; throughput is unaffected.

Offload off is unchanged on the 27B FP8: master binary against the merged head, batch 1, 128 prompt and 128 new tokens, 3 alternating rounds with a second master copy as the A/A control. Decode B/A 0.994, 0.998, 0.998 (median 0.998), A/A median 0.994; prefill B/A median 0.997, A/A median 0.991. B/A sits inside the A/A band.

Streaming decode runs at the link ceiling: the copy of layer N+1 hides behind compute of layer N, and the link is the limit. Prefill is unaffected (about 1.78k tok/s in every row) because 2048 rows of compute per layer take longer than the copy.

The 48 GB row of the plan is the resident row: the 27B FP8 fits 33 GB, so a 48 GB cap streams nothing.

### Ceiling per format at batch 1

Bytes per token over the measured 57.8 GB/s, for a 27B model with nothing resident (worst case) and with half resident:

| Format | Weights | All streamed | Half resident |
| --- | --- | --- | --- |
| bf16 | 54 GB | 1.1 tok/s | 2.1 tok/s |
| FP8 | 27 GB | 2.1 tok/s | 4.3 tok/s |
| NVFP4 | 15 GB | 3.9 tok/s | 7.7 tok/s |

From NVMe at 10 GB/s the same table divides by 5.8. Lower bits stream faster in proportion; speculative decoding (MTP) does not change bytes per step, so it multiplies tokens per streamed byte by the accepted draft length.

## Design notes

- Dense decode reads every layer once per token, so tok/s is bounded by link bandwidth over streamed bytes. Overlap is all the design can add, and the sweep shows it is complete.
- Staging slots are reused round robin. A layer waits on an event recorded when its copy finishes; a slot is refilled only after an event recorded when the layer that last used it finished.
- Pinned buffers come from `hostpool::HostPool`, the pool KV swap and session snapshots use.
- The tiered store in `offload.rs` is the bookkeeping that the MoE expert cache will drive (pages keyed by layer and expert).

## Expert cache (MoE)

With `vram_budget > 0` on a MoE model the routed experts of every layer live in pinned host memory and the kernels read them from a slot pool through the router's index buffer (the index is remapped to slots after routing, so the kernels are unchanged). `expert_slots` sets the pool size (0 derives it from `vram_budget`); `hotlist` names a JSON file of use counts (`Runtime::save_expert_hotlist` writes it after a calibration run) whose hottest experts fill the pool at load and guide prefetch until the router has history. Prefill with more distinct experts than slots runs in groups. Not run yet: GLM-4.5-Air real weights (not downloaded here; the tiny GLM-4 MoE fixture with shared experts is bit identical with the cache).

NVMe tier: `host_budget` caps the pinned host memory for expert pages. A layer whose pages do not fit is written to a page file under `nvme_path` (removed on drop) and each missing expert is read from it with `pread` into a pinned staging buffer and copied to its slot. Full Qwen3.5-122B NVFP4 on GPU 1 (`bench --vram-gb 90 --expert-slots 4500 --host-gb 4 --nvme-path ~/ib-nvme1025 512 128`): loads and runs with 13 GB host RSS, 20.5 GB device, 2.4 tok/s decode, 39.8 tok/s prefill, hit ratio 0.47 with the synchronous one-expert pread reader (#1025). With the reader on the shared `diskio` O_DIRECT pool (#1022, 4 threads, 4 MB chunks, a layer's missing experts read concurrently into pinned staging): 104.6 tok/s prefill (median of 6, 2.6x), 2.0 tok/s decode (median of 6, no gain, within noise of 2.4 and a bit lower), hit ratio 0.51, same 13 GB host RSS and 20.5 GB device. Prefill gains from the concurrent big reads; decode misses are a few small page reads per layer in series with compute, so it needs the overlap work in #1024, not more read bandwidth.

Drive (Samsung 9100 PRO, root fs, `diskio::tests::drive_bandwidth_by_threads_and_chunk`, 4 GB file, O_DIRECT in use, `dd iflag=direct bs=4M` gives 9.9 GB/s): 1 thread 9.0 / 10.8 / 12.3 GB/s for 1 / 2 / 4 MB chunks; 2 threads 12.8 / 14.2 / 14.6; 4 threads 14.5 / 14.6 / 14.5; 8 threads 14.6 / 14.6 / 14.0. The pool defaults to 4 threads and 4 MB chunks. If O_DIRECT is refused, `DirectFile` logs once and falls back to buffered reads. Layer streaming from disk charges its pinned staging slots against `host_budget` in `plan_layers` (`host_staging_bytes`), and page files are named `<prefix>-<pid>-<random>.pages` with files of dead pids removed at start. The 8 layer cut through the NVMe path gives the same first 64 tokens as resident. `tests/offload.rs` has `expert_pages_on_nvme_are_bit_identical` and `expert_cache_with_glm4_moe_experts_is_bit_identical`.

Qwen3.5-122B NVFP4 (Sehyo, 76 GB) on GPU 1, first 8 of 48 layers (`bench --layers 8`, every MoE layer kept, the full model was not run), 512 token prompt, 128 new tokens, batch 1, 3 alternating rounds of 2 runs each, greedy. Cache is 256 slots shared by all 8 layers (2048 experts, 12.5%), `bench --vram-gb 90 --expert-slots 256`.

| Mode | Decode tok/s (3 rounds) | Prefill tok/s | Device memory | Cache hit ratio | Host to VRAM |
| --- | --- | --- | --- | --- | --- |
| resident | 368.6, 368.5, 368.6 | 27.6k | 22.75 GB | n/a | 0 |
| 256 expert slots | 154.1, 154.8, 154.5 | 2.1k | 13.20 GB | 0.659 | 77.3 GB over the whole bench (prefill, warm up, 2 runs) |

The first 64 greedy tokens are identical to resident in all three rounds. Missing experts cost about 5.3 MB each (gate, up, down plus scales), so 64 routed experts per token with a 34% miss rate is about 0.12 GB per token and a link ceiling of about 480 tok/s at 57.8 GB/s; the measured 154 tok/s is far below it because loads are issued after the router and the compute stream waits for them (only the next layer's predicted experts overlap). Closing that gap is the follow-up (issue filed). Prefill is slow for the same reason: 512 distinct tokens touch nearly every expert in every layer. Prefill (17 or more tokens) now clears the unpinned slots when most of the layer's experts are missing and copies the misses in file order, merging neighbours into one copy per tensor: 2.1k to 2.5k tok/s (same bench, GPU 1). That is the link ceiling: a prefill must move 8 layers x about 256 experts x 5.3 MB = 11 GB, 0.19 s at 57.8 GB/s, so streaming whole layers through a staging ring cannot beat it. Only a larger pool (fewer layers cut) or a faster link raises it. First 64 greedy tokens unchanged.

Hotlist: a calibration run saved by `--save-hotlist` and loaded by `--hotlist` (or found as `<spec>.hotlist`) changed misses over a 16 token run from 4160 to 4104 on the real model (the hottest 256 of 2048 experts only cover so much), and on the tiny fixture the first decode token has zero misses with the hotlist against 6 or more cold (`tests/offload.rs`).

## Tests

- `cargo test -p inference-bench-specrt --lib offload` (CPU, 17 tests: residency, eviction under budget, prefetch order, no re-read of a resident page, throttled fake NVMe).
- `tests/offload.rs` (GPU, `--include-ignored`): logits and per layer outputs bit identical to fully resident with 1 and 2 resident layers (3 layer, 4 layer and FP8 fixtures, slot reuse), identical greedy tokens, no streaming when the budget holds everything.
- `core` render test and `server` config test for the metrics and the config table.

## Not done here

- NVMe tier on the GPU path (needs the I/O thread pool shared with KV offload and snapshots).
- MoE expert cache and hotlist, CPU compute tier, Models page readout (filed as issues).
