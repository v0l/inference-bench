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

Offload counters (`OffloadStats`, printed by `bench` as `offload stats` and `nvme:` lines and shown in the serving panel; layer streaming and the expert reader both feed them): `nvme_reads` is the number of read requests submitted to the disk (one per layer from disk, or one per contiguous expert run per weight part, so a read is not always one expert); `nvme_read_bytes` is the bytes those requests asked for; `nvme_read_failures` is the requests that returned an error; `nvme_wait_nanos` is the time the expert reader spent blocked on read tickets (the cost the compute stream pays for a miss). `host_to_vram_bytes` is what was copied to the card, whether it came from pinned host memory or from a disk read.

Drive (Samsung 9100 PRO, root fs, `diskio::tests::drive_bandwidth_by_threads_and_chunk`, 4 GB file, O_DIRECT in use, `dd iflag=direct bs=4M` gives 9.9 GB/s): 1 thread 9.0 / 10.8 / 12.3 GB/s for 1 / 2 / 4 MB chunks; 2 threads 12.8 / 14.2 / 14.6; 4 threads 14.5 / 14.6 / 14.5; 8 threads 14.6 / 14.6 / 14.0. The pool defaults to 4 threads and 4 MB chunks. If O_DIRECT is refused, `DirectFile` logs once and falls back to buffered reads. Layer streaming from disk charges its pinned staging slots against `host_budget` in `plan_layers` (`host_staging_bytes`), and page files are named `<prefix>-<pid>-<random>.pages` with files of dead pids removed at start. The 8 layer cut through the NVMe path gives the same first 64 tokens as resident. `tests/offload.rs` has `expert_pages_on_nvme_are_bit_identical` and `expert_cache_with_glm4_moe_experts_is_bit_identical`.

Qwen3.5-122B NVFP4 (Sehyo, 76 GB) on GPU 1, first 8 of 48 layers (`bench --layers 8`, every MoE layer kept, the full model was not run), 512 token prompt, 128 new tokens, batch 1, 3 alternating rounds of 2 runs each, greedy. Cache is 256 slots shared by all 8 layers (2048 experts, 12.5%), `bench --vram-gb 90 --expert-slots 256`.

| Mode | Decode tok/s (3 rounds) | Prefill tok/s | Device memory | Cache hit ratio | Host to VRAM |
| --- | --- | --- | --- | --- | --- |
| resident | 368.6, 368.5, 368.6 | 27.6k | 22.75 GB | n/a | 0 |
| 256 expert slots | 154.1, 154.8, 154.5 | 2.1k | 13.20 GB | 0.659 | 77.3 GB over the whole bench (prefill, warm up, 2 runs) |

The first 64 greedy tokens are identical to resident in all three rounds. Missing experts cost about 5.3 MB each (gate, up, down plus scales), so 64 routed experts per token with a 34% miss rate is about 0.12 GB per token and a link ceiling of about 480 tok/s at 57.8 GB/s; the measured 154 tok/s is far below it because loads are issued after the router and the compute stream waits for them (only the next layer's predicted experts overlap). Closing that gap is the follow-up (issue filed). Prefill is slow for the same reason: 512 distinct tokens touch nearly every expert in every layer. Prefill (17 or more tokens) now clears the unpinned slots when most of the layer's experts are missing and copies the misses in file order, merging neighbours into one copy per tensor: 2.1k to 2.5k tok/s (same bench, GPU 1). That is the link ceiling: a prefill must move 8 layers x about 256 experts x 5.3 MB = 11 GB, 0.19 s at 57.8 GB/s, so streaming whole layers through a staging ring cannot beat it. Only a larger pool (fewer layers cut) or a faster link raises it. First 64 greedy tokens unchanged.

Hotlist: a calibration run saved by `--save-hotlist` and loaded by `--hotlist` (or found as `<spec>.hotlist`) changed misses over a 16 token run from 4160 to 4104 on the real model (the hottest 256 of 2048 experts only cover so much), and on the tiny fixture the first decode token has zero misses with the hotlist against 6 or more cold (`tests/offload.rs`).

### Score policy: lagged refill off the token's path (#1169)

`policy = "score"` replaces the slot book with `offload::ExpertBook`. Each expert (layer, expert) keeps a use score: every forward pass adds its share of the routed rows and the score decays by half every `score_half_life` passes (default 64). The pool is split into cache slots and a few scratch slots (`scratch_slots`: the larger of top_k and a sixteenth of the pool). A routed expert that is resident is read from its cache slot; a miss is copied into a scratch slot on the copy stream for this pass only (the same staged path as before), so a miss still costs its copy but never displaces a hot expert. Prefill with more misses than scratch slots borrows the coldest unrouted cache slots, or runs in groups.

After the last cached MoE layer of a pass, `end_step` picks up to `promote_per_step` (default 16) of the best scored non-resident experts and copies each into the cache slot of the coldest resident on a lowest-priority stream, recording one event per copy. A resident is evicted only when the newcomer's score is over `promote_margin` (default 1.15) times its own. A slot is published as resident only when its event has completed (`CudaEvent::is_complete`, polled at the start of every cached MoE layer, the stream is never synchronised); until then the expert is served through scratch. With a hotlist the scores are seeded from it and the cache filled before the first token.

CPU tests in `offload.rs` with a fake copy engine and clock: `promotion_evicts_a_resident_only_past_the_margin`, `an_expert_still_copying_is_served_from_scratch`, `scores_decay_by_half_each_half_life`, `alternating_experts_do_not_thrash_the_cache` (at most 2 evictions in 1000 alternating steps with the margin, over 400 without), `hotlist_seed_fills_the_cache_before_the_first_token`, `misses_past_the_scratch_slots_split_into_groups_or_borrow_cold_slots`, `a_borrowed_slot_reused_by_a_later_group_belongs_to_the_last_expert_copied`. GPU tests in `tests/offload.rs`: `score_policy_expert_cache_is_bit_identical_and_promotes_between_steps` and `score_policy_hotlist_seed_warms_the_cache_and_stays_identical`. The default stays `lru` until the 122B A/B below is run.

With `pointer_table = true` (#1165) the score policy feeds the table from `ExpertBook`. (a) The table is rebuilt from the book at the start of every cached MoE layer, after polling the promotion events, and uploaded on the compute stream only when it changed. An eviction leaves the table at once, a promoted expert enters it only once its copy event completed, and residency only changes between steps (`end_step` after the last cached MoE layer, on the table route too), never inside a layer. A layer decoded from the table credits every one of its experts `top_k / local` per pass, so table-routed residents keep their scores. (b) When the pool holds every cached expert (`expert_slots >= local experts x cached layers`), there are no scratch slots and `promote_per_step` is forced to 0: everything is resident, the score policy has nothing to evict, and it behaves like the table on its own. (c) Scratch slots sit above the cache slots and are never booked as resident, so a borrowed or scratch slot never appears in the table; a layer with a miss takes the staged path. Tests: `the_table_changes_only_when_a_promotion_lands_or_an_expert_leaves`, `a_pool_that_holds_every_cached_expert_has_no_scratch_and_promotes_nothing`, `scratch_slots_never_appear_in_the_pointer_table`, and on GPU `score_policy_with_pointer_table_is_bit_identical`.

Not measured yet (GPU 1 was held by the user's app): `bench --vram-gb 90 --expert-slots 256 --layers 8 --ab --a-set runtime.offload.policy=lru --b-set runtime.offload.policy=score <122B spec> 1 2000`, plus the hit ratio of each arm over the 2000 tokens.

## CPU compute tier for cold experts (evaluation)

Question: for a missing expert, is a GEMV on the CPU from host RAM faster than copying it over PCIe (57.8 GB/s pinned H2D)? Measured with `crates/specrt/src/bin/cpubench.rs` (`cargo build --release --bin cpubench`, `cpubench --threads N --pool-mb 3072`), on this desktop: AMD Threadripper 9960X (24 cores, 48 threads, Zen 5), 24 MiB L2, 128 MiB L3, 62 GB RAM.

CPU features: AVX-512 F/BW/VL/DQ/VNNI/BF16/VBMI/IFMA are present, AMX is not (AMD). The bench kernel is an AVX-512 VNNI int8 GEMV on NVFP4-packed rows (nibble LUT via `vpshufb`, then `vpdpbusd`), checked equal to a scalar reference. The expert is 3 matrices of 1024 x 3072 FP4 (4.72 MB, same size class as the 5.3 MB Qwen3.5-122B expert).

| Measurement | Result |
| --- | --- |
| Host RAM streaming read, 48 threads | 159 GB/s (24 threads 158, 8 threads 137) |
| Pinned host to device (from above) | 57.8 GB/s |
| CPU GEMV on cold experts (4.7 MB each, pool 3 GB, beyond L3), 48 threads | 156 GB/s of weights, 33.0k experts/s, 30 us per expert |
| Same, 24 threads / 8 threads | 151 / 131 GB/s |
| Copy over PCIe, one expert | 12.2k experts/s, 82 us per expert |
| CPU GEMV on an expert held in cache (compute ceiling, 48 threads) | 1.9 TB/s of weights |

Split of the work: the CPU GEMV is memory bound, not compute bound. It runs at the host read bandwidth (2.7x the PCIe copy rate) and the compute ceiling is 12x above that, so extra tokens that hit the same expert are nearly free on the CPU once its weights are in cache (about 2.5 us per extra token against 30 us for the first). The data that must cross PCIe for a CPU expert is only the activation row (3072 values) down and the 3072 outputs back, a few KB.

Break-even:

- Batch 1 decode: a cold expert costs 30 us on the whole CPU against 82 us to copy, so CPU wins on bandwidth by 2.7x, provided all 48 threads are free and the host link and the CPU share the memory controller. The pinned copy engine also reads host RAM, so running both at once splits the 159 GB/s: copy plus CPU together can serve about 159 / 4.72 MB = 33k experts/s, against 12k for the copy alone.
- Prefill and large batches: the copy is paid once per expert and the compute is shared by every token routed to it, while the CPU cost grows with tokens per expert (about 2.5 us per token past the first, against the GPU doing it for free). Prefill touches nearly every expert with hundreds of tokens each, so CPU loses there. The cutoff is roughly 30 tokens on one expert (82 us copy against 30 + 2.5 x n us).
- The CPU tier does not touch the measured gap in the cache bench above (154 tok/s against a 480 tok/s link ceiling): that gap is the compute stream waiting on loads issued after the router, which a CPU path would still pay (GPU to CPU sync of the router output, CPU GEMV, CPU to GPU result, per layer, per token).

Decision: not built. The bandwidth win is real (2.7x, up to about 2.8x combined with the copy engine) but only for decode with few tokens per expert, and today's decode loss is dominated by load issue latency rather than link rate, so the first fix is issuing loads before the router completes or speculating from the previous layer. Revisit once decode reaches the link ceiling. Cost to build if revisited: an AVX-512 VNNI NVFP4 expert GEMV (the bench kernel is the start), activation quantisation to int8 on the CPU path (numerics differ from the GPU NVFP4 activation path and would need its own parity test), a worker thread pool pinned off the engine threads, and a per layer GPU to CPU to GPU hop, which costs at least two stream syncs per MoE layer. The 62 GB host also has to hold the pinned expert pool the CPU reads, which the machine only fits for models up to about 40 GB of experts (9 GB was available during this run).

The link is also memory-bandwidth shared with the CPU on this desktop; on the Spark (unified memory, no PCIe) the question does not apply.

## Tests

- `cargo test -p inference-bench-specrt --lib offload` (CPU, 17 tests: residency, eviction under budget, prefetch order, no re-read of a resident page, throttled fake NVMe).
- `tests/offload.rs` (GPU, `--include-ignored`): logits and per layer outputs bit identical to fully resident with 1 and 2 resident layers (3 layer, 4 layer and FP8 fixtures, slot reuse), identical greedy tokens, no streaming when the budget holds everything.
- `core` render test and `server` config test for the metrics and the config table.

## Not done here

- NVMe tier on the GPU path (needs the I/O thread pool shared with KV offload and snapshots).
- MoE expert cache and hotlist, CPU compute tier, Models page readout (filed as issues).
