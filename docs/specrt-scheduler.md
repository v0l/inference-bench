# specrt serving core: paged KV, continuous batching, preemption

Issue #335. This is the part of specrt that serves many concurrent requests: the KV layout, the
scheduler, admission and preemption, the server wiring, and a load generator that runs the same
workload against specrt and vLLM.

## Layout

| piece | where | what |
|---|---|---|
| KV pool | `runtime.rs` (`Layout`, `load_layout`, `set_block_table`) | fixed blocks of 16 tokens, block-major: `[block][attention layer][kv head][16][head_dim]`, K and V in two pools |
| block tables | `Buffers::block_table` | one row of `max_model_len / 16` entries per slot; the host keeps a mirror and copies only the entries that changed |
| recurrent state | `GdnBuffers` | one conv window and one delta-net state per slot, indexed by slot in `gdn_conv_step`, `gdn_step` and the chunked prefill |
| step interface | `runtime/step.rs` | `prefill_slot(slot, tokens, start, sample)` and `decode(rows)`, each row `{slot, token, pos, sampling, step}`; sampling (greedy or Gumbel with a per-row seed) runs on the device in `sample_rows` |
| scheduler | `sched.rs` | `BlockPool`, `Scheduler<E: Executor>`, events; knows nothing about CUDA |
| serving thread | `serve.rs` | `Core` owns the scheduler on its own thread; `Serving` tokenises, submits a job and detokenises the stream in the caller's thread |
| server | `crates/core/src/serving.rs`, `crates/server` | `ServingTarget` shares one `Core` between `max_num_seqs` leases; `/metrics` adds the scheduler gauges and histograms |
| load generator | `crates/core/src/loadtest.rs`, `inference-bench loadtest` | open-loop Poisson arrivals over any OpenAI-compatible URL |

Block-major order puts one block of every attention layer together, so a block is a single unit
to allocate and free, and a table entry is the same for all layers. The kernels (`kv_slot` in
`ops.cu`) compute `((block * ATTN_LAYERS + layer) * KV_HEADS + head) * 16 + pos % 16`.

## Decisions

- **One f16 copy of K and V in the serving layout.** The old layout stored a hi/lo pair of f16
  (128 KiB per token on the 27B). Serving stores one plane: 16 layers x 4 heads x 256 x (K, V) x
  2 bytes = 64 KiB per token, checked by `serving_kv_is_one_f16_copy_per_token`. `KV_PLANES` is a
  codegen constant: the single-stream `Runtime::load` and `load_batched` keep 2 planes so the
  reference tests stay at their 1e-3 bounds, and `flash_mma` drops the lo mma when there is one
  plane. The serving path on the tiny fixtures is still greedy-identical to the 2-plane path.
- **Blocks of 16 tokens.** The existing `PAGE`, and the same as vLLM's default. The prefill
  attention tiles (`flash_mma` reads 16 or 32 keys) stay inside a block or two.
- **Recurrent state per slot, not per block.** The delta-net state is 151 MB per sequence on the
  27B regardless of context, so it lives in a pool of `max_num_seqs` slots allocated at load.
  That fixes `max_num_seqs` as a memory cost up front: 32 slots are 5 GB on the 27B.
- **KV budget.** `--kv-cache-gb` (server `kv_cache_gb`) fixes the pool. Without it the pool takes
  all free device memory after the weights, the slots and the scratch, minus 1 GB. The pool is
  allocated last, after measuring what is free, so the number of blocks is exact.
- **Admission.** A waiting request is admitted when a slot is free and the pool has blocks for
  its whole prompt plus one, plus a reserve (1 percent of the pool, at least 1 block) once anything
  else is running. A request that could never fit (prompt plus `max_tokens` larger than the whole
  pool, or than `max_model_len`) is refused at submit with a 400.
- **Preemption by recompute.** When a decoder needs a new block and the pool is empty, the most
  recently admitted sequence is preempted: its blocks and slot are freed and it goes back to the
  front of the waiting queue with its prompt and every token generated so far. When it is admitted
  again it prefills all of that and continues; tokens already streamed are not sent twice. Swap to
  host memory is not implemented.
- **Chunked prefill and the inter-token target.** Each step decodes every running decoder, then
  spends a prefill budget on the waiting prefills in FCFS order, in chunks of at most
  `max_batched_tokens - decoders` and the runtime chunk (2048). With decoders running, the budget
  is `itl_target` (default 100 ms) times the measured prefill rate, in multiples of 64 tokens, so
  a step costs about one decode plus one `itl_target` of prefill. Until a rate has been measured
  the first chunk next to decoders is 64 tokens. The prefill time is measured after a stream
  synchronise: without it the asynchronous launches made prefill look free, the rate estimate ran
  away, and one step took 3.3 s on the 27B. Unit tests:
  `prefill_beside_decoders_stays_inside_the_inter_token_target_from_the_first_step`,
  `a_first_token_mid_step_caps_the_rest_of_that_steps_prefill`.
  Subtracting the decode time from the target was tried and dropped: on the 27B eight decoders
  alone take 95 ms, which left a 64-token chunk per step and prefill at a crawl.
- **Prefill and decode in one step.** The batched decode and the per-sequence chunked prefill are
  separate launches on one stream, decode first. They are not fused into one batch: the delta-net
  chunked prefill and the per-token recurrent step are different kernels.
- **Cancel.** The serving thread checks every live job's cancel flag before each step. A dropped
  SSE stream sets it (the send fails), so its blocks are back in the pool within one step.
  Server test: `a_dropped_stream_frees_its_kv_blocks_within_a_step`.

## Correctness tests

| test | what it proves |
|---|---|
| `tests/serving.rs` `*_shuffled_kv_blocks_match_the_contiguous_layout` (tiny, hd256; 1 and 2 planes) | a shuffled block table gives logits within 1e-5 of the contiguous table and identical greedy |
| `*_mixed_length_step_decode_matches_single_sequence_greedy` (tiny, hd256, fp8) | 4 prompts of 5 to 300 tokens, chunked at 37 and 128, decoded together through `decode()` give the single-sequence greedy tokens |
| `*_sixty_four_scheduled_sequences_match_single_sequence_greedy_with_preemption` (tiny, hd256) | 64 sequences of 3 to 400 tokens through the scheduler with a pool of 120 blocks: preemption happens, every output equals single-sequence greedy, no block is left over |
| `the_serving_core_streams_concurrent_jobs_and_cancels_within_a_step` | the serving thread streams the same tokens, and a cancel ends a 500-token job within a few steps |
| `sched.rs` unit tests | 1000 random alloc/free cycles leak nothing; 64 sequences under preemption match their solo output on a fake executor that checks every KV read goes through the right block table; decoders are never skipped while a 40k prompt prefills; stop tokens; cancel; refusal of requests that never fit; admission waits instead of overcommitting |
| `crates/server/tests/serving.rs` (Qwen3-0.6B) | 6 concurrent HTTP completions are batched and equal their solo output; a dropped stream frees its blocks; oversize requests are 400s |

## Configuration

Server config keys (or the same flags on `inference-bench serve`): `max_num_seqs` turns the
serving core on (without it the old single-stream engine serves); `max_model_len` (default
131072, at most 262144, falls back to `max_ctx`); `kv_cache_gb`; `max_batched_tokens` (default
8192); `itl_target_ms` (default 100). Requests accept `ignore_eos` like vLLM.

`/metrics` adds `inference_running_seqs`, `inference_waiting_seqs`, `inference_kv_blocks_used`,
`inference_kv_blocks_total`, `inference_preemptions_total`, `inference_steps_total`, and the
histograms `inference_queue_seconds`, `inference_inter_token_seconds`, `inference_batch_size`
(decode rows per step), `inference_prefill_tokens_per_step` and `inference_step_seconds`.

## Load generator

```
inference-bench loadtest --url http://127.0.0.1:8080 --model qwen3.8-27b \
    --requests 200 --rate 2 --clients 32 --mix 0.6:1k-8k,0.3:8k-32k,0.1:32k-128k --output 128-1024
```

Arrivals are Poisson at `--rate` per second (0 sends everything at once), at most `--clients` in
flight. Prompt lengths are in words, drawn from weighted buckets (the default is the journal
shape); every request sets `ignore_eos` so replies run to their length. It reports aggregate
decode tok/s (completion tokens over wall time), TTFT p50/p95, inter-token gaps p50/p95/p99/max,
and from `/metrics` sampled every second: preemptions, KV use and running sequences. It reads
both specrt's and vLLM's metric names, so the same command runs against both.

## Measurements

### Qwen3-0.6B (bf16), shared GPU

`SPECRT_BENCH_SERVE=1 bench specs/qwen3-0.6b.toml <prompt> <new> <batch>` drives the scheduler
in-process. vLLM was serving beside it the whole time, so these are indicative.

| prompt | batch | decode tok/s aggregate | decode step |
|---|---|---|---|
| 128 | 8 | 2,026 | 3.9 ms |
| 128 | 32 | 3,013 | 10.5 ms |
| 8,192 | 1 | 113 to 155 | 6.3 to 8.7 ms |
| 8,192 | 8 | 416 to 468 | 11.7 to 12.9 ms |

The graph-captured `generate_batch` reaches 2,073 tok/s at batch 8 and 3,282 at 32, so the
eager per-step path is within 3 to 8 percent of it at those sizes. `BATCH_DECODE_SPLITS` of 8, 16
and 32 were inside the noise of each other.

Through the HTTP server with the journal-shaped mix capped at 28k words (96 requests at 4/s,
32 clients, 4 GB of KV = 2,340 blocks): all 96 succeeded, KV use peaked at 99.8 percent, 1
preemption, inter-token p50 13 ms, p99 22 ms. That run predates the prefill timing fix and had
one 1.4 s gap.

### Real Qwen3.8-27B FP8, beside the live vLLM

`scripts/serve27b.sh` under `flock /tmp/gpu27.lock`, 9 Nov 2026. vLLM (`local-llm.service`) was
serving the whole time at about 100 percent GPU use and the 300 W cap, so 38.9 GB were free.
That fits the 30.2 GB of weights, 8 slots of recurrent state (1.25 GB) and 3 GB of KV (3,072
blocks, 49k tokens) with `max_model_len` 32768 and `max_batched_tokens` 2048. 16 or 32 slots do
not fit beside vLLM: each slot costs 157 MB of delta-net state. The service journal logged no
errors and free memory returned to 38.9 GB after each run.

| run | result |
|---|---|
| 1 stream, 512 prompt, 256 out | inter-token p50 57 ms (17.4 tok/s), TTFT 0.8 s |
| 4 streams | inter-token p50 68 ms, 59 tok/s aggregate while decoding (48.7 over the wall clock) |
| 8 streams | inter-token p50 94 ms, 85 tok/s aggregate while decoding (69.3 over the wall clock) |
| 32 requests, journal mix capped at 28k words, 0.3/s, 8 slots | all succeeded, 234k prompt tokens, KV use peaked at 97.8 percent, no preemption, TTFT p50 146 s (the queue: arrivals outran the 250 tok/s chunked prefill) |
| 7 decoders while a 29k prompt prefills | before the timing fix: inter-token max 3.27 s; after: p50 234 ms, p99 419 ms, max 446 ms; the 29k prompt took 133 s |

Against the issue's targets this is far off: 85 tok/s at 8 running against 410, and 446 ms of
decoder stall against 200 ms. The single-stream graph path gets 22 tok/s on the same shared card,
so the scheduler's eager path costs about 20 percent at batch 1. The gaps that remain are not in
the scheduler:

- Decode bytes: every step reads 27 GB of FP8 weights; NVFP4 (#321) and MTP cut that.
- Shared card: vLLM at 100 percent beside it.
- Chunked prefill next to decoders runs in small chunks (about `itl_target x rate` tokens), and
  the GEMMs at 64 to 256 rows are much slower per token than the 2048-row chunk, so a 29k prompt
  prefills at 220 tok/s beside 7 decoders against 550 to 1,300 tok/s alone. With eight 27B
  decoders taking 95 ms per step, the 100 ms target cannot be met with any prefill at all.

### Commands

```
cd crates/specrt
SPECRT_REF_DIR=~/.cache/specrt-ref-335 cargo test --release --test serving -- --include-ignored --test-threads=1
cargo test --release --lib sched
cargo test --release -p inference-bench-server --test serving -- --include-ignored --test-threads=1
SPECRT_BENCH_SERVE=1 cargo run --release --bin bench -- specs/qwen3-0.6b.toml 128 128 8
flock /tmp/gpu27.lock ./scripts/serve27b.sh              # KV_GB, SEQS, MAX_LEN, ONLY=stall
```
