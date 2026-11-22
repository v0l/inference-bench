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
  is `(step_cost - 1)` times the measured plain decode step, times the measured prefill rate, in multiples of 64 tokens, so a step costs about `step_cost` decode steps (server config `step_cost`, default 3, 1 or less is off; `itl_target_ms`, default off, is an optional absolute cap). Until a rate has been measured
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
- **Graph-captured decode.** `Runtime::decode` pads the rows to the next of 1, 2, 4, 8, 16, 32
  (`GRAPH_SIZES`) and replays one CUDA graph per size, captured on first use, with tokens, positions,
  slots and sampling params in the `Stepper`'s fixed device buffers. Pad rows point at a spare slot
  (`slots`) whose block table names a spare KV block (`kv_blocks`), so they never touch live state.
  `[runtime] graphs = false` falls back to eager launches. Test:
  `rows_padded_to_a_captured_graph_size_match_single_sequence_greedy`.
- **Fused decode and prefill step.** When decoders run and a prefill chunk is due, the scheduler
  calls `Executor::fused`, which `Runtime::fused_step` runs as one batch: the embed, every norm, QKV,
  O, gate/up, down, the delta-net projections and the lm_head see `decode rows + chunk tokens` rows, so
  each weight matrix is read once per step. Attention (`batch_decode_attention` for the decode rows,
  `flash_mma` for the chunk) and the delta-net step (`gdn_step` against `gdn_chunk_*`) stay separate
  launches per kind of row. The lm_head runs on the decode rows plus the last chunk row only when it
  samples. The fused step is not graph-captured (the chunk length varies); `[runtime] fuse = false` goes back
  to decode then prefill. Tests (`tests/fused.rs`): 3 decode rows plus a 200 token chunk give the same
  logits as the separate launches, within 1e-5 on the fixtures when both use one GEMM tile
  (`[runtime] tile = "big"`) and 1e-4 across tile choices; the 64 sequence preemption test runs through the
  fused path.
- **Recurrent state on demand.** The conv and delta-net state of a slot (157 MB on the 27B, 156,893,184
  bytes) is a virtual range reserved for all slots at load; physical memory is mapped when the
  scheduler admits a sequence (`StatePool`, CUDA virtual memory) and unmapped when it finishes, is
  cancelled or is preempted. `max_num_seqs` 32 costs 5 GB only if 32 sequences run. Admission stops,
  and the request waits, when `acquire_state` fails or free device memory is under the state size plus
  256 MB. The serving layout keeps room for 8 resident slots in its KV headroom. Tests:
  `tests/statepool.rs` (1000 random acquire/release cycles end with nothing mapped and the free memory
  back), `admission_stops_cleanly_when_no_recurrent_state_can_be_allocated` and
  `recurrent_states_are_freed_by_finish_cancel_and_preemption_with_no_growth` in `sched.rs`.
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
the model's own maximum, falls back to `max_ctx`); `kv_cache_gb`; `max_batched_tokens` (default
8192); `itl_target_ms` (default 100); `[speculative]` with `enabled`, `draft_tokens` (1 to 3) and
`max_batch` over the spec's `[mtp]` block (see `specrt-mtp.md`). Requests accept `ignore_eos` like vLLM.

`/metrics` adds `inference_running_seqs`, `inference_waiting_seqs`, `inference_kv_blocks_used`,
`inference_kv_blocks_total`, `inference_preemptions_total`, `inference_steps_total`,
`inference_spec_drafted_tokens_total`, `inference_spec_accepted_tokens_total`, and the
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

`bench --serve specs/qwen3-0.6b.toml <prompt> <new> <batch>` drives the scheduler
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
cargo run --release --bin bench -- --serve specs/qwen3-0.6b.toml 128 128 8
flock /tmp/gpu27.lock ./scripts/serve27b.sh              # KV_GB, SEQS, MAX_LEN, ONLY=stall
```

## #601 batched decode at 8 concurrent

27B NVFP4, `crates/specrt/scripts/bench601.sh`: 8 clients, 24 requests of 1000-1500 words (about 1.3k
tokens) arriving at once, 300-900 output tokens, 3 GB KV, greedy, MTP 2 drafts. Run 15 Nov 00:05-00:22 with the
user's app unloaded, but sharing the card with other workers' short jobs, so treat differences under 5 percent as noise.
Aggregate is completion tokens over wall time. Per-request decode is the median of the server's own
`decode_tok_s` per request, which is (completion tokens - 1) / (end - first token), so queue and prefill time are not in it.

| Run | Aggregate tok/s | Per-request decode med / min | TTFT p50 / p95 | ITL p95 |
|---|---|---|---|---|
| before: master, itl 100 | 366 | 55 / 36 | 0.66 / 2.76 s | 44 ms |
| itl 50 | 268 | n/a | 3.24 / 9.09 s | 81 ms |
| itl 100 (decode-aware budget) | 326 | 50 / 35 | 1.27 / 3.91 s | 91 ms |
| itl 200 | 374 | n/a | 0.61 / 2.46 s | 43 ms |
| itl 400 | 388 | 55 / 36 | 0.37 / 2.14 s | 43 ms |
| after: itl 100, whole prompt 4096 (new default) | 386 | 56 / 35 | 0.38 / 2.01 s | 43 ms |
| after, MTP max_batch 4 | 250 | 34 / 32 | 0.33 / 1.82 s | 28 ms |
| after, 16 slots, 32 requests, MTP at all sizes | 557 | 45 / 35 | 0.41 / 4.05 s | 50 ms |
| after, 16 slots, MTP max_batch 8 | 360 | 26 / 23 | 0.38 / 3.98 s | 40 ms |

What the numbers say:

- A small `itl_target` is the cause of the slow TTFT. Beside 8 decoders a step takes about 45 ms on this
  card, so itl 50 leaves almost no prefill room: p50 TTFT 3.2 s and aggregate down to 268. Itl 200 and 400
  behave like whole-prompt prefill, because a 1.3k prompt then fits in one step.
- Whole-prompt prefill (`whole_prompt_tokens`, default 4096) gives the itl 400 result (TTFT p50 0.38 s) without
  giving up the 100 ms target for long prompts, which stay chunked. ITL p95 stays at 43 ms because a 1.3k prefill
  step is shorter than the p95 step anyway.
- The prefill budget now subtracts the measured decode step time from `itl_target` (it used to give prefill the full
  target on top of a decode step). On its own it makes TTFT worse at itl 100 (1.27 s), so it only pays with the
  whole-prompt limit.
- MTP stays on at every batch size (`max_batch` unset means all slots). Capping it at 4 or at 8 with 16 slots drops
  everyone to plain decode: 250 and 360 tok/s against 386 and 557. Nothing to raise.
- Targets at 8 concurrent: per-request decode 56 against at least 50 met, TTFT p50 0.38 s against under 1.5 s met,
  aggregate 386 against at least 450 not met. At 16 concurrent aggregate is 557, above vLLM's 512 at 8, with
  per-request decode 45. The 8-slot ceiling is the decode step itself (45 ms for 8 rows times
  about 2.5 tokens per step with MTP), not the scheduler: prefill is a small share of steps at this load. Raising it
  needs a faster batched MTP verify (kernel work, not this change).
- The user's screenshot (23-28 tok/s per request, TTFT 3.3 s) is not reproduced here: this load gives 55 and 0.66 s
  on master. It matches the 16-slot MTP max_batch 8 row (26) and the itl 50 row (3.2 s) in shape, so the live app was
  probably running with a lower MTP batch limit or `itl_target_ms`, or with 20 slots and other prompts; check its
  config before blaming the scheduler.

Defaults: `itl_target_ms` stays 100, `whole_prompt_tokens` 1024 (see #655), `max_batch` unset. Both are in the load dialog
(serving group) and in the server config (`whole_prompt_tokens`, `itl_target_ms`, `--whole-prompt-tokens`,
`--itl-target-ms`). `prefill_burst_tokens` (default 16384, 0 turns it off) and `max_long_in_flight` (default 4, 1 is the old one-at-a-time gate) set how prompts over 2,048
tokens are read beside decoders (#603, `docs/perf/603-prefill-suspects.md`). `RUST_LOG=inference_bench_specrt=debug` logs rows, MTP or plain, prefill tokens and step ms per step.

## Guaranteed prefill share (#655)

Under load the decode step alone fills `itl_target_ms`, so the budget left for prefill was the 64 token floor and a 1.2k prompt took about 20 steps (TTFT 3-5 s). Three changes in `sched.rs`:

- `min_prefill_tokens` (default 512, `serve.min_prefill_tokens` in the options schema, `min_prefill_tokens` in the server config, `--min-prefill-tokens`): prompt tokens read every step beside decoders, whatever the decode step costs. 0 restores the old target-only behaviour.
- `swap_space_gb` (default 0, `serve.swap_space_gb`, server config `swap_space_gb`, `--swap-space-gb`): pinned host memory for parking preempted sequences. A preempted sequence copies its KV blocks and recurrent state (and MTP pending hidden) to the host, frees its blocks and slot, returns to the front of the waiting queue, and is restored byte for byte on resume with no re-prefill. When the pool is full, or 0, it recomputes as before.
- `admit_lookahead` (default 0 until the 27B gate402 run shows a win, `serve.admit_lookahead`, `admit_lookahead`, `--admit-lookahead`): admission keeps free blocks for the newcomer's next `min(max_tokens, lookahead)` tokens and for what running rows are still promised, so rows admitted together stop preempting each other. 0 restores prompt-only admission.
- `prefill_rate` is only updated from steps that read at least `max(min_prefill_chunk, min_prefill_tokens / 2)` tokens, so small steps no longer drag the EWMA down.
- Short prompts go first: running prefills are served shortest-remaining-first, `whole_prompt_remaining` looks at the shortest unfinished prompt rather than the head, and a prompt larger than the step budget is not admitted while another long one is already in flight. The long prompt still gets the leftover budget every step, so it is slowed, not starved.

Simulated (CPU, `Fake` engine, `print_load_table` and `print_sweep_table` in `sched.rs`). Decode step 0.56 ms per row, prefill 20 us per token, `itl_target` 10 ms; all times are multiplied by 10 so decode steps read 45-100 ms like the 27B and `itl_target` reads 100 ms, so `room` is 0 as on the user's app. Load: N long decoders, one 20k prompt at t0, eight 1.2k prompts 10 steps apart, three 8-12k agent prompts every 25 steps. "master" is bf7e6b3 with its shipping defaults (`whole_prompt_tokens` 4096). Times in seconds (scaled).

| decoders | variant | 1.2k TTFT p50 / p95 | long TTFT p50 / max | long tok per step | step p95 |
|---|---|---|---|---|---|
| 8 | master | 21.6 / 35.3 | 21.6 / 35.6 | 110 | 0.07 |
| 8 | shortest + 512, whole 4096 | 0.30 / 2.01 | 5.4 / 9.6 | 859 | 0.48 |
| 8 | shortest + 512, whole 2048 | 0.30 / 2.03 | 6.0 / 9.8 | 736 | 0.48 |
| 8 | shortest + 512, whole 1024 (default) | 2.9 / 4.7 | 10.4 / 11.8 | 532 | 0.24 |
| 8 | shortest + 512, whole 0 | 3.4 / 4.7 | 10.8 / 12.0 | 509 | 0.23 |
| 8 | shortest + 256, whole 0 | 8.5 / 12.1 | 15.5 / 20.3 | 256 | 0.14 |
| 16 | master | 36.0 / 57.3 | 36.2 / 58.2 | 107 | 0.12 |
| 16 | shortest + 512, whole 4096 | 0.34 / 2.23 | 6.1 / 10.9 | 859 | 0.52 |
| 16 | shortest + 512, whole 2048 | 0.34 / 2.24 | 7.2 / 11.3 | 736 | 0.53 |
| 16 | shortest + 512, whole 1024 (default) | 3.5 / 5.6 | 12.9 / 14.5 | 532 | 0.28 |
| 16 | shortest + 512, whole 0 | 4.3 / 6.0 | 13.6 / 14.8 | 509 | 0.28 |
| 16 | shortest + 256, whole 0 | 11.7 / 17.0 | 21.5 / 28.3 | 256 | 0.18 |
| 20 | master | 42.8 / 67.8 | 43.1 / 69.0 | 107 | 0.14 |
| 20 | shortest + 512, whole 4096 | 0.36 / 2.36 | 6.6 / 11.6 | 859 | 0.56 |
| 20 | shortest + 512, whole 2048 | 0.36 / 2.38 | 7.9 / 12.1 | 736 | 0.55 |
| 20 | shortest + 512, whole 1024 (default) | 3.9 / 6.1 | 14.3 / 15.9 | 532 | 0.31 |
| 20 | shortest + 512, whole 0 | 4.7 / 6.6 | 15.0 / 16.4 | 509 | 0.30 |
| 20 | shortest + 256, whole 0 | 13.2 / 19.4 | 24.4 / 32.3 | 256 | 0.21 |

Master reads about 110 tokens per step for every prompt, which is the user's plateau. Shortest-first alone and the 512 floor alone are each worse than both together (see measured at 16 decoders before the sweep: 1.2k p95 4.0 s and long prompts at 306 tokens per step for shortest-first alone, 1.2k p95 9.3 s for the floor alone). A second prompt over 2048 tokens is not admitted while another is in flight, so the long ones are served in turn and all complete.

What bounds the step: a step is the decode plus every prefill token it carries, so the step p95 is decode plus the floor (512 tokens, about 0.1 s) plus any whole-prompt remainder. `whole_prompt_tokens` 4096 lets whole prompts ride in one step (0.5 s steps); 1024 roughly halves the step p95 for a 0.5-1.5 s worse 1.2k p95. A cap on prefill tokens per step was also swept (1024 and 512 on top of whole 4096): it gave the same step p95 as the matching whole limit with worse TTFT, so it is not a separate option. The 2x `itl_target` bound (0.2 s) is not reachable together with a useful TTFT: only 256 reaches it at 8-16 decoders, at 12-19 s 1.2k p95. Defaults chosen: `min_prefill_tokens` 512, `whole_prompt_tokens` 1024 (was 4096), step p95 about 2.4-3.1x `itl_target` at 8-20 decoders, against 1.2k TTFT p95 of 5-6 s in place of master's 35-68 s. Lower `min_prefill_tokens` to 256 to hold the 2x bound at the cost of slower prefill. The regression test `prompts_stay_bounded_behind_each_other_with_full_decoders` asserts 1.2k TTFT p95 under 8 s, long TTFT max under 20 s, over 400 long-prompt tokens per step and step p95 at most 0.35 s; on bf7e6b3 it fails (`short ttft p95 57.331s`).

Real-27B A/B for the supervisor (GPU free, under `flock /tmp/gpu27.lock`, 20 slots, classify-shaped 1.2k prompts plus a few 10-20k agent prompts): run `crates/specrt/scripts/serve27b.sh` with `--min-prefill-tokens 0` then `512` (and 256), and drive it with `loadtest` at 16-20 concurrent; compare TTFT p50/p95, `inference_prefill_tokens_per_step` and per-request decode tok/s.

## #360 measurements

Not measured on the real 27B: another process held 24 GB of the GPU next to vLLM's 51 GB (free
memory 14 GB against the 36 GB `serve27b.sh` needs) for the whole window, and the one run that fit
failed its model loads with CUDA out of memory, so its numbers are not recorded. Still to run, under
`flock /tmp/gpu27.lock`, shared GPU with vLLM serving, 7 alternating runs and medians:
single stream tok/s with graphs against `[runtime] graphs = false` (57 ms against 45 ms per token),
`[runtime] fuse = false` against fused for 8 decoders beside a 29k prefill (aggregate tok/s and decoder
stall max), and the vLLM comparison of the same `loadtest` mix (#307).

## Structured output

`response_format` accepts `json_object` and `json_schema` (`json_schema.schema`). A schema compiles into the grammar in `specrt::grammar` (type, properties, required, enum, const, items, anyOf/oneOf, single-entry allOf, local `$ref`, additionalProperties). Objects are strict by default, like xgrammar: keys not listed in `properties` are rejected. `additionalProperties: true` (or a schema) allows extra keys, and an object with no `properties` accepts any keys. Properties are emitted in declaration order. Schemas that cannot be compiled are a 400.

`tool_choice: "required"` or `{"type":"function","function":{"name":...}}` compiles a grammar of the model's own tool call wrapper (Qwen XML `<tool_call><function=..><parameter=..>` or Hermes JSON, chosen by the tool parser) around each tool's parameter schema, so the reply is always a valid call to a listed tool (the named one for a named choice). It cannot be combined with `response_format`. Compiled grammars are cached per loaded model, LRU, size `grammar_cache` in the server config (default 32, 0 turns the cache off). Constrained rows decode without speculation. Mask time on a 248k vocabulary (release, `mask_time_on_a_248k_vocabulary`): about 40 us for the first mask of a state, 59 ms for a state inside a free string (not yet cached; later steps in the same state hit the cache). Mask overhead against the 27B step at batch 8 is unmeasured (GPU was in use by the user's app).
