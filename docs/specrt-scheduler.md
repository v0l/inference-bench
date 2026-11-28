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
  pool, or than `max_model_len`) is refused at submit with a 400. Admission scans the queue in
  arrival order instead of stopping at the head (#1518, see "Admission past a blocked head").
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

### step_cost measurement (#404, 22 Nov 2026)

`serve27b.sh ONLY=stall` on the 27B (GPU1 shared, 7 decoders of 700 tokens plus one 29k prompt; prefill is the long request TTFT).

| setting | 29k prefill | decoder itl p99 | itl max |
|---|---|---|---|
| server defaults, old absolute 100 ms | 19.4 s | 1.34 s | 1.43 s |
| server defaults, step_cost 3 | 19.5 s | 1.35 s | 1.43 s |
| server defaults, step_cost 4 | 19.8 s | 1.36 s | 1.46 s |
| rule alone, old absolute | 27.5 / 28.1 s | 0.17 s | 0.18 s |
| rule alone, step_cost 2 | 26.8 / 26.0 s | 0.11 s | 0.17 s |
| rule alone, step_cost 3 | 28.7 / 29.6 s | 0.18 s | 0.18 s |
| rule alone, step_cost 4 | 30.3 / 30.1 s | 0.22 s | 0.22 s |

No setting cuts prefill 2x, so none qualifies and the default stays 3. With the defaults (`prefill_burst_tokens` 16384, `min_prefill_tokens` 512, `whole_prompt_tokens` 1024) the long prompt already reads at the alone rate (1,490 tok/s) whatever the rule says, so the 220 tok/s in the issue is gone; the 1.3 s decoder stall is unchanged by `step_cost`. With those three off the rule alone decides, and step_cost 2 gives the lowest inter-token p99 at the same prefill time. This is not a `bench --ab` result: that tool times single-process decode, not a served stall.

First-step fix: before any decode step is measured, `decoders_run_every_step_while_a_long_prompt_prefills_in_chunks` saw `decode_rows: 0` with a 1024-token prefill under the new default; the prefill rate is now measured net of decode time and that test sets `step_cost: 0.0` to pin the max-batched-tokens path, while the new tests pin the relative rule.

### Long-prompt stall defaults (#989, 22 Nov 2026)

Same `serve27b.sh ONLY=stall` run on GPU1 (7 decoders of 700 tokens, one 29k prompt, 8 slots, 3 GB KV, `--max-batched-tokens 2048`). Burst, min and whole are `prefill_burst_tokens`, `min_prefill_tokens`, `whole_prompt_tokens`. The 29k prefill column is the long request's time to first token.

| burst / min / whole | 29k prefill | decoder itl p99 | itl max |
|---|---|---|---|
| 16384 / 512 / 1024 (default) | 19.3 s | 1.33 s | 1.41 s |
| 0 / 512 / 1024 | 22.9 s | 0.42 s | 0.77 s |
| 0 / 1024 / 1024 | 20.9 s | 0.76 s | 0.95 s |
| 768 / 512 / 1024 | 21.8 s | 0.59 s | 0.77 s |
| 640 / 256 / 1024 | 22.7 s | 0.51 s | 0.66 s |
| 512 / 256 / 1024 | 23.3 s | 0.42 s | 0.77 s |
| 512 / 256 / 256 | 23.8 s | 0.42 s | 0.43 s |
| 512 / 0 / 0 | 26.3 s | 0.43 s | 0.44 s |
| 448 / 256 / 256 | 25.8 s | 0.40 s | 0.41 s |
| 384 / 256 / 256 | 26.2 s | 0.35 s | 0.36 s |
| 0 / 256 / 1024 | 27.4 s | 0.27 s | 0.28 s |

The burst causes the step: with the default a step carries up to 16384 prompt tokens, about 1.3 s at 1,490 tok/s, and every decoder waits for it. A step of N prompt tokens stalls decoders for about N / 1,490 s plus the decode, so p99 under 400 ms needs bursts of about 450 tokens or fewer, and those cost 25 s to 26 s of prefill (30% to 35% over the alone rate). No tested setting has p99 under 400 ms with prefill within 15% of 19.3 s (22.2 s): the two targets conflict on this model, because the prefill rate falls once chunks are small. The default therefore stays at burst 16384 (fastest long prefill, right for batch and agent workloads).

The other side, #603's long-prompt mix (`--requests 40 --rate 1 --clients 8 --mix 1:15000-18000 --output 256-512`, 26 seqs, 22.5 GB KV, fp8 KV, MTP, `--max-batched-tokens 16384`, `--max-long-in-flight 4`, GPU1), against the stall workload above with only the burst changed (min 512, whole 1024). The stall workload runs with `--max-batched-tokens 2048`, so bursts of 2048 and more are the same there.

| burst | long TTFT p50 | long TTFT p99 | long ITL p95 | long ITL max | stall decoder itl p99 | stall 29k prefill | chat control ITL p95 |
|---|---|---|---|---|---|---|---|
| 16384 (default) | 3.40 s | 21.1 s | 0.044 s | 2.97 s | 1.33 s | 19.3 s | 0.146 s |
| 4096 | 3.80 s | 21.9 s | 0.72 s | 0.82 s | 1.33 s (not run; `--max-batched-tokens 2048` caps the step, same as 2048) | 19.3 s | 0.144 s |
| 2048 | 3.86 s | 23.4 s | 0.41 s | 0.48 s | 1.33 s | 19.2 s | 0.143 s |
| 1024 | 6.45 s | 28.6 s | 0.27 s | 0.80 s | 0.74 s | 20.7 s | 0.143 s |
| 0 | 9.02 s | 116.6 s | 0.16 s | 0.28 s | 0.42 s | 22.9 s | 0.143 s |

Rule: smallest burst with long TTFT p50 within 25% of the burst-16384 value (4.25 s) and stall decoder p99 under 0.5 s. Bursts 4096 and 2048 keep TTFT (3.8 s, +12%) but the stall stays at 1.33 s, since one 2048-token step already costs that long at the 29k prompt's attention depth. Bursts 1024 and 0 cut the stall to 0.74 s and 0.42 s but lose the #603 win (TTFT p50 +90% and +165%, p99 116 s at 0). Only burst 0 is under 0.5 s on the stall, and it fails the TTFT bound; no burst satisfies both, so the default is unchanged. Giving up the stall is the cheaper side: burst 0 returns the 24.9 s class TTFT that #603 removed.

For interactive chat beside long prompts use the latency profile, `--prefill-burst-tokens 384 --min-prefill-tokens 256 --whole-prompt-tokens 256` (same names as server.toml keys and in the load dialog serving options): decoder itl p99 0.35 s, 29k prefill 26 s. `--prefill-burst-tokens 0` alone is the middle point (0.42 s, 23 s with min 512 and whole 1024) but loses the #603 long-mix TTFT.

### Decode-aware stall cap (#1007, 22 Nov 2026)

`serve.stall_budget_ms` (`--stall-budget-ms`, server.toml `stall_budget_ms`, default 0 = off). Each step the longest-waiting decoder's time since its last token is measured; the prefill in that step is capped at `(budget - waited - decode step) * measured prefill rate`, rounded to 64 tokens, at least 64. A step that fits the budget carries the full burst; with no decoder running nothing is capped.

GPU1, 27B, 3 rounds per arm, arms alternated. Stall: `serve27b.sh ONLY=stall` (FP8 exact spec, 7 decoders of 700 tokens, one 29k prompt). Long mix and chat control: #603's shapes (NVFP4, fp8 KV, MTP, 26 seqs, 22.5 GB KV, burst 16384, long 4), with batch-8 decode (48 x 512 in, 256 out). Scripts: `crates/specrt/scripts/stall1007.sh`, `mixrounds1007.sh`.

| stall_budget_ms | stall itl p99 | itl max | 29k prefill | long TTFT p50 | long TTFT p99 | chat itl p95 | batch-8 tok/s |
|---|---|---|---|---|---|---|---|
| 0 (A/A control) | 1.30 / 1.31 / 1.31 s | 1.38 s | 18.8 / 19.0 / 19.0 s | 3.56 / 3.56 / 3.56 s | 22.7 / 22.4 / 22.9 s | 0.148 / 0.146 / 0.147 s | 471 / 467 / 469 |
| 400 | 0.33 / 0.25 / 0.25 s | 0.34-0.38 s | 27.8 / 29.0 / 29.0 s | 7.57 / 7.44 / 7.88 s | 31.1 / 31.3 / 31.1 s | 0.147 / 0.147 / 0.147 s | 468 / 469 / 471 |
| 800 | 0.51 / 0.51 / 0.49 s | 0.57-0.59 s | 22.6 / 22.6 / 22.8 s | 5.36 / 4.51 / 4.44 s | 25.4 / 25.3 / 25.6 s | 0.148 / 0.148 / 0.147 s | 466 / 464 / 470 |

Bars: stall p99 under 0.5 s, long TTFT p50 at most 4.25 s, long TTFT p99 at most 25 s, chat unchanged. Budget 400 meets the stall bar and the chat control but costs the long mix (TTFT p50 +110%, p99 +38%, 29k prefill +50%). Budget 800 misses all but chat. Batch-8 decode and chat are unchanged within noise at every budget. No tested budget meets both the stall and the long-mix bars, the same conflict #989 found with fixed bursts: capping a step to the budget shrinks chunks, and the prefill rate falls with small chunks. The default stays 0. Setting 400 gives the stall bound (0.25 to 0.33 s) for interactive use at the price above.

#### Floor on the capped chunk (`serve.min_stall_chunk_tokens`)

The cap never goes below the floor. Same rig, 3 rounds, budget 0 as the control (run in the same sessions; its mix row is 3.74-3.76 s p50, 22.7-22.9 s p99 on this master).

| budget / floor | stall itl p99 | itl max | 29k prefill | long TTFT p50 | long TTFT p99 | chat itl p95 | batch-8 tok/s |
|---|---|---|---|---|---|---|---|
| 0 / 0 (control) | 1.30-1.31 s | 1.39 s | 18.8-18.9 s | 3.74-3.76 s | 22.7-22.9 s | 0.150 s | 514 |
| 400 / 256 | 0.25 s | 0.38 s | 23.5-23.9 s | not run | not run | | |
| 400 / 512 | 0.41 s | 0.52 s | 22.4-22.5 s | 7.64-8.25 s | 31.7-33.5 s | 0.150 s | 510 |
| 400 / 1024 | 0.73-0.74 s | 0.76 s | 20.3-20.5 s | 7.89-8.00 s | 29.7-29.8 s | 0.150 s | 512 |
| 400 / 2048 | 1.31 s | 1.39 s | 18.9-19.0 s | not run | not run | | |
| 600 / 1024 | 0.74 s | 0.76 s | 20.5 s | not run | not run | | |
| 600 / 2048 | 1.31 s | 1.39 s | 19.0 s | not run | not run | | |

The stall p99 tracks the floor, not the budget: a step of N tokens stalls decoders for about N / 1,490 s, so a floor of 2048 or more is the same as no cap (1.31 s) and a floor of 1024 gives 0.74 s. Only floors of 512 or less get under 0.5 s, and at 512 the long-mix TTFT p50 is 7.6 to 8.3 s and p99 31.7 to 33.5 s, the same as floor 1024 (7.9 s, 29.7 s): the long-mix loss is not the chunk size falling, it is any cap at all while decoders are running, because eight long prompts all run beside decoders that keep hitting the budget. Floors 256, 2048 and the 600 budget were not run through the long mix because their stall rows already fail the stall bar (2048, 600/1024) or follow the 400/512 result (256 is a smaller chunk than 512, so TTFT can only be worse). No (budget, floor) passes both bars, so the default stays 0 (off). Budget 200 was not run: with any floor the stall p99 is set by the floor.

## Admission past a blocked head (#1518, 27 Nov 2026)

Admission used to serve the waiting queue only from the front. A long prompt waiting for the long prompt cap, or a prompt that did not fit the KV pool yet, held back every request behind it while slots and KV were free. Now `Scheduler::admit_until` walks the queue in arrival order and admits every request that fits:

- A request refused for the long prompt cap or for KV is skipped and the scan goes on. Full slots and a planned prefill budget still end the scan, and so do a refused recurrent state and a parked sequence that cannot resume yet.
- The long prompt cap counts the tokens left to read after the prefix cache or a disk session, the same count used once a prompt is running. A 30k agent turn with 29k cached is not long. The cache is read with `PrefixCache::peek`, which does not touch the LRU, so a skipped prompt does not keep its blocks warm.
- Parked (swapped) and preempted sequences (admitted before) are never overtaken: when one of them cannot go, the scan stops.
- Fairness: each time a later request starts ahead of one held for KV, that one's overtake count goes up. At `serve.max_overtakes` (default 32; server.toml `max_overtakes`, `--max-overtakes`, load dialog "queue overtakes"; 0 is strict arrival order) it blocks the queue again, so a large prompt behind a steady stream of small ones still starts. A prompt held for the long cap is not counted: it starts as soon as a long prompt finishes, and short prompts never take the long cap from it.
- `inference_admission_blocked_steps_total{reason}` counts the reason that ended the scan, or when the scan reached the end of the queue, the first reason that held a request. Each request's held time is charged to its own reason, not the head's.

Tests (`sched.rs`): `a_short_prompt_is_admitted_past_a_long_prompt_waiting_for_the_long_cap`, `a_mostly_cached_long_prompt_does_not_count_against_the_long_cap`, `a_prompt_that_does_not_fit_the_kv_pool_does_not_block_smaller_ones` (all three fail on the old head-only rule), `a_large_prompt_behind_a_steady_stream_of_small_ones_still_starts_within_a_bound` (no bound: never starts in 400 steps; bound 4: starts within 80) and `parked_sequences_are_never_overtaken`.

Simulated tables (Fake executor; `print_load_table` "both" row, the same as `print_sweep_table` whole 1024 min 512, which are the server defaults; 1.2k prompts arriving beside a 20k prompt and three 6k to 12k prompts, long cap 1):

| decoders | 1.2k TTFT p50 before / after | 1.2k TTFT p95 before / after | long TTFT p50 before / after | long TTFT max before / after | step p95 before / after |
|---|---|---|---|---|---|
| 8 | 2.43 / 0.33 s | 3.72 / 0.33 s | 8.73 / 8.65 s | 9.27 / 9.61 s | 0.182 / 0.182 s |
| 16 | 3.15 / 0.42 s | 4.79 / 0.42 s | 11.28 / 11.20 s | 12.00 / 12.43 s | 0.227 / 0.227 s |
| 20 | 3.51 / 0.46 s | 5.33 / 0.46 s | 12.56 / 12.48 s | 13.37 / 13.84 s | 0.250 / 0.250 s |

The 1.2k TTFT p95 is lower or equal in every row of both tables at every decoder count. Rows with `shortest_first` off ("master", "min512") have no long cap and are unchanged; every other row drops to under 0.9 s (the worst is `whole 0 min 512` at 20 decoders, 6.02 to 0.86 s). The long prompts lose 3 to 4% at the max because the short ones now read their prompts in steps the long ones used to have alone; p50 does not move. With `whole_prompt_tokens` 2048 or more the long rows lose more (8 decoders, whole 4096: long p50 4.44 to 6.64 s, max 7.29 to 8.07 s), since every short prompt is then read whole in one step ahead of them; the server default is 1024. `print_long_prompt_table` (24 prompts of 15k to 18k, nothing short to overtake) is identical before and after.

### Real 27B check (GPU1, 28 Nov 2026)

Rig: `crates/specrt/scripts/admit1518_rounds.sh` (one server per arm, arms alternated, 3 rounds). Qwen3.8-27B NVFP4 with MTP and the prefix cache on, 8 slots, 26 GB KV, server defaults otherwise (`max_long_in_flight` 4, burst 16384, whole prompt 1024, min prefill 512), `--max-concurrent 64` so the HTTP layer does not hold requests back. Load: 60 requests, Poisson 1/s, 12 clients, prompts 55% 1k to 8k words and 45% 15k to 30k, 4 conversations whose prompts share their first 90% (prefix hit rate 33%), 128 to 384 tokens out. Master is `2bc21c2d` with the loadtest and log commit on top. Queue wait is the server's `queue_ms` per request (arrival to first prefill step), the rest is the client.

| | master (3 rounds) | branch (3 rounds) | mean master / branch |
|---|---|---|---|
| queue wait p50 | 0.05 / 0.05 / 0.05 s | 0.05 / 0.05 / 0.05 s | 0.047 / 0.046 s |
| queue wait p95 | 0.22 / 1.28 / 0.32 s | 2.80 / 0.36 / 0.32 s | 0.61 / 1.16 s |
| TTFT p50 | 10.46 / 10.76 / 11.16 s | 12.05 / 11.31 / 11.18 s | 10.80 / 11.51 s |
| TTFT p95 | 18.17 / 19.30 / 21.06 s | 21.42 / 21.06 / 21.08 s | 19.51 / 21.18 s |
| long prompt TTFT p95 | 18.85 / 22.92 / 22.70 s | 24.14 / 22.30 / 22.75 s | 21.49 / 23.06 s |
| decode tok/s per request p50 | 18.1 / 18.5 / 16.9 | 16.7 / 16.1 / 16.9 | 17.9 / 16.6 |
| ITL p95 | 48.7 / 48.6 / 49.5 ms | 48.3 / 49.4 / 49.6 ms | 48.9 / 49.1 ms |

On this mix admission never blocks: `inference_admission_blocked_steps_total` is 0 for every reason in every run, and the longest arrival to admission time is 15 ms. The queue wait is the time an admitted prompt waits for its first prefill step behind longer prompts already being read, which this change does not touch, and its p95 is two or three requests out of 60 (the round to round spread is larger than any arm difference). The third rounds are the same to two digits, including decode tok/s per request (16.89 / 16.88). Rounds 1 and 2 have the branch 1.5 to 2.4 tok/s lower with a 6 to 12 s longer wall time, so two more alternated rounds were run: master 16.9 / 16.9 tok/s, TTFT p95 18.98 / 21.04 s, long TTFT p95 20.55 / 22.79 s, wall 142.8 / 138.0 s; branch 16.9 / 17.0 tok/s, TTFT p95 21.05 / 21.13 s, long TTFT p95 22.79 / 22.36 s, wall 137.8 / 142.3 s. The same two run shapes (about 138 s and about 142 s of wall) turn up in both arms, so the gap in rounds 1 and 2 is run to run spread, not a cost of the change. Over all five rounds the long TTFT p95 mean is 21.76 s on master and 22.87 s on the branch (+5%). The issue's bar (queue wait p95 down 30%) is not met because there is no admission wait to remove here; long TTFT p95 (+5% over five rounds) and ITL p95 (+0.4%) are inside its bounds. With `max_long_in_flight` 1 and 24 clients (one run per arm) the long cap does block (8 to 10 steps), and TTFT p50 is 29.3 s on the branch against 29.6 s on master: with every slot busy, admitting past the head changes little. Which part of the 7.9 s on the user's app is admission, prefill wait or full slots is read from the #1519 series once the app runs a build that has them.