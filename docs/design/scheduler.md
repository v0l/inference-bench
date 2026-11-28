# Scheduler and serving core

The part of specrt that serves many requests at once: the paged KV pool, the scheduler, admission and preemption, the serving thread and the load generator. Options and defaults are the `serve.*` rows of [../generated/engine-options.md](../generated/engine-options.md); the same names are server.toml keys and `inference-bench serve` flags.

## Pieces

| Piece | Where | What |
|---|---|---|
| KV pool | `crates/specrt/src/runtime.rs` (`Layout`, `kv_block_bytes`) | blocks of 16 tokens, block major `[block][attention layer][kv head][16][head_dim]`, K and V pools, one f16 plane in the serving layout (fp8 options in [quantisation.md](quantisation.md)); state kinds from `[[state]]` |
| block tables | `Buffers::block_table` | one row of `max_model_len / 16` entries per slot, host mirror, only changed entries copied |
| recurrent state | `crates/specrt/src/statepool.rs` | conv and delta-net state per slot (157 MB on the 27B) in a virtual range reserved at load, mapped on admission and unmapped on finish, cancel or preemption |
| step interface | `crates/specrt/src/runtime/step.rs` | `prefill_slot` and `decode(rows)`; sampling on the device in `sample_rows` |
| scheduler | `crates/specrt/src/sched.rs` | `BlockPool`, `Scheduler<E: Executor>`, events; no CUDA |
| serving thread | `crates/specrt/src/serve.rs` | `Core` owns the scheduler on its own thread; `Serving` tokenises, submits and detokenises |
| server | `crates/core/src/serving.rs`, `crates/server` | `ServingTarget` shares one `Core`; `/metrics` from `crates/core/src/metrics_text.rs` |
| load generator | `crates/core/src/loadtest.rs`, `inference-bench loadtest` | open loop Poisson arrivals against any OpenAI compatible URL |

## A step

1. Cancel flags are checked; a dropped SSE stream frees its blocks within one step.
2. Every running decoder decodes. `Runtime::decode` pads rows to the next of `GRAPH_SIZES` (1, 2, 4, 8, 16, 32) and replays one CUDA graph per size; pad rows point at a spare slot and spare block.
3. Prefill budget beside decoders: `(step_cost - 1)` times the measured decode step, times the measured prefill rate, in multiples of 64, at least `min_prefill_tokens`; `itl_target_ms` is an optional absolute cap; a prompt up to `whole_prompt_tokens` is read whole; while a prompt over 2,048 tokens waits or reads, up to `prefill_burst_tokens` per step, at most `max_long_in_flight` such prompts at once. Running prefills are served shortest remaining first. `stall_budget_ms` with `min_stall_chunk_tokens` caps the chunk by the longest decoder wait. The prefill rate is measured after a stream sync and only from steps of a useful size.
4. With `[runtime] fuse`, decode rows and the prefill chunk run as one batch through every linear (`Runtime::fused_step`), so each weight is read once; attention and the delta-net step stay per kind of row. The fused step is not graph captured.

## Admission and preemption

- A waiting request is admitted when a slot and a recurrent state are free and the pool has blocks for its prompt plus one, plus a 1 percent reserve once anything runs, plus `admit_lookahead` tokens of growth for it and the running rows. A request that can never fit (prompt plus `max_tokens` over the pool or `max_model_len`) is a 400 at submit.
- `Scheduler::admit_until` scans the queue in arrival order and admits everything that fits; a request held for the long prompt cap or for KV is skipped. Full slots, a refused state and a parked sequence that cannot resume end the scan; parked and preempted sequences are never overtaken. A request held for KV blocks the queue again after `max_overtakes` later starts. The long prompt cap counts tokens left after the prefix cache (`PrefixCache::peek`).
- When a decoder needs a block and the pool is empty, the most recently admitted sequence is preempted. With `swap_space_gb` above 0 its KV blocks and state (and MTP pending hidden) go to pinned host memory and come back byte for byte; otherwise it is recomputed from its prompt and generated tokens, without resending streamed tokens.
- `serve.kv_cache_bytes` (server `kv_cache_gb`) fixes the pool; without it the pool takes free memory after weights, states and scratch, minus 1 GB, allocated last so the block count is exact.

Structured output and constrained tool calls compile a grammar (`crates/specrt/src/grammar.rs`) cached per model (`grammar_cache`, default 32); constrained rows decode without speculation. JSON schema objects are strict unless `additionalProperties` allows more.

## Why the defaults

| Default | Reason |
|---|---|
| `max_num_seqs` 8 | single stream decodes at full speed and aggregate still grows at 8 (#598) |
| `step_cost` 3 | no step cost cut the 29k prefill 2x on the 27B (#404) |
| `min_prefill_tokens` 512, `whole_prompt_tokens` 1024 | 1.2k TTFT p95 5 to 6 s at 8 to 20 decoders instead of 35 to 68 s, step p95 2.4 to 3.1x the target (#655) |
| `prefill_burst_tokens` 16384, `max_long_in_flight` 4 | fastest long prefill (29k at 19.3 s); no burst kept both decoder p99 under 0.5 s and long TTFT within 25 percent (#989) |
| `stall_budget_ms` 0 | any cap while decoders run cost the long mix 2x TTFT p50 (#1007) |
| `swap_space_gb` 0, `admit_lookahead` 0 | recompute until a 27B run shows a win |
| `max_overtakes` 32 | 1.2k TTFT p95 under 0.9 s beside long prompts; a large prompt still starts (#1518) |

Latency profile for chat beside long prompts: `--prefill-burst-tokens 384 --min-prefill-tokens 256 --whole-prompt-tokens 256` (decoder ITL p99 0.35 s, 29k prefill 26 s on the 27B).

## Measured state

27B NVFP4, MTP, prefix cache, 8 slots, 26 GB KV, server defaults, desktop GPU1, 28 Nov 2026, master 2bc21c2d, 60 requests at 1/s, 12 clients, 55 percent 1k to 8k word prompts and 45 percent 15k to 30k (`crates/specrt/scripts/admit1518_rounds.sh`, 5 rounds):

| Measure | Value |
|---|---|
| queue wait p50 | 0.05 s |
| TTFT p50 / p95 | 10.8 / 19.5 s |
| long prompt TTFT p95 | 21.5 s |
| decode tok/s per request p50 | 16.9 to 18.5 |
| ITL p95 | 49 ms |
| admission blocked steps | 0 |

## Tests

`crates/specrt/src/sched.rs` unit tests (leaks, preemption on a fake executor, admission, the load tables `print_load_table` and `print_sweep_table`), `crates/specrt/tests/serving.rs` (shuffled block tables, mixed length decode, 64 sequences with preemption, streaming and cancel), `fused.rs`, `statepool.rs`, `crates/server/tests/serving.rs`.

```
inference-bench loadtest --url http://127.0.0.1:8080 --model qwen3.8-27b \
    --requests 200 --rate 2 --clients 32 --mix 0.6:1k-8k,0.3:8k-32k,0.1:32k-128k --output 128-1024
```
