# Prefix caching in specrt

Issue #325. Agents resend the same system prompt, tool schema and history every turn, so the
prefix of a request has usually been prefilled before. Prefix caching skips that prefill.

## What is already there

The KV cache is paged. `codegen::PAGE` is 16 tokens, `sched::BlockPool` hands out block ids, each
sequence has a block table (`Runtime::set_block_table`) and the attention kernels read through it.
Any physical block can sit at any logical position, which `tests/serving.rs` proves with shuffled
tables. Blocks were owned by exactly one sequence. Sharing them needs refcounts and nothing else
in the kernels.

The 16 full-attention layers of the Qwen3.8 27B have this paged KV. The 48 delta-net layers have a
recurrent state per sequence (`GdnBuffers`: a conv state and an ssm state per slot, see
`docs/specrt-linear-attention.md`). It exists only for the end of the tokens processed so far, so a
prefix can be reused only where a copy of that state was taken.

State per slot on the 27B: ssm `48 layers x 48 value heads x 128 x 128 x 4 B` is about 151 MB, plus
the conv state (a few MB). `Runtime::state_slot_bytes` reports the exact figure for a spec. That is
why snapshots are taken at chosen boundaries, never at every block.

## What vLLM does

Read from `vllm/v1/core` in the local venv.

- Every KV cache group (full attention, mamba/linear attention) has a block size and a
  `SingleTypeKVCacheManager`. `HybridKVCacheCoordinator` runs them together and aligns block sizes
  so the hit length is a multiple of both (`_cache_hit_alignment_tokens`).
- Block hash: `hash_block_tokens(parent_hash, token_ids, extra_keys)`. `extra_keys` carries LoRA
  id, multimodal hashes and `cache_salt`. The first block chains from a fixed `NONE_HASH`.
- `BlockPool` keeps a hash to block map and a free list in LRU order. A block with refcount zero
  stays in the map until it is reused, so a later request can revive it.
- A mamba group, with `mamba_cache_mode = "align"`, stores one recurrent state per block slot at
  block boundaries. `MambaManager.find_longest_cache_hit` searches right to left and takes the
  LAST block whose hash is cached (the state exists only there), and the full attention group is
  truncated to that length. Earlier mamba blocks are null placeholders.
- The hybrid coordinator iterates to a fixed point: each group may shorten the candidate hit, and
  the check repeats until every group accepts it. Full attention is downward closed (any shorter
  prefix is also cached), so for the simple hybrid case one pass suffices.
- Sparse retention (`reachable_block_mask`, retention interval) keeps only some boundary states so
  mamba state memory stays bounded.

specrt takes the same shape: a block hash chain, refcounted shared KV blocks, and a state snapshot
that decides the usable hit length. The differences: snapshots live in a pool with their own
budget, not in KV blocks, and they are taken at an interval chosen by config.

SGLang's RadixAttention uses a radix tree over token ids. A hash chain over fixed blocks gives the
same sharing without a tree, and it is what vLLM and jitllm (`kvpage.go`) use, so we use it.

## Design

All of it is in `crates/specrt/src/prefix.rs`, free of CUDA, so the policy is unit tested on the CPU.

### Keys

`namespace(spec, kv_planes, salt)` is a SHA-256 over the model name and repo, layer counts, KV
heads, head dim, linear attention dims, KV planes, block size and the salt. It is the root of every
chain, so two specs with equal block bytes can never alias, and `cache_salt` isolates tenants.

`block_hashes(root, tokens)` is `h[i] = sha256(h[i-1] || tokens of block i)` for each FULL block.
A partial tail block is never hashed. The key is computed on the final templated token ids, so the
Qwen template stripping earlier reasoning changes the key naturally (slice 2 checks this against
real transcripts).

### KV blocks

`BlockPool` gains refcounts. `alloc` returns a block at refcount 1, `retain` adds a holder,
`release` drops one and frees at zero. The cache holds one reference per registered block, each
running sequence holds one per block in its table.

`PrefixCache` maps chain hash to block id. After a prefill the sequence registers its full blocks.
A request starts by `lookup`, which returns the retained blocks of the longest usable prefix.
Eviction takes the least recently used block that only the cache references. Ticks are assigned
deepest block first so a parent is always newer than its children: a chain loses its tail before
its head, and no block is stranded behind an evicted parent. `max_blocks` is the budget in blocks.

### Recurrent snapshots

`SnapshotStore` is a pool keyed by the chain hash of the block boundary the state belongs to, with
its own LRU and a budget in snapshots (`max_snapshots`, each about 150 MB on the 27B, held in
host memory). A snapshot is only inserted for a hash whose KV block is cached, and it is dropped
when that block is evicted.

`lookup` walks the leading run of cached blocks, then takes the deepest boundary in it that has a
snapshot. A model with no linear layers needs no snapshot and uses the whole run. The hit never
covers the last token of the prompt: at least one token is prefilled to produce logits.

`snapshot_points(start, prompt_len, interval)` lists the block aligned positions where prefill
must stop to take a snapshot: every `interval` tokens (default 2048) and the last full block
boundary of the prompt, which is the position the next chat turn will hit. The tail after the last
boundary is prefilled normally and is not cached. Cutting snapshots on chat template message
boundaries (jitllm's proposal) is a later refinement; the end of request snapshot covers the
common agent loop.

`Runtime::snapshot_state(slot)` copies a slot's conv and ssm state to host, `restore_state` writes
it into any live slot. The `StateSnapshot` is a plain pair of `Vec<f32>`, which is also what the
idle session disk snapshot issue needs to persist.

### Config

`[runtime.prefix_cache]` in the model spec (and the server `runtime` table that maps onto it):

```toml
[runtime.prefix_cache]
enabled = false
snapshot_interval = 2048
max_snapshots = 4
max_blocks = 0
```

`max_blocks = 0` lets the cache use the whole KV pool, bounded by eviction under allocation
pressure. No env knobs.

## Correctness test

`tests/prefix.rs`, tiny hybrid fixture: prefill A then B in one slot gives the same logits, bit for
bit, as prefill A in a second slot, snapshot, restore into a third slot whose block table shares A's
blocks, then prefill B there. Greedy decode afterwards matches too.

## Scheduler integration

`sched::Scheduler` owns the `PrefixCache` when `[runtime.prefix_cache] enabled = true` and the
executor allows it (`Executor::supports_prefix_cache`).

- `submit` computes the namespace root from the request salt.
- `admit` hashes the prompt, calls `lookup` (never the last prompt token), evicts unreferenced
  cached blocks until the remainder fits, restores the hit's snapshot into the new slot and starts
  prefill at the hit length. `Finished` carries `cached_tokens`.
- Hybrid prefill is cut at `snapshot_points`, and `cache_progress` registers the blocks and takes a
  snapshot at each. A non-hybrid model registers its blocks after every chunk.
- On finish, every full block (prompt and generated tokens) is registered. A hybrid model also
  takes a snapshot there if the end is block aligned, otherwise its next hit reaches the last
  snapshot point of the prefill. So a chat turn hits the previous prompt to its last full block.
- Block allocation (`alloc_block`) evicts LRU cached blocks before preempting anyone.

## Server

- `usage.prompt_tokens_details.cached_tokens` is in every response and in the streamed usage chunk.
- `/metrics`: `inference_prefix_cache_queries_total` (prompt tokens looked up),
  `inference_prefix_cache_hit_tokens_total`, `inference_prefix_cache_hit_rate` (gauge), and gauges
  for cached blocks, snapshots and snapshot host bytes.
- `[runtime.prefix_cache]` in the server config maps onto the spec table, validated at load.
- `cache_salt` in the request body narrows the namespace. With API keys configured the key is
  hashed into the namespace too, so keys never share cache entries unless the operator sets
  `prefix_cache_share_across_keys = true`. Without keys the body salt alone is used.

### Timing side channel

A hit makes the first token faster, so a client that can send prompts can tell whether another
client recently sent the same prefix: send a guess, compare TTFT with a cold prompt. This is the
same side channel as any shared prefix cache (vLLM documents it for `cache_salt`). Mitigations here:
per key isolation by default when keys exist, `cache_salt` for finer tenancy, and no sharing
across models or specs because the namespace includes them. An operator who needs no leakage at all
turns the cache off. Hits are only visible to the caller that earned them: `cached_tokens` is in the
response for that request only and the metrics are aggregates.

## Chat templates

The key is computed on the final templated token ids, after the template has dropped the reasoning
of earlier turns. `tests/prefix.rs::the_cache_key_follows_the_final_templated_ids_across_thinking_turns`
renders a Qwen thinking transcript for two turns. With `preserve_thinking` the turn 1 prompt is a
prefix of turn 2. With it off (the template drops the old reasoning) the chain forks at the first
changed block, so the cache only shares the part that really is equal. Equal ids from different
request shapes (string or content parts) share keys, and thinking on and off never share a tail.

## Measurements

`prefixbench` (`crates/specrt/src/bin/prefixbench.rs`): `prefixbench <spec> <turns|clients> <on|off> [snapshot_interval] [kv_gb]`.
Greedy, synthetic token ids, one warm-up request. The GPU was shared with other workers' jobs and the
live vLLM during these runs (the lock serialises our runs, not theirs), so absolute times are noisy;
the 27B off runs vary from 8.5 s to 14.8 s at turn 20 between runs.

20 turn agent transcript: 1000 token system prompt, each turn appends 32 generated and 500 new tokens,
TTFT of each request (ms), snapshot interval 2048.

| model | cache | turn 1 | turn 2 | turn 10 | turn 20 | median | hit rate |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | off | 14.2 | 18.8 | 207 | 543 | 256 | 0 |
| Qwen3-0.6B | on | 13.6 | 10.1 | 18.8 | 31.1 | 20.8 | 0.912 |
| Qwen3.8 27B nvfp4, MTP off | off | 1339 | 1039 | 7479 | 14833 | 7479 | 0 |
| Qwen3.8 27B nvfp4, MTP off | on | 779 | 496 | 528 | 592 | 561 | 0.907 |

With the cache the TTFT is flat in the context length (the remainder of a turn is about 530 new
tokens) while without it TTFT grows linearly with the transcript. The 27B hit is 992 tokens at turn 2:
the hybrid hit stops at the last block boundary of the previous prefill.

16 clients sharing a 2000 token system prompt, 200 unique tokens each, 16 new tokens (Qwen3-0.6B, a
second wave of 16 new clients with the same system prompt follows the first):

| cache | wave | TTFT mean (ms) | TTFT p95 (ms) | wall (s) | hit rate |
|---|---|---|---|---|---|
| off | first | 803 | 1373 | 1.56 | 0 |
| off | second | 798 | 1377 | 1.56 | 0 |
| on | first | 434 | 565 | 0.67 | 0.682 |
| on | second | 318 | 318 | 0.43 | 0.909 |

Within the first wave the first admitted request prefills the whole prompt and registers its blocks
as each chunk completes, so requests admitted in later steps already hit (hit rate 0.682 of all
prompt tokens before any wave finished).

Snapshot memory on the 27B: 8 snapshots held 1255 MB of host memory, 157 MB each (conv plus ssm
state, f32). `max_snapshots` is the budget: the default of 4 is 0.63 GB. Snapshots live in host
memory, not in the KV pool, so they do not reduce KV blocks. Eviction under pressure is covered by
`cached_blocks_are_evicted_when_new_requests_need_the_pool_and_output_stays_exact` and the GPU test
`tiny_prefix_cached_scheduling_matches_single_sequence_greedy` (90 blocks, 6 slots, 24 requests with
a shared prefix, exact greedy output).

The 27B rows above ran with `[mtp] speculate = false`. The cache now also works with MTP speculation
on, which is the stock serving config. A snapshot carries three parts: the conv state, the ssm state
and the MTP pending hidden row of the slot (the input of the next draft round). The draft layer's KV
needs no extra copy because it lives in the same paged blocks as the target layers (`kv_layers`
includes the MTP layer), so a shared block already holds both. Restoring clears the slot's draft
flag, so the first speculative step after a hit re-primes the drafts from the restored hidden.

Qwen3.8 27B nvfp4 with `speculate = true`, `draft_tokens = 2` (the stock spec), same 20 turn transcript:

| cache | turn 1 | turn 2 | turn 10 | turn 20 | median | hit rate | snapshots |
|---|---|---|---|---|---|---|---|
| off | 1328 | 1994 | 7741 | 15047 | 8577 | 0 | 0 |
| on | 738 | 410 | 473 | 533 | 487 | 0.913 | 8 (1255 MB) |

Tests: `a_prefix_cache_hit_with_speculation_on_gives_the_output_of_a_run_without_the_cache` and
`the_cached_state_includes_the_draft_layer_kv_and_pending_hidden` (tests/mtp.rs, tiny MTP fixture,
4 chat turns, greedy output identical with the cache on and off, every later turn hits).

### Request ends that are not block aligned

When a request finishes, the full blocks are registered and the partial last block is registered as
a tail: its parent hash, its tokens, its block, and for hybrid models a state snapshot taken at that
exact position. A lookup that finds the tail's parent and a prompt that continues the tail's tokens
gets `tokens = full blocks + tail length`. Admission copies the tail block into a fresh block on the
device (`copy_kv_block`, a block is never shared while a sequence still writes into it) and starts
prefill at the exact end of the previous request. Tails are evicted before their parents, die with
them, count against `max_blocks`, and hybrid tail states count against `max_snapshots`.
`the_next_chat_turn_hits_the_end_of_the_previous_request_even_when_it_is_not_block_aligned`
checks 189 of 190 tokens for plain and hybrid models (the previous prompt plus its generated
tokens, minus the last token that is always recomputed).

### Snapshot copy off the serving thread

`async_snapshots = true` (default) in `[runtime.prefix_cache]`: the state is copied device to device
into a staging buffer on the compute stream, then a side stream copies it to a pinned host buffer
and records an event. The serving thread does not wait; the snapshot reads synchronise on that
event only when a hit restores it. Pinned buffers are recycled through a pool when snapshots are
evicted. `async_snapshots = false` falls back to a blocking copy.

Decode inter-token gaps of one steady decoder (ms) while six 3072 token prompts arrive one by one,
snapshot interval 512 (so each long prompt takes several snapshots), 27B nvfp4 with MTP on:

| mode | p50 | p99 | max |
|---|---|---|---|
| cache off | 99.4 | 756 | 940 |
| cache on, async copy | 99.5 | 105 | 136 |
| cache on, blocking copy | 98.5 | 213 | 231 |

The async copy halves the worst gap against the blocking copy. The cache off row has larger gaps
than both cache rows, and that is probably not the copy: with the cache on, hybrid prefill is cut
at every snapshot point (512 tokens here), so each step runs a shorter prefill chunk beside the
decoder. That explanation is untested (no run with the cache on and the interval above the prompt
length). The comparison this table supports is async against blocking copy, both with the same
chunking: p99 105 against 213 ms.

## Status

Done: refcounted blocks, hash chain, LRU cache, snapshot store, snapshot and restore on the runtime,
the config table, scheduler integration, server usage, metrics and salts, the template test, the
benchmarks above, bit identical A+B equals A, snapshot, B test, the cache with MTP on, tail
snapshots at non block aligned request ends, asynchronous snapshot copies.

Left (issue #326): snapshots at chat template message boundaries, and persisting `StateSnapshot`
and cached blocks to disk. `StateSnapshot` is already one flat f32 buffer plus the three lengths.
