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

## Slice 1 status and what is left

Done in slice 1: refcounted blocks, hash chain, LRU cache, snapshot store, snapshot and restore on
the runtime, the config table, the equality test.

Left for slice 2: scheduler integration (lookup before admission, chunk prefill at snapshot points,
register on finish, eviction under allocation pressure), `usage.prompt_tokens_details.cached_tokens`,
the `inference_prefix_cache_*` metrics, `cache_salt` per API key and the timing side channel note,
the 20 turn transcript and 16 client shared system prompt benchmarks.
