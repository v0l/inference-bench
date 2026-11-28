# Prefix cache and session snapshots

Agents resend the same system prompt, tools and history every turn. The prefix cache reuses KV blocks and recurrent state already computed; session snapshots write an idle conversation to disk and restore it on the next turn. Both live in `crates/specrt/src/prefix.rs` and `crates/specrt/src/sessions.rs` (no CUDA, unit tested on CPU) and are driven by `sched.rs`.

## Prefix cache

- Keys: `namespace(spec, kv_planes, salt)` is a SHA-256 over the model, layer counts, KV heads, head dim, linear attention dims, KV planes, block size and the salt, and roots every chain. `block_hashes` is `h[i] = sha256(h[i-1] || tokens of block i)` over full 16 token blocks, on the final templated ids (so a template that drops earlier reasoning forks the chain where the ids change).
- KV blocks: `BlockPool` blocks are refcounted; the cache holds one reference per registered block and each sequence one per block in its table. Eviction takes the least recently used block only the cache holds, deepest first, so a chain loses its tail before its head. Allocation evicts cached blocks before preempting anyone.
- Recurrent state: models with linear layers can reuse a prefix only where a state copy exists. `SnapshotStore` keeps states (conv, ssm, and the MTP pending hidden) keyed by the chain hash of their block boundary, in host memory with its own LRU and `max_snapshots` budget; a snapshot dies with its block. Prefill stops at `snapshot_points`: every `snapshot_interval` tokens and the last full block of the prompt. Copies run on a side stream into pinned buffers (`async_snapshots`), so the serving thread does not wait.
- Lookup: the leading run of cached blocks, cut to the deepest boundary with a snapshot on hybrid models, never covering the last prompt token. A finished request also registers its partial last block as a tail (with a state taken at that exact position); a later prompt that continues it gets the full blocks plus the tail, and admission copies the tail block before writing into it.
- The MTP draft layer's KV is in the same paged blocks, so a hit carries it; restoring clears the draft flag and the first round re-primes drafts.
- Server: `usage.prompt_tokens_details.cached_tokens` in every response; `/metrics` `inference_prefix_cache_queries_total`, `inference_prefix_cache_hit_tokens_total`, `inference_prefix_cache_hit_rate` and gauges for blocks, snapshots and snapshot bytes. `cache_salt` narrows the namespace; with API keys the key is hashed in too unless `prefix_cache_share_across_keys = true`. A hit makes TTFT faster, which a client can observe for its own guesses; per key isolation, salts and turning the cache off are the mitigations.

`[runtime.prefix_cache]`:

| Key | Default |
|---|---|
| `enabled` | false in the spec, on in the max performance profile |
| `snapshot_interval` | 2048 |
| `max_snapshots` | 32 (about 5 GB of host memory on the 27B; 4 evicted a chat's end state before its next turn) |
| `max_blocks` | 0, the whole pool, bounded by eviction |
| `async_snapshots` | true |

## Session snapshots

`[sessions]` (server config, `SessionOptions`): `enabled` false, `dir`, `idle_seconds` 300, `ttl_hours` 8, `min_tokens` 4096, `max_bytes` (0 is 20 percent of free space), `keep_after_restore` false.

- A finished request of at least `min_tokens` stays held on the GPU. `Janitor::tick` writes sessions idle for `idle_seconds` (KV blocks of every attention layer plus the recurrent state, through pinned memory on a side stream), releases the GPU side only after the write succeeds, expires after `ttl_hours` and evicts least recently used past `max_bytes`. Under block pressure the oldest held session is written first.
- Files: `<dir>/sessions/<model hash prefix>/<session id>.kvs`, 0600 in a 0700 directory, header with dtype, layers, block size, tokens, model and spec hashes, crc32 per section, written to a temp file, fsynced and renamed. Open drops truncated, corrupt or mismatched files.
- At admission `find_prefix` finds the longest stored session that is a strict prefix of the prompt within the salt scope; the payload is restored into fresh blocks and only the suffix is prefilled. A failed restore deletes the file and prefills.
- Saves and restores share the pinned host pool with swapped sequences (`swap_space_gb` is the one limit).
- Server routes: `GET /admin/sessions`, `DELETE /admin/sessions/{id}`, `POST /admin/sessions/{id}/save|restore|strip`, `inference-bench-server --purge-sessions`, Prometheus `inference_session_*`. The app's server tab lists sessions.

## Measured state

| Case | Without | With | Source |
|---|---|---|---|
| 27B NVFP4, MTP on, 20 turn agent transcript, TTFT median | 8,577 ms | 487 ms (hit rate 0.913) | #325 |
| decoder ITL p99 while six 3k prompts arrive, async against blocking snapshot copy | 213 ms | 105 ms | #325 |
| 27B NVFP4 at 32k on the Spark, session restore TTFT | 72.4 s full prefill | 7.7 s (2.27 GiB file, 5.2 s restore) | #326 |

## Tests

`crates/specrt/tests/prefix.rs` (A then B equals A, snapshot, restore, B bit for bit; template keys), `crates/specrt/src/prefix.rs` and `sessions.rs` unit tests, `sched.rs` (eviction, sessions with a fake clock), `crates/specrt/tests/mtp.rs` (cache with speculation), `crates/specrt/tests/sessions.rs` (GPU roundtrip, 64 greedy tokens and logprobs bit for bit), `crates/server/tests/api.rs`. Benches: `crates/specrt/src/bin/prefixbench.rs`, `sessionbench.rs`, `crates/specrt/scripts/sessions326.sh`.
