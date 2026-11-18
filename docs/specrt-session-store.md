# Session snapshots to disk

Issue #326. An idle conversation is written to disk after 5 minutes, kept for 8 hours counted from
last use, and restored on the next request that continues it, so only the new suffix is prefilled.

## Done: the store (`crates/specrt/src/sessions.rs`, CPU only, 16 unit tests)

- `SessionStore` owns `<dir>/sessions/<model hash prefix>/<session id>.kvs`. The session id is a
  SHA-256 over model hash, spec hash and token ids. Directory mode 0700, files 0600.
- File: fixed header (magic, version, kv dtype, layers, block size, token count, created, last
  used, model hash, spec hash, section count, stripped flag, crc32), token ids with crc32, then each
  payload section as length, bytes, crc32, then a footer with the total file length. The payload
  sections are opaque bytes: the engine fills them with the packed KV blocks of each full attention
  layer followed by the recurrent state (`StateSnapshot` floats).
- Writes go to `.tmp-*`, fsync, rename, fsync of the directory. Leftover temp files are deleted at
  open. A failed write (disk full, directory gone) returns an error, bumps
  `snapshot_failures` and leaves no file.
- Open builds the in-memory index from headers and tokens and removes files that are truncated,
  corrupt, of another version, or whose model, spec, dtype, layers or block size differ.
- `find_prefix(prompt)` returns the longest stored session whose tokens are a strict prefix of the
  prompt. `load` verifies every checksum and the id, and drops the file on any failure so the
  caller prefills instead. `touch` updates last use in place.
- `Janitor::tick(now, store, source)` takes the clock as an argument (fake clock in tests): writes
  sessions idle for `idle_seconds` and at least `min_tokens` long through the `Capture` trait,
  releases the GPU side only after the write succeeded, expires after `ttl_hours`, then enforces
  `max_bytes` by evicting least recently used first.
- `strip` keeps header and tokens, drops the payload. `purge_root` deletes everything.
- `SessionOptions` is the `[sessions]` table: `enabled`, `dir`, `idle_seconds` (300), `ttl_hours`
  (8), `min_tokens` (4096), `max_bytes` (0 = unlimited here, to be derived from free space with
  `free_space_budget`), `keep_after_restore`.
- Counters in `Metrics` back the Prometheus names `inference_session_snapshots_total`,
  `..._restores_total{result}`, `..._snapshot_bytes`, `..._restore_seconds`,
  `..._prefill_tokens_saved_total`.

## Wired in

- `Executor::session_compat`, `save_session(blocks, state)` and `restore_session(slot, blocks, sections)`.
  `GpuExecutor` copies each KV block of the K and V caches through pinned memory on a forked side
  stream (`Runtime::save_kv_blocks`) and packs four sections: meta, K, V, recurrent state (conv, ssm,
  MTP pending). Restore (`load_kv_blocks`, `restore_state`) is a straight copy, no re-encoding.
- `Scheduler::enable_sessions`: a finished request that reached `min_tokens` stays held on the GPU
  (blocks retained) and is tracked by the janitor. `tick_sessions` (called every loop turn in
  `serve::run`) writes sessions idle for `idle_seconds`, then releases their blocks, expires after
  `ttl_hours` and enforces `max_bytes` (default 20 percent of free space). Under block pressure the
  oldest held session is written first. At admission `find_prefix` matches on salt scope plus token
  prefix, the blocks are allocated fresh, the payload is restored, only the suffix is prefilled and
  `cached_tokens` reports the restored length. A failed restore deletes the snapshot and prefills.
- Server: `[sessions]` table, `GET /admin/sessions`, `DELETE /admin/sessions/{id}`,
  `POST /admin/sessions/{id}/save|restore|strip`, Prometheus `inference_session_*`,
  `inference-bench-server --purge-sessions`. Sessions are off by default.
- UI: Sessions section on the server tab (size, idle, expires in, restore, strip, delete).

## Shared pinned pool (#402)

Session saves and restores take their pinned staging memory from `rt.host_pool()`, the same pool
`park`/`resume` use for swapped sequences. `swap_space_gb` is the one limit for both
(`Executor::set_host_limit`), so a parked sequence and a session write compete for the same bytes;
a save that does not fit fails, the session stays held on the GPU and is retried on the next tick.

## Tests

CPU: `sessions::tests` (store, janitor with a fake clock), `sched::tests` (two turns 6 minutes apart
on the fake executor, 8 h expiry, corrupt snapshot, salt isolation), `tests/api.rs` (routes, config,
purge), `serving::tests` (metrics), ui `the_server_tab_lists_saved_sessions_...`.
GPU (ignored): `tests/sessions.rs` roundtrips the tiny hybrid and a dense fixture through a fresh
engine and compares 64 greedy tokens and every logprob bit for bit.
Run both, and the 32k measurement table, with `crates/specrt/scripts/sessions326.sh` (`MEASURE=1`
adds `sessionbench`, which prints snapshot bytes, write time, restore time and TTFT with and without
the snapshot).

## Measurement (Spark GB10, Qwen3.8 27B NVFP4, 32k context)

| context | snapshot size | write | restore | TTFT restored | TTFT full prefill | speedup |
|---|---|---|---|---|---|---|
| 32768 tokens | 2.27 GiB | 10.3 s | 5.2 s | 7.7 s | 72.4 s | 9.4x |

The roundtrip tests compare the restored engine with an uninterrupted engine that ran the same
history through the same path (prefill, decode, then the prefix cache hit at the end of the previous
request), so logits are compared bit for bit against a run that never saved or restored.

## Left

- Sidecar index file for faster start (the index is built from the file headers, which is fast enough
  until thousands of sessions exist).
- Optional zstd for the recurrent state.
