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

## Not done yet

- Engine side `Capture`: copy a sequence's KV blocks per full attention layer (packed layout,
  side stream to pinned memory, as `snapshot_state_async` does for the state) and restore into
  fresh blocks plus `restore_state`. Needs the GPU.
- Scheduler hook: record requests in `Tracker`, call `Janitor::tick` from a timer, look up
  `find_prefix` at admission when the prefix cache misses, report `cached_tokens`.
- Server: `[sessions]` config mapping, `/admin/sessions` routes, metrics export,
  `--purge-sessions`, UI section.
- Roundtrip logits test on the tiny hybrid fixture, end to end test, 32k measurement table.
