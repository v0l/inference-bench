# AGENTS.md

A Rust LLM server and desktop app on its own engine, `specrt`. Crates: `specrt` (engine), `kir`, `core`, `server`, `cli`, `ui`.

## Build

```bash
export CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-<issue>
export CARGO_BUILD_JOBS=4
cargo build --release
cargo test -p inference-bench-specrt
```

- Use a per-issue target dir. The default one is shared and parallel sessions overwrite each other's binaries.
- Run cargo under `systemd-run --user --wait --collect -P -q -p MemoryMax=10G`. Anything that loads weights uses `MemoryMax=16G -p MemorySwapMax=0`.
- Put every GPU run under `flock /tmp/gpu27.lock`.
- Everything configurable lives in the model spec TOML or the server config. Env vars are deprecated dev overrides.
- No code comments. Commit subjects are short, no trailers.

## Testing

Read `docs/testing.md`. It has the tiers, every test file, the exact commands, reference data and speed measurement rules.

## Do not block on the 27B

Never wait for a 27B GPU window. Run the cheapest tier that can fail because of your change (CPU, then tiny fixtures on GPU, then a synthetic 27B-shaped layer), paste the `test result:` lines, and give the exact real-27B command in your report. The supervisor runs it in a window.

Never stop or restart the user's app on :8001, `local-llm.service` or anything on the Spark. Check `nvidia-smi` free memory first.

## Merging

Work on your issue branch and commit there. Do not push or merge. Merges happen through the supervisor.
