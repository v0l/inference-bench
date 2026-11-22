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

## Perf passes

The standard per-architecture speed loop is `docs/perf-pass.md` (`crates/specrt/scripts/perfpass.sh <family> --host spark`); results land in `docs/perf/scoreboard.md`.

## GPUs

The desktop has two GPUs. GPU0 belongs to the user's running app (inference-bench-ui on :8001). Never touch it, never stop or restart the app, never start work on GPU0. `running_seqs` on :8001 says nothing about GPU1.

GPU1 is free for workers. Always use it through the per-GPU lock: `~/.agents/bin/gpu-run 1 -- <cmd>`. Never use `gpu-run any` (it can pick GPU0) and never set `CUDA_VISIBLE_DEVICES=0`.

Compile first (`cargo test --no-run` or a build). Take the lock only around the GPU binary, never while compiling. Several workers share GPU1, so keep windows short and release the lock the moment the run is done.

Run the cheapest tier that can fail because of your change: CPU, then tiny fixtures on GPU, then a synthetic 27B-shaped layer, then the real model. The real 27B NVFP4 weights fit one 95 GB card, so the real27b, serving, mtp and kv_dtype suites and loadtests run on GPU1. Running them and pasting the `test result:` lines is your job. Only a run that needs both GPUs (tp2, two-GPU placement) is left to the supervisor: give the exact command in your report.

Never stop or restart `local-llm.service` or touch anything on the Spark without the user.

## Merging

Work on your issue branch and commit there. Do not push or merge. Merges happen through the supervisor.
