#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/910
target=$HOME/ib-gates-target/910
mkdir -p "$target"
exec 9>/tmp/gpu27.lock
flock 9
systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 -p WorkingDirectory="$root" \
    --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" --setenv=CARGO_BUILD_JOBS=6 \
    bash -lc "cargo test -p inference-bench-specrt --release --test vision_multi_gpu -- --include-ignored --test-threads=1 --nocapture 2>&1 | grep -E '^test |KL|panicked|test result|^error'"
