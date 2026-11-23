#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-1136}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-3}
run() { systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$PWD" -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E HOME nice -n 19 "$@"; }
run cargo test -p inference-bench-specrt --test no_arch_names --test codegen --test golden --test spec --test linear_unchanged 2>&1 | grep -E '^test result|FAILED|panicked|^error' || true
run cargo test -p inference-bench-kir 2>&1 | grep -E '^test result|FAILED|panicked|^error' || true
