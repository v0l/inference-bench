#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
target=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-763}
spec=$here/../tests/fixtures/qwen3.8-27b-nvfp4-exact.toml
mode=${1:?usage: gate763_desktop.sh <loop|kl|needle> args... [kvgate flags]}
shift
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 \
    --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
    "$target/release/kvgate" "$spec" "$mode" --corpus "${KVGATE_CORPUS:-$here/../..}" "$@"
