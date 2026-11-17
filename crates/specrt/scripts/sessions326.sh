#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-326}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
out=${OUT:-$root/crates/specrt/out/sessions326}
mkdir -p "$out"
cd "$root"
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "${FORCE:-}" ]; then
    echo "the GPU has compute apps running, set FORCE=1 inside a granted window" >&2
    exit 1
fi
exec 9>/tmp/gpu27.lock
flock 9
systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$root" \
    -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH \
    cargo test -p inference-bench-specrt --release --test sessions -- --include-ignored --test-threads=1 2>&1 | tee "$out/roundtrip.log"
if [ -z "${MEASURE:-}" ]; then
    exit 0
fi
spec=${SPEC:-$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4-exact.toml}
tokens=${TOKENS:-32768}
systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$root" \
    -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH \
    cargo run -p inference-bench-specrt --release --bin sessionbench -- "$spec" "$tokens" "$out/store" 2>&1 | tee "$out/table.md"
