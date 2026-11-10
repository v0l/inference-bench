#!/usr/bin/env bash
set -euo pipefail
dest=${1:?usage: build462.sh <dest dir>}
here=$(cd "$(dirname "$0")/../../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-462}
systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$here" -E CARGO_TARGET_DIR="$CARGO_TARGET_DIR" -E CARGO_BUILD_JOBS=4 \
    bash -lc 'cargo build --release -p inference-bench-specrt --bins 2>&1 | grep -E "^error|^warning: unused" -A6 || true'
mkdir -p "$dest"
cp "$CARGO_TARGET_DIR/release/bench" "$CARGO_TARGET_DIR/release/logitdump" "$dest/"
