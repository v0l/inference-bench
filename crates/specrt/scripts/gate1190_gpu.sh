#!/usr/bin/env bash
set -uo pipefail
dir=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-1190}/debug/deps
status=0
for t in mxfp8 fp8_pb_wo w4a16_nvfp4; do
    bin=$(ls -t "$dir"/$t-* | grep -v '\.d$' | head -1)
    echo "== $t"
    "$bin" --include-ignored --test-threads=1 --nocapture 2>&1 | grep -E "test |max err|panicked|result" || status=1
done
exit $status
