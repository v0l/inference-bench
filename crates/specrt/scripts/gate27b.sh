#!/usr/bin/env bash
set -uo pipefail
out=${1:?usage: gate27b.sh <outfile>}
spec=$(cd "$(dirname "$0")" && pwd)/../tests/fixtures/qwen3.8-27b-nvfp4.toml
bin=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench}/release
idle() { for _ in $(seq 120); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); [ "$u" -lt 15 ] && return; sleep 2; done; }
run() { systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 "$@"; }
{
    idle; run "$bin/logitdump" "$spec" 17,300 4
    for _ in 1 2; do
        idle; SPECRT_BENCH_RUNS=5 run "$bin/bench" "$spec" 1024,4096 128 | grep prompt=
        idle; SPECRT_BENCH_RUNS=5 run "$bin/bench" "$spec" 512 128 8 | grep batch=
    done
} >"$out" 2>&1
