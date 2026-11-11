#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
B=${BENCH:-/mnt/data/cargo-target/inference-bench/release/bench}
spec=tests/fixtures/qwen3.8-27b-nvfp4-exact.toml
act=tests/fixtures/qwen3.8-27b-nvfp4.toml
log=${LOG:-out/bench27b-nvfp4.log}
mkdir -p "$(dirname "$log")"
: > "$log"
start=$(date '+%Y-%m-%d %H:%M:%S')
guard() {
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt 36000 ]; then
        echo "free $free MiB, stopping" >> "$log"
        exit 1
    fi
    errors=$(journalctl -u local-llm.service --since "$start" --no-pager -q | grep -ciE " error|oom|out of memory")
    if [ "$errors" -gt 0 ]; then
        echo "vLLM logged $errors errors, stopping" >> "$log"
        exit 2
    fi
    echo "== $* free ${free} MiB util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) $(date +%T)" >> "$log"
}
for round in 1 2; do
    for mode in weight act; do
        guard single $mode round $round
        if [ "$mode" = act ]; then
            SPECRT_BENCH_RUNS=3 nice -n 15 "$B" "$act" 128,1024,4096 128 >> "$log" 2>&1
        else
            SPECRT_BENCH_RUNS=3 nice -n 15 "$B" "$spec" 128,1024,4096 128 >> "$log" 2>&1
        fi
    done
done
for batch in 4 8 16; do
    guard batch $batch
    SPECRT_BENCH_RUNS=3 nice -n 15 "$B" "$act" 128 128 "$batch" >> "$log" 2>&1
done
echo "done free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date +%T)" >> "$log"
