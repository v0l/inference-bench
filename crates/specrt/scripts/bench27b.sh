#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
B=/mnt/data/cargo-target/inference-bench/release/bench
log=out/bench27b.log
: > "$log"
start=$(date +%T)
for round in 1 2 3; do
    for mode in weight act; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free" -lt 36000 ]; then
            echo "free $free MiB, stopping" >> "$log"
            exit 1
        fi
        echo "== $mode round $round free ${free} MiB $(date +%T)" >> "$log"
        if [ "$mode" = act ]; then
            SPECRT_BENCH_RUNS=2 SPECRT_FP8_ACT=1 nice -n 15 "$B" tests/fixtures/qwen3.8-27b.toml 128,1024,4096 128 >> "$log" 2>&1
        else
            SPECRT_BENCH_RUNS=2 env -u SPECRT_FP8_ACT nice -n 15 "$B" tests/fixtures/qwen3.8-27b.toml 128,1024,4096 128 >> "$log" 2>&1
        fi
        errors=$(journalctl -u local-llm.service --since "$start" --no-pager -q | grep -ciE " error|oom|out of memory")
        if [ "$errors" -gt 0 ]; then
            echo "vLLM logged $errors errors, stopping" >> "$log"
            exit 2
        fi
    done
done
echo "done free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date +%T)" >> "$log"
