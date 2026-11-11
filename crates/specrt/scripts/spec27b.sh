#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
B=${BENCH:-${CARGO_TARGET_DIR:-../../target}/release/bench}
spec=${SPEC:-tests/fixtures/qwen3.8-27b-nvfp4-exact.toml}
log=${LOG:-out/spec27b.log}
mkdir -p "$(dirname "$log")"
: > "$log"
start=$(date '+%Y-%m-%d %H:%M:%S')
guard() {
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt 36000 ]; then
        echo "free $free MiB, stopping" | tee -a "$log"
        exit 1
    fi
    errors=$(journalctl -u local-llm.service --since "$start" --no-pager -q | grep -ciE " error|oom|out of memory")
    if [ "$errors" -gt 0 ]; then
        echo "vLLM logged $errors errors, stopping" | tee -a "$log"
        exit 2
    fi
    echo "== $* free ${free} MiB util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) $(date +%T)" | tee -a "$log"
}
for batch in ${BATCHES:-1 4 8}; do
    guard batch "$batch"
    nice -n 15 "$B" --speculative "$spec" 8192 "${NEW:-128}" "$batch" 2>&1 | grep -v "^ *$" | tee -a "$log"
done
echo "done free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date +%T)" | tee -a "$log"
