#!/usr/bin/env bash
set -uo pipefail
model=${1:?usage: ab7b.sh <model.gguf> <bin-a> <bin-b> [prompts] [new] [rounds]}
a=${2:?}; b=${3:?}
prompts=${4:-1024}; new=${5:-128}; rounds=${6:-3}
idle() { for _ in $(seq 150); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); [ "$u" -lt 15 ] && return; sleep 2; done; }
for _ in $(seq "$rounds"); do
    for bin in "$a" "$b"; do
        idle
        echo "$(basename "$(dirname "$bin")")/$(basename "$bin") $(SPECRT_BENCH_RUNS=5 "$bin" "$model" "$prompts" "$new" 2>&1 | grep -o 'prompt=[0-9]* prefill=[0-9.]* tok/s decode=[0-9]* tokens [0-9.]* tok/s' | tr '\n' ' ')"
    done
done
