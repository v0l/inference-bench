#!/usr/bin/env bash
set -uo pipefail
model=${1:?usage: vsllama.sh <model.gguf> <specrt bench> [rounds]}
bench=${2:?}
rounds=${3:-3}
llama=${LLAMA_BENCH:-$HOME/git/llama.cpp/build/bin/llama-bench}
idle() { for _ in $(seq 150); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); [ "$u" -lt 10 ] && return; sleep 2; done; }
for _ in $(seq "$rounds"); do
    idle
    "$llama" -m "$model" -ngl 99 -fa 1 -p 1024,4096 -n 0 -r 3 -o json 2>/dev/null | jq -r '.[] | "llama.cpp prefill p=\(.n_prompt) \(.avg_ts | floor) +- \(.stddev_ts | floor)"'
    idle
    "$llama" -m "$model" -ngl 99 -fa 1 -p 0 -n 128 -d 1024,4096 -r 3 -o json 2>/dev/null | jq -r '.[] | "llama.cpp decode d=\(.n_depth) \(.avg_ts * 10 | floor / 10) +- \(.stddev_ts * 10 | floor / 10)"'
    idle
    SPECRT_BENCH_RUNS=3 "$bench" "$model" 1024,4096 128 2>&1 | grep -o 'prompt=[0-9]* prefill=[0-9.]* tok/s decode=[0-9]* tokens [0-9.]* tok/s' | sed 's/^/specrt /'
done
