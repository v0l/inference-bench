#!/bin/bash
set -u
IB=${IB:-$HOME/git/ib273}
LL=${LL:-$HOME/git/llama.cpp/build/bin}
GG=${GG:-$HOME/git/specrt/out/gguf}
OUT=${OUT:-/tmp/ib273/out}
ROUNDS=${ROUNDS:-6}
BIN=$IB/target/release
mkdir -p "$OUT"

declare -A HF=([qwen3]=Qwen/Qwen3-0.6B [qwen25]=Qwen/Qwen2.5-0.5B)
declare -A GGUF=([qwen3]=Qwen3-0.6B-bf16.gguf [qwen25]=Qwen2.5-0.5B-bf16.gguf)
declare -A SPEC=([qwen3]=qwen3-0.6b [qwen25]=qwen2.5-0.5b)

drop() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }

wait_ready() {
  for _ in $(seq 1 150); do
    curl -sf "$1/v1/models" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}

llama_round() {
  local m=$1 r=$2 id=${HF[$1]}
  drop
  "$LL/llama-server" -m "$GG/${GGUF[$m]}" --alias "$id" -ngl 99 --host 127.0.0.1 --port 18080 \
    -np 1 -c 4096 --no-cache-prompt > "$OUT/$m-llama-server-r$r.log" 2>&1 &
  local pid=$!
  wait_ready http://127.0.0.1:18080 $pid || { kill $pid; return 1; }
  for p in 128 1024; do
    "$BIN/inference-bench" run --backend openai --url http://127.0.0.1:18080 --label llama.cpp --model "$id" \
      --prompt-len $p --max-tokens 128 --json --no-save > "$OUT/$m-llama-run$p-r$r.json"
  done
  kill $pid; wait $pid 2>/dev/null
}

specrt_round() {
  local m=$1 r=$2 id=${HF[$1]}
  drop
  for p in 128 1024; do
    "$BIN/inference-bench" run --backend specrt --worker "$BIN/inference-bench-specrt" --max-ctx 4096 --model "$id" \
      --prompt-len $p --max-tokens 128 --json --no-save > "$OUT/$m-specrt-run$p-r$r.json"
  done
  drop
  (cd "$IB/crates/specrt" && for p in 128 1024; do "$BIN/bench" specs/${SPEC[$m]}.toml $p 128; done) \
    > "$OUT/$m-specrtbench-r$r.txt" 2>/dev/null
}

for m in ${MODELS:-qwen3 qwen25}; do
  for r in $(seq 1 "$ROUNDS"); do
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader >> "$OUT/state.log"
    if [ $((r % 2)) = 1 ]; then order="llama specrt"; else order="specrt llama"; fi
    for e in $order; do
      echo "$m round $r $e start $(date +%s)"
      ${e}_round $m $r
    done
  done
done
echo finished
