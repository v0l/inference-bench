#!/bin/bash
set -u
IB=${IB:-$HOME/git/ib265}
LL=${LL:-$HOME/git/llama.cpp/build/bin}
GG=${GG:-$HOME/git/specrt/out/gguf}
OUT=${OUT:-/tmp/ib265/out}
ROUNDS=${ROUNDS:-6}
BIN=$IB/target/release
export PATH=$HOME/.venv-vllm/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$OUT"

declare -A HF=([qwen3]=Qwen/Qwen3-0.6B [qwen25]=Qwen/Qwen2.5-0.5B)
declare -A GGUF=([qwen3]=Qwen3-0.6B-bf16.gguf [qwen25]=Qwen2.5-0.5B-bf16.gguf)
declare -A SPEC=([qwen3]=qwen3-0.6b [qwen25]=qwen2.5-0.5b)

wait_ready() {
  for _ in $(seq 1 150); do
    curl -sf "$1/v1/models" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}

state() {
  echo "== $1 $(date -Is)"
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  curl -s localhost:8001/metrics | grep -E '^vllm:num_requests_(running|waiting)' | tr '\n' ' '
  echo
  "$HOME/.venv-vllm/bin/python" -c 'import torch;f,t=torch.cuda.mem_get_info();print(f"cuda free {f/2**30:.1f} GiB")'
}

vllm_round() {
  local m=$1 r=$2 id=${HF[$1]}
  for p in 128 1024; do
    "$BIN/inference-bench" run --backend openai --url http://127.0.0.1:18000 --label vllm --model "$id" \
      --prompt-len $p --max-tokens 128 --json --no-save > "$OUT/$m-vllm-run$p-r$r.json"
  done
  "$BIN/inference-bench" batch --backend openai --url http://127.0.0.1:18000 --label vllm --model "$id" \
    --sizes 1,4,16 --json > "$OUT/$m-vllm-batch-r$r.json"
}

llama_round() {
  local m=$1 r=$2 id=${HF[$1]}
  for p in 128 1024; do
    "$BIN/inference-bench" run --backend openai --url http://127.0.0.1:18080 --label llama.cpp --model "$id" \
      --prompt-len $p --max-tokens 128 --json --no-save > "$OUT/$m-llama-run$p-r$r.json"
  done
  "$BIN/inference-bench" batch --backend openai --url http://127.0.0.1:18080 --label llama.cpp --model "$id" \
    --sizes 1,4,16 --json > "$OUT/$m-llama-batch-r$r.json"
  "$LL/llama-bench" -m "$GG/${GGUF[$m]}" -ngl 99 -p 128,1024 -n 128 -r 1 -o json > "$OUT/$m-llamabench-r$r.json" 2>/dev/null
  "$LL/llama-batched-bench" -m "$GG/${GGUF[$m]}" -ngl 99 -c 8192 -npp 128 -ntg 128 -npl 1,4,16 --output-format jsonl \
    2>/dev/null | grep '^{' > "$OUT/$m-llamabatched-r$r.jsonl"
}

specrt_round() {
  local m=$1 r=$2 id=${HF[$1]}
  for p in 128 1024; do
    "$BIN/inference-bench" run --backend specrt --worker "$BIN/inference-bench-specrt" --max-ctx 4096 --model "$id" \
      --prompt-len $p --max-tokens 128 --json --no-save > "$OUT/$m-specrt-run$p-r$r.json"
  done
  (cd "$IB/crates/specrt" && for p in 128 1024; do "$BIN/bench" specs/${SPEC[$m]}.toml $p 128; done
   for n in 1 4 16; do "$BIN/bench" specs/${SPEC[$m]}.toml 128 128 $n; done) > "$OUT/$m-specrtbench-r$r.txt" 2>/dev/null
}

sampler() {
  while true; do
    printf '%s %s\n' "$(date +%s)" \
      "$(curl -s -m 1 localhost:8001/metrics | awk '/^vllm:num_requests_running/ {print $2}')"
    sleep 1
  done
}
sampler >> "$OUT/load.log" &
spid=$!
trap 'kill $spid 2>/dev/null' EXIT

for m in ${MODELS:-qwen3 qwen25}; do
  id=${HF[$m]}
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 nohup "$HOME/.venv-vllm/bin/vllm" serve "$id" \
    --served-model-name "$id" --host 127.0.0.1 --port 18000 --dtype bfloat16 --max-model-len 4096 \
    --gpu-memory-utilization 0.08 --max-num-seqs 16 --no-enable-prefix-caching > "$OUT/$m-vllm.log" 2>&1 &
  vpid=$!
  "$LL/llama-server" -m "$GG/${GGUF[$m]}" --alias "$id" -ngl 99 --host 127.0.0.1 --port 18080 \
    -np 16 -c 24576 --no-cache-prompt > "$OUT/$m-llama-server.log" 2>&1 &
  lpid=$!
  wait_ready http://127.0.0.1:18000 $vpid && wait_ready http://127.0.0.1:18080 $lpid || { echo "servers did not start"; kill $vpid $lpid; exit 1; }
  state "$m start" >> "$OUT/state.log"
  vllm_round $m warm; llama_round $m warm
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  specrt_round $m warm
  engines=(vllm llama specrt)
  for r in $(seq 1 "$ROUNDS"); do
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    state "$m round $r" >> "$OUT/state.log"
    for k in 0 1 2; do
      e=${engines[$(((r + k) % 3))]}
      sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
      echo "$m round $r $e start $(date +%s)"
      ${e}_round $m $r
      echo "$m round $r $e end $(date +%s)"
    done
  done
  kill $vpid $lpid
  wait $vpid $lpid 2>/dev/null
  sleep 5
done
echo finished
