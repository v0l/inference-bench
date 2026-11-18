#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/gate776_guard.sh"
bin=${CARGO_TARGET_DIR:-$here/../../../target}/release/inference-bench
model=${MODEL:-orcarouter/Qwen3.8-27B-Uncensored-NVFP4}
port=${PORT:-18776}
seqs=${SEQS:-21}
ctx=${CTX:-262144}
kv_gb=${KV_GB:-95}
plan_ctx=${PLAN_CTX:-$((kv_gb * 1073741824 / 33280 / seqs))}
out=${OUT:-$here/../out/gate776}
mkdir -p "$out"
plan_only=0
[ "${1:-}" = --plan-only ] && plan_only=1
cfg=$out/server.toml
cat >"$cfg" <<EOF
max_num_seqs = $seqs
max_model_len = $ctx
kv_cache_gb = $kv_gb
max_batched_tokens = 2048

[runtime]
kv_dtype = "fp8"
kv_f16_layers = 0

[placement]
context = $plan_ctx
slots = $seqs
max_tp = 1

[[placement.devices]]
name = "gpu0"
memory_gib = 86.4
reserve_gib = 0
bandwidth_gbs = 1792
tflops = 500
ordinal = 0

[[placement.devices]]
name = "gpu1"
memory_gib = 86.4
reserve_gib = 0
bandwidth_gbs = 1792
tflops = 500
ordinal = 1

[[models]]
id = "qwen3.8-27b"
model = "$model"
EOF
"$bin" plan --spec "${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}" --config "$cfg" --objective latency | tee "$out/plan.txt"
[ $plan_only = 1 ] && exit 0
require_idle_gpus || exit 1
if curl -sf -m 2 localhost:8001/v1/models >/dev/null; then
    echo "the app on :8001 is serving, stop it first" >&2
    exit 1
fi
"$bin" serve --config "$cfg" --bind 127.0.0.1:$port >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 600); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    kill -0 $server 2>/dev/null || { tail -20 "$out/server.log"; exit 1; }
    sleep 1
done
curl -sf "localhost:$port/v1/models" >/dev/null
curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
    -d '{"model":"qwen3.8-27b","prompt":"hello","max_tokens":8}' >/dev/null
echo "== VRAM per GPU" | tee "$out/report.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee -a "$out/report.txt"
grep -E "pipelined serving layout|stage_kv_bytes" "$out/server.log" | tee -a "$out/report.txt" || true
echo "== one request above the planned per-slot context ($plan_ctx)" | tee -a "$out/report.txt"
long_words=${LONG_WORDS:-$((plan_ctx + 8000))}
awk -v n="$long_words" 'BEGIN{for(i=0;i<n;i++)printf "word "}' | jq -Rs '{model:"qwen3.8-27b",prompt:.,max_tokens:16,temperature:0}' >"$out/long.json"
curl -s -m 900 "localhost:$port/v1/completions" -H 'content-type: application/json' -d @"$out/long.json" | tee "$out/long.out" | jq -c '{status:(if .error then "error" else "ok" end),usage,error}' | tee -a "$out/report.txt"
echo "== parallel $seqs" | tee -a "$out/report.txt"
"$bin" loadtest --url "http://127.0.0.1:$port" --model qwen3.8-27b --requests "$seqs" --rate 0 --clients "$seqs" --mix "1:512-512" --output 128-128 --json >"$out/parallel.json"
curl -s "localhost:$port/metrics" | grep -E '^(inference_(running_seqs|kv_blocks_total|pipeline_stages|preemptions_total)|vllm:num_requests_running)' | tee -a "$out/report.txt"
jq -c '{requests,failed,running_max,decode_tok_s}' "$out/parallel.json" | tee -a "$out/report.txt"
grep -ci 'parallel requests off' "$out/server.log" | sed 's/^/parallel-off lines: /' | tee -a "$out/report.txt" || true
nvidia-smi --query-gpu=index,memory.used --format=csv | tee -a "$out/report.txt"
