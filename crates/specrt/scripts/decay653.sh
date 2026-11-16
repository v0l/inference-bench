#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-/mnt/data/cargo-target/inference-bench-653/release/inference-bench}
spec=${SPEC:-$here/../out/serve653/spec.toml}
out=${OUT:-$here/../out/serve653/run}
secs=${SECS:-900}
port=${PORT:-18653}
kv_gb=${KV_GB:-6}
mkdir -p "$out"
exec 9>/tmp/gpu27.lock
flock 9
"$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs 8 --max-model-len 16384 --kv-cache-gb "$kv_gb" --max-batched-tokens 2048 >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 180); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    kill -0 $server 2>/dev/null || { tail -5 "$out/server.log"; exit 1; }
    sleep 1
done
model=$(curl -s localhost:$port/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
"$bin" loadtest --url "http://127.0.0.1:$port" --model "$model" --json --requests 100000 --rate 3 --clients 6 --mix "1:700-1000" --output 16-64 --seed 5 >"$out/classify.json" 2>/dev/null &
c1=$!
"$bin" loadtest --url "http://127.0.0.1:$port" --model "$model" --json --requests 100000 --rate 1 --clients 2 --mix "1:700-1000" --output 2000-5000 --seed 6 >"$out/agent.json" 2>/dev/null &
c2=$!
trap 'kill $c1 $c2 $server 2>/dev/null; wait 2>/dev/null || true' EXIT
t0=$(date +%s)
echo "t_s gen_tokens steps step_sum prefix_blocks kv_used kv_total running waiting preempt prefill_sum batch_sum batch_cnt" >"$out/series.txt"
while [ $(( $(date +%s) - t0 )) -lt "$secs" ]; do
    m=$(curl -s localhost:$port/metrics)
    g() { echo "$m" | awk -v k="$1" '$1==k {print $2; exit}'; }
    echo "$(( $(date +%s) - t0 )) $(g inference_decode_tokens_total) $(g inference_step_seconds_count) $(g inference_step_seconds_sum) $(g inference_prefix_cache_blocks) $(g inference_kv_blocks_used) $(g inference_kv_blocks_total) $(g inference_running_seqs) $(g inference_waiting_seqs) $(g inference_preemptions_total) $(g inference_prefill_tokens_per_step_sum) $(g inference_batch_size_sum) $(g inference_batch_size_count)" >>"$out/series.txt"
    sleep 15
done
