#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${CARGO_TARGET_DIR:-$here/../../../target}/release/inference-bench
spec=$here/../tests/fixtures/qwen3.8-27b-exact.toml
port=${PORT:-18336}
seqs=${SEQS:-8}
kv_gb=${KV_GB:-3}
max_len=${MAX_LEN:-32768}
out=${OUT:-$here/../out/serve27b}
mkdir -p "$out"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 36000 ]; then
    echo "only $free MiB free, need 36000" >&2
    exit 1
fi
start=$(date '+%Y-%m-%d %H:%M:%S')
"$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs "$seqs" --max-model-len "$max_len" \
    --kv-cache-gb "$kv_gb" --max-batched-tokens 2048 >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 120); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    kill -0 $server 2>/dev/null || { tail -5 "$out/server.log"; exit 1; }
    sleep 1
done
curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
    -d '{"model":"qwen3.8-27b","prompt":"hello","max_tokens":4}' >/dev/null
run() {
    local name=$1; shift
    echo "== $name" | tee -a "$out/report.txt"
    "$bin" loadtest --url "http://127.0.0.1:$port" --model qwen3.8-27b --json "$@" >"$out/$name.json"
    python3 -c "import json,sys; r=json.load(open(sys.argv[1])); [r.pop(k) for k in ('samples',)]; print(json.dumps(r))" "$out/$name.json" | tee -a "$out/report.txt"
    curl -s "localhost:$port/metrics" | grep -E '^inference_(preemptions_total|kv_blocks_total|batch_size_(sum|count)|prefill_tokens_per_step_(sum|count))' | tee -a "$out/report.txt"
}
only=${ONLY:-all}
for n in 1 4 $seqs; do
    [ "$only" = all ] || break
    run "decode$n" --requests "$n" --rate 0 --clients "$n" --mix "1:512-512" --output 256-256
done
[ "$only" = all ] && run "mixed" --requests 32 --rate 0.3 --clients "$seqs" --mix "0.6:1k-8k,0.3:8k-16k,0.1:16k-28k" --output 128-512
d=$((seqs - 1))
"$bin" loadtest --url "http://127.0.0.1:$port" --model qwen3.8-27b --json --requests $d --rate 0 --clients $d \
    --mix "1:256-256" --output 700-700 >"$out/stall.json" &
decoders=$!
sleep 8
echo "== stall" | tee -a "$out/report.txt"
"$bin" loadtest --url "http://127.0.0.1:$port" --model qwen3.8-27b --json --requests 1 --rate 0 --clients 1 \
    --mix "1:28000-28000" --output 8-8 --seed 9 >"$out/stall-long.json"
wait $decoders
for f in stall stall-long; do
    python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r.pop('samples'); print(json.dumps(r))" "$out/$f.json" | tee -a "$out/report.txt"
done
errors=$(journalctl -u local-llm.service --since "$start" --no-pager -q | grep -ciE " error|oom|out of memory" || true)
echo "local-llm errors during the run: $errors" | tee -a "$out/report.txt"
