#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${CARGO_TARGET_DIR:-$here/../../../target}/release/inference-bench
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
name=${NAME:-qwen3.8-27b-nvfp4}
port=${PORT:-18598}
seqs=${SEQS:-8}
kv_gb=${KV_GB:-4}
out=${OUT:-$here/../out/seqs598}
mkdir -p "$out"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 30000 ]; then
    echo "only $free MiB free, need 30000" >&2
    exit 1
fi
"$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs "$seqs" --max-model-len 8192 \
    --kv-cache-gb "$kv_gb" --max-batched-tokens 2048 >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 180); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    kill -0 $server 2>/dev/null || { tail -5 "$out/server.log"; exit 1; }
    sleep 1
done
curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$name\",\"prompt\":\"hello\",\"max_tokens\":8}" >/dev/null
for n in ${NS:-1 2 4 8}; do
    for rep in 1 2 3; do
        "$bin" loadtest --url "http://127.0.0.1:$port" --model "$name" --json --requests "$n" --rate 0 --clients "$n" \
            --mix "1:512-512" --output 256-256 >"$out/decode$n-$rep.json"
        python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r.pop('samples',None); print(sys.argv[2], sys.argv[3], json.dumps(r))" "$out/decode$n-$rep.json" "$n" "$rep" | tee -a "$out/report.txt"
    done
done
