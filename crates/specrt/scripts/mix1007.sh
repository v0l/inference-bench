#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${CARGO_TARGET_DIR:-$here/../../../target}/release/inference-bench
out=${OUT:-/tmp/ib1007-mix}
port=${PORT:-18337}
arm=${ARM:?stall budget ms}
tag=${TAG:-arm$arm}
mkdir -p "$out"
spec=$out/$tag.toml
python3 - "$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml" "$spec" <<'P'
import sys
t = open(sys.argv[1]).read()
t = t.replace("prefill_chunk = 4096", 'prefill_chunk = 4096\nkv_dtype = "fp8"', 1)
open(sys.argv[2], "w").write(t)
P
model=$tag
"$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs 26 --max-model-len 32768 --kv-cache-gb 22.5 \
    --max-batched-tokens 16384 --max-long-in-flight 4 --stall-budget-ms "$arm" ${SERVE_ARGS:-} >"$out/$tag.server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 240); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    kill -0 $server 2>/dev/null || { tail -5 "$out/$tag.server.log"; exit 1; }
    sleep 1
done
curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$model\",\"prompt\":\"hello\",\"max_tokens\":4}" >/dev/null
run() {
    local name=$1; shift
    "$bin" loadtest --url "http://127.0.0.1:$port" --model $model --json "$@" >"$out/$tag.$name.json"
    python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r.pop('samples',None); print(sys.argv[2], json.dumps(r))" "$out/$tag.$name.json" "$tag.$name" | tee -a "$out/report.txt"
}
run long --requests 40 --rate 1 --clients 8 --mix "1:15000-18000" --output 256-512
run chat --requests 80 --rate 4 --clients 16 --mix "1:500-1500" --output 256-512
run batch8 --requests 48 --rate 0 --clients 8 --mix "1:512-512" --output 256-256
