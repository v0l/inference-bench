#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:?inference-bench binary}
tag=${TAG:?arm name}
out=${OUT:-/tmp/ib1518}
port=${PORT:-18338}
requests=${REQUESTS:-60}
mkdir -p "$out"
spec=$out/admit1518.toml
cp "$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml" "$spec"
printf '\n[runtime.prefix_cache]\nenabled = true\n' >>"$spec"
model=admit1518
"$bin" --log-format json serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs 8 --max-model-len 40960 --kv-cache-gb 26 \
    ${SERVE_ARGS:-} >"$out/$tag.server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 300); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    kill -0 $server 2>/dev/null || { tail -5 "$out/$tag.server.log"; exit 1; }
    sleep 1
done
curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$model\",\"prompt\":\"hello\",\"max_tokens\":4}" >/dev/null || exit 1
"$bin" loadtest --url "http://127.0.0.1:$port" --model $model --json --requests "$requests" --rate ${RATE:-0.5} --clients ${CLIENTS:-10} \
    --mix "${MIX:-0.8:1k-8k,0.2:15k-30k}" --output ${OUTPUT:-128-384} --prefixes ${PREFIXES:-8} --prefix-share ${SHARE:-0.85} --seed ${SEED:-5} >"$out/$tag.json"
curl -s "localhost:$port/metrics" | grep -E '^inference_(admission_blocked_steps_total|preemptions_total|prefix)' >"$out/$tag.metrics"
python3 "$here/admit1518.py" "$out/$tag.json" "$out/$tag.server.log" "$tag" | tee -a "$out/report.txt"
