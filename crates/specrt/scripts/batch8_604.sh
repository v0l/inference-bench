#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-604}
bin=$CARGO_TARGET_DIR/release/inference-bench
out=${OUT:-$here/../out/batch8_604}
port=${PORT:-18604}
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
model=${MODEL:-qwen3.8-27b-nvfp4}
mkdir -p "$out"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${NEED_MIB:-36000}" -gt "$free" ]; then
    echo "only $free MiB free, need ${NEED_MIB:-36000}" >&2
    exit 1
fi
exec 9>/tmp/gpu27.lock
flock 9

serve() {
    "$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs 8 --max-model-len 8192 --kv-cache-gb 3 "$@" >"$out/$tag.server.log" 2>&1 &
    server=$!
    for _ in $(seq 180); do
        curl -sf "localhost:$port/v1/models" >/dev/null && break
        kill -0 $server 2>/dev/null || { tail -5 "$out/$tag.server.log"; return 1; }
        sleep 1
    done
}

stop() {
    kill $server 2>/dev/null || true
    wait $server 2>/dev/null || true
}

load() {
    "$bin" loadtest --url "http://127.0.0.1:$port" --model "$model" --json --requests "${1:-24}" --rate 0 --clients 8 \
        --mix "1:1000-1500" --output 300-900 >"$out/$tag.json"
}

tag=bench601
trap stop EXIT
serve
curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' -d "{\"model\":\"$model\",\"prompt\":\"hello\",\"max_tokens\":4}" >/dev/null
load 24
python3 - "$out/$tag.json" <<'P' | tee "$out/report.txt"
import json, sys
r = json.load(open(sys.argv[1]))
r.pop("samples", None)
print(json.dumps(r))
P
stop

tag=nsys
nsys profile --cuda-graph-trace=node --trace=cuda -o "$out/round8" --force-overwrite true \
    "$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs 8 --max-model-len 8192 --kv-cache-gb 3 >"$out/nsys.server.log" 2>&1 &
server=$!
for _ in $(seq 240); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    sleep 1
done
load 8
kill -INT $server
wait $server 2>/dev/null || true
trap - EXIT
nsys stats --report cuda_gpu_kern_sum --format csv "$out/round8.nsys-rep" >"$out/round8.kernels.csv"
nsys export --type sqlite --force-overwrite true -o "$out/round8.sqlite" "$out/round8.nsys-rep"
python3 "$here/nsys_top.py" "$out/round8.sqlite" 2>&1 | head -60 | tee "$out/round8.top.txt"
