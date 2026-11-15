#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${CARGO_TARGET_DIR:-$here/../../../target}/release/inference-bench
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
model=${MODEL:-qwen3.8-27b-nvfp4}
port=${PORT:-18601}
seqs=${SEQS:-8}
kv_gb=${KV_GB:-3}
out=${OUT:-$here/../out/bench601}
requests=${REQUESTS:-24}
mkdir -p "$out"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${NEED_MIB:-36000}" -gt "$free" ]; then
    echo "only $free MiB free, need ${NEED_MIB:-36000}" >&2
    exit 1
fi
run_config() {
    local name=$1; shift
    RUST_LOG=${RUST_LOG:-inference_bench_specrt=debug,info} "$bin" serve --bind 127.0.0.1:$port --model "$spec" \
        --max-num-seqs "$seqs" --max-model-len 8192 --kv-cache-gb "$kv_gb" "$@" >"$out/$name.server.log" 2>&1 &
    local server=$!
    trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' RETURN
    for _ in $(seq 180); do
        curl -sf "localhost:$port/v1/models" >/dev/null && break
        kill -0 $server 2>/dev/null || { tail -5 "$out/$name.server.log"; return 1; }
        sleep 1
    done
    curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$model\",\"prompt\":\"hello\",\"max_tokens\":4}" >/dev/null
    "$bin" loadtest --url "http://127.0.0.1:$port" --model "$model" --json --requests "$requests" --rate 0 --clients "$seqs" \
        --mix "1:1000-1500" --output 300-900 >"$out/$name.json"
    python3 - "$out/$name.json" "$out/$name.server.log" "$name" <<'P' | tee -a "$out/report.txt"
import json, re, sys
r = json.load(open(sys.argv[1]))
r.pop("samples", None)
steps = [l for l in open(sys.argv[2]) if " step" in l and "rows=" in l]
mtp = sum(1 for l in steps if "mtp=true" in l)
pf = sum(1 for l in steps if not re.search(r"prefill_tokens=0\b", l))
print(sys.argv[3], json.dumps(r), f"steps={len(steps)} mtp_steps={mtp} prefill_steps={pf}")
P
    kill $server 2>/dev/null; wait $server 2>/dev/null || true
    trap - RETURN
}
if [ -n "${BASELINE:-}" ]; then
    run_config "${TAG:-baseline}"
    exit 0
fi
for itl in ${ITLS:-50 100 200 400}; do
    run_config "${TAG:-}itl$itl" --itl-target-ms "$itl"
done
run_config "${TAG:-}whole4k" --itl-target-ms 100 --whole-prompt-tokens 4096
