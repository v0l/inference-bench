#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "${GPU27_LOCK_HELD:-}" ]; then
    export GPU27_LOCK_HELD=1
    exec flock /tmp/gpu27.lock "$0" "$@"
fi
root=$here/../../..
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-402}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
bin=$CARGO_TARGET_DIR/release/inference-bench
spec=$here/../tests/fixtures/qwen3.8-27b-exact.toml
port=${PORT:-18402}
seqs=${SEQS:-8}
kv_gb=${KV_GB:-1.5}
swap_gb=${SWAP_GB:-16}
lookahead=${LOOKAHEAD:-512}
max_len=${MAX_LEN:-32768}
requests=${REQUESTS:-32}
out=${OUT:-$here/../out/gate402}
mkdir -p "$out"
: >"$out/report.txt"
cd "$root"

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 36000 ]; then
    echo "only $free MiB free, need 36000" >&2
    exit 1
fi

tests_ok=1
run_test() {
    local name=$1
    echo "== $name" | tee -a "$out/report.txt"
    if systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$PWD" \
        -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E SPECRT_REF_DIR \
        cargo test --release -p inference-bench-specrt --test serving -- --ignored --exact "$name" 2>&1 | tee "$out/$name.log" | grep -q "test result: ok. 1 passed"; then
        echo "PASS $name" | tee -a "$out/report.txt"
    else
        echo "FAIL $name" | tee -a "$out/report.txt"
        tests_ok=0
    fi
}
run_test tiny_hybrid_park_then_resume_gives_bit_identical_logits
run_test tiny_mtp_park_then_resume_gives_bit_identical_logits
run_test repeated_parking_leaves_no_pinned_memory_behind

systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$PWD" -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH \
    cargo build --release -p inference-bench-cli

arm() {
    local name=$1 swap=$2 look=$3
    "$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs "$seqs" --max-model-len "$max_len" \
        --kv-cache-gb "$kv_gb" --max-batched-tokens 2048 --swap-space-gb "$swap" --admit-lookahead "$look" >"$out/$name.server.log" 2>&1 &
    local server=$!
    trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' RETURN
    for _ in $(seq 180); do
        curl -sf "localhost:$port/v1/models" >/dev/null && break
        kill -0 $server 2>/dev/null || { tail -5 "$out/$name.server.log"; return 1; }
        sleep 1
    done
    curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
        -d '{"model":"qwen3.8-27b","prompt":"hello","max_tokens":4}' >/dev/null
    "$bin" loadtest --url "http://127.0.0.1:$port" --model qwen3.8-27b --json --requests "$requests" --rate 0.5 --clients "$seqs" \
        --mix "0.6:1k-8k,0.3:8k-16k,0.1:16k-28k" --output 128-512 --seed 402 >"$out/$name.json"
    curl -s "localhost:$port/metrics" | grep -E '^inference_(preemptions_total|swap)' >"$out/$name.metrics" || true
    kill $server 2>/dev/null; wait $server 2>/dev/null || true
    trap - RETURN
    sleep 3
}
arm recompute 0 0
arm swap "$swap_gb" 0
arm lookahead 0 "$lookahead"
arm both "$swap_gb" "$lookahead"

python3 - "$out" <<'PY' | tee -a "$out/report.txt"
import json, sys
out = sys.argv[1]
print(f"{'arm':<10} {'ttft p99 s':>10} {'itl max ms':>11} {'tok/s':>8} {'preempt':>8} {'failed':>7}")
for arm in ("recompute", "swap", "lookahead", "both"):
    r = json.load(open(f"{out}/{arm}.json"))
    print(f"{arm:<10} {r['ttft_p99']:>10.2f} {r['itl_max']*1e3:>11.0f} {r['decode_tok_s']:>8.1f} {r['preemptions'] or 0:>8.0f} {r['failed']:>7}")
PY
if [ "$tests_ok" = 1 ]; then
    echo "bit-identical and leak tests: PASS" | tee -a "$out/report.txt"
else
    echo "bit-identical and leak tests: FAIL" | tee -a "$out/report.txt"
    exit 1
fi
