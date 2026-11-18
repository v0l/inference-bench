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
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-exact.toml}
model=${MODEL:-qwen3.8-27b-exact}
port=${PORT:-18402}
seqs=${SEQS:-8}
swap_gb=${SWAP_GB:-16}
lookahead=${LOOKAHEAD:-512}
max_len=${MAX_LEN:-32768}
requests=${REQUESTS:-32}
probe_gb=${PROBE_GB:-1.5}
pressure_requests=${PRESSURE_REQUESTS:-16}
pressure_output=${PRESSURE_OUTPUT:-2048-3072}
pressure_prompt=${PRESSURE_PROMPT:-4k-6k}
pressure_arms=${PRESSURE_ARMS:-recompute swap}
out=${OUT:-$here/../out/gate402}
mkdir -p "$out"
[ "${SKIP_TESTS:-0}" = 1 ] || : >"$out/report.txt"
cd "$root"

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ "$free" =~ ^[0-9]+$ ]]; then
    [ "$free" -ge 36000 ] || { echo "only $free MiB free, need 36000" >&2; exit 1; }
else
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 40 ] || { echo "only $avail GB MemAvailable, need 40" >&2; exit 1; }
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
if [ "${SKIP_TESTS:-0}" != 1 ]; then
    echo "== serving suite" | tee -a "$out/report.txt"
    if systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$PWD" \
        -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E SPECRT_REF_DIR \
        cargo test --release -p inference-bench-specrt --test serving -- --include-ignored --test-threads=1 2>&1 | tee "$out/serving-suite.log" | grep -q "test result: ok\."; then
        echo "PASS serving suite" | tee -a "$out/report.txt"
    else
        echo "FAIL serving suite" | tee -a "$out/report.txt"
        tests_ok=0
    fi
    run_test tiny_hybrid_park_then_resume_gives_bit_identical_logits
    run_test tiny_mtp_park_then_resume_gives_bit_identical_logits
    run_test repeated_parking_leaves_no_pinned_memory_behind
fi

systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$PWD" -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH \
    cargo build --release -p inference-bench-cli

port_free() {
    for _ in $(seq 90); do
        curl -s -m 2 "localhost:$port/v1/models" >/dev/null 2>&1 || return 0
        sleep 1
    done
    echo "port $port still answers" >&2
    return 1
}

stop_server() {
    kill "$1" 2>/dev/null || true
    for _ in $(seq 40); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
    kill -9 "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
    port_free
}

serve() {
    exec "$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs "$seqs" --max-model-len "$max_len" \
        --max-batched-tokens 2048 "$@"
}

if [ -n "${KV_GB:-}" ]; then
    kv_gb=$KV_GB
else
    "$bin" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs "$seqs" --max-model-len "$max_len" --max-batched-tokens 2048 --kv-cache-gb "$probe_gb" >"$out/probe.log" 2>&1 &
    probe=$!
    for _ in $(seq 60); do
        curl -sf "localhost:$port/v1/models" >/dev/null && break
        sleep 1
    done
    curl -s -m 300 "localhost:$port/v1/completions" -H 'content-type: application/json' \
        -d '{"model":"'"$model"'","prompt":"hi","max_tokens":2}' >"$out/probe.response" || true
    stop_server $probe
    held=$(grep -oE 'KV cache holds [0-9]+ tokens' "$out/probe.response" | grep -oE '[0-9]+' | head -1 || true)
    if [ -z "$held" ]; then
        echo "could not read the KV tokens per GB from $out/probe.response; set KV_GB" >&2
        exit 1
    fi
    kv_gb=$(python3 -c "import sys; held=int(sys.argv[1]); probe=float(sys.argv[2]); max_len=int(sys.argv[3]); print(round(probe*(max_len+4096)/held,3))" "$held" "$probe_gb" "$max_len")
fi
echo "KV budget $kv_gb GB (just above one $max_len token sequence plus 4096; eight journal-mix rows need about 70k tokens)" | tee -a "$out/report.txt"

scrape() {
    curl -s "localhost:$port/metrics" | grep -E '^(inference_preemptions_total|specrt_)' >"$out/$1.metrics" || true
}

arm() {
    port_free
    local name=$1 swap=$2 look=$3
    serve --kv-cache-gb "$kv_gb" --swap-space-gb "$swap" --admit-lookahead "$look" >"$out/$name.server.log" 2>&1 &
    local server=$!
    trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' RETURN
    for _ in $(seq 180); do
        kill -0 $server 2>/dev/null || { tail -5 "$out/$name.server.log"; return 1; }
        curl -sf "localhost:$port/v1/models" >/dev/null && break
        sleep 1
    done
    curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
        -d '{"model":"'"$model"'","prompt":"hello","max_tokens":4}' >/dev/null
    scrape "$name.journal.before"
    "$bin" loadtest --url "http://127.0.0.1:$port" --model "$model" --json --requests "$requests" --rate 0.5 --clients "$seqs" \
        --mix "0.6:1k-8k,0.3:8k-16k,0.1:16k-28k" --output 128-512 --seed 402 >"$out/$name.journal.json"
    scrape "$name.journal.after"
    case " $pressure_arms " in
    *" $name "*)
        scrape "$name.pressure.before"
        "$bin" loadtest --url "http://127.0.0.1:$port" --model "$model" --json --requests "$pressure_requests" --rate 0 --clients "$seqs" \
            --mix "1:$pressure_prompt" --output "$pressure_output" --seed 403 >"$out/$name.pressure.json"
        scrape "$name.pressure.after"
        ;;
    esac
    stop_server $server
    trap - RETURN
}
arm recompute 0 0
arm swap "$swap_gb" 0
arm lookahead 0 "$lookahead"
arm both "$swap_gb" "$lookahead"

python3 - "$out" <<'PY' | tee -a "$out/report.txt"
import json, sys, re
out = sys.argv[1]
def metrics(name):
    m = {}
    try:
        for line in open(f"{out}/{name}.metrics"):
            k, v = line.rsplit(" ", 1)
            m[k] = float(v)
    except FileNotFoundError:
        pass
    return m
for mix in ("journal", "pressure"):
    print(f"\n{mix} mix")
    print(f"{'arm':<10} {'ttft p99 s':>10} {'itl max ms':>11} {'tok/s':>7} {'swap':>5} {'recomp':>6} {'reprefill tok':>13} {'swapped GB':>10} {'swap s':>7} {'failed':>6}")
    for arm in ("recompute", "swap", "lookahead", "both"):
        try:
            r = json.load(open(f"{out}/{arm}.{mix}.json"))
        except FileNotFoundError:
            continue
        b, a = metrics(f"{arm}.{mix}.before"), metrics(f"{arm}.{mix}.after")
        d = lambda k: a.get(k, 0.0) - b.get(k, 0.0)
        print(f"{arm:<10} {r['ttft_p99']:>10.2f} {r['itl_max']*1e3:>11.0f} {r['decode_tok_s']:>7.1f} {d('specrt_preemptions_total{kind=\"swap\"}'):>5.0f} {d('specrt_preemptions_total{kind=\"recompute\"}'):>6.0f} {d('specrt_reprefilled_tokens_total'):>13.0f} {d('specrt_swapped_bytes_total')/2**30:>10.2f} {d('specrt_swap_seconds'):>7.1f} {r['failed']:>6}")
PY
if [ "$tests_ok" = 1 ]; then
    echo "bit-identical and leak tests: PASS" | tee -a "$out/report.txt"
else
    echo "bit-identical and leak tests: FAIL" | tee -a "$out/report.txt"
    exit 1
fi
min_pressure=${MIN_PRESSURE_PREEMPTIONS:-20}
for arm in recompute swap; do
    n=$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1]))['preemptions'] or 0))" "$out/$arm.pressure.json")
    if [ "$n" -lt "$min_pressure" ]; then
        echo "FAIL arm $arm saw $n preemptions on the pressure mix, need $min_pressure: the table is void" | tee -a "$out/report.txt"
        exit 1
    fi
done
n=$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1]))['preemptions'] or 0))" "$out/recompute.journal.json")
echo "journal mix recompute preemptions: $n" | tee -a "$out/report.txt"
