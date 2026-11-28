#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
bin=${BIN:-${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-1520}/release/inference-bench}
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
out=${OUT:-$root/docs/generated/qwen27-run}
port=${PORT:-18520}
model=qwen3.8-27b
kvs=${KVS:-fp8 f16}
modes=${MODES:-mtp nomtp}
drafts=${DRAFTS:-2}
long=${LONG:-0}
py=${PY:-python3}
work=$(mktemp -d /tmp/specrt-baseline.XXXX)
unit=specrt-baseline-$port
mkdir -p "$out"

[ "${CUDA_VISIBLE_DEVICES:-}" = 1 ] || { echo "run under ~/.agents/bin/gpu-run 1 --min-free 60000 --" >&2; exit 1; }

stop_server() {
    systemctl --user stop "$unit" 2>/dev/null || true
    systemctl --user reset-failed "$unit" 2>/dev/null || true
}
trap 'stop_server; rm -rf "$work"' EXIT

kv_gb() {
    case $1 in
        fp8) echo "${KV_GB_FP8:-14}" ;;
        f16) echo "${KV_GB_F16:-28}" ;;
    esac
}

config() {
    local kv=$1 mode=$2
    local speculate=true
    [ "$mode" = nomtp ] && speculate=false
    local f16_layers=""
    [ "$kv" = fp8 ] && f16_layers="kv_f16_layers = 0"
    cat <<EOF
bind = "127.0.0.1:$port"
max_num_seqs = 32
max_model_len = 262144
kv_cache_gb = $(kv_gb "$kv")
max_batched_tokens = 16384
tool_parser = "xml_function"
reasoning_parser = "think_tags"

[speculative]
enabled = $speculate
draft_tokens = $drafts

[runtime]
prefill_chunk = 4096
kv_dtype = "$kv"
$f16_layers

[runtime.prefix_cache]
enabled = true
max_snapshots = 16

[[models]]
id = "$model"
model = "$spec"
preload = true
EOF
}

start_server() {
    local tag=$1
    stop_server
    systemd-run --user --unit="$unit" --collect -q -p MemoryMax=16G -p MemorySwapMax=0 \
        -E CUDA_VISIBLE_DEVICES=1 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E HOME="$HOME" -E PATH="$PATH" \
        -p StandardOutput="file:$work/$tag.server.log" -p StandardError="file:$work/$tag.server.log" \
        "$bin" serve --config "$work/$tag.toml"
    for _ in $(seq 900); do
        if curl -sf "localhost:$port/v1/models" | jq -e --arg m "$model" '.data[] | select(.id == $m and .loaded)' >/dev/null 2>&1; then
            return 0
        fi
        systemctl --user is-active "$unit" >/dev/null || { tail -20 "$work/$tag.server.log" >&2; return 1; }
        sleep 1
    done
    return 1
}

counter() {
    curl -sf "localhost:$port/metrics" | awk -v n="$1" '$1 == n {print $2}'
}

suite() {
    local name=$1 file=$2
    shift 2
    "$py" "$here/vllm_baseline.py" --base "http://127.0.0.1:$port" --model "$model" --ratio "${RATIO:-1.0}" \
        --label "$label" --suite "$name" --out "$out/$file" "$@" || echo "$label $name failed" >&2
}

for kv in $kvs; do
    for mode in $modes; do
        tag=$mode
        [ "$kv" = fp8 ] || tag=$kv-$mode
        label=specrt-$tag
        suites=${SUITES:-single concurrent prefix held}
        [ "$kv" = fp8 ] || suites=${SUITES:-single concurrent}
        [ "$long" = 1 ] && suites=${SUITES:-single concurrent64k}
        run=specrt-run-$tag.json
        [ "$long" = 1 ] && run=specrt-run-long-$tag.json
        config "$kv" "$mode" >"$work/$tag.toml"
        state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm --format=csv,noheader)
        started=$(date -Is)
        t0=$(date +%s)
        start_server "$tag" || { echo "$tag: server did not come up" >&2; cp "$work/$tag.server.log" "$out/specrt-$tag.server.log"; continue; }
        load_s=$(( $(date +%s) - t0 ))
        loaded_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm --format=csv,noheader)
        d0=$(counter inference_spec_drafted_tokens_total)
        a0=$(counter inference_spec_accepted_tokens_total)
        lengths=128,1024,4096,16384
        [ "$long" = 1 ] && lengths=$lengths,32768,65536,131072,196608,260000
        has() { [[ " $suites " == *" $1 "* ]]; }
        has single && suite single "specrt-single-$tag.json" --lengths "$lengths"
        has concurrent && suite concurrent "specrt-concurrent-$tag.json" --sizes 1,4,8,16,32 --conc-len 128
        has prefix && suite prefix "specrt-prefix-$tag.json" --lengths 8192,131072 --runs 3
        has held && suite held "specrt-held-$tag.json" --lengths 131072,260000
        has concurrent64k && suite concurrent "specrt-concurrent64k-$tag.json" --sizes 4 --conc-len 65536
        d1=$(counter inference_spec_drafted_tokens_total)
        a1=$(counter inference_spec_accepted_tokens_total)
        jq -n --arg label "$label" --arg kv "$kv" --arg mode "$mode" --arg started "$started" --arg finished "$(date -Is)" \
            --arg commit "$(git -C "$root" rev-parse --short HEAD)" --arg dirty "$(git -C "$root" status --porcelain --untracked-files=no | wc -l)" \
            --arg before "$state" --arg loaded "$loaded_state" --arg spec "$(basename "$spec")" \
            --argjson load_s "$load_s" --argjson drafts "$drafts" --argjson kv_gb "$(kv_gb "$kv")" \
            --argjson drafted "$(( ${d1:-0} - ${d0:-0} ))" --argjson accepted "$(( ${a1:-0} - ${a0:-0} ))" \
            --arg layout "$(grep -m1 -o 'kv_tokens=[0-9]*' "$work/$tag.server.log" || true)" --arg suites "$suites" \
            '{label: $label, kv_dtype: $kv, mode: $mode, spec: $spec, commit: $commit, dirty_files: ($dirty | tonumber), started: $started, finished: $finished,
              load_s: $load_s, kv_cache_gb: $kv_gb, kv_layout: $layout, draft_tokens: $drafts,
              gpu_before: $before, gpu_loaded: $loaded, spec_drafted: $drafted, spec_accepted: $accepted, suites: $suites}' >"$out/$run"
        grep -E "WARN|ERROR" "$work/$tag.server.log" | head -20 >"$out/specrt-$tag.server.log" || true
        stop_server
        sleep 10
    done
done
"$py" "$here/baseline_report.py" "$out" --update-doc "$root/docs/generated/qwen27-baseline.md"
