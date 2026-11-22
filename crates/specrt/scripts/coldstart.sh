#!/usr/bin/env bash
# usage: coldstart.sh <bench binary> <spec.toml> [prompt_len] [new_tokens]
# Runs `bench --cold` in four states: both caches warm, specrt PTX cache cold,
# driver cache cold, both cold. Run under ~/.agents/bin/gpu-run 1 --.
set -eu
bench=$1
spec=$2
prompt=${3:-128}
new=${4:-16}
work=$(mktemp -d /tmp/coldstart.XXXX)
trap 'rm -rf "$work"' EXIT
warm_dir=$work/ptx-warm
mkdir -p "$warm_dir"

with_cache() {
    local dir=$1 out=$2
    if grep -q '^\[runtime\]' "$spec"; then
        sed "s|^\[runtime\]|[runtime]\ncache_dir = \"$dir\"|" "$spec" > "$out"
    else
        { cat "$spec"; printf '\n[runtime]\ncache_dir = "%s"\n' "$dir"; } > "$out"
    fi
}

run() {
    local label=$1 ptx=$2 driver=$3
    local s=$work/spec-$label.toml
    with_cache "$ptx" "$s"
    echo "== $label"
    if [ -n "$driver" ]; then
        CUDA_CACHE_PATH=$driver "$bench" --cold --kv-gb 2 "$s" "$prompt" "$new" 2>&1 | grep -E "cold_start|ttft_first|graph_capture|rror"
    else
        "$bench" --cold --kv-gb 2 "$s" "$prompt" "$new" 2>&1 | grep -E "cold_start|ttft_first|graph_capture|rror"
    fi
}

run prime "$warm_dir" ""
run warm "$warm_dir" ""
run ptx-cold "$work/ptx-fresh1" ""
run driver-cold "$warm_dir" "$work/driver1"
run both-cold "$work/ptx-fresh2" "$work/driver2"
