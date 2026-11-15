#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-617}/release/bench}
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
out=${OUT:-/tmp/i617/long}
new=${NEW:-}
mkdir -p "$out"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 40000 ]; then
    echo "only $free MiB free, need 40000" >&2
    exit 1
fi
for case in ${CASES:-1x8192 4x8192 12x8192 1x32768 4x32768 12x32768 1x131072 4x131072}; do
for bin in ${BINS:-$bin}; do
    seqs=${case%x*}
    ctx=${case#*x}
    gen=${new:-$([ "$seqs" -gt 1 ] && echo 512 || echo 256)}
    prompt=$((ctx - gen))
    log="$out/$(basename "$bin")-$case.txt"
    if [ -n "${NSYS:-}" ]; then
        flock /tmp/gpu27.lock nsys profile -t cuda --cuda-graph-trace=node -o "$out/$(basename "$bin")-$case" --force-overwrite true \
            env SPECRT_BENCH_SERVE=1 SPECRT_BENCH_RUNS=1 "$bin" "$spec" "$prompt" "$gen" "$seqs" >"$log" 2>&1
        nsys stats -r cuda_gpu_kern_sum -f csv -o "$out/$(basename "$bin")-$case" "$out/$(basename "$bin")-$case.nsys-rep" >/dev/null 2>&1 || true
    else
        SPECRT_BENCH_SERVE=1 SPECRT_BENCH_RUNS=1 SPECRT_BENCH_KV_GB=${KV_GB:-40} flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q -p MemoryMax=24G -p MemorySwapMax=0 \
            -E SPECRT_BENCH_SERVE -E SPECRT_BENCH_RUNS -E SPECRT_BENCH_KV_GB -E HOME="$HOME" "$bin" "$spec" "$prompt" "$gen" "$seqs" >"$log" 2>&1 || true
    fi
    grep "serve round" "$log" | sed "s|^|$case $(basename "$bin") |" || tail -3 "$log"
done
done
