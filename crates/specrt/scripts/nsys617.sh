#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
out=${OUT:-/tmp/i617/nsys}
seqs=${SEQS:-12}
ctx=${CTX:-32768}
gen=${GEN:-256}
mkdir -p "$out"
for bin in ${BINS:-/tmp/i617/bench-base2 /tmp/i617/bench-new2}; do
    name=$(basename "$bin")-${seqs}x$ctx
    flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q -p MemoryMax=24G -p MemorySwapMax=0 \
        -E SPECRT_BENCH_SERVE=1 -E SPECRT_BENCH_RUNS=1 -E SPECRT_BENCH_PROFILE=1 -E SPECRT_BENCH_KV_GB=${KV_GB:-40} -E HOME="$HOME" -E PATH="$PATH" \
        nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi -o "$out/$name" --force-overwrite true \
        "$bin" "$spec" $((ctx - gen)) "$gen" "$seqs" >"$out/$name.log" 2>&1 || true
    grep "serve round" "$out/$name.log" | sed "s|^|$name |" | cut -c1-400
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o "$out/$name" "$out/$name.nsys-rep" >/dev/null 2>&1 || true
    python3 "$here/nsys_top.py" "$out/${name}_cuda_gpu_kern_sum.csv" 8 || true
done
