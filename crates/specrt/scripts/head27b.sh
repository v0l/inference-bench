#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
B=${BENCH:-/mnt/data/cargo-target/inference-bench/release/bench}
NSYS=${NSYS:-/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys}
out=${OUT:-out/nsys363}
mkdir -p "$out"
for head in bf16 fp8 nvfp4; do
    case $head in
        bf16) spec=tests/fixtures/qwen3.8-27b-nvfp4-exact.toml ;;
        fp8) spec=tests/fixtures/qwen3.8-27b-nvfp4-head-fp8.toml ;;
        nvfp4) spec=tests/fixtures/qwen3.8-27b-nvfp4-head-nvfp4.toml ;;
    esac
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt 36000 ]; then
        echo "free $free MiB, stopping before $head"
        exit 1
    fi
    echo "== $head free $free MiB util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) $(date +%T)"
    SPECRT_BENCH_RUNS=3 nice -n 15 "$B" "$spec" 128 128
    SPECRT_BENCH_RUNS=1 nice -n 15 "$NSYS" profile --cuda-graph-trace=node -t cuda -o "$out/$head" -f true "$B" "$spec" 128 64 > /dev/null 2>&1
    "$NSYS" stats -q -r cuda_gpu_kern_sum -f csv -o "$out/$head" "$out/$head.nsys-rep" > /dev/null 2>&1
done
