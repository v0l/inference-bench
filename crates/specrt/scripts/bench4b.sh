#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-635}
model=${MODEL_DIR:-$HOME/.cache/specrt-models/Qwen3.5-4B-NVFP4}
out=${OUT:-$here/../out/bench4b}
runs=${SPECRT_BENCH_RUNS:-3}
mkdir -p "$out"
if [ ! -f "$model/model.safetensors.index.json" ]; then
    flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 \
        "$HOME/.venv-vllm/bin/python" "$here/quantize4b.py" "$model"
fi
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${NEED_MIB:-12000}" -gt "$free" ]; then
    echo "only $free MiB free, need ${NEED_MIB:-12000}" >&2
    exit 1
fi
weight=$out/qwen3.5-4b-nvfp4-exact.toml
act=$out/qwen3.5-4b-nvfp4.toml
sed "s#@CHECKPOINT@#$model#" "$here/qwen3.5-4b-nvfp4.toml.in" | sed '/^act_quant = /d' > "$weight"
sed "s#@CHECKPOINT@#$model#" "$here/qwen3.5-4b-nvfp4.toml.in" > "$act"
CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4} systemd-run --user --wait --collect -P -p MemoryMax=16G -p "WorkingDirectory=$(dirname "$here")" \
    -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E HOME cargo build --release -p inference-bench-specrt --bin bench
bin=$CARGO_TARGET_DIR/release/bench
exec 9>/tmp/gpu27.lock
flock 9
log=$out/bench4b.log
: > "$log"
{
    echo "== $(date '+%F %T') $(git -C "$here" rev-parse --short HEAD) free ${free} MiB"
    for spec in "$weight" "$act"; do
        echo "== single $(basename "$spec")"
        SPECRT_BENCH_RUNS=$runs "$bin" "$spec" 128,1024,4096 128
    done
    for batch in 4 8 16; do
        echo "== batch $batch act"
        SPECRT_BENCH_RUNS=$runs "$bin" "$act" 128 128 "$batch"
    done
    echo "== speculative act"
    SPECRT_BENCH_RUNS=$runs "$bin" --speculative "$act" 1024 128 1
} 2>&1 | tee -a "$log"
