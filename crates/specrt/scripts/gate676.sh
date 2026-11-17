#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
file=${1:?usage: gate676.sh <qwen35 model.gguf> [tokens for greedy, default 64]}
n_gen=${2:-64}
llama=${LLAMA_DIR:-$HOME/git/llama.cpp}
out=${OUT:-/tmp/gate676}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-676}
export LD_LIBRARY_PATH=$llama/build/bin
mkdir -p "$out"

if curl -sf -m3 localhost:8001/metrics | grep -q '^inference_running_seqs [1-9]'; then
    echo "the app on :8001 is serving, stop here" >&2
    exit 1
fi
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
size_mib=$(( $(stat -c %s "$file") / 1048576 + 8000 ))
if [ "$free" -lt "$size_mib" ]; then
    echo "only $free MiB free, need $size_mib" >&2
    exit 1
fi

gcc -O2 "$root/scripts/gguf_reference.c" -I"$llama/include" -I"$llama/ggml/include" -L"$llama/build/bin" -lllama -lggml -lggml-base -lm -Wl,-rpath,"$llama/build/bin" -o "$out/gguf_reference"
prompt="785 6722 315 9625 374 12095 13 9645 264 2805 32794 911 279 9396 323 279 9759 13"
"$out/gguf_reference" "$file" "$out/ref" 99 "$n_gen" $prompt

cd "$root"
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 \
    -p "WorkingDirectory=$root" -E CARGO_TARGET_DIR -E PATH -E HOME \
    -E SPECRT_GGUF_FILE="$file" -E SPECRT_GGUF_REF="$out/ref" \
    cargo test --release --test gguf_qwen35_gpu -- --ignored --nocapture --test-threads=1
