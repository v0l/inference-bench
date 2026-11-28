#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
crate=$(cd "$here/.." && pwd)
spec=${SPEC:-$here/../tests/fixtures/qwen3.8-27b-nvfp4.toml}
bin=${BIN:-${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-1516}/release/bench}
out=${OUT:-/tmp/nsys1516}
seqs=${SEQS:-1}
mkdir -p "$out"
for mode in ${MODES:-fixed measured}; do
    s="$out/$mode.toml"
    sed "s#^\[runtime\]#[runtime]\ngemm_select = \"$mode\"#" "$spec" > "$s"
    name=$mode-b$seqs
    ~/.agents/bin/gpu-run 1 --min-free 40000 -- systemd-run --user --wait --collect -P -q -p MemoryMax=24G -p MemorySwapMax=0 \
        -E CUDA_VISIBLE_DEVICES=1 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E HOME="$HOME" -E PATH="$PATH" --working-directory="$crate" \
        nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi -o "$out/$name" --force-overwrite true \
        "$bin" --serve --runs 1 --profile "$s" 128 256 "$seqs" >"$out/$name.log" 2>&1 || true
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o "$out/$name" "$out/$name.nsys-rep" >/dev/null 2>&1 || true
    python3 "$here/nsys_top.py" "$out/${name}_cuda_gpu_kern_sum.csv" 12 || true
done
