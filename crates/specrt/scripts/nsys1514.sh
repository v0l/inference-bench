#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
crate=$(cd "$here/.." && pwd)
root=$(cd "$crate/../.." && pwd)
spec=${SPEC:-$crate/tests/fixtures/qwen3.8-27b-nvfp4.toml}
bin=${BIN:-${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-1514}/release/bench}
out=${OUT:-/tmp/nsys1514}
tag=${TAG:-base}
mkdir -p "$out"
for b in ${BATCHES:-1 8 32}; do
    name=$tag-b$b
    ~/.agents/bin/gpu-run 1 --min-free 40000 -- systemd-run --user --wait --collect -P -q -p MemoryMax=24G -p MemorySwapMax=0 \
        -E CUDA_VISIBLE_DEVICES=1 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E HOME="$HOME" -E PATH="$PATH" --working-directory="$root" \
        nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o "$out/$name" --force-overwrite true \
        "$bin" --speculative "$spec" 1024 64 "$b" >"$out/$name.log" 2>&1 || true
    nsys export --type sqlite --force-overwrite true -o "$out/$name.sqlite" "$out/$name.nsys-rep" >/dev/null 2>&1 || true
done
