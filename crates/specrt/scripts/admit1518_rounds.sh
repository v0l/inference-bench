#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
rounds=${ROUNDS:-3}
for r in $(seq "$rounds"); do
    for arm in master branch; do
        if [ "$arm" = master ]; then b=$MASTER_BIN; else b=$BRANCH_BIN; fi
        ~/.agents/bin/gpu-run 1 --min-free 40000 -- systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 \
            -E CUDA_VISIBLE_DEVICES=1 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E HOME="$HOME" -E PATH="$PATH" -E BIN="$b" -E TAG="$arm$r" \
            -E OUT="${OUT:-/tmp/ib1518}" -E REQUESTS="${REQUESTS:-60}" -E RATE="${RATE:-0.5}" -E CLIENTS="${CLIENTS:-10}" -E SEED="${SEED:-5}" \
            -E MIX="${MIX:-0.8:1k-8k,0.2:15k-30k}" -E OUTPUT="${OUTPUT:-128-384}" -E PREFIXES="${PREFIXES:-8}" -E SHARE="${SHARE:-0.85}" \
            --working-directory="$PWD" "$here/admit1518.sh"
    done
done
