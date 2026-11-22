#!/usr/bin/env bash
set -u
B=${B:?test binary}
OUT=${OUT:-/tmp/rq605}
for set in "$@"; do
    tag=$(echo "$set" | tr '=,' '__')
    ~/.agents/bin/gpu-run 1 -- systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --setenv=CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        --setenv=REQUANT_WORK=$OUT --setenv=REQUANT_SET="$set" --setenv=SPECRT_REF_DIR=$HOME/.cache/specrt-ref-375 \
        "$B" --include-ignored --nocapture > "$OUT/$tag.log" 2>&1
    grep -E "REQUANT_REPORT|set \"|panicked" "$OUT/$tag.log"
done
