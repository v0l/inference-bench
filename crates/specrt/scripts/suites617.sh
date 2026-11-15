#!/usr/bin/env bash
set -uo pipefail
deps=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-617}/release/deps
export SPECRT_REF_DIR=${SPECRT_REF_DIR:-$HOME/.cache/specrt-ref-375}
cd "$(dirname "$0")/.."
out=${OUT:-/tmp/i617/suites}
mkdir -p "$out"
for t in ${SUITES:-decode_attention mtp serving linear truncation generate kv_dtype fused head_quant prefix logits}; do
    exe=$(ls -t "$deps/$t"-* 2>/dev/null | grep -v '\.d$' | head -1)
    [ -n "$exe" ] || { echo "$t: no binary"; continue; }
    filter=()
    [ "$t" = decode_attention ] && filter=(reference)
    flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 \
        --working-directory="$PWD" -E SPECRT_REF_DIR -E HOME="$HOME" \
        "$exe" --include-ignored --skip real_27b --test-threads=1 "${filter[@]}" >"$out/$t.txt" 2>&1
    echo "$t: $(grep -E '^test result:' "$out/$t.txt" | tail -1)"
    grep -E "^test .* FAILED|panicked" "$out/$t.txt" | head -5
done
