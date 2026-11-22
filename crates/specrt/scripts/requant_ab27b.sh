#!/usr/bin/env bash
set -u
BN=${BN:?bench binary}
SPEC=${SPEC:-tests/fixtures/qwen3.8-27b-nvfp4.toml}
OUT=${OUT:-/tmp/rq605}
SET=${SET:?comma free list of --b-set args, e.g. "--b-set mlp_tail=nvfp4_g16"}
for b in ${BATCHES:-1 4 8}; do
    ~/.agents/bin/gpu-run 1 -- bash -c 'exec systemd-run --user --wait --collect -P -q -p MemoryMax=30G -p MemorySwapMax=0 --setenv=CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" --working-directory="$PWD" "$@"' _ \
        "$BN" --ab --rounds "${ROUNDS:-5}" $SET "$SPEC" "$b" "${NEW:-192}" > "$OUT/ab_b$b.log" 2>&1
    grep -E "verdict|decode|gpu-run|Error" "$OUT/ab_b$b.log" | tail -6
done
