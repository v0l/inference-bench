#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
bin=${BIN:?BIN=<prefixbench>}
probe=${PROBE:-}
out=${OUT:-$here/out/p555}
lengths=${LENGTHS:-4096,32768,131072}
rounds=${ROUNDS:-3}
specs=${SPECS:-qwen3.8-27b-nvfp4 qwen3.8-27b}
alt=${ALT_BIN:-}
mkdir -p "$out"
run() { systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$here" "$@"; }
summary() { grep -E "SUMMARY.*cold|rror|panick"; }
for r in $(seq "$rounds"); do
    for s in $specs; do
        echo "round $r $s master"
        run "$bin" "tests/fixtures/$s.toml" long on 0 12 "$lengths" 2>&1 | summary
        if [ -n "$alt" ]; then
            echo "round $r $s branch"
            run "$alt" "tests/fixtures/$s.toml" long on 0 12 "$lengths" 2>&1 | summary
        fi
    done
done
if [ -n "${NSYS:-1}" ] && [ "${NSYS:-1}" != 0 ]; then
    for len in ${NSYS_LENGTHS:-4096 32768}; do
        for b in "$bin" $alt; do
            tag=$(basename "$(dirname "$(dirname "$b")")")-$len
            run nsys profile --trace=cuda -o "$out/$tag" -f true --stats=false "$b" tests/fixtures/qwen3.8-27b-nvfp4.toml long off 0 12 "$len" 2>&1 | summary
            nsys export --type sqlite -o "$out/$tag.sqlite" -f true "$out/$tag.nsys-rep" >/dev/null 2>&1
        done
    done
fi
if [ -n "$probe" ]; then
    SPECRT_GEMM_TOKENS=2048,4096 run -E SPECRT_GEMM_TOKENS "$probe" prefill_gemm_fp4_in_model_probe --ignored --nocapture 2>&1 | grep -E "f4a|test result"
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
