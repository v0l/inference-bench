#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-/tmp/ib1007}
rounds=${ROUNDS:-3}
arms=${ARMS:-"0 400 200"}
mkdir -p "$out"
for r in $(seq "$rounds"); do
    for a in $arms; do
        ONLY=stall OUT="$out/stall-$a-r$r" SERVE_ARGS="--stall-budget-ms $a" \
            ~/.agents/bin/gpu-run 1 -- "$here/serve27b.sh" >"$out/stall-$a-r$r.log" 2>&1
        echo "stall arm $a round $r exit $?"
    done
done
