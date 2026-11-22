#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-/tmp/ib1007}
rounds=${ROUNDS:-3}
arms=${ARMS:-"0:0 400:0 200:0"}
mkdir -p "$out"
for r in $(seq "$rounds"); do
    for arm in $arms; do
        a=${arm%%:*}
        f=${arm##*:}
        ONLY=stall OUT="$out/stall-$a-f$f-r$r" SERVE_ARGS="--stall-budget-ms $a --min-stall-chunk-tokens $f" \
            ~/.agents/bin/gpu-run 1 -- "$here/serve27b.sh" >"$out/stall-$a-f$f-r$r.log" 2>&1
        echo "stall arm $a floor $f round $r exit $?"
    done
done
