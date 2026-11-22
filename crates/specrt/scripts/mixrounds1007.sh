#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-/tmp/ib1007-mix}
rounds=${ROUNDS:-3}
arms=${ARMS:-"0:0 400:0"}
mkdir -p "$out"
for r in $(seq "$rounds"); do
    for arm in $arms; do
        a=${arm%%:*}
        f=${arm##*:}
        OUT=$out ARM=$a TAG=a$a-f$f-r$r SERVE_ARGS="--min-stall-chunk-tokens $f" ~/.agents/bin/gpu-run 1 -- "$here/mix1007.sh" >"$out.$a-f$f-r$r.log" 2>&1
        echo "mix arm $a floor $f round $r exit $?"
    done
done
