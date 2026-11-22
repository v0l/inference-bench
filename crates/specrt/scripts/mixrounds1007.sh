#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-/tmp/ib1007-mix}
rounds=${ROUNDS:-3}
arms=${ARMS:-"0 400"}
for r in $(seq "$rounds"); do
    for a in $arms; do
        OUT=$out ARM=$a TAG=a$a-r$r ~/.agents/bin/gpu-run 1 -- "$here/mix1007.sh" >"$out.$a-r$r.log" 2>&1
        echo "mix arm $a round $r exit $?"
    done
done
