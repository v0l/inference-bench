#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
bin=${BIN:?BIN=<prefixbench>}
spec=${SPEC:-$here/tests/fixtures/qwen3.8-27b.toml}
len=${LEN:-16384}
out=${OUT:-/tmp/flashprof}
mkdir -p "$out"
flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$here" \
    nsys profile -o "$out/p" -f true --stats=false "$bin" "$spec" long on 0 6 "$len" 2>&1 | grep -E "SUMMARY.*cold|rror|Failed|not found"
nsys stats --force-export=true -r cuda_gpu_kern_sum --format csv "$out/p.nsys-rep" 2>/dev/null | python3 -c "
import sys,csv
rows=list(csv.reader(l for l in sys.stdin if ',' in l))
tot=sum(float(r[1]) for r in rows[1:])
for r in rows[1:int(sys.argv[1])+1]: print(f'{r[0]:>5}% total {float(r[1])/1e6:9.1f} ms n {r[2]:>5} med {float(r[4])/1e6:8.3f} min {float(r[5])/1e6:8.3f} ms  {r[-1][:50]}')
" "${TOP:-10}"
