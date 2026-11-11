#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base=${BASE:?BASE=<baseline prefixbench>}
branch=${BRANCH:?BRANCH=<branch prefixbench>}
base_spec=${BASE_SPEC:-$here/../tests/fixtures/qwen3.8-27b.toml}
branch_spec=${BRANCH_SPEC:-$here/../tests/fixtures/qwen3.8-27b-act.toml}
lengths=${LENGTHS:-4096,32768,65536}
rounds=${ROUNDS:-3}
kv_gb=${KV_GB:-6}
run() { flock /tmp/gpu27.lock systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 "$@"; }
for r in $(seq "$rounds"); do
    echo "round $r base"
    run "$base" "$base_spec" long on 0 "$kv_gb" "$lengths" 2>&1 | grep -v "^\s*$" | grep -iv warn
    echo "round $r branch"
    run "$branch" "$branch_spec" long on 0 "$kv_gb" "$lengths" 2>&1 | grep -v "^\s*$" | grep -iv warn
done
