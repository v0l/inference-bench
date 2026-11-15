#!/usr/bin/env bash
set -uo pipefail
out=${1:?usage: kvgate27b.sh <outfile> <bin dir> [part: kl|needle|speed|mtp|all]}
bin=${2:?usage: kvgate27b.sh <outfile> <bin dir> [part]}
part=${3:-all}
spec=$(cd "$(dirname "$0")" && pwd)/../tests/fixtures/qwen3.8-27b-nvfp4-exact.toml
tmp=$(mktemp -d)
cp "$spec" "$tmp/f16.toml"
{ cat "$spec"; printf '\n[runtime]\nkv_dtype = "fp8"\n'; } >"$tmp/fp8.toml"
run() { systemd-run --user --wait --collect -P -p MemoryMax=16G -p MemorySwapMax=0 "$@"; }
want() { [ "$part" = all ] || [ "$part" = "$1" ]; }
{
    want kl && { run "$bin/kvgate" "$spec" kl 8192 20; run "$bin/kvgate" "$spec" kl 32768 10; }
    want needle && { run "$bin/kvgate" "$spec" needle 32768 0.1,0.5,0.9; run "$bin/kvgate" "$spec" needle 131072 0.5; }
    want speed && { run "$bin/kvgate" "$spec" batch 8192 8 48; run "$bin/kvgate" "$spec" batch 32768 8 48; }
    want mtp && for d in f16 fp8; do echo "mtp kv=$d"; run "$bin/bench" --speculative "$tmp/$d.toml" 4096 128 1 | grep -v first; done
} >"$out" 2>&1
rm -rf "$tmp"
