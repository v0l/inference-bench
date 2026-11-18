#!/usr/bin/env bash
set -uo pipefail
side=${1:?usage: gate763_serve_spark.sh <ours|old> <f16|fp8|fp8:N> <seeds> <new_tokens> [prefix=on|off] [mtp=on|off] [warm=120000] [len=150000]}
arm=${2:?}
seeds=${3:-2}
new=${4:-1024}
prefix=${5:-on}
mtp=${6:-on}
warm=${7:-120000}
len=${8:-150000}
root=$HOME/ib763/$side
target=$HOME/ib-gates-target/763-$side
out=$HOME/ib763/out
port=${PORT:-18763}
mkdir -p "$out"
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }
run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$root/crates/specrt" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" --setenv=KVGATE_CORPUS="$root/crates" "$@"
}
base=$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4-exact.toml
tag=$side-${arm//:/_}-p$prefix-m$mtp
spec=$out/$tag.toml
cp "$base" "$spec"
case $arm in
    f16) ;;
    fp8) printf '\n[runtime]\nkv_dtype = "fp8"\n' >>"$spec" ;;
    fp8:*) printf '\n[runtime]\nkv_dtype = "fp8"\nkv_f16_layers = %s\n' "${arm#fp8:}" >>"$spec" ;;
esac
if [ "$prefix" = on ]; then
    if grep -q '^\[runtime\]' "$spec"; then printf '[runtime.prefix_cache]\nenabled = true\n' >>"$spec"; else printf '\n[runtime]\n[runtime.prefix_cache]\nenabled = true\n' >>"$spec"; fi
fi
[ "$mtp" = off ] && sed -i 's/^speculate = true/speculate = false/' "$spec"
run -p MemoryMax=20G cargo build --release --bin kvgate 2>&1 | tail -1
run -p MemoryMax=20G cargo build --release -p inference-bench-cli --bin inference-bench 2>&1 | tail -1
if [ ! -f "$out/tx$len.txt" ] || [ ! -f "$out/tx$warm.txt" ]; then
    run "$target/release/kvgate" "$base" transcript "$len" "$out/tx$len.txt"
    run "$target/release/kvgate" "$base" transcript "$warm" "$out/tx$warm.txt"
fi
exec 9>/tmp/gpu27.lock
echo "waiting for gpu27.lock"; flock 9; echo "lock held"
log=$out/$tag.server.log
systemd-run --user --collect -q --unit=ib763-server -p MemoryMax=60G -p MemorySwapMax=0 --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
    "$target/release/inference-bench" serve --bind 127.0.0.1:$port --model "$spec" --max-num-seqs 2 --max-model-len $((len + new + 64)) \
    --kv-cache-gb 12 --max-batched-tokens 8192
trap 'systemctl --user stop ib763-server 2>/dev/null' EXIT
for _ in $(seq 300); do
    curl -sf "localhost:$port/v1/models" >/dev/null && break
    sleep 2
done
id=$(curl -s localhost:$port/v1/models | jq -r '.data[0].id')
echo "model $id tag $tag"
ask() {
    local file=$1 seed=$2 max=$3 dest=$4
    jq -n --rawfile p "$file" --arg m "$id" --argjson s "$seed" --argjson n "$max" \
        '{model:$m,prompt:$p,max_tokens:$n,temperature:1.0,top_p:0.95,top_k:20,seed:$s}' >"$out/req.json"
    curl -s --max-time 3000 "localhost:$port/v1/completions" -H 'content-type: application/json' -d @"$out/req.json" | jq -r '.choices[0].text' >"$dest"
}
if [ "$prefix" = on ]; then
    echo "warm $warm"
    ask "$out/tx$warm.txt" 1 8 /dev/null
    curl -s localhost:$port/metrics | grep -E "prefix" | head -5
fi
for seed in $(seq 1 "$seeds"); do
    dest=$out/$tag.seed$seed.txt
    ask "$out/tx$len.txt" "$seed" "$new" "$dest"
    echo "-- seed $seed"
    run "$target/release/kvgate" "$base" detect "$dest"
    head -c 200 "$dest" | tr '\n' ' '; echo
    tail -c 200 "$dest" | tr '\n' ' '; echo
    curl -s localhost:$port/metrics | grep -E "prefix.*hit|cached" | head -4
done
