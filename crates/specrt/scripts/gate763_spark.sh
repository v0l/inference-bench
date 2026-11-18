#!/usr/bin/env bash
set -uo pipefail
side=${1:?usage: gate763_spark.sh <ours|old> <loop|kl|needle> [args...]}
mode=${2:?}
shift 2
root=$HOME/ib763/ours
[ "$side" = old ] && root=$HOME/ib763/old
target=$HOME/ib-gates-target/763-$side
spec=$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4-exact.toml
out=$HOME/ib763/out
mkdir -p "$out"
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }
run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$root/crates/specrt" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" \
        --setenv=KVGATE_CORPUS="${KVGATE_CORPUS:-$root/crates}" ${KVGATE_STEPS:+--setenv=KVGATE_STEPS=$KVGATE_STEPS} \
        ${KVGATE_F16_LAYERS:+--setenv=KVGATE_F16_LAYERS=$KVGATE_F16_LAYERS} "$@"
}
run -p MemoryMax=20G cargo build --release --bin kvgate 2>&1 | tail -1
exec 9>/tmp/gpu27.lock
echo "waiting for gpu27.lock"; flock 9; echo "lock held"
run "$target/release/kvgate" "$spec" "$mode" "$@"
