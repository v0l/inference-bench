#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/623
master=$HOME/ib-gates/623-master
vis=$HOME/ib-gates/vis498
out=$HOME/ib-gates/out623
ours=$HOME/ib-gates-target/623
base=$HOME/ib-gates-target/623-master
hf=$HOME/.cache/huggingface/hub
q8b=$(ls $hf/models--unsloth--Qwen3-8B-GGUF/snapshots/*/Qwen3-8B-Q4_K_M.gguf)
spec27=tests/fixtures/qwen3.8-27b-nvfp4.toml
export PATH=$HOME/ib-gates/bin/ff7/bin:$PATH
mkdir -p "$out"
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }
run() {
    local dir=$1 target=$2; shift 2
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$dir" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" "$@"
}
quiet() {
    for _ in $(seq 1 90); do
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        [ "${u:-100}" -le 3 ] && return 0
        sleep 20
    done
    echo "GPU not quiet (util ${u}%)"
}
bench() {
    local side=$1 d=$root tg=$ours
    [ $side = master ] && { d=$master; tg=$base; }
    shift
    run "$d/crates/specrt" "$tg" "$tg/release/bench" --runs 3 "$@"
}
benchours() {
    run "$root/crates/specrt" "$ours" "$ours/release/bench" --runs 3 "$@"
}
cell() {
    local side=$1; shift
    quiet
    echo "util_before=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)"
    if [ $side = master ]; then bench master "$@"; else benchours "$@"; fi
}
echo "== build"
for side in ours master; do
    d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
    run "$d/crates/specrt" "$tg" -p MemoryMax=20G cargo build --release --bin bench 2>&1 | tail -1
    run "$d/crates/specrt" "$tg" -p MemoryMax=20G cargo test --release --no-run --test vision_lm 2>&1 | tail -1
done
exec 9>/tmp/gpu27.lock
echo "waiting for gpu27.lock"; flock 9; echo "lock held $(date +%T)"
if [ -z "${SKIP_VISION:-}" ]; then
    for side in master ours; do
        d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
        echo "== vision_lm $side"
        quiet
        run "$d/crates/specrt" "$tg" --setenv=SPECRT_VISION_REF_DIR="$vis" \
            cargo test --release --test vision_lm -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "^test |test result|panicked|skipped" | tee "$out/$side.vision_lm.txt"
    done
fi
echo "== speed ABBA, 3 rounds"
for r in ${ROUNDS:-1 2 3}; do
    for side in master ours ours master; do
        echo "-- $side round $r 27B nvfp4 1k prefill + decode b1"
        cell $side "$spec27" 1024 128 2>&1 | grep -E "util_before|prompt=|rror|panick"
        echo "-- $side round $r 27B nvfp4 decode b8"
        cell $side "$spec27" 128 128 8 2>&1 | grep -E "util_before|batch=|rror|panick"
        echo "-- $side round $r gguf Qwen3-8B Q4_K_M"
        cell $side "$q8b" 1024 128 2>&1 | grep -E "util_before|prompt=|rror|panick"
    done
done
