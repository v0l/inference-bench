#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/676
master=$HOME/ib-gates/676-master
file=$HOME/ib-gates/676-gguf/27b-q5km.gguf
ref=$HOME/ib-gates/ref676
out=$HOME/ib-gates/out676
llama=$HOME/git/llama.cpp
mkdir -p "$out"
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }
run() {
    local dir=$1 target=$2; shift 2
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$dir" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" "$@"
}
ours=$HOME/ib-gates-target/676
base=$HOME/ib-gates-target/676-master
export LD_LIBRARY_PATH=$llama/build/bin
echo "== gate676 llama.cpp reference"
gcc -O2 "$root/crates/specrt/scripts/gguf_reference.c" -I"$llama/include" -I"$llama/ggml/include" -L"$llama/build/bin" -lllama -lggml -lggml-base -lm -Wl,-rpath,"$llama/build/bin" -o "$out/gguf_reference" || exit 1
prompt="785 6722 315 9625 374 12095 13 9645 264 2805 32794 911 279 9396 323 279 9759 13"
"$out/gguf_reference" "$file" "$out/ref" 99 64 $prompt 2>&1 | tail -2
echo "== gguf_qwen35_gpu 27B"
run "$root/crates/specrt" "$ours" --setenv=SPECRT_GGUF_FILE="$file" --setenv=SPECRT_GGUF_REF="$out/ref" \
    cargo test --release --test gguf_qwen35_gpu -- --ignored --nocapture --test-threads=1 a_qwen35_gguf_matches a_qwen35_gguf_native 2>&1 | grep -E "KL|test |panicked|result"
echo "== gguf_qwen35_gpu tiny"
run "$root/crates/specrt" "$ours" cargo test --release --test gguf_qwen35_gpu -- --ignored --nocapture a_tiny 2>&1 | grep -E "KL|test |panicked|result"
for t in linear mtp tiny27b; do
    echo "== $t"
    run "$root/crates/specrt" "$ours" --setenv=SPECRT_REF_DIR="$ref" cargo test --release --test "$t" -- --include-ignored --test-threads=1 --skip real_27b 2>&1 | grep -E "^test |test result|panicked" | grep -v "ok$" 
done
echo "== gdn_speed paired"
for r in 1 2 3; do
    for side in master ours; do
        d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
        echo "-- $side round $r"
        run "$d/crates/specrt" "$tg" cargo test --release --test gdn_speed -- --ignored --nocapture 2>&1 | grep "^gdn tokens"
    done
done
