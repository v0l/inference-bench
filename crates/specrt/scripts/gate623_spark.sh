#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/623
master=$HOME/ib-gates/623-master
ref=$HOME/ib-gates/ref375
out=$HOME/ib-gates/out623
ours=$HOME/ib-gates-target/623
base=$HOME/ib-gates-target/623-master
hf=$HOME/.cache/huggingface/hub
q8b=$(ls $hf/models--unsloth--Qwen3-8B-GGUF/snapshots/*/Qwen3-8B-Q4_K_M.gguf)
q35=$HOME/ib-gates/q35-q8.gguf
llama=$HOME/git/llama.cpp
spec27=tests/fixtures/qwen3.8-27b-nvfp4.toml
export PATH=$HOME/ib-gates/bin/ffdir/7.0/bin:$PATH
mkdir -p "$out" "$HOME/ib-gates/gguf"
llq=$llama/build/bin/llama-quantize
if [ ! -f $q35 ]; then LD_LIBRARY_PATH=$llama/build/bin $llq --allow-requantize "$(ls $hf/models--unsloth--Qwen3.5-4B-GGUF/snapshots/*/*.gguf)" $q35 Q8_0; fi
g=$HOME/ib-gates/gguf
if [ ! -f $g/Qwen3-0.6B-Q8_0.gguf ]; then
    ln -sf $HOME/ib-gguf/Qwen3-0.6B-bf16.gguf $g/Qwen3-0.6B-BF16.gguf
    for s in Q8_0 Q4_K_M; do LD_LIBRARY_PATH=$llama/build/bin $llq $g/Qwen3-0.6B-BF16.gguf $g/Qwen3-0.6B-$s.gguf $s; done
fi
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }
run() {
    local dir=$1 target=$2; shift 2
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$dir" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" "$@"
}
bench() {
    local side=$1 d=$2 tg=$3; shift 3
    if [ "$side" = master ]; then
        run "$d/crates/specrt" "$tg" --setenv=SPECRT_BENCH_RUNS=3 "$tg/release/bench" "$@"
    else
        run "$d/crates/specrt" "$tg" "$tg/release/bench" --runs 3 "$@"
    fi
}
suite=${SUITE:-"kv_dtype truncation mtp serving tiny27b linear logits prefix gguf gguf_logits gguf_qwen35_gpu vision_lm vision"}
tests=""; for t in $suite real27b; do tests="$tests --test $t"; done
echo "== build"
for side in ours master; do
    d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
    run "$d/crates/specrt" "$tg" -p MemoryMax=20G cargo build --release --bin bench 2>&1 | tail -1
    run "$d/crates/specrt" "$tg" -p MemoryMax=20G cargo test --release --no-run $tests 2>&1 | tail -1
done
exec 9>/tmp/gpu27.lock
echo "waiting for gpu27.lock"; flock 9; echo "lock held $(date +%T)"
export LD_LIBRARY_PATH=$llama/build/bin
gcc -O2 "$root/crates/specrt/scripts/gguf_reference.c" -I"$llama/include" -I"$llama/ggml/include" -L"$llama/build/bin" -lllama -lggml -lggml-base -lm -Wl,-rpath,"$llama/build/bin" -o "$out/gguf_reference" || exit 1
prompt="785 6722 315 9625 374 12095 13 9645 264 2805 32794 911 279 9396 323 279 9759 13"
"$out/gguf_reference" "$q35" "$out/ref35" 99 64 $prompt 2>&1 | tail -1
for s in Q8_0 Q4_K_M BF16; do "$out/gguf_reference" $g/Qwen3-0.6B-$s.gguf $g/q3-$s 0 64 $prompt 2>&1 | tail -1; done
for side in ours master; do
    d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
    for t in $suite; do
        echo "== $side $t"
        run "$d/crates/specrt" "$tg" --setenv=SPECRT_REF_DIR="$ref" --setenv=SPECRT_GGUF_DIR="$HOME/ib-gates/gguf" \
            --setenv=SPECRT_GGUF_FILE="$q35" --setenv=SPECRT_GGUF_REF="$out/ref35" \
            cargo test --release --test "$t" -- --include-ignored --test-threads=1 --skip real_27b 2>&1 | grep -E "^test |test result|panicked" > "$out/$side.$t.txt"
        grep "test result" "$out/$side.$t.txt"
    done
done
echo "== real27b ours (NVFP4, skip needle)"
if [ -z "${SKIP_REAL27B:-}" ]; then run "$root/crates/specrt" "$ours" --setenv=SPECRT_REF_DIR="$ref" \
    cargo test --release --test real27b -- --include-ignored --test-threads=1 --skip needle 2>&1 | grep -E "^test |test result|panicked" | tee "$out/ours.real27b.txt"; fi
echo "== failing tests, ours vs master"
for t in $suite; do
    diff <(grep -E "^test .* \.\.\. " "$out/master.$t.txt" | sort) <(grep -E "^test .* \.\.\. " "$out/ours.$t.txt" | sort) > "$out/diff.$t.txt" && echo "$t: same outcomes" || { echo "$t: DIFFERS"; cat "$out/diff.$t.txt"; }
done
echo "== paired speed, 3 rounds"
for r in 1 2 3; do
    for side in master ours; do
        d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
        echo "-- $side round $r 27B nvfp4 prefill 1024 + decode b1"
        bench $side "$d" "$tg" "$spec27" 1024 128 2>&1 | grep "prompt="
        echo "-- $side round $r 27B nvfp4 decode b8"
        bench $side "$d" "$tg" "$spec27" 128 128 8 2>&1 | grep "batch="
        echo "-- $side round $r gguf Qwen3-8B Q4_K_M"
        bench $side "$d" "$tg" "$q8b" 1024 128 2>&1 | grep "prompt="
    done
done
