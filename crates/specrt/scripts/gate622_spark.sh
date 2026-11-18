#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/622
master=$HOME/ib-gates/622-master
vis=$HOME/ib-gates/vis498
hf=$HOME/.cache/huggingface/hub
q35=$HOME/ib-gates/q35-q8.gguf
q8b=$(ls $hf/models--unsloth--Qwen3-8B-GGUF/snapshots/*/Qwen3-8B-Q4_K_M.gguf)
out=$HOME/ib-gates/out622
llama=$HOME/git/llama.cpp
ours=$HOME/ib-gates-target/622
base=$HOME/ib-gates-target/622-master
mkdir -p "$out"
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }
run() {
    local dir=$1 target=$2; shift 2
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$dir" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" "$@"
}
echo "== build"
for side in ours master; do
    d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
    run "$d/crates/specrt" "$tg" -p MemoryMax=20G cargo build --release --bin bench 2>&1 | tail -1
    run "$d/crates/specrt" "$tg" -p MemoryMax=20G cargo test --release --no-run --test vision_encoder --test gguf_logits --test gguf --test gguf_qwen35_gpu 2>&1 | tail -1
done
exec 9>/tmp/gpu27.lock
echo "waiting for gpu27.lock"; flock 9; echo "lock held"
export LD_LIBRARY_PATH=$llama/build/bin
echo "== llama.cpp reference for $(basename "$q35")"
gcc -O2 "$root/crates/specrt/scripts/gguf_reference.c" -I"$llama/include" -I"$llama/ggml/include" -L"$llama/build/bin" -lllama -lggml -lggml-base -lm -Wl,-rpath,"$llama/build/bin" -o "$out/gguf_reference" || exit 1
prompt="785 6722 315 9625 374 12095 13 9645 264 2805 32794 911 279 9396 323 279 9759 13"
"$out/gguf_reference" "$q35" "$out/ref" 99 64 $prompt 2>&1 | tail -2
echo "== gguf_qwen35_gpu (Qwen3.5-4B Q8_0)"
run "$root/crates/specrt" "$ours" --setenv=SPECRT_GGUF_FILE="$q35" --setenv=SPECRT_GGUF_REF="$out/ref" \
    cargo test --release --test gguf_qwen35_gpu -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "KL|^test |panicked|result"
[ -n "${SKIP_VISION:-}" ] || true
echo "== gguf_logits (Qwen3-0.6B Q4_K_M, Q8_0 against llama.cpp refs)"
run "$root/crates/specrt" "$ours" cargo test --release --test gguf_logits -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "KL|^test |panicked|result"
echo "== gguf"
run "$root/crates/specrt" "$ours" cargo test --release --test gguf -- --include-ignored --test-threads=1 2>&1 | grep -E "^test |test result|panicked"
for t in ${VISION_TESTS-vision_encoder vision_lm vision}; do
    echo "== $t"
    run "$root/crates/specrt" "$ours" --setenv=SPECRT_VISION_REF_DIR="$vis" cargo test --release --test "$t" -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "^test |test result|panicked|max abs|rms"
done
echo "== paired speed, 3 rounds"
for r in 1 2 3; do
    for side in master ours; do
        d=$root; tg=$ours; [ $side = master ] && { d=$master; tg=$base; }
        for f in "$q8b" "$q35"; do
            echo "-- $side round $r $(basename "$f")"
            run "$d/crates/specrt" "$tg" "$tg/release/bench" "$f" 1024 128 2>&1 | grep "prompt="
        done
        echo "-- $side round $r vision encode"
        run "$d/crates/specrt" "$tg" cargo test --release --test vision_encoder -- --ignored --nocapture a_960x1280 2>&1 | grep -E "best of 5"
    done
done
