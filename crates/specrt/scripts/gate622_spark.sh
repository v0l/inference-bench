#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/622
ours=$HOME/ib-gates-target/622
file=$HOME/ib-gates/676-gguf/27b-q5km.gguf
ref=$HOME/ib-gates/out676/ref
vis=$HOME/ib-gates/vis498
run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 -p WorkingDirectory="$root/crates/specrt" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$ours" "$@"
}
exec 9>/tmp/gpu27.lock; flock 9
echo "== gguf_qwen35_gpu"
run --setenv=SPECRT_GGUF_FILE="$file" --setenv=SPECRT_GGUF_REF="$ref" cargo test --release --test gguf_qwen35_gpu -- --ignored --nocapture --test-threads=1 2>&1 | grep -E "KL|test |panicked|result"
echo "== gguf"
run cargo test --release --test gguf --test gguf_logits -- --include-ignored --test-threads=1 2>&1 | grep -E "^test |test result|panicked"
for t in vision vision_encoder vision_lm; do
    echo "== $t"
    run --setenv=SPECRT_VIS_DIR="$vis" cargo test --release --test "$t" -- --include-ignored --test-threads=1 2>&1 | grep -E "^test |test result|panicked"
done
