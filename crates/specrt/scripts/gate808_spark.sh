#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/808
ref=$HOME/ib-gates/ref808
target=$HOME/ib-gates-target/808
py=$HOME/.venv-vllm/bin/python
mkdir -p "$ref" "$target"
run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 -p WorkingDirectory="$root" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" --setenv=CARGO_BUILD_JOBS=6 \
        --setenv=SPECRT_REF_DIR="$ref" "$@"
}
cd "$root/crates/specrt"
CUDA_VISIBLE_DEVICES= $py scripts/tiny_mamba.py "$ref" 2>&1 | tail -1
if [ "${1:-}" = "all" ]; then
    cp -n "$HOME"/ib-gates/ref830/* "$ref"/ 2>/dev/null
    CUDA_VISIBLE_DEVICES= $py scripts/tiny_variants.py "$ref" 2>&1 | tail -1
    CUDA_VISIBLE_DEVICES= $py scripts/reference.py tiny tests/fixtures/tiny-qwen3.5 "$ref" 256 2>&1 | tail -1
fi
exec 9>/tmp/gpu27.lock
flock 9
tests=("--test mamba -- --include-ignored")
if [ "${1:-}" = "all" ]; then
    tests+=("--test variants -- --include-ignored" "--test linear -- --include-ignored" "--test tiny27b -- --include-ignored" "--test sessions -- --include-ignored" "--test prefix -- --include-ignored" "--test serving -- --include-ignored" "--test described -- --include-ignored")
fi
for t in "${tests[@]}"; do
    echo "== $t"
    run bash -lc "cargo test -p inference-bench-specrt --release $t --test-threads=1 --nocapture 2>&1 | grep -E '^test .*(FAILED|failed)|max_abs|agree|panicked|test result|^error'"
done
