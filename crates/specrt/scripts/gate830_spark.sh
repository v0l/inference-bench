#!/usr/bin/env bash
set -uo pipefail
root=$HOME/ib-gates/830
ref=$HOME/ib-gates/ref830
target=$HOME/ib-gates-target/830
py=$HOME/.venv-vllm/bin/python
mkdir -p "$ref" "$target"
run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=20G -p MemorySwapMax=0 -p WorkingDirectory="$root" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" --setenv=CARGO_BUILD_JOBS=6 \
        --setenv=SPECRT_REF_DIR="$ref" "$@"
}
cd "$root/crates/specrt"
$py scripts/tiny_variants.py "$ref" | tail -4
$py scripts/reference.py tiny-moe tests/fixtures/tiny-qwen3.5-moe "$ref" 256 | tail -1
$py scripts/reference.py tiny-moe-fp8 tests/fixtures/tiny-qwen3.5-moe-fp8 "$ref" 256 | tail -1
exec 9>/tmp/gpu27.lock
flock 9
for t in "--test variants -- --include-ignored" "--test moe_mma -- --include-ignored" "--test linear -- --include-ignored moe" "--test tp -- --include-ignored" "--test codegen -- --include-ignored" "--lib" "--test golden" "--test spec"; do
    echo "== $t"
    run bash -lc "cargo test -p inference-bench-specrt $t 2>&1 | grep -E '^test .*(FAILED|failed)|test result|error'"
done
echo "== clippy"
run bash -lc "cargo clippy -p inference-bench-specrt -p inference-bench-kir --all-targets -- -D warnings 2>&1 | tail -15"
