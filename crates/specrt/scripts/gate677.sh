#!/usr/bin/env bash
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
repo=HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-677}
export CARGO_BUILD_JOBS=4
export SPECRT_REF_DIR=${SPECRT_REF_DIR:?reference dir with vllm-modelopt.json, see step 4}
failed=0
step() {
    local name=$1; shift
    if "$@"; then echo "PASS $name"; else echo "FAIL $name"; failed=1; fi
}
cargo_test() { systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$root" cargo test -p inference-bench-specrt "$@"; }
snapshot() { ls -d "$HOME"/.cache/huggingface/hub/models--${repo//\//--}/snapshots/* 2>/dev/null | head -1; }

download() {
    local need_gb=23 free_gb
    free_gb=$(df -BG --output=avail "$HOME/.cache/huggingface" | tail -1 | tr -dc 0-9)
    [ "$free_gb" -ge $((need_gb + 5)) ] || { echo "only ${free_gb} GB free, the checkpoint is about 22 GB"; return 1; }
    huggingface-cli download "$repo"
}

modelopt_tiny() { cargo_test --test modelopt --no-fail-fast -- --include-ignored --nocapture; }
tiny27b() { cargo_test --test tiny27b -- --include-ignored --test-threads=1; }

vllm_compare() {
    [ -f "$SPECRT_REF_DIR/vllm-modelopt.json" ] || { echo "missing $SPECRT_REF_DIR/vllm-modelopt.json: run vllm_logprobs.py with VLLM_REPO=$repo and VLLM_MODEL=<served name> against vLLM serving this checkpoint, then copy its vllm-nvfp4.json there as vllm-modelopt.json"; return 1; }
    local dir tmp=$SPECRT_REF_DIR/modelopt27b
    dir=$(snapshot) || return 1
    mkdir -p "$tmp" && cp "$SPECRT_REF_DIR/vllm-modelopt.json" "$tmp/vllm-nvfp4.json"
    sed -e "s#^repo = .*#repo = \"$repo\"#" "$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml" >"$root/crates/specrt/tests/fixtures/qwen3.8-27b-modelopt.toml"
    SPECRT_REF_DIR=$tmp SPECRT_27B_NVFP4_SPEC=qwen3.8-27b-modelopt \
        systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$root" \
        env SPECRT_REF_DIR=$tmp SPECRT_27B_NVFP4_SPEC=qwen3.8-27b-modelopt CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
        cargo test -p inference-bench-specrt --test real27b -- --ignored --exact --nocapture real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs
}

record() {
    printf '%s\n' "$1" >"$root/crates/specrt/modelopt-verified.txt"
    echo "recorded $1 in crates/specrt/modelopt-verified.txt"
}

run_all() {
    step download download
    step modelopt-tiny-gpu modelopt_tiny
    step tiny27b tiny27b
    step darkstar-27b-vs-vllm vllm_compare
    if [ $failed -eq 0 ]; then record pass; else record pending; fi
    exit $failed
}

if [ -n "${GPU27_LOCK_HELD:-}" ]; then run_all; fi
export GPU27_LOCK_HELD=1
exec flock /tmp/gpu27.lock bash -c "$(declare -f step cargo_test snapshot download modelopt_tiny tiny27b vllm_compare record run_all); failed=0; repo=$repo; root=$root; run_all"
