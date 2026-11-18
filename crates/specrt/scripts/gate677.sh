#!/usr/bin/env bash
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
repo=HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-677}
export CARGO_BUILD_JOBS=4
export SPECRT_REF_DIR=${SPECRT_REF_DIR:-${TMPDIR:-/tmp}/gate677-ref}
failed=0
step() {
    local name=$1; shift
    case " ${GATE_SKIP:-} " in *" $name "*) echo "SKIP $name"; failed=1; return 0;; esac
    if "$@"; then echo "PASS $name"; else echo "FAIL $name"; failed=1; fi
}
cargo_test() { systemd-run --user --wait --collect -P -q -p MemoryMax=10G -E CARGO_TARGET_DIR="$CARGO_TARGET_DIR" --working-directory="$root" cargo test -p inference-bench-specrt "$@"; }
snapshot() { ls -d "$HOME"/.cache/huggingface/hub/models--${repo//\//--}/snapshots/* 2>/dev/null | head -1; }

download() {
    local need_gb=23 free_gb
    free_gb=$(df -BG --output=avail "$HOME/.cache/huggingface" | tail -1 | tr -dc 0-9)
    [ "$free_gb" -ge $((need_gb + 5)) ] || { echo "only ${free_gb} GB free, the checkpoint is about 22 GB"; return 1; }
    if command -v huggingface-cli >/dev/null; then huggingface-cli download "$repo"; else hf download "$repo"; fi
}

modelopt_tiny() { cargo_test --test modelopt --no-fail-fast -- --include-ignored --nocapture; }
tiny27b() { cargo_test --test tiny27b -- --include-ignored --test-threads=1; }

cli() { echo "$CARGO_TARGET_DIR/release/inference-bench"; }
build_cli() { systemd-run --user --wait --collect -P -q -p MemoryMax=10G -E CARGO_TARGET_DIR="$CARGO_TARGET_DIR" --working-directory="$root" cargo build --release -p inference-bench-cli; }

derive_spec() {
    local out=$1
    build_cli || return 1
    "$(cli)" spec --model "$repo" --out "$out" --expect-act-quant none
}

vllm_reference() {
    local out=$1 port=${VLLM_PORT:-18727} py=${VLLM_PYTHON:-$HOME/.venv-vllm/bin/python} pid ok=0
    [ -f "$out/vllm-nvfp4.json" ] && { echo "reusing $out/vllm-nvfp4.json"; return 0; }
    mkdir -p "$out"
    systemd-run --user --collect -p MemoryMax=${VLLM_MEM:-24G} -p MemorySwapMax=0 --unit=gate677-vllm \
        -E MAX_JOBS=${VLLM_JOBS:-1} -E NVCC_THREADS=${VLLM_NVCC_THREADS:-1} -E HF_HUB_OFFLINE=1 -E PATH="$(dirname "$py"):/usr/local/cuda/bin:$PATH" \
        "$py" -m vllm.entrypoints.openai.api_server --model "$repo" --served-model-name darkstar \
        --port "$port" --max-model-len 2048 --gpu-memory-utilization "${VLLM_UTIL:-0.25}" --enforce-eager >"$out/vllm.log" 2>&1 || return 1
    for _ in $(seq 180); do
        curl -sf "localhost:$port/v1/models" >/dev/null && { ok=1; break; }
        systemctl --user is-active --quiet gate677-vllm || break
        sleep 5
    done
    if [ $ok -eq 1 ]; then
        VLLM_URL=http://127.0.0.1:$port/v1/completions VLLM_MODEL=darkstar VLLM_REPO=$repo "$py" "$here/vllm_logprobs.py" "$out" || ok=0
    else
        tail -20 "$out/vllm.log"
    fi
    systemctl --user stop gate677-vllm 2>/dev/null
    [ $ok -eq 1 ]
}

vllm_compare() {
    local tmp=$SPECRT_REF_DIR/modelopt27b spec=$SPECRT_REF_DIR/modelopt27b/darkstar-spec.toml
    mkdir -p "$tmp"
    derive_spec "$spec" || return 1
    vllm_reference "$tmp" || return 1
    systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 --working-directory="$root" \
        env SPECRT_REF_DIR="$tmp" SPECRT_27B_NVFP4_SPEC="$spec" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
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
exec flock /tmp/gpu27.lock bash -c "$(declare -f step cargo_test snapshot download modelopt_tiny tiny27b cli build_cli derive_spec vllm_reference vllm_compare record run_all); failed=0; repo=$repo; root=$root; here=$here; run_all"
