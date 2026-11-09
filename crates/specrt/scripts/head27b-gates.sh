#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export SPECRT_REF_DIR=${SPECRT_REF_DIR:?}
T=${TESTBIN:?}
PY=${PY:-$HOME/.venv-vllm/bin/python}
guard() {
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt 36000 ]; then
        echo "free $free MiB, stopping before $*"
        exit 1
    fi
    echo "== $* free $free MiB $(date +%T)"
}
gpu() {
    name=$1
    shift
    guard "$name"
    flock /tmp/gpu27.lock env SPECRT_27B_SPEC="$name" nice -n 15 "$T" --ignored --exact --nocapture "$@" 2>&1 | grep -vE "^\s*$"
}
gpu qwen3.8-27b-nvfp4 real_27b_generates_greedy_and_dumps_logits
if [ ! -f "$SPECRT_REF_DIR/real27b-ref.logits.bin" ]; then
    PYTHONDONTWRITEBYTECODE=1 REF_THREADS=16 nice -n 15 "$PY" scripts/reference.py real-full orcarouter/Qwen3.8-27B-Uncensored-NVFP4 "$SPECRT_REF_DIR" "$SPECRT_REF_DIR/real27b-specrt-weight.json" 2>&1 | tail -1
    PYTHONDONTWRITEBYTECODE=1 REF_THREADS=16 nice -n 15 "$PY" scripts/reference.py real-full orcarouter/Qwen3.8-27B-Uncensored-NVFP4 "$SPECRT_REF_DIR" "$SPECRT_REF_DIR/real27b-specrt-weight.json" act 2>&1 | tail -1
fi
gpu qwen3.8-27b-nvfp4-head-fp8 real_27b_generates_greedy_and_dumps_logits
gpu qwen3.8-27b-nvfp4-head-nvfp4 real_27b_generates_greedy_and_dumps_logits
SPECRT_27B_SPEC=qwen3.8-27b-nvfp4 "$T" --ignored --exact --nocapture real_27b_greedy_and_logits_match_the_fp8_reference 2>&1 | grep -E "^(weight|act)|test result|panicked"
echo GATES_DONE
