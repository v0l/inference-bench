#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export SPECRT_REF_DIR=${SPECRT_REF_DIR:?}
T=${TESTBIN:?}
for spec in ${SPECS:-qwen3.8-27b-nvfp4 qwen3.8-27b-nvfp4-head-fp8 qwen3.8-27b-nvfp4-head-nvfp4}; do
    for try in $(seq 1 ${TRIES:-90}); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free" -ge 36000 ] && break
        sleep 20
    done
    echo "== $spec free $free MiB $(date +%T)"
    flock /tmp/gpu27.lock bash -c '
        f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$f" -ge 36000 ] || { echo "free $f MiB under the lock, skipped"; exit 1; }
        SPECRT_27B_SPEC='"$spec"' nice -n 15 '"$T"' --ignored --exact --nocapture real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs
    ' 2>&1 | grep -E "^\{|test result|panicked|skipped"
done
echo VLLM_DONE
