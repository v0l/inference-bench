#!/bin/bash
set -u
MODEL=${1:?model dir}
OUT=${2:?output dir}
PORT=18639
GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$GPU_FREE_MIB" -lt 42000 ]; then
    echo "only $GPU_FREE_MIB MiB free, need 42000: not starting vLLM"
    exit 3
fi
cd "$(dirname "$0")"
LOG=$(mktemp /tmp/vllm-gemma3.XXXX.log)
cleanup() {
    [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null
    sleep 3
    pkill -f "vllm serve $MODEL" 2>/dev/null
}
trap cleanup EXIT
MAX_JOBS=3 NVCC_THREADS=2 systemd-run --user --wait --collect -P -q -p MemoryMax=24G -p MemorySwapMax=0 \
    --setenv=MAX_JOBS --setenv=NVCC_THREADS --setenv=HOME --setenv=PATH \
    ~/.venv-vllm/bin/vllm serve "$MODEL" --served-model-name gemma27 --port $PORT --max-model-len 4096 --max-num-seqs 2 \
    --kv-cache-memory-bytes 3G --enforce-eager --limit-mm-per-prompt '{"image":0}' > "$LOG" 2>&1 &
VPID=$!
for i in $(seq 1 90); do
    if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
    sleep 5
    if ! kill -0 "$VPID" 2>/dev/null; then
        echo "vLLM exited early, log $LOG"
        tail -20 "$LOG"
        exit 4
    fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "vLLM never answered, log $LOG"; exit 5; }
~/.venv-vllm/bin/python vllm_gemma3.py "$MODEL" "$OUT"
echo "reference written to $OUT, vLLM log $LOG"
