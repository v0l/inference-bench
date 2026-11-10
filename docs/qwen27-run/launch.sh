#!/bin/bash
MODE=${1:?mtp|nomtp|kvauto}
source ~/.venv-vllm/bin/activate
export HF_HOME=$HOME/.cache/huggingface
export MAX_JOBS=3
export NVCC_THREADS=2
export PYTHONPATH=$HOME/vllm-middleware${PYTHONPATH:+:$PYTHONPATH}
KV=fp8
SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
[ "$MODE" = nomtp ] && SPEC=()
[ "$MODE" = kvauto ] && { SPEC=(); KV=auto; }
exec vllm serve orcarouter/Qwen3.8-27B-Uncensored-NVFP4 \
  --served-model-name "qwen3.8-27b" \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --host 0.0.0.0 --port 8011 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-memory-bytes 14G \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --kv-cache-dtype $KV \
  "${SPEC[@]}" \
  --enable-prompt-tokens-details \
  --mm-encoder-tp-mode data \
  --middleware null_detail.StripNullImageDetail
