#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
hub=$HOME/.cache/huggingface/hub
py=${SGLANG_PY:-$HOME/.venv-sglang/bin/python}
out=${OUT:-$HOME/ib-gates/817/sglang}
rounds=${ROUNDS:-3}
lens=${LENS:-1024,8192}
mkdir -p "$out"

snap() { ls -d "$hub/models--$1/snapshots/"* 2>/dev/null | head -1; }

model_for() {
    case $1 in
        qwen2) snap Qwen--Qwen2.5-0.5B ;;
        qwen3) snap Qwen--Qwen3-0.6B ;;
        qwen3_5) echo "$HOME/models/Qwen3.5-9B" ;;
        llama) snap unsloth--Llama-3.2-1B ;;
        mistral) snap Locutusque--TinyMistral-248M ;;
        gemma3) echo "$HOME/ib-gates/817/models/gemma-3-1b-it" ;;
        phi3) echo "$HOME/ib-gates/817/models/Phi-3-mini-4k-instruct" ;;
        smollm3) echo "$HOME/ib-gates/817/models/SmolLM3-3B" ;;
        olmo2) echo "$HOME/ib-gates/817/models/OLMo-2-0425-1B" ;;
        granite) echo "$HOME/ib-gates/817/models/granite-3.3-2b-base" ;;
        qwen3_moe) echo "$HOME/ib-gates/817/models/Qwen3-30B-A3B-FP8" ;;
        qwen3_next) echo "$HOME/ib-gates/817/models/Qwen3-Next-80B-A3B-Instruct-NVFP4" ;;
    esac
}

util_for() {
    case $1 in
        qwen3_5) echo 0.45 ;;
        qwen3_moe) echo 0.6 ;;
        smollm3|granite|phi3|olmo2) echo 0.2 ;;
        qwen3_next) echo 0.7 ;;
        *) echo 0.12 ;;
    esac
}

exec 9>/tmp/gpu27.lock
flock 9

for family in "$@"; do
    model=$(model_for "$family")
    if [ -z "$model" ] || [ ! -e "$model/config.json" ]; then
        echo "$family: no checkpoint"
        continue
    fi
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 40 ] || { echo "$family: MemAvailable ${avail} GB, need 40"; continue; }
    echo "== sglang $family"
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 \
        --setenv=PATH="$(dirname "$py"):/usr/local/cuda/bin:/usr/bin:/bin" --setenv=HOME="$HOME" --setenv=SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 --setenv=XDG_CACHE_HOME="$out/xdg" \
        "$py" "$here/sglang_perf.py" "$model" --lens "$lens" --rounds "$rounds" --util "$(util_for "$family")" ${ATTN:+--attn $ATTN} --out "$out/$family.json" >"$out/$family.log" 2>&1 \
        || echo "$family failed: $(grep -E 'Error|error' "$out/$family.log" | tail -1 | cut -c1-200)"
done
