#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${WORK:-$HOME/ib-gates/821}
target=${CARGO_TARGET_DIR:-$HOME/ib-gates-target/821}
llama=${LLAMA:-$HOME/git/llama.cpp}
py=$HOME/.venv-vllm/bin/python
models=${*:-llama-3.2-1b gemma-3-1b-it granite-3.3-2b-base SmolLM3-3B Phi-3-mini-4k-instruct}

run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=40G -p MemorySwapMax=0 \
        --setenv=FP32_DEVICE=${FP32_DEVICE:-cuda} --setenv=PATH="$HOME/.venv-vllm/bin:$PATH" --setenv=HOME="$HOME" --setenv=VLLM_CACHE_ROOT=$work/vllm-cache --setenv=FLASHINFER_WORKSPACE_BASE=$work/flashinfer \
        --setenv=TORCHINDUCTOR_CACHE_DIR=$work/inductor --setenv=XDG_CACHE_HOME=$work/xdg --setenv=MAX_JOBS=2 --setenv=NVCC_THREADS=2 "$@"
}

exec 9>/tmp/gpu27.lock
flock 9
for m in $models; do
    out=$work/env/$m
    mkdir -p "$out"
    echo "== $m"
    "$py" "$here/envelope.py" prompts "$work/models/$m" "$out"
    run "$py" "$here/envelope.py" fp32 "$work/models/$m" "$out"
    run "$py" "$here/envelope.py" vllm "$work/models/$m" "$out" 2>&1 | grep -E "^vllm|Error" | tail -3
    "$target/release/envelope" "$work/models/$m" "$out/prompts.json" "$out" split unsplit4 unsplit 2>&1 | tail -3
    args=$("$py" -c "import json,sys;print(' '.join(map(str,[t for p in json.load(open('$out/prompts.json')) for t in p])))")
    : >"$out/llama.f32"
    for i in $(seq 0 7); do
        ids=$("$py" -c "import json;print(' '.join(map(str,json.load(open('$out/prompts.json'))[$i])))")
        "$work/gguf_reference" "$work/gguf/$m.gguf" "$out/llama-$i" 99 1 $ids >/dev/null 2>&1
        cat "$out/llama-$i.logits.bin" >>"$out/llama.f32"
    done
    "$py" "$here/envelope.py" compare "$work/models/$m" "$out" vllm="$out/vllm.npy" llama.cpp="$out/llama.f32" split="$out/specrt-split.f32" unsplit4="$out/specrt-unsplit4.f32" unsplit="$out/specrt-unsplit.f32" | tee "$out/envelope.txt"
done
