#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)

usage() {
    echo "usage: perfpass.sh <family> [--model DIR|GGUF|SPEC] [--host spark|desktop|mac] [--rounds N] [--lens 1024,8192]"
    echo "       perfpass.sh --ab <benchA> <benchB> <model> [--rounds N] [--lens 1024,8192]"
    echo "families: qwen2 qwen3 qwen3_moe qwen3_5 qwen3_next llama mistral gemma3 phi3 smollm3 olmo2 granite gguf-q4 gguf-q8"
    exit 2
}

[ $# -ge 1 ] || usage
mode=family
if [ "$1" = --ab ]; then
    mode=ab
    shift
    [ $# -ge 3 ] || usage
    bin_a=$1; bin_b=$2; model=$3
    shift 3
    family=ab
else
    family=$1
    shift
    model=${MODEL:-}
fi
host=local
rounds=${ROUNDS:-3}
lens=${LENS:-1024,8192}
while [ $# -gt 0 ]; do
    case $1 in
        --model) model=$2; shift 2 ;;
        --host) host=$2; shift 2 ;;
        --rounds) rounds=$2; shift 2 ;;
        --lens) lens=$2; shift 2 ;;
        *) usage ;;
    esac
done

issue=${ISSUE:-803}
case $host in
    local) ;;
    spark)
        ssh spark "mkdir -p ~/ib-gates/$issue/src"
        rsync -az --delete --exclude target --exclude .git --exclude out --exclude docs/perf "$repo/" spark:ib-gates/$issue/src/
        args="$family --rounds $rounds --lens $lens"
        [ -n "$model" ] && args="$args --model $model"
        ssh spark "cd ~/ib-gates/$issue/src && ISSUE=$issue COMMIT=$(git -C "$repo" rev-parse --short HEAD) HOST_LABEL=spark ./crates/specrt/scripts/perfpass.sh $args"
        rc=$?
        mkdir -p "$repo/docs/perf"
        rsync -az --exclude scoreboard.md spark:ib-gates/$issue/src/docs/perf/ "$repo/docs/perf/"
        for row in "$repo"/docs/perf/"$family"/*.row; do
            [ -f "$row" ] && python3 "$here/perfpass_report.py" merge "$repo/docs/perf/scoreboard.md" "$row"
        done
        exit "$rc"
        ;;
    desktop)
        echo "the desktop cards serve the user's models: run this by hand in a supervisor window, see docs/perf-pass.md"
        exit 1
        ;;
    mac)
        echo "the mac has no CUDA: specrt perf passes are CUDA only"
        exit 1
        ;;
    *) usage ;;
esac

commit=${COMMIT:-$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)}
work=${WORK:-$HOME/ib-gates/$issue}
target=${CARGO_TARGET_DIR:-$HOME/ib-gates-target/$issue}
models=${MODELS:-$work/models}
hub=$HOME/.cache/huggingface/hub
llama=${LLAMA_BIN:-$HOME/git/llama.cpp/build/bin}
llama_src=${LLAMA_SRC:-$HOME/git/llama.cpp}
vllm_py=${VLLM_PY:-$HOME/.venv-vllm/bin/python}
conv_py=${CONV_PY:-$work/venv/bin/python}
roof=${ROOF_GBS:-273}
batch=${BATCH:-8}
new=${NEW:-128}
vllm_util=${VLLM_UTIL:-0.12}
sglang_py=${SGLANG_PY:-$HOME/.venv-sglang/bin/python}
date=$(date +%F)
out=$repo/docs/perf/$family
raw=$work/raw/$family-$date
mkdir -p "$out" "$raw"

snap() { ls -d "$hub/models--$1/snapshots/"* 2>/dev/null | head -1; }

pick_model() {
    case $1 in
        qwen2) snap Qwen--Qwen2.5-0.5B ;;
        qwen3) echo "$repo/crates/specrt/specs/qwen3-0.6b.toml" ;;
        qwen3_moe) echo "$models/Qwen3-30B-A3B" ;;
        qwen3_5) echo "$HOME/models/Qwen3.5-9B" ;;
        qwen3_next) echo "" ;;
        llama) snap unsloth--Llama-3.2-1B ;;
        mistral) snap Locutusque--TinyMistral-248M ;;
        gemma3) echo "$models/gemma-3-1b-it" ;;
        phi3) echo "$models/Phi-3-mini-4k-instruct" ;;
        smollm3) echo "$models/SmolLM3-3B" ;;
        olmo2) echo "$models/OLMo-2-0425-1B" ;;
        granite) echo "$models/granite-3.3-2b-base" ;;
        gguf-q4) ls "$(snap unsloth--Qwen3-0.6B-GGUF)"/*Q4_K_M.gguf 2>/dev/null | head -1 ;;
        gguf-q8) echo "$models/Qwen3-0.6B-Q8_0.gguf" ;;
        *) usage ;;
    esac
}

[ -n "$model" ] || model=$(pick_model "$family")
if [ "$mode" = family ] && { [ -z "$model" ] || [ ! -e "$model" ]; }; then
    echo "no model for $family ($model): download it first, see docs/perf-pass.md"
    exit 1
fi

avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "MemAvailable ${avail} GB"
[ "$avail" -ge 40 ] || { echo "need 40 GB available"; exit 1; }

run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=60G -p MemorySwapMax=0 \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" "$@"
}

if [ "$mode" = family ]; then
    systemd-run --user --wait --collect -P -q -p MemoryMax=20G -p "WorkingDirectory=$repo" \
        --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=CARGO_TARGET_DIR="$target" --setenv=CARGO_BUILD_JOBS=6 \
        cargo build --release -p inference-bench-specrt --bin bench 2>&1 | tail -1
fi
bench=$target/release/bench

exec 9>/tmp/gpu27.lock
echo "waiting for gpu27.lock"; flock 9; echo "lock held"

if [ "$mode" = ab ]; then
    for r in $(seq "$rounds"); do
        for side in A B; do
            b=$bin_a; [ $side = B ] && b=$bin_b
            run "$b" --runs 3 "$model" "$lens" "$new" 2>&1 | grep -E "^\s+prompt=" | sed "s/^/round=$r side=$side /" | cut -c1-200
        done
    done | tee "$raw/ab.log"
    python3 "$here/perfpass_report.py" ab "$raw/ab.log"
    exit 0
fi

echo "== ours: prefill and decode b1"
run "$bench" --runs "$rounds" "$model" "$lens" "$new" >"$raw/ours-single.log" 2>&1
cut -c1-240 "$raw/ours-single.log"
echo "== ours: decode b$batch"
run "$bench" --serve --runs "$rounds" --kv-gb 4 "$model" 128 "$new" "$batch" >"$raw/ours-batch.log" 2>&1
grep "serve round" "$raw/ours-batch.log" | cut -c1-240

prof() {
    local tag=$1; shift
    run nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi -o "$raw/$tag" --force-overwrite true \
        "$bench" "$@" >"$raw/$tag.log" 2>&1
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o "$raw/$tag" "$raw/$tag.nsys-rep" >/dev/null 2>&1
}
echo "== nsys decode b1 / b$batch"
prof nsys-b1 --serve --profile --runs 1 --kv-gb 4 "$model" 128 "$new" 1
prof "nsys-b$batch" --serve --profile --runs 1 --kv-gb 4 "$model" 128 "$new" "$batch"
echo "== nsys prefill"
first=${lens%%,*}
last=${lens##*,}
for len in $first $last; do
    run nsys profile -t cuda -o "$raw/nsys-p$len" --force-overwrite true "$bench" --runs 1 "$model" "$len" 2 >"$raw/nsys-p$len.log" 2>&1
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o "$raw/nsys-p$len" "$raw/nsys-p$len.nsys-rep" >/dev/null 2>&1
done

ckpt=$model
case $model in
    *.toml) ckpt=$(snap "$(sed -n 's/^repo = "\(.*\)"/\1/p' "$model" | head -1 | sed 's#/#--#')") ;;
esac
gguf=""
case $model in
    *.gguf) gguf=$model ;;
    *)
        if [ -f "$ckpt/config.json" ] && [ -f "$llama_src/convert_hf_to_gguf.py" ]; then
            gguf=$work/models/$family-bf16.gguf
            mkdir -p "$work/models"
            [ -f "$gguf" ] || PYTHONPATH=$llama_src/gguf-py run "$conv_py" "$llama_src/convert_hf_to_gguf.py" "$ckpt" --outtype bf16 --outfile "$gguf" >"$raw/convert.log" 2>&1 \
                || { gguf=""; echo "gguf conversion failed, see $raw/convert.log"; }
        fi
        ;;
esac
if [ -n "$gguf" ]; then
    echo "== llama.cpp"
    run "$llama/llama-bench" -m "$gguf" -ngl 99 -fa 1 -p "$lens" -n 0 -r "$rounds" -o json 2>/dev/null >"$raw/llama-prefill.json"
    run "$llama/llama-bench" -m "$gguf" -ngl 99 -fa 1 -p 0 -n "$new" -r "$rounds" -o json 2>/dev/null >"$raw/llama-decode.json"
    run "$llama/llama-batched-bench" -m "$gguf" -c $((batch * (128 + new + 16))) -b 2048 -ub 512 -npp 128 -ntg "$new" -npl "$batch" -ngl 99 -fa on --output-format jsonl 2>/dev/null >"$raw/llama-batch.jsonl"
else
    echo "llama.cpp: not run (no GGUF)"
fi

if [ -f "$ckpt/config.json" ] && [ -x "$vllm_py" ] && [ -z "${NO_VLLM:-}" ]; then
    echo "== vllm"
    run -p MemoryMax=40G --setenv=PATH="$(dirname "$vllm_py"):$PATH" --setenv=VLLM_CACHE_ROOT=$work/vllm-cache --setenv=FLASHINFER_WORKSPACE_BASE=$work/flashinfer --setenv=TORCHINDUCTOR_CACHE_DIR=$work/inductor --setenv=XDG_CACHE_HOME=$work/xdg --setenv=MAX_JOBS=2 --setenv=NVCC_THREADS=2 "$vllm_py" "$here/vllm_perf.py" "$ckpt" --lens "$lens" --new "$new" --batch "$batch" --rounds "$rounds" --util "$vllm_util" --out "$raw/vllm.json" >"$raw/vllm.log" 2>&1 \
        || echo "vllm failed: $(grep -E 'Error|error' "$raw/vllm.log" | tail -1 | cut -c1-200)"
else
    echo "vllm: not run (not a safetensors checkpoint dir)"
fi

if [ -f "$ckpt/config.json" ] && [ -x "$sglang_py" ] && [ -z "${NO_SGLANG:-}" ]; then
    echo "== sglang"
    run -p MemoryMax=40G --setenv=PATH="$(dirname "$sglang_py"):/usr/local/cuda/bin:/usr/bin:/bin" --setenv=XDG_CACHE_HOME=$work/xdg "$sglang_py" "$here/sglang_perf.py" "$ckpt" --lens "$lens" --new "$new" --batch "$batch" --rounds "$rounds" --util "$vllm_util" --out "$raw/sglang.json" >"$raw/sglang.log" 2>&1 \
        || echo "sglang failed: $(grep -E 'Error|error' "$raw/sglang.log" | tail -1 | cut -c1-200)"
else
    echo "sglang: not run (no venv or not a safetensors checkpoint dir)"
fi

python3 "$here/perfpass_report.py" family "$family" "$model" "$raw" "$out" "$commit" "$roof" "$batch" "$lens" \
    "crates/specrt/scripts/perfpass.sh $family --model $model --host ${HOST_LABEL:-local}"
