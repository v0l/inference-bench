#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$here/../../..
out=${OUT:-$here/../out/nsys413}
port=${PORT:-18401}
mtp=${MTP:-on}
phase=${1:-vllm}
bench=${BENCH:-/mnt/data/cargo-target/inference-bench/release/bench}
venv=${VENV:-$HOME/.venv-vllm}
model=${MODEL:-orcarouter/Qwen3.8-27B-Uncensored-NVFP4}
unit=vllm27b-nsys-$port
mkdir -p "$out"
log=$out/$phase-$mtp.log
start=$(date '+%Y-%m-%d %H:%M:%S')

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

guard() {
    local free
    free=$(free_mib)
    if [ "$free" -lt 36000 ]; then
        echo "only $free MiB free, need 36000" >&2
        exit 1
    fi
    echo "== $* free ${free} MiB util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) $(date +%T)" | tee -a "$log"
}

stop_vllm() {
    if systemctl --user is-active "$unit.service" >/dev/null 2>&1; then
        systemctl --user kill -s INT --kill-whom=main "$unit.service" 2>/dev/null
        for _ in $(seq 24); do
            systemctl --user is-active "$unit.service" >/dev/null 2>&1 || break
            sleep 5
        done
    fi
    systemctl --user stop "$unit.service" 2>/dev/null
    systemctl --user reset-failed "$unit.service" 2>/dev/null
    pkill -f "vllm serve .*--port $port" 2>/dev/null
    return 0
}

ids() {
    local n=$1 salt=$2 i s=""
    for ((i = 0; i < n; i++)); do s+="$((1000 + (i * 7 + salt * 131) % 5000)),"; done
    echo "[${s%,}]"
}

complete() {
    local len=$1 new=$2 salt=$3
    curl -sf "localhost:$port/v1/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"qwen3.8-27b\",\"prompt\":$(ids "$len" "$salt"),\"max_tokens\":$new,\"min_tokens\":$new,\"temperature\":0,\"ignore_eos\":true}" >/dev/null
}

scenarios=0

scenario() {
    local name=$1 len=$2 new=$3 batch=$4 i base
    scenarios=$((scenarios + 1))
    base=$((scenarios * 1000))
    for i in $(seq "$batch"); do complete "$len" "$new" "$((base + i))" & done
    wait
    curl -sf -X POST "localhost:$port/start_profile" >/dev/null || { echo "start_profile failed" >&2; return 1; }
    local t0 t1
    t0=$(date +%s.%N)
    for i in $(seq "$batch"); do complete "$len" "$new" "$((base + 500 + i))" & done
    wait
    t1=$(date +%s.%N)
    curl -sf -X POST "localhost:$port/stop_profile" >/dev/null
    echo "vllm $name mtp=$mtp prompt=$len new=$new batch=$batch wall=$(echo "$t1 - $t0" | bc) s" | tee -a "$log"
    sleep 5
}

metrics() {
    curl -s "localhost:$port/metrics" | grep -E '^vllm:spec_decode_num_(drafts|accepted_tokens|draft_tokens)_total' | tee -a "$log"
}

summarise() {
    local f
    for f in "$1"*.nsys-rep; do
        nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output "${f%.nsys-rep}" "$f" >/dev/null 2>&1
    done
}

run_vllm() {
    guard vllm start mtp=$mtp
    local spec=()
    [ "$mtp" = on ] && spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
    local prefix=$out/vllm-mtp-$mtp
    local nsys="nsys profile --cuda-graph-trace=node --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=repeat:12 --force-overwrite=true -o ${prefix}-%n"
    [ "${NSYS:-1}" = 0 ] && nsys=
    rm -f "$prefix"*.nsys-rep
    trap stop_vllm EXIT
    systemd-run --user --collect --unit "$unit" -p MemoryMax=24G -p MemorySwapMax=0 \
        -E MAX_JOBS=3 -E NVCC_THREADS=2 -E HF_HOME="$HOME/.cache/huggingface" \
        -E PYTHONPATH="$HOME/vllm-middleware" -E PATH="$venv/bin:/usr/local/cuda/bin:$PATH" \
        $nsys \
        "$venv/bin/vllm" serve "$model" \
        --served-model-name qwen3.8-27b \
        --max-num-batched-tokens 16384 --max-num-seqs 8 --max-model-len 8192 \
        --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
        --host 127.0.0.1 --port "$port" \
        --gpu-memory-utilization 0.38 --kv-cache-memory-bytes 3G \
        --skip-mm-profiling --mm-processor-cache-gb 0 --safetensors-load-strategy lazy --enable-prefix-caching --kv-cache-dtype fp8 "${spec[@]}" \
        --profiler-config '{"profiler":"cuda"}' >/dev/null 2>&1
    for _ in $(seq 120); do
        curl -sf "localhost:$port/v1/models" >/dev/null && break
        systemctl --user is-active "$unit.service" >/dev/null || { journalctl --user -u "$unit.service" -n 30 --no-pager; exit 1; }
        sleep 5
    done
    curl -sf "localhost:$port/v1/models" >/dev/null || { echo "vllm did not come up" >&2; exit 1; }
    echo "vllm up, free $(free_mib) MiB" | tee -a "$log"
    scenario single128 128 64 1
    scenario batch4 128 64 4
    scenario batch8 128 64 8
    scenario prefill1024 1024 2 1
    scenario prefill4096 4096 2 1
    metrics
    stop_vllm
    trap - EXIT
    sleep 10
    echo "vllm stopped, free $(free_mib) MiB" | tee -a "$log"
    summarise "$prefix"
}

run_specrt() {
    local act=$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4.toml
    local def=$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4-exact.toml
    local spec_name spec flag args name extra
    for spec_name in ${SPECS:-act default}; do
        spec=$act
        [ "$spec_name" = default ] && spec=$def
        for flag in ${FLAGS:-plain spec}; do
            extra=
            [ "$flag" = spec ] && extra=--speculative
            local runs=("128 64" "128 64 4" "128 64 8" "1024 2" "4096 2")
            [ "$flag" = spec ] && runs=("256 64" "256 64 4" "256 64 8")
            for args in "${runs[@]}"; do
                guard specrt "$spec_name" "$flag" "$args"
                name=specrt-$spec_name-$flag-${args// /_}
                SPECRT_BENCH_RUNS=3 nice -n 15 nsys profile --cuda-graph-trace=node --trace=cuda,nvtx \
                    --force-overwrite=true -o "$out/$name" \
                    "$bench" $extra "$spec" $args >"$out/$name.txt" 2>&1
                nsys stats --report cuda_gpu_kern_sum --format csv --output "$out/$name" "$out/$name.nsys-rep" >/dev/null 2>&1
                tail -3 "$out/$name.txt" | tee -a "$log"
            done
        done
    done
}

case $phase in
    vllm) run_vllm ;;
    specrt) run_specrt ;;
    *) echo "usage: [MTP=on|off] flock /tmp/gpu27.lock vllm27b-nsys.sh vllm|specrt" >&2; exit 2 ;;
esac
echo "served vLLM errors during the window: $(journalctl -u local-llm.service --since "$start" --no-pager -q 2>/dev/null | grep -ciE ' error|oom|out of memory')" | tee -a "$log"
