#!/usr/bin/env bash
set -euo pipefail
passenv() { for v in "$@"; do if [ -n "${!v:-}" ]; then printf -- "-E\n%s\n" "$v"; fi; done; }
cd "$(dirname "$0")/../../.."
root=$PWD
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-658}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
out=${PAIR_OUT:-/tmp/pair658}
lock=/tmp/gpu27.lock
steps=${PAIR_STEPS:-p2p allreduce tp27b vllm}
model=${PAIR_MODEL:-$root/crates/specrt/tests/fixtures/qwen3.8-27b-nvfp4-exact.toml}
mkdir -p "$out"
log() { echo "[pair658 $(date +%H:%M:%S)] $*" | tee -a "$out/pair.log"; }

cargo_run() {
  systemd-run --user --wait --collect -P -q -p MemoryMax=10G -p WorkingDirectory="$root" \
    -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E HOME cargo "$@"
}

guarded() {
  systemd-run --user --wait --collect -P -q -p MemoryMax=${GUARD_MEM:-16G} -p MemorySwapMax=0 -p WorkingDirectory="$root" \
    -E PATH -E HOME $(passenv SPECRT_CACHE CUDA_VISIBLE_DEVICES) "$@"
}

check_free() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -Ev 'helium|signal|agentbench|lens|Xwayland|quickshell' || true)
  if [ -n "$apps" ]; then
    log "GPU compute apps still running, refusing:"
    echo "$apps" | tee -a "$out/pair.log"
    exit 1
  fi
  if curl -sf -m 2 localhost:8001/v1/models >/dev/null; then
    log "something serves on :8001, refusing"
    exit 1
  fi
}

build() {
  log "building"
  /usr/local/cuda/bin/nvcc -O2 -o "$out/p2p658" "$root/crates/specrt/scripts/p2p658.cu"
  cargo_run test --release -p inference-bench-specrt --test allreduce --no-run 2>&1 | tail -2
  cargo_run test --release -p inference-bench-specrt --test tp --no-run 2>&1 | tail -2
  cargo_run build --release -p inference-bench-specrt --bin tpbench 2>&1 | tail -2
}

bin_of() { ls -t "$CARGO_TARGET_DIR"/release/deps/"$1"-* | grep -v '\.d$' | head -1; }

step_p2p() {
  nvidia-smi topo -m | tee "$out/topo.txt"
  nvidia-smi topo -p2p r | tee -a "$out/topo.txt"
  nvidia-smi topo -p2p w | tee -a "$out/topo.txt"
  guarded "$out/p2p658" | tee "$out/p2p.txt"
}

step_allreduce() {
  SPECRT_CACHE=$out/jit guarded "$(bin_of allreduce)" --ignored --nocapture --test-threads=1 two_gpu 2>&1 | tee "$out/allreduce.txt"
}

step_tp27b() {
  local b=$CARGO_TARGET_DIR/release/tpbench
  for tp in 1 2; do
    for rows in 1 8; do
      SPECRT_CACHE=$out/jit GUARD_MEM=24G guarded "$b" "$model" --tp "$tp" --batch "$rows" --decode 128 --prompt 512 2>&1 | tee -a "$out/tp27b.txt"
    done
    SPECRT_CACHE=$out/jit GUARD_MEM=24G guarded "$b" "$model" --tp "$tp" --batch 1 --decode 8 --prompt 4096 2>&1 | tee -a "$out/tp27b.txt"
  done
  SPECRT_CACHE=$out/jit GUARD_MEM=24G guarded "$b" "$model" --compare 1,2 --prompt 256 2>&1 | tee -a "$out/tp27b.txt"
}

step_vllm() {
  local port=8658
  log "starting vLLM tp=2 on :$port (read-only ~/.venv-vllm)"
  systemd-run --user --collect -q --unit=vllm658 -p MemoryMax=24G -p MemorySwapMax=0 -E HOME -E PATH -E MAX_JOBS=3 -E NVCC_THREADS=2 \
    "$HOME/.venv-vllm/bin/vllm" serve orcarouter/Qwen3.8-27B-Uncensored-NVFP4 --tensor-parallel-size 2 --port "$port" --max-model-len 8192 --gpu-memory-utilization 0.85
  trap 'systemctl --user stop vllm658 2>/dev/null || true' EXIT
  for _ in $(seq 1 180); do curl -sf -m 2 "localhost:$port/v1/models" >/dev/null && break; sleep 5; done
  curl -sf "localhost:$port/v1/models" >/dev/null || { log "vLLM did not come up"; journalctl --user -u vllm658 --no-pager | tail -30; exit 1; }
  python3 - "$port" <<'PY' | tee "$out/vllm.txt"
import json, sys, time, urllib.request, concurrent.futures as cf
port = sys.argv[1]
def run(n, prompt_tokens, max_tokens):
    body = json.dumps({"model": "orcarouter/Qwen3.8-27B-Uncensored-NVFP4", "prompt": "hello " * prompt_tokens, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}).encode()
    def one(_):
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/completions", body, {"Content-Type": "application/json"})))
        return time.time() - t, r["usage"]["completion_tokens"]
    with cf.ThreadPoolExecutor(n) as ex:
        t = time.time()
        res = list(ex.map(one, range(n)))
    wall = time.time() - t
    return wall, sum(c for _, c in res)
run(1, 16, 8)
for n in (1, 8):
    wall, toks = run(n, 512, 128)
    print(f"vllm tp2 batch {n}: {toks / wall:.1f} tok/s total, {toks / wall / n:.1f} tok/s per stream (512 prompt, 128 decode, wall time includes prefill)")
wall, toks = run(1, 4096, 1)
print(f"vllm tp2 4k prefill: {4096 / wall:.0f} tok/s ({wall * 1e3:.0f} ms)")
PY
  systemctl --user stop vllm658
  trap - EXIT
}

build
exec 9>"$lock"
log "waiting for $lock"
flock 9
check_free
log "lock held, steps: $steps"
for s in $steps; do
  log "step $s"
  "step_$s"
done
log "done, results in $out"
