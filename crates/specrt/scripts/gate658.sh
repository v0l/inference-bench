#!/usr/bin/env bash
set -euo pipefail
passenv() { for v in "$@"; do if [ -n "${!v:-}" ]; then printf -- "-E\n%s\n" "$v"; fi; done; }
cd "$(dirname "$0")/../../.."
root=$PWD
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-658}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
mode=${1:-virtual}
mem_cap=${GATE_MEMORY_MAX:-10G}
min_avail_gb=${GATE_MIN_AVAIL_GB:-0}
lock=/tmp/gpu27.lock
log() { echo "[gate658 $(date +%H:%M:%S)] $*"; }

wait_for_memory() {
  [ "$min_avail_gb" -gt 0 ] || return 0
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge "$min_avail_gb" ] && { log "MemAvailable ${avail} GB"; return 0; }
    log "MemAvailable ${avail} GB below ${min_avail_gb} GB, waiting"
    sleep 60
  done
}

cargo_run() {
  systemd-run --user --wait --collect -P -q -p MemoryMax="$mem_cap" -p MemorySwapMax=0 -p WorkingDirectory="$root" \
    -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E HOME $(passenv CUDA_VISIBLE_DEVICES SPECRT_TP_DEVICE SPECRT_CACHE) \
    cargo "$@"
}

suites=${GATE_SUITES:-allreduce tp}

build_tests() {
  for s in $suites; do cargo_run test --release -p inference-bench-specrt --test "$s" --no-run; done
}

virtual_tests() {
  for s in $suites; do
    case "$s" in
    allreduce) cargo_run test --release -p inference-bench-specrt --test allreduce -- --ignored --nocapture --test-threads=1 --skip two_gpu ;;
    *) cargo_run test --release -p inference-bench-specrt --test "$s" -- --ignored --nocapture --test-threads=1 ;;
    esac
  done
}

case "$mode" in
virtual)
  log "building"
  build_tests
  wait_for_memory
  exec 9>"$lock"
  log "waiting for $lock"
  flock 9
  wait_for_memory
  log "running virtual-rank tests"
  virtual_tests
  log "done"
  ;;
pair)
  exec "$root/crates/specrt/scripts/pair658.sh"
  ;;
*) echo "usage: $0 virtual|pair"; exit 1 ;;
esac
