#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../gate776_guard.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

printf '0, GPU-aaa, 2399\n1, GPU-bbb, 1026\n' >"$tmp/gpus"
export GATE_GPUS_FILE="$tmp/gpus" GATE_APPS_FILE="$tmp/apps"

printf 'GPU-bbb, 1368070, VLLM::Worker, 1016\n' >"$tmp/apps"
status=0
out=$(require_idle_gpus 2>&1) || status=$?
[ "$status" -eq 1 ] || fail "vLLM worker on GPU 1 must refuse with 1, got $status"
grep -q 'another process holds GPU 1 (pid 1368070, VLLM::Worker, 1016 MiB): free both GPUs first' <<<"$out" || fail "message: $out"

printf 'GPU-aaa, 18982, /opt/helium/helium --type=gpu-process --flag, 178\nGPU-aaa, 19808, agentbench, 249\n' >"$tmp/apps"
require_idle_gpus || fail "small desktop apps must not block"

: >"$tmp/apps"
require_idle_gpus || fail "an idle machine must pass"

printf '0, GPU-aaa, 61000\n1, GPU-bbb, 1026\n' >"$tmp/gpus"
status=0
out=$(require_idle_gpus 2>&1) || status=$?
[ "$status" -eq 1 ] || fail "61 GB used must refuse"
grep -q 'GPU 0 has 61000 MiB in use' <<<"$out" || fail "message: $out"

echo "gate776 guard: ok"
