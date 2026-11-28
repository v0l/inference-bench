#!/bin/bash
D=$(cd "$(dirname "$0")" && pwd)
PY=$HOME/.venv-vllm/bin/python
B=$D/../../crates/specrt/scripts/vllm_baseline.py
LOG=$D/window.log
exec >>"$LOG" 2>&1

restart_service() {
  systemctl --user stop vllm-bench.service 2>/dev/null
  sleep 5
  sudo systemctl start local-llm.service
  echo "service start requested $(date -Is)"
}

state() {
  echo "== $1 $(date -Is)"
  nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | cut -c1-80
}

launch() {
  local mode=$1
  systemctl --user stop vllm-bench.service 2>/dev/null
  systemd-run --user --unit=vllm-bench --collect -p MemoryMax=24G -p MemorySwapMax=0 \
    --property=StandardOutput=file:$D/vllm-$mode.log --property=StandardError=file:$D/vllm-$mode.log \
    $D/launch.sh $mode
  for i in $(seq 1 180); do
    curl -sf localhost:8011/v1/models >/dev/null && return 0
    systemctl --user is-active vllm-bench.service >/dev/null || return 1
    sleep 5
  done
  return 1
}

state "window start"


for mode in mtp nomtp; do
  launch $mode || { echo "launch $mode failed"; continue; }
  state "$mode up"
  grep -E "GPU KV cache size|Maximum concurrency|vllm_version|version" $D/vllm-$mode.log | head -5
  $PY $B --base http://127.0.0.1:8011 --label vllm-$mode --suite single --lengths 128,1024,4096,16384,32768,65536,131072,196608,260000 --out $D/single-$mode.json
  $PY $B --base http://127.0.0.1:8011 --label vllm-$mode --suite concurrent --sizes 1,4,8,16,32 --conc-len 128 --out $D/concurrent-$mode.json
  $PY $B --base http://127.0.0.1:8011 --label vllm-$mode --suite concurrent --sizes 4 --conc-len 65536 --out $D/concurrent64k-$mode.json
  $PY $B --base http://127.0.0.1:8011 --label vllm-$mode --suite prefix --lengths 8192,131072 --runs 3 --out $D/prefix-$mode.json
  $PY $B --base http://127.0.0.1:8011 --label vllm-$mode --suite held --lengths 131072,260000 --out $D/held-$mode.json
  state "$mode done"
  systemctl --user stop vllm-bench.service
  sleep 15
done
echo "window done $(date -Is)"
