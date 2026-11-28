#!/bin/bash
D=$(cd "$(dirname "$0")" && pwd)
exec 9>/tmp/gpu27.lock
until flock -w 20 9; do echo "waiting for gpu lock $(date -Is)" >>$D/window.log; done
restart() {
  echo "restarting service $(date -Is)" >>$D/window.log
  systemctl --user stop vllm-bench.service 2>/dev/null
  sleep 5
  sudo systemctl start local-llm.service
}
trap restart EXIT
sudo systemctl stop local-llm.service
sleep 8
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >>$D/window.log
if nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi vllm; then
  echo "vllm still on GPU" >>$D/window.log
  exit 1
fi
bash $D/window.sh
