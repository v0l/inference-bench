#!/bin/bash
D=$(cd "$(dirname "$0")" && pwd)
ROOT=$D/../..
exec 9>/tmp/gpu27.lock
until flock -w 20 9; do echo "waiting for gpu lock $(date -Is)" >>$D/window496.log; done
exec >>$D/window496.log 2>&1
restart() {
  echo "restarting service $(date -Is)"
  systemctl --user stop vllm-bench.service 2>/dev/null
  sleep 5
  sudo systemctl start local-llm.service
}
trap restart EXIT
sudo systemctl stop local-llm.service
sleep 8
nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi vllm && { echo "vllm still on GPU"; exit 1; }
systemd-run --user --unit=vllm-bench --collect -p MemoryMax=24G -p MemorySwapMax=0 \
  --property=StandardOutput=file:$D/vllm-kvauto.log --property=StandardError=file:$D/vllm-kvauto.log \
  $D/launch.sh kvauto
up=0
for i in $(seq 1 180); do
  curl -sf localhost:8011/v1/models >/dev/null && { up=1; break; }
  systemctl --user is-active vllm-bench.service >/dev/null || break
  sleep 5
done
[ $up = 1 ] || { echo "kvauto launch failed"; exit 1; }
grep -E "GPU KV cache size|Maximum concurrency" $D/vllm-kvauto.log | head -3
rm -rf /tmp/vis496
VLLM_URL=http://127.0.0.1:8011 $HOME/.venv-vllm/bin/python $ROOT/crates/specrt/scripts/vllm_vision.py /tmp/vis496 || { echo "reference failed"; exit 1; }
systemctl --user stop vllm-bench.service
sleep 20
cd $ROOT
export CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-307 CARGO_BUILD_JOBS=4
SPECRT_VISION_REF_DIR=/tmp/vis496 nice cargo test --release -p inference-bench-specrt --test vision_lm -- --include-ignored --nocapture --test-threads=1 >$D/vision-lm-kvauto.log 2>&1
echo "tests exit $?"
for kv in 24; do
  nice /mnt/data/cargo-target/inference-bench-307/release/prefixbench crates/specrt/tests/fixtures/qwen3.8-27b.toml long on 8192 $kv 4096,32768,65536,131072,196608,260000 >$D/specrt-long.log 2>&1
  echo "specrt long exit $?"
done
echo "window496 done $(date -Is)"
