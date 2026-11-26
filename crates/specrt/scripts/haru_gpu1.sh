#!/usr/bin/env bash
set -euo pipefail
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "run under gpu-run 1" >&2; exit 2; }
[ "$CUDA_VISIBLE_DEVICES" != "0" ] || { echo "refusing GPU0" >&2; exit 2; }
exec systemd-run --user --wait --collect -P -q -p MemoryMax=${MEMORY_MAX:-16G} -p MemorySwapMax=0 \
  -E CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" -E HF_HUB_OFFLINE=1 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --working-directory="$PWD" "$@"
