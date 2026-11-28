#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
model=${MODEL_DIR:-$HOME/.cache/specrt-models/Qwen3.5-4B-NVFP4}
out=${OUT:-/tmp/gemm-select-4b}
mkdir -p "$out/cache"
spec() {
    sed "s#@CHECKPOINT@#$model#" "$here/qwen3.5-4b-nvfp4.toml.in" | sed "s#^\[runtime\]#[runtime]\ncache_dir = \"$out/cache\"#"
}
spec > "$out/act.toml"
spec | sed '/^act_quant = /d' > "$out/exact.toml"
echo "$out"
