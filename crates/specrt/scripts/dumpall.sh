#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$here/..
bin=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench}/release/logitdump
g=$HOME/.cache/specrt-gguf
for m in "$root/specs/qwen3-0.6b.toml" "$root/specs/qwen3-0.6b-q8.toml" "$root/specs/llama-3.2-1b.toml" \
    "$g/Qwen3-0.6B-Q4_K_M.gguf" "$g/Qwen3-0.6B-Q8_0.gguf" "$g/Llama-3.2-1B-Instruct-Q4_K_M.gguf"; do
    "$bin" "$m" 17,300,1100 4
done
for m in tiny-qwen3.5 tiny-qwen3.5-fp8 tiny-qwen3.5-nvfp4; do
    "$bin" "$root/tests/fixtures/$m" 17,300,900 4
done
