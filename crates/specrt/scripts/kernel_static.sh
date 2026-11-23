#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
out=${1:-/tmp/kernel-static}
doc=${2:-$here/docs/perf/kernel-static-tables.md}
gpus=${KERNEL_STATIC_GPUS:-sm_120,sm_121,sm_75}
venv=${VENV:-$HOME/.venv-vllm}
target=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-kernel-static}
export CUDA_VISIBLE_DEVICES=
arches=$(echo "$gpus" | sed 's/sm_12\([0-9]\)/sm_12\1a/g')
mkdir -p "$out"
rm -rf "$out/ours"
(cd "$here" && systemd-run --user --wait --collect -P -q -p MemoryMax=14G --setenv=CUDA_VISIBLE_DEVICES= --setenv=CARGO_TARGET_DIR="$target" --setenv=CARGO_BUILD_JOBS=4 \
    --setenv=KERNEL_STATIC_OUT="$out/ours" --setenv=KERNEL_STATIC_GPUS="$gpus" --working-directory="$here" \
    cargo test --release -q -p inference-bench-specrt --test kernel_static -- --ignored dump_static_kernels)
systemd-run --user --wait --collect -P -q -p MemoryMax=14G -p MemorySwapMax=0 --setenv=CUDA_VISIBLE_DEVICES= --working-directory="$here" \
    "$venv/bin/python" "$here/crates/specrt/scripts/kernel_static_theirs.py" "$out/theirs" "$arches"
"$venv/bin/python" "$here/crates/specrt/scripts/kernel_static_triton.py" "$out/theirs" "$arches" >/dev/null
site=$("$venv/bin/python" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
rm -rf "$out/vllm"
mkdir -p "$out/vllm/C" "$out/vllm/moe"
(cd "$out/vllm/C" && cuobjdump -xelf all "$site/_C_stable_libtorch.abi3.so" >/dev/null)
(cd "$out/vllm/moe" && cuobjdump -xelf all "$site/_moe_C_stable_libtorch.abi3.so" >/dev/null)
python3 "$here/crates/specrt/scripts/kernel_static.py" "$out" "$doc" "$out/vllm"
echo "wrote $doc"
