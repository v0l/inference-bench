#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
out=${1:-/tmp/kernel-vs}
doc=${2:-$here/docs/perf/kernel-vs-tables.md}
only=${KERNEL_VS_ONLY:-}
iters=${KERNEL_VS_ITERS:-200}
venv=${VENV:-$HOME/.venv-vllm}
target=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-kernel-vs}
gpu=${KERNEL_VS_GPU:-1}
mkdir -p "$out"
commit=$(git -C "$here" rev-parse --short HEAD)$(git -C "$here" diff --quiet HEAD -- crates || echo "+dirty")
cmd="crates/specrt/scripts/kernel_vs.sh $*"
rm -rf "$out/ours"
(cd "$here" && systemd-run --user --wait --collect -P -q -p MemoryMax=10G --setenv=CUDA_VISIBLE_DEVICES= --setenv=CARGO_TARGET_DIR="$target" --setenv=CARGO_BUILD_JOBS=4 \
    --setenv=KERNEL_VS_OUT="$out/ours" --working-directory="$here" \
    nice -n 19 cargo test --release -q -p inference-bench-specrt --test kernel_vs -- --ignored dump_vs_kernels)
env=(--setenv=FLASHINFER_WORKSPACE_BASE="$out/fi" --setenv=FLASHINFER_CUDA_ARCH_LIST=12.0f --setenv=TRITON_CACHE_DIR="$out/triton" --setenv=MAX_JOBS=2 --setenv=FLASHINFER_NVCC_THREADS=2)
systemd-run --user --wait --collect -P -q -p MemoryMax=14G -p MemorySwapMax=0 "${env[@]}" --working-directory="$here" \
    nice -n 19 "$venv/bin/python" "$here/crates/specrt/scripts/kernel_vs.py" prepare
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU$gpu free: $free MiB"
"$HOME/.agents/bin/gpu-run" "$gpu" --min-free 20000 -- systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 "${env[@]}" \
    --setenv=CUDA_VISIBLE_DEVICES="$gpu" --working-directory="$here" \
    "$venv/bin/python" "$here/crates/specrt/scripts/kernel_vs.py" run "$out/ours" "$out/kernel-vs.json" --iters "$iters" --only "$only" --commit "$commit" --command "$cmd" || status=$?
"$venv/bin/python" "$here/crates/specrt/scripts/kernel_vs.py" report "$out/kernel-vs.json" "$doc"
echo "wrote $doc"
exit "${status:-0}"
