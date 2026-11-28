#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ "${1:-}" == "--fetch" ]]; then
  python3 crates/specrt/scripts/popular_fetch.py
elif [[ ! -d "${POPULAR_CACHE:-$HOME/.cache/specrt-popular}/files" ]]; then
  python3 crates/specrt/scripts/popular_fetch.py --reuse-date
fi
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-coverage}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
POPULAR_WRITE=1 nice cargo test -q -p inference-bench-specrt --test popular_models -- --ignored --exact popular_model_coverage_report > /dev/null
cargo test -q -p inference-bench-specrt --test popular_models
echo "wrote docs/generated/model-coverage.md and crates/specrt/tests/fixtures/popular/classes.tsv"
