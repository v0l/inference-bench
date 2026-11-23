#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/mnt/data/cargo-target/inference-bench-1136}
export SPECRT_REF_DIR=${SPECRT_REF_DIR:-/tmp/ref1136}
export SPECRT_122B_DIR=${SPECRT_122B_DIR:-$(ls -d ~/.cache/huggingface/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4/snapshots/* | head -1)}
base=${BASE:-master}
rounds=${ROUNDS:-3}
export SPECRT_122B_REF=${SPECRT_122B_REF:-$HOME/.cache/specrt-ref-644/vllm122.json}
if [ -z "${SPECRT_122B_PROMPT:-}" ]; then
    export SPECRT_122B_PROMPT=/tmp/ref1136/prompt122.json
    python3 -c "import json,sys;json.dump(json.load(open(sys.argv[1]))['prompt'],open(sys.argv[2],'w'))" "$SPECRT_122B_REF" "$SPECRT_122B_PROMPT"
fi
gpu() { ~/.agents/bin/gpu-run 1 -- "$@"; }
bin() { ls -t "$CARGO_TARGET_DIR"/release/deps/"$1"-* | grep -v '\.d$' | head -1; }
for t in nvfp4_quant moe_route moe_mma; do
    gpu "$(bin "$t")" --include-ignored --test-threads=1 2>&1 | grep -E '^test result|FAILED|panicked'
done
gpu "$(bin variants)" --include-ignored --test-threads=1 gpt_oss 2>&1 | grep -E '^test result|FAILED|panicked'
gpu "$(bin real122b)" --ignored --test-threads=1 real_122b_top1_and_kl_match_vllm 2>&1 | grep -E 'top-1|KL|^test result|FAILED'
if [ ! -x /tmp/ab1136/old ]; then
    wt=$(mktemp -d /tmp/ab1136-base.XXXX)
    git worktree add -q --detach "$wt" "$base"
    (cd "$wt" && CARGO_TARGET_DIR=/mnt/data/cargo-target/inference-bench-1136-base systemd-run --user --wait --collect -P -q -p MemoryMax=10G --working-directory="$wt" -E CARGO_TARGET_DIR -E PATH -E HOME nice -n 19 cargo test --release --no-run -p inference-bench-specrt --test real122b)
    mkdir -p /tmp/ab1136
    cp "$(ls -t /mnt/data/cargo-target/inference-bench-1136-base/release/deps/real122b-* | grep -v '\.d$' | head -1)" /tmp/ab1136/old
    git worktree remove --force "$wt"
fi
new=$(bin real122b)
for r in $(seq "$rounds"); do
    for arm in old new old; do
        exe=/tmp/ab1136/old
        [ "$arm" = new ] && exe=$new
        echo "round $r $arm: $(gpu "$exe" --ignored --exact real_122b_batch_8_decode_speed --nocapture 2>&1 | grep -o '[0-9.]* tok/s')"
    done
done
