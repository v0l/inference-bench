#!/usr/bin/env bash
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
branch_target=${BRANCH_TARGET:-/mnt/data/cargo-target/inference-bench-620}
master_target=${MASTER_TARGET:-/mnt/data/cargo-target/inference-bench-620m}
master_tree=${MASTER_TREE:-$repo/../inference-bench-gate620-master}
ref_tiny=${SPECRT_REF_TINY:-/tmp/ref387}
ref_real=${SPECRT_REF_REAL:-$HOME/.cache/specrt-ref-375}
spec27=${SPEC27:-tests/fixtures/qwen3.8-27b-nvfp4-exact.toml}
max_slower_pct=${MAX_SLOWER_PCT:-2}
runs=${GATE_RUNS:-3}
out=${OUT:-$here/../out/gate620}
mkdir -p "$out"
summary=$out/summary.txt
: > "$summary"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}

cargo_run() {
    systemd-run --user --wait --collect -P -q -p MemoryMax=16G -p MemorySwapMax=0 \
        -E CARGO_TARGET_DIR -E CARGO_BUILD_JOBS -E PATH -E HOME -E SPECRT_REF_DIR -E SPECRT_GDN_TOKENS \
        -p "WorkingDirectory=$PWD" "$@"
}

record() {
    printf '%-44s %s\n' "$1" "$2" | tee -a "$summary"
}

gate() {
    local name=$1
    shift
    if "$@" > "$out/$name.log" 2>&1; then
        record "$name" PASS
    else
        record "$name" "FAIL (see $out/$name.log)"
    fi
}

ratio_gate() {
    local name=$1 branch_val=$2 master_val=$3 higher_is_better=$4
    awk -v n="$name" -v b="$branch_val" -v m="$master_val" -v hib="$higher_is_better" -v lim="$max_slower_pct" 'BEGIN {
        if (b == "" || m == "" || b == 0 || m == 0) { printf "%-44s FAIL (no numbers: branch=%s master=%s)\n", n, b, m; exit 1 }
        r = hib ? b / m : m / b
        slower = (1 - r) * 100
        printf "%-44s %s ratio branch/master %.4f (%.2f%% slower)\n", n, (slower <= lim ? "PASS" : "FAIL"), r, slower
        exit (slower <= lim ? 0 : 1)
    }' | tee -a "$summary"
}

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 36000 ]; then
    echo "only $free MiB free, need 36000" >&2
    exit 1
fi
if curl -s -m 3 localhost:8001/metrics | grep -q "^inference_running_seqs [1-9]"; then
    echo "the app on :8001 is serving, stop here" >&2
    exit 1
fi

if [ ! -d "$master_tree" ]; then
    git -C "$repo" worktree add -q --detach "$master_tree" master || exit 1
fi
if [ -e "$master_tree/.git" ]; then
    git -C "$master_tree" checkout -q --detach master || exit 1
fi

for refdir in "$ref_tiny" "$ref_real"; do
    [ -d "$refdir" ] || { echo "missing reference dir $refdir (see project notes: reference.py)" >&2; exit 1; }
done

echo "== build (outside the lock)" | tee -a "$summary"
(cd "$repo" && CARGO_TARGET_DIR=$branch_target cargo_run cargo build --release -p inference-bench-specrt --bin bench) || exit 1
(cd "$master_tree" && CARGO_TARGET_DIR=$master_target cargo_run cargo build --release -p inference-bench-specrt --bin bench) || exit 1
(cd "$repo" && CARGO_TARGET_DIR=$branch_target cargo_run cargo test --release -p inference-bench-specrt --no-run) || exit 1
(cd "$master_tree" && CARGO_TARGET_DIR=$master_target cargo_run cargo test --release -p inference-bench-specrt --test gdn_speed --no-run) || exit 1

exec 9>/tmp/gpu27.lock
flock 9
export GPU27_LOCK_HELD=1
echo "== gates $(date '+%F %T') branch $(git -C "$repo" rev-parse --short HEAD) master $(git -C "$master_tree" rev-parse --short HEAD) free ${free} MiB" | tee -a "$summary"

run_tests() {
    local tree=$1 target=$2 refdir=$3
    shift 3
    (cd "$tree/crates/specrt" && CARGO_TARGET_DIR=$target SPECRT_REF_DIR=$refdir cargo_run cargo test --release -p inference-bench-specrt "$@")
}

echo "== paired gdn_speed (branch vs master, $runs runs each, interleaved)" | tee -a "$summary"
export SPECRT_GDN_TOKENS=${SPECRT_GDN_TOKENS:-1024,4096}
: > "$out/gdn_branch.log"
: > "$out/gdn_master.log"
for i in $(seq "$runs"); do
    run_tests "$master_tree" "$master_target" "$ref_tiny" --test gdn_speed -- --ignored --nocapture --test-threads=1 >> "$out/gdn_master.log" 2>&1
    run_tests "$repo" "$branch_target" "$ref_tiny" --test gdn_speed -- --ignored --nocapture --test-threads=1 >> "$out/gdn_branch.log" 2>&1
done
for tok in ${SPECRT_GDN_TOKENS//,/ }; do
    for k in prep scan; do
        b=$(grep "gdn tokens=$tok:" "$out/gdn_branch.log" | grep -oE "$k min [0-9.]+ ms median [0-9.]+" | awk '{print $3}' | sort -n | head -1)
        m=$(grep "gdn tokens=$tok:" "$out/gdn_master.log" | grep -oE "$k min [0-9.]+ ms median [0-9.]+" | awk '{print $3}' | sort -n | head -1)
        ratio_gate "gdn_speed tokens=$tok $k (min ms)" "$b" "$m" 0
    done
done

gate kir_speed run_tests "$repo" "$branch_target" "$ref_tiny" --test kir_speed -- --ignored --nocapture --test-threads=1

echo "== paired MTP speed (bench --speculative on the 27B, batches 1 4 8)" | tee -a "$summary"
for batch in 1 4 8; do
    : > "$out/mtp_b$batch.branch.log"
    : > "$out/mtp_b$batch.master.log"
    for i in $(seq "$runs"); do
        (cd "$master_tree/crates/specrt" && "$master_target/release/bench" --runs 3 --speculative "$spec27" 8192 128 "$batch") >> "$out/mtp_b$batch.master.log" 2>&1
        (cd "$repo/crates/specrt" && "$branch_target/release/bench" --runs 3 --speculative "$spec27" 8192 128 "$batch") >> "$out/mtp_b$batch.branch.log" 2>&1
    done
    b=$(grep -oE "decode=[0-9.]+ tok/s aggregate" "$out/mtp_b$batch.branch.log" | grep -oE "[0-9.]+" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    m=$(grep -oE "decode=[0-9.]+ tok/s aggregate" "$out/mtp_b$batch.master.log" | grep -oE "[0-9.]+" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    ratio_gate "mtp_speed batch=$batch (tok/s median)" "$b" "$m" 1
done

gate linear run_tests "$repo" "$branch_target" "$ref_tiny" --test linear -- --include-ignored --skip real_27b --test-threads=1
gate mtp_greedy_identity_chi_square run_tests "$repo" "$branch_target" "$ref_tiny" --test mtp -- --include-ignored --skip real_27b --test-threads=1
gate tiny27b_all_27b_paths run_tests "$repo" "$branch_target" "$ref_tiny" --test tiny27b -- --include-ignored --skip real_27b --test-threads=1
gate serving run_tests "$repo" "$branch_target" "$ref_tiny" --test serving -- --include-ignored --skip real_27b --test-threads=1
gate truncation run_tests "$repo" "$branch_target" "$ref_tiny" --test truncation -- --include-ignored --skip real_27b --test-threads=1
gate real_27b_nvfp4_vs_vllm run_tests "$repo" "$branch_target" "$ref_real" --test real27b -- --ignored --exact real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs --test-threads=1
gate real_27b_mtp_acceptance run_tests "$repo" "$branch_target" "$ref_real" --test real27b -- --ignored --exact real_27b_mtp_acceptance_on_vllms_own_greedy_tokens --test-threads=1

echo "== summary" 
cat "$summary"
if grep -q FAIL "$summary"; then
    echo "gate620: FAIL"
    exit 1
fi
echo "gate620: PASS"
