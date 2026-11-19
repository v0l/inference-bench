#!/bin/bash
set -e
base=${1:-origin/master}
here=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
trap 'git -C "$here" worktree remove --force "$work/base" 2>/dev/null; rm -rf "$work"' EXIT
git -C "$here" worktree add --detach "$work/base" "$base" >/dev/null
cp "$here/crates/specrt/tests/moe_ptx.rs" "$work/base/crates/specrt/tests/moe_ptx.rs"
base_target=${BASE_TARGET_DIR:-$work/target-base}
here_target=${HERE_TARGET_DIR:-$work/target-here}
run() {
    local tree=$1 out=$2 target=$3
    find "$tree/crates/kir/src" "$tree/crates/specrt/src" -name '*.rs' -o -name '*.cu' | xargs touch
    rm -rf "$out"
    (cd "$tree" && CARGO_TARGET_DIR="$target" MOE_PTX_OUT="$out" cargo test -q -p inference-bench-specrt --test moe_ptx -- --include-ignored >/dev/null)
    local n
    n=$(find "$out" -name '*.ptx' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "no PTX written to $out"; exit 2; }
    local newest_src oldest_ptx
    newest_src=$(find "$tree/crates/kir/src" "$tree/crates/specrt/src" -type f -printf '%T@\n' | sort -n | tail -1)
    oldest_ptx=$(find "$out" -name '*.ptx' -printf '%T@\n' | sort -n | head -1)
    awk -v p="$oldest_ptx" -v s="$newest_src" 'BEGIN { exit !(p >= s) }' || { echo "PTX in $out is older than the source"; exit 2; }
}
run "$work/base" "$work/old" "$base_target"
run "$here" "$work/new" "$here_target"
python3 - "$work/old" "$work/new" <<'PY'
import os, re, sys
def funcs(p):
    t = open(p).read()
    parts = re.split(r'(?=\.visible \.entry |\.func )', t)
    out = {}
    for x in parts[1:]:
        m = re.match(r'\.(?:visible \.entry|func)\s+(?:\([^)]*\)\s*)?(\S+?)\(', x)
        out[m.group(1)] = x
    return out
bad = 0
total = 0
for f in sorted(os.listdir(sys.argv[1])):
    a, b = funcs(os.path.join(sys.argv[1], f)), funcs(os.path.join(sys.argv[2], f))
    for n in sorted(set(a) | set(b)):
        total += 1
        if a.get(n) != b.get(n):
            bad += 1
            print("DIFF", f, n)
print(f"{total} functions compared, {bad} differ")
if total == 0:
    print("no functions parsed")
    sys.exit(2)
sys.exit(1 if bad else 0)
PY
