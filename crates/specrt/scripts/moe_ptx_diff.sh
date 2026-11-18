#!/bin/bash
set -e
base=${1:-origin/master}
here=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
trap 'git -C "$here" worktree remove --force "$work/base" 2>/dev/null; rm -rf "$work"' EXIT
git -C "$here" worktree add --detach "$work/base" "$base" >/dev/null
cp "$here/crates/specrt/tests/moe_ptx.rs" "$work/base/crates/specrt/tests/moe_ptx.rs"
run() {
    (cd "$1" && MOE_PTX_OUT="$2" cargo test -q -p inference-bench-specrt --test moe_ptx -- --include-ignored >/dev/null)
}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$work/target}
run "$work/base" "$work/old"
run "$here" "$work/new"
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
sys.exit(1 if bad else 0)
PY
