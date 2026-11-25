import json
import sys
from pathlib import Path

ours, theirs = Path(sys.argv[1]), Path(sys.argv[2])
total = same = 0
gaps = []
ok = True
for f in sorted(ours.glob("p*.json")):
    a = json.loads(f.read_text())["choices"][0]["logprobs"]
    b = json.loads((theirs / f.name).read_text())["choices"][0]["logprobs"]
    n = min(len(a["tokens"]), len(b["tokens"]))
    first = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), n)
    for i in range(first):
        ta, tb = a["top_logprobs"][i], b["top_logprobs"][i]
        gaps += [abs(ta[t] - tb[t]) for t in ta.keys() & tb.keys()]
    matched = sum(a["tokens"][i] == b["tokens"][i] for i in range(n))
    total += n
    same += matched
    ok &= first >= 16
    print(f"{f.stem}: first divergence at {first} of {n}, top-1 equal {matched}/{n}")
gaps.sort()
share = same / max(total, 1)
p99 = gaps[int(0.99 * (len(gaps) - 1))] if gaps else 0.0
print(f"top-1 equal {same}/{total} ({100 * share:.1f}%), logprob gap over shared top-20 before divergence: max {max(gaps, default=0):.4f}, p99 {p99:.4f}, n {len(gaps)}")
ok &= share >= 0.95 and max(gaps, default=0) < 0.1
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
