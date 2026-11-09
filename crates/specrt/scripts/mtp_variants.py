import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mtp_diff


def context(d):
    full = d["prompt"] + d["gen"]
    table = {}
    steps = []
    for c in d["calls"]:
        n = len(c["input_ids"])
        pos = c["positions"][0] if c["positions"].dim() > 1 else c["positions"]
        for j in range(n):
            p = int(pos[j])
            if p + 1 >= len(full) or int(c["input_ids"][j]) != full[p + 1]:
                break
            table.setdefault(p, c["hidden"][j])
        if n > 1 and len(steps) < 10**6:
            m = 0
            while m < n and int(pos[m]) + 2 < len(full) and int(c["input_ids"][m]) == full[int(pos[m]) + 1]:
                m += 1
            if m and c is not d["calls"][0]:
                steps.append((int(pos[m - 1]), int(c["argmax"][m - 1])))
    return full, table, steps


def main():
    ref = Path(sys.argv[1])
    prompts = [int(x) for x in sys.argv[2].split(",")]
    variants = sys.argv[3].split(",")
    runs = {v: mtp_diff.build(v) for v in variants}
    totals = {v: [0, 0] for v in variants + ["vllm"]}
    for i in prompts:
        d = torch.load(ref / f"vllm-mtp-{i}.pt")
        full, table, steps = context(d)
        top = max(p for p, _ in steps)
        missing = [p for p in range(top + 1) if p not in table]
        assert not missing, (i, missing[:5])
        hidden = torch.stack([table[p] for p in range(top + 1)])
        call = {"input_ids": torch.tensor(full[1 : top + 2]), "positions": torch.arange(top + 1), "hidden": hidden}
        for v in variants:
            logits = mtp_diff.forward(runs[v], call)
            for p, vl in steps:
                totals[v][0] += int(logits[p].argmax()) == full[p + 2]
                totals[v][1] += 1
        for p, vl in steps:
            totals["vllm"][0] += vl == full[p + 2]
            totals["vllm"][1] += 1
    for v, (h, n) in totals.items():
        print(v, h, n, round(h / max(n, 1), 4), flush=True)


main()
