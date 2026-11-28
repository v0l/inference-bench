import argparse
import sqlite3
import statistics
from collections import Counter, defaultdict


def kernels(path):
    con = sqlite3.connect(path)
    return con.execute(
        "select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName order by k.start"
    ).fetchall()


def cut(rows, mark):
    marks = [i for i, r in enumerate(rows) if r[0] == mark]
    return [rows[a + 1 : b + 1] for a, b in zip(marks, marks[1:])]


def spec_rounds(rows):
    return cut(rows, "spec_accept")


def plain_steps(rows):
    last = max((i for i, r in enumerate(rows) if r[0] == "spec_accept"), default=-1)
    return cut(rows[last + 1 :], "sample_rows")


def table(units, want, key, top, label, path):
    print(f"{label} rows per unit:", sorted(Counter(u[-1][1] for u in units).items()))
    same = [u for u in units if u[-1][1] == want]
    if not same:
        print(f"{label}: no unit with {want} sequences")
        return
    mode = Counter(key(u) for u in same).most_common(1)[0][0]
    pure = [u for u in same if key(u) == mode]
    wall = statistics.median(u[-1][6] - u[0][5] for u in pure)
    busy = statistics.median(sum(k[6] - k[5] for k in u) for u in pure)
    per = defaultdict(list)
    calls = defaultdict(list)
    grids = {}
    for u in pure:
        acc = defaultdict(int)
        n = Counter()
        for name, gx, gy, gz, bx, s, e in u:
            acc[name] += e - s
            n[name] += 1
            grids[name] = (gx, gy, gz, bx)
        for name in acc:
            per[name].append(acc[name])
            calls[name].append(n[name])
    print(
        f"{label} seqs={want} {path}: units={len(units)} pure={len(pure)} kernels/unit={len(pure[0])} "
        f"wall={wall / 1e6:.2f} ms kernel-sum={busy / 1e6:.2f} ms"
    )
    ranked = sorted(((statistics.median(v), k) for k, v in per.items()), reverse=True)
    for ns, name in ranked[:top]:
        c = statistics.median(calls[name])
        print(
            f"  {ns / 1e6:7.3f} ms {100 * ns / busy:5.1f}%  n={c:4.0f}  {ns / c / 1e3:8.1f} us/call  "
            f"grid={grids[name]}  {name}"
        )


def main():
    ap = argparse.ArgumentParser(description="Per round kernel breakdown from an nsys sqlite export (--cuda-graph-trace=node)")
    ap.add_argument("sqlite")
    ap.add_argument("top", nargs="?", type=int, default=28)
    ap.add_argument("seqs", nargs="?", type=int)
    ap.add_argument("--plain", action="store_true", help="plain decode steps cut at sample_rows instead of MTP rounds")
    a = ap.parse_args()
    rows = kernels(a.sqlite)
    if a.plain:
        units = plain_steps(rows)
        key = lambda u: tuple(sorted(Counter(k[0] for k in u).items()))
        label = "plain"
    else:
        units = spec_rounds(rows)
        key = len
        label = "spec"
    if not units:
        print(f"{label}: no units in {a.sqlite}")
        return
    want = a.seqs if a.seqs is not None else max(u[-1][1] for u in units)
    table(units, want, key, a.top, label, a.sqlite)


if __name__ == "__main__":
    main()
