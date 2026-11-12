import re
import sqlite3
import sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
con = sqlite3.connect(path)
rows = con.execute(
    "select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start"
).fetchall()


def is_prefill(name):
    return name.endswith(("_big", "_huge")) or "f4a_big" in name or "GemmUniversal" in name or "device_kernel" in name


marks = [(s, e) for n, s, e in rows if is_prefill(n)]
windows = []
for s, e in marks:
    if windows and s - windows[-1][1] < 50_000_000:
        windows[-1][1] = max(windows[-1][1], e)
    else:
        windows.append([s, e])


def cls(name):
    name = re.sub(r"_(big|huge|narrow|tiny)$", "", name)
    return name[:60]


for w0, w1 in windows:
    ks = [(n, s, e) for n, s, e in rows if s >= w0 and e <= w1]
    busy = 0
    cur_s, cur_e = None, None
    for _, s, e in ks:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = w1 - w0
    total = sum(e - s for _, s, e in ks)
    gaps = [ks[i + 1][1] - ks[i][2] for i in range(len(ks) - 1)]
    big_gaps = sum(g for g in gaps if g > 20_000)
    print(f"window {span / 1e6:.1f} ms, busy {busy / 1e6:.1f} ms ({100 * busy / span:.1f}%), kernel sum {total / 1e6:.1f} ms, "
          f"{len(ks)} kernels, idle {(span - busy) / 1e6:.1f} ms, gaps over 20 us {big_gaps / 1e6:.1f} ms")
    agg = {}
    for n, s, e in ks:
        a = agg.setdefault(cls(n), [0, 0])
        a[0] += e - s
        a[1] += 1
    for n, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {100 * t / total:5.1f}%  {t / 1e6:8.2f} ms  n {c:5d}  mean {t / c / 1e3:8.1f} us  {n}")
