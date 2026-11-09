import sqlite3
import statistics
import sys

path, steps = sys.argv[1], int(sys.argv[2])
con = sqlite3.connect(path)
rows = con.execute(
    "select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
    "join StringIds s on s.id = k.shortName order by k.start"
).fetchall()
groups = {}
for name, gx, gy, gz, bx, start, end in rows:
    groups.setdefault((name, gx, gy, gz, bx), []).append(end - start)
per_step = {k: v for k, v in groups.items() if len(v) >= steps * 0.9 and not k[0].endswith(("_big", "_narrow"))}
median_sum = sum(statistics.median(v) * len(v) for v in per_step.values()) / steps
mean_sum = sum(sum(v) for v in per_step.values()) / steps
count = sum(len(v) for v in per_step.values()) / steps
small = sum(statistics.median(v) * len(v) for v in per_step.values() if statistics.median(v) < 20000) / steps
gaps = [rows[i + 1][5] - rows[i][6] for i in range(len(rows) - 1)]
gaps = [g for g in gaps if g >= 0]
tight = sum(g for g in gaps if g < 20000) / steps
loose = sum(g for g in gaps if g >= 20000) / steps
span = (rows[-1][6] - rows[0][5]) / steps
print(f"{path} steps={steps}")
print(f"  kernels/step {count:.0f}  median-sum {median_sum / 1e6:.2f} ms  mean-sum {mean_sum / 1e6:.2f} ms  span {span / 1e6:.2f} ms")
print(f"  kernels under 20 us {small / 1e6:.2f} ms/step  gaps under 20 us {tight / 1e6:.2f} ms/step  gaps over 20 us {loose / 1e6:.2f} ms/step")
