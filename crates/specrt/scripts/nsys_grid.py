import sqlite3
import statistics
import sys

con = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rows = con.execute(
    "select s.value, d.value, k.gridX, k.gridY, k.gridZ, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
    "join StringIds s on s.id = k.shortName join StringIds d on d.id = k.demangledName"
).fetchall()
groups = {}
for short, dem, gx, gy, gz, dur in rows:
    kind = short
    if "e2m1" in dem and "GemmUniversal" in dem:
        kind = "fp4_gemm"
    elif "e4m3" in dem and "GemmUniversal" in dem:
        kind = "fp8_gemm"
    groups.setdefault((kind, gx, gy, gz), []).append(dur)
total = sum(sum(v) for v in groups.values())
print(f"{sys.argv[1]} kernel_sum_ms={total / 1e6:.2f}")
for key, v in sorted(groups.items(), key=lambda kv: -sum(kv[1]))[:top]:
    print(f"  n={len(v):6d} med={statistics.median(v) / 1e3:9.1f}us tot={sum(v) / 1e6:8.2f}ms {key[0][:40]} grid={key[1:]}")
