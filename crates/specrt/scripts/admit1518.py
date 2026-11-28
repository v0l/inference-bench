import json
import sys


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, round(p * (len(v) - 1)))] if v else float("nan")


report = json.load(open(sys.argv[1]))
queue = []
for line in open(sys.argv[2]):
    try:
        f = json.loads(line).get("fields", {})
    except ValueError:
        continue
    if f.get("message") == "generation finished" and f.get("prompt_tokens", 0) > 100:
        queue.append(f["queue_ms"] / 1e3)
rows = report["per_request"]
long = [r["ttft_secs"] for r in rows if r["prompt_tokens"] > 12000]
ttft = [r["ttft_secs"] for r in rows]
tok = [r["decode_tok_s"] for r in rows if r["completion_tokens"] > 8]
print(json.dumps({
    "arm": sys.argv[3],
    "ok": len(rows),
    "failed": report["failed"],
    "queue_p50": pct(queue, 0.5),
    "queue_p95": pct(queue, 0.95),
    "ttft_p50": pct(ttft, 0.5),
    "ttft_p95": pct(ttft, 0.95),
    "long_n": len(long),
    "long_ttft_p95": pct(long, 0.95),
    "decode_tok_s_p50": pct(tok, 0.5),
    "itl_p95": report["itl_p95"],
    "wall": report["wall_secs"],
}))
