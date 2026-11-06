import json
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
rounds = [str(r) for r in range(1, int(sys.argv[2]) + 1)] if len(sys.argv) > 2 else [str(r) for r in range(1, 7)]
models = {"qwen3": "Qwen3-0.6B", "qwen25": "Qwen2.5-0.5B"}

load = {}
for line in (out / "load.log").read_text().split("\n"):
    parts = line.split()
    if len(parts) == 2:
        load[int(parts[0])] = float(parts[1])
windows = {}
for line in (out.parent / "run.log").read_text().split("\n"):
    m = re.match(r"(\w+) round (\d+) (\w+) (start|end) (\d+)", line)
    if m:
        windows.setdefault((m[1], m[2], m[3]), {})[m[4]] = int(m[5])


def busy(model, r, engine):
    w = windows.get((model, r, engine), {})
    secs = [v for t, v in load.items() if w.get("start", 0) <= t <= w.get("end", -1)]
    return sum(1 for v in secs if v > 0) / len(secs) if secs else None


def med(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def run_file(model, engine, p, r):
    f = out / f"{model}-{engine}-run{p}-r{r}.json"
    try:
        x = json.loads(f.read_text())[0]
    except (OSError, ValueError, IndexError):
        return None
    return None if x.get("error") else x


def batch_file(model, engine, r):
    try:
        return {x["batch"]: x for x in json.loads((out / f"{model}-{engine}-batch-r{r}.json").read_text())}
    except (OSError, ValueError):
        return {}


def llamabench(model, r):
    try:
        rows = json.loads((out / f"{model}-llamabench-r{r}.json").read_text())
    except (OSError, ValueError):
        return {}
    return {(x["n_prompt"], x["n_gen"]): x["avg_ts"] for x in rows}


def llamabatched(model, r):
    rows = {}
    for line in (out / f"{model}-llamabatched-r{r}.jsonl").read_text().split("\n"):
        if line.startswith("{"):
            x = json.loads(line)
            rows[x["pl"]] = x["speed_tg"]
    return rows


def specrtbench(model, r):
    text = (out / f"{model}-specrtbench-r{r}.txt").read_text()
    res = {}
    for m in re.finditer(r"prompt=(\d+) prefill=([\d.]+) tok/s decode=\d+ tokens ([\d.]+)", text):
        res[("prefill", int(m[1]))] = float(m[2])
        res[("decode", int(m[1]))] = float(m[3])
    for m in re.finditer(r"batch=(\d+) .* decode=([\d.]+) tok/s aggregate", text):
        res[("batch", int(m[1]))] = float(m[2])
    return res


summary = {}
for key, name in models.items():
    if not (out / f"{key}-vllm-run128-r1.json").exists():
        continue
    s = {"http": {}, "native": {}, "batch_http": {}, "batch_native": {}, "busy_27b": {}}
    for engine, label in [("vllm", "vLLM"), ("llama", "llama.cpp"), ("specrt", "specrt")]:
        s["busy_27b"][label] = [busy(key, r, engine) for r in rounds]
        for p in (128, 1024):
            runs = [run_file(key, engine, p, r) for r in rounds]
            s["http"][f"{label} @{p}"] = {
                "prefill_tok_s": med([x and x["prefill_tok_s"] for x in runs]),
                "decode_tok_s": med([x and x["decode_tok_s"] for x in runs]),
                "ttft_ms": med([x and x["ttft_secs"] * 1000 for x in runs]),
                "n": sum(1 for x in runs if x),
            }
        if engine != "specrt":
            for n in (1, 4, 16):
                rows = [batch_file(key, engine, r).get(n) for r in rounds]
                s["batch_http"][f"{label} x{n}"] = {
                    "decode_tok_s": med([x and x["decode_tok_s"] for x in rows]),
                    "throughput_tok_s": med([x and x["throughput_tok_s"] for x in rows]),
                    "ttft_ms": med([x and x["median_ttft_secs"] * 1000 for x in rows]),
                }
    lb = [llamabench(key, r) for r in rounds]
    sb = [specrtbench(key, r) for r in rounds]
    lbb = [llamabatched(key, r) for r in rounds]
    for p in (128, 1024):
        s["native"][f"prefill @{p}"] = {
            "llama.cpp": med([x.get((p, 0)) for x in lb]),
            "specrt": med([x.get(("prefill", p)) for x in sb]),
        }
    s["native"]["decode 128"] = {
        "llama.cpp": med([x.get((0, 128)) for x in lb]),
        "specrt": med([x.get(("decode", 128)) for x in sb]),
    }
    for n in (1, 4, 16):
        s["batch_native"][f"x{n}"] = {
            "llama.cpp": med([x.get(n) for x in lbb]),
            "specrt": med([x.get(("batch", n)) for x in sb]),
        }
    summary[name] = s

print(json.dumps(summary, indent=2))
