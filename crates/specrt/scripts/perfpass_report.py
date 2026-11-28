import csv
import json
import re
import statistics
import sys
from datetime import date
from pathlib import Path

HEADER = (
    "| family | model | ours decode b1 | ours prefill | ours decode b8 | best competitor | their decode b1 | their prefill | their decode b8 "
    "| decode ratio | prefill ratio | b8 ratio | date | commit | command |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def label(model):
    parts = Path(model).parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") - 1].replace("models--", "").replace("--", "/")
    return Path(model).name


def fnum(x):
    return "n/a" if x is None else f"{x:.1f}"


def ratio(a, b):
    return None if not a or not b else a / b


def fratio(r):
    return "n/a" if r is None else f"{r:.2f}"


def ours_single(path):
    text = Path(path).read_text() if Path(path).exists() else ""
    head = re.search(r"weights=([0-9.]+) GB", text)
    rows = {}
    for m in re.finditer(r"prompt=(\d+) prefill=([0-9.]+) tok/s decode=\d+ tokens ([0-9.]+) tok/s", text):
        rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return float(head.group(1)) if head else None, rows


def ours_batch(path):
    text = Path(path).read_text() if Path(path).exists() else ""
    vals = [float(m.group(1)) for m in re.finditer(r"full-batch decode=([0-9.]+) tok/s", text)]
    steps = [float(m.group(1)) for m in re.finditer(r"over \d+ steps of ([0-9.]+) ms", text)]
    return (statistics.median(vals) if vals else None), (statistics.median(steps) if steps else None)


def llama(raw):
    out = {"prefill": {}, "decode": None, "batch": None}
    p = raw / "llama-prefill.json"
    if p.exists() and p.read_text().strip():
        for e in json.loads(p.read_text()):
            if e.get("n_prompt"):
                out["prefill"][e["n_prompt"]] = e["avg_ts"]
    p = raw / "llama-decode.json"
    if p.exists() and p.read_text().strip():
        for e in json.loads(p.read_text()):
            if e.get("n_gen"):
                out["decode"] = e["avg_ts"]
    p = raw / "llama-batch.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("{"):
                out["batch"] = json.loads(line)["speed_tg"]
    return out


def vllm(raw, name="vllm.json"):
    p = raw / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def kernels(csv_path, top=8):
    p = Path(str(csv_path) + "_cuda_gpu_kern_sum.csv")
    if not p.exists():
        return None, []
    rows = list(csv.DictReader(p.open()))
    total = sum(float(r["Total Time (ns)"]) for r in rows)
    short = []
    for r in rows[:top]:
        name = re.sub(r"\s+", " ", r["Name"])[:70]
        short.append((float(r["Time (%)"]), int(r["Instances"]), float(r["Med (ns)"]) / 1e3, name))
    return total, short


def family_report(family, model, raw, out, commit, roof, batch, lens, command, board):
    raw, out = Path(raw), Path(out)
    roof = float(roof)
    batch = int(batch)
    lens = [int(x) for x in lens.split(",")]
    weights_gb, single = ours_single(raw / "ours-single.log")
    b8, step_ms = ours_batch(raw / "ours-batch.log")
    ll = llama(raw)
    vl = vllm(raw)
    sg = vllm(raw, "sglang.json")
    first = lens[0]
    ours_p = single.get(first, (None, None))[0]
    ours_d = single.get(first, (None, None))[1]
    comp = {}
    if ll["decode"] or ll["prefill"]:
        comp["llama.cpp"] = (ll["decode"], ll["prefill"].get(first), ll["batch"])
    if vl:
        comp["vllm"] = (vl["decode"].get("1"), vl["prefill"].get(str(first)), vl["decode"].get(str(batch)))
    if sg:
        comp["sglang"] = (sg["decode"].get("1"), sg["prefill"].get(str(first)), sg["decode"].get(str(batch)))
    def top(i):
        ranked = [(v[i], k) for k, v in comp.items() if v[i]]
        return max(ranked) if ranked else (None, None)

    theirs = (top(0)[0], top(1)[0], top(2)[0])
    best = "/".join(dict.fromkeys(n for n in (top(0)[1], top(1)[1], top(2)[1]) if n)) or "none"
    today = date.today().isoformat()
    lines = [f"# {family} perf pass {today}", "", f"model: `{model}`, commit `{commit}`, device roofline {roof} GB/s", "", f"command: `{command}`", ""]
    lines += ["## Side by side", "", "| engine | prefill " + " | prefill ".join(map(str, lens)) + " | decode b1 | decode b" + str(batch) + " |", "|---|" + "---|" * (len(lens) + 2)]
    lines.append("| specrt | " + " | ".join(fnum(single.get(n, (None, None))[0]) for n in lens) + f" | {fnum(ours_d)} | {fnum(b8)} |")
    if ll["decode"] or ll["prefill"]:
        lines.append("| llama.cpp | " + " | ".join(fnum(ll["prefill"].get(n)) for n in lens) + f" | {fnum(ll['decode'])} | {fnum(ll['batch'])} |")
    else:
        lines.append("| llama.cpp | " + " | ".join("not run" for _ in lens) + " | not run | not run |")
    if vl:
        lines.append("| vllm | " + " | ".join(fnum(vl["prefill"].get(str(n))) for n in lens) + f" | {fnum(vl['decode'].get('1'))} | {fnum(vl['decode'].get(str(batch)))} |")
    else:
        lines.append("| vllm | " + " | ".join("not run" for _ in lens) + " | not run | not run |")
    if sg:
        lines.append("| sglang | " + " | ".join(fnum(sg["prefill"].get(str(n))) for n in lens) + f" | {fnum(sg['decode'].get('1'))} | {fnum(sg['decode'].get(str(batch)))} |")
    else:
        lines.append("| sglang | " + " | ".join("not run" for _ in lens) + " | not run | not run |")
    lines.append("")
    if weights_gb and ours_d:
        lines += [
            "## Decode roofline",
            "",
            f"weights {weights_gb:.2f} GB, decode b1 {ours_d:.1f} tok/s is {weights_gb * ours_d:.0f} GB/s of weight traffic, {weights_gb * ours_d / roof * 100:.0f}% of {roof} GB/s.",
            "",
        ]
    for tag, title in (("nsys-b1", "decode batch 1"), (f"nsys-b{batch}", f"decode batch {batch}"), (f"nsys-p{lens[0]}", f"prefill {lens[0]}"), (f"nsys-p{lens[-1]}", f"prefill {lens[-1]}")):
        total, ks = kernels(raw / tag)
        if not ks:
            continue
        lines += [f"## Kernels, {title}", "", f"kernel sum {total / 1e6:.2f} ms", "", "| share | launches | median us | kernel |", "|---|---|---|---|"]
        lines += [f"| {s:.1f}% | {n} | {m:.1f} | `{name}` |" for s, n, m, name in ks]
        lines.append("")
    if step_ms:
        total, _ = kernels(raw / f"nsys-b{batch}")
        log = (raw / f"nsys-b{batch}.log").read_text() if (raw / f"nsys-b{batch}.log").exists() else ""
        m = re.search(r"over (\d+) steps", log)
        if total and m:
            per_step = total / 1e6 / int(m.group(1))
            lines += [f"Batch {batch} step: wall {step_ms:.2f} ms, kernel sum {per_step:.2f} ms, gap {step_ms - per_step:.2f} ms.", ""]
    (out / f"{today}.md").write_text("\n".join(lines) + "\n")
    drat = ratio(ours_d, theirs[0])
    prat = ratio(ours_p, theirs[1])
    brat = ratio(b8, theirs[2])
    row = (
        f"| {family} | `{label(model)}` | {fnum(ours_d)} | {fnum(ours_p)} | {fnum(b8)} | {best} | {fnum(theirs[0])} | {fnum(theirs[1])} | {fnum(theirs[2])} "
        f"| {fratio(drat)} | {fratio(prat)} | {fratio(brat)} | {today} | {commit} | `{command}` |"
    )
    (out / f"{today}.row").write_text(row + "\n")
    print("\n".join(lines))
    print(row)
    merge(board, out / f"{today}.row")


SG_MARK = "## SGLang"
CLAIMS_MARK = "## Paired A/B claims"
SG_HEADER = [
    "| family | sglang prefill | sglang decode b1 | sglang decode b8 | prefill ratio | decode ratio | b8 ratio | backend | date |",
    "|---|---|---|---|---|---|---|---|---|",
]
SG_NOTE = (
    "Indicative only: SGLang ran with the radix cache off at batch 8, median of 3, while another job shared the Spark GPU. "
    "Ratios are specrt / SGLang against the row above. `triton` marks runs where the default flashinfer backend failed."
)
PREFIX = ["# Perf scoreboard", "", "Ratios are specrt / best competitor, above 1.00 means we are faster. Rows are replaced in place; the command is beside each row.", ""]


def split_board(board):
    text = Path(board).read_text() if Path(board).exists() else ""
    text = text.partition(CLAIMS_MARK)[0]
    main, _, sg = text.partition(SG_MARK)
    rows = [l for l in main.splitlines() if l.startswith("| ") and not l.startswith("| family")]
    sg_rows = [l for l in sg.splitlines() if l.startswith("| ") and not l.startswith("| family")]
    prefix = [l for l in main.splitlines() if not l.startswith("|")]
    while prefix and not prefix[-1].strip():
        prefix.pop()
    return (prefix or PREFIX[:-1]), rows, sg_rows


def write_board(board, prefix, rows, sg_rows):
    existing = Path(board).read_text() if Path(board).exists() else ""
    claims = existing.partition(CLAIMS_MARK)[2]
    key = lambda l: l.split("|")[1].strip()
    out = prefix + [""] + HEADER.rstrip("\n").splitlines() + sorted(rows, key=key)
    if sg_rows:
        out += ["", SG_MARK, "", SG_NOTE, ""] + SG_HEADER + sorted(sg_rows, key=key)
    if claims:
        out += ["", CLAIMS_MARK + claims.rstrip("\n")]
    Path(board).write_text("\n".join(out) + "\n")


def merge(board, row_path):
    row = Path(row_path).read_text().strip()
    key = row.split("|")[1].strip()
    prefix, rows, sg_rows = split_board(board)
    rows = [l for l in rows if l.split("|")[1].strip() != key]
    rows.append(row)
    write_board(board, prefix, rows, sg_rows)


def merge_sglang(board, family, json_path, backend="flashinfer"):
    prefix, rows, sg_rows = split_board(board)
    sg = json.loads(Path(json_path).read_text())
    mine = next((l for l in rows if l.split("|")[1].strip() == family), None)
    ours = [None, None, None]
    if mine:
        cells = [c.strip() for c in mine.split("|")]
        ours = [float(cells[i]) if re.fullmatch(r"[0-9.]+", cells[i]) else None for i in (3, 4, 5)]
    first = sorted(int(k) for k in sg["prefill"])[0]
    batch = max(int(k) for k in sg["decode"])
    p, d1, d8 = sg["prefill"].get(str(first)), sg["decode"].get("1"), sg["decode"].get(str(batch))
    row = (
        f"| {family} | {fnum(p)} | {fnum(d1)} | {fnum(d8)} | {fratio(ratio(ours[1], p))} | {fratio(ratio(ours[0], d1))} | {fratio(ratio(ours[2], d8))} "
        f"| {backend} | {date.today().isoformat()} |"
    )
    sg_rows = [l for l in sg_rows if l.split("|")[1].strip() != family] + [row]
    write_board(board, prefix, rows, sg_rows)


def ab(log):
    rounds = {}
    for line in Path(log).read_text().splitlines():
        m = re.match(r"round=(\d+) side=(\w) .*prompt=(\d+) prefill=([0-9.]+) tok/s decode=\d+ tokens ([0-9.]+) tok/s", line)
        if m:
            rounds.setdefault((int(m.group(3)), "prefill"), {}).setdefault(m.group(2), []).append(float(m.group(4)))
            rounds.setdefault((int(m.group(3)), "decode"), {}).setdefault(m.group(2), []).append(float(m.group(5)))
    for (length, kind), sides in sorted(rounds.items()):
        a, b = sides.get("A", []), sides.get("B", [])
        n = min(len(a), len(b))
        if n == 0:
            continue
        ratios = sorted(b[i] / a[i] for i in range(n))
        med = statistics.median(ratios)
        q = statistics.quantiles(ratios, n=4) if n >= 4 else [ratios[0], med, ratios[-1]]
        iqr = q[-1] - q[0]
        verdict = "refuse (noisy)" if med and iqr / med > 0.10 else ("keep" if med >= 1.05 else "drop")
        print(f"{kind} len={length}: B/A median {med:.3f} IQR {iqr:.3f} rounds {n} -> {verdict}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "family":
        family_report(*sys.argv[2:12])
    elif cmd == "merge":
        merge(sys.argv[2], sys.argv[3])
    elif cmd == "sglang":
        merge_sglang(*sys.argv[2:6])
    elif cmd == "ab":
        ab(sys.argv[2])
