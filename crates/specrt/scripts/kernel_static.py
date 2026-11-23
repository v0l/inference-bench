import json
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(sys.argv[1])
DOC = Path(sys.argv[2]) if len(sys.argv) > 2 else None
VLLM = Path(sys.argv[3]) if len(sys.argv) > 3 else None

GPUS = {
    "sm_120": {"name": "RTX PRO 6000 Blackwell Max-Q", "host": "desktop", "arch": "sm_120a", "sms": 188, "clock_ghz": 3.09, "bw_gbs": 1792, "regs_sm": 65536, "smem_sm": 100 * 1024, "threads_sm": 1536, "blocks_sm": 32, "bf16_tflops": 439, "fp4_tflops": 1755, "fp32_tflops": 110},
    "sm_121": {"name": "DGX Spark GB10", "host": "spark", "arch": "sm_121a", "sms": 48, "clock_ghz": 2.4, "bw_gbs": 273, "regs_sm": 65536, "smem_sm": 100 * 1024, "threads_sm": 1536, "blocks_sm": 32, "bf16_tflops": 125, "fp4_tflops": 500, "fp32_tflops": 31},
    "sm_75": {"name": "RTX 2060", "host": "windows", "arch": "sm_75", "sms": 30, "clock_ghz": 1.68, "bw_gbs": 336, "regs_sm": 65536, "smem_sm": 64 * 1024, "threads_sm": 1024, "blocks_sm": 16, "bf16_tflops": 0, "fp4_tflops": 0, "fp32_tflops": 6.5},
}

SPEC_SOURCES = {
    "sm_120": "nvidia.com RTX PRO 6000 Blackwell Max-Q page: 1792 GB/s, 3511 sparse FP4 TOPS (dense FP4 1755, bf16 dense taken as a quarter of that), 110 TFLOP/s FP32; 188 SMs and 3090 MHz max clock from nvidia-smi",
    "sm_121": "NVIDIA DGX Spark spec: 273 GB/s LPDDR5x, 1 PFLOP sparse FP4 (dense 500), bf16 dense a quarter of that; 48 SMs",
    "sm_75": "TU106 RTX 2060: 336 GB/s, 30 SMs, 6.5 TFLOP/s FP32, no bf16 or FP4 tensor cores",
}

CLASSES = [
    ("ldgsts", r"^LDGSTS"),
    ("ldg128", r"^LDG\..*128"),
    ("ldg64", r"^LDG\..*64"),
    ("ldg32", r"^LDG\b"),
    ("ldsm", r"^LDSM"),
    ("lds", r"^LDS\b"),
    ("sts", r"^STS\b"),
    ("stg128", r"^STG\..*128"),
    ("stg", r"^STG\b"),
    ("local", r"^(LDL|STL)\b"),
    ("mma", r"^(HMMA|QMMA|OMMA|IMMA|UTCMMA)"),
    ("fma", r"^(FFMA|FMUL|FADD|HFMA2|HADD2|HMUL2|FMNMX)"),
    ("mufu", r"^MUFU"),
    ("shfl", r"^SHFL"),
    ("bar", r"^(BAR|DEPBAR|LDGDEPBAR|WARPSYNC|SYNCS)"),
]
SHOW = ["ldg128", "ldg64", "ldg32", "ldgsts", "ldsm", "lds", "mma", "fma", "mufu", "shfl", "bar", "local"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


@lru_cache(maxsize=None)
def resources(cubin):
    out = {}
    cur = None
    for line in run(["cuobjdump", "--dump-resource-usage", cubin]).splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            cur = m.group(1)
        elif cur and "REG:" in line:
            v = dict(re.findall(r"(\w+):(\d+)", line))
            out[cur] = {"regs": int(v.get("REG", 0)), "smem": int(v.get("SHARED", 0)), "spill": int(v.get("LOCAL", 0)) + int(v.get("STACK", 0))}
            cur = None
    return out


@lru_cache(maxsize=None)
def sass(cubin, fn):
    lines = run(["cuobjdump", "-sass", "-fun", fn, cubin]).splitlines()
    insts = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* (0x[0-9a-f]{16}) \*/", line)
        if not m:
            continue
        nxt = re.search(r"/\* (0x[0-9a-f]{16}) \*/", lines[i + 1]) if i + 1 < len(lines) else None
        hi = int(nxt.group(1), 16) if nxt else 0
        body = re.sub(r"^@!?U?P\w+\s+", "", m.group(2).strip())
        insts.append((int(m.group(1), 16), body.split()[0] if body else "", body, (hi >> 41) & 0xF))
    return tuple(insts)


def classify(op):
    for name, pat in CLASSES:
        if re.match(pat, op):
            return name
    return "other"


def hot_loop(insts):
    best = ()
    for addr, op, text, _ in insts:
        if op.startswith("BRA"):
            m = re.search(r"0x([0-9a-f]+)", text)
            if m and int(m.group(1), 16) <= addr:
                body = tuple(x for x in insts if int(m.group(1), 16) <= x[0] <= addr)
                if len(body) > len(best):
                    best = body
    return best


def shape_of(insts):
    mma = Counter(op for _, op, _, _ in insts if classify(op) == "mma")
    return ", ".join(f"{k} x{v}" for k, v in sorted(mma.items()))


def occupancy(gpu, regs, smem, threads):
    g = GPUS[gpu]
    warps = (threads + 31) // 32
    per_warp = (max(regs, 1) * 32 + 255) // 256 * 256
    by = {
        "regs": g["regs_sm"] // (per_warp * warps),
        "smem": g["smem_sm"] // (smem + 1024) if smem else g["blocks_sm"],
        "threads": g["threads_sm"] // max(threads, 1),
        "blocks": g["blocks_sm"],
    }
    lim = min(by, key=by.get)
    return by[lim], lim, by[lim] * warps * 32


def measure(gpu, cubin, fn, threads, dyn_smem=0):
    r = resources(str(cubin)).get(fn)
    if r is None:
        return None
    insts = sass(str(cubin), fn)
    hot = hot_loop(insts)
    smem = r["smem"] + dyn_smem
    blocks, lim, threads_sm = occupancy(gpu, r["regs"], smem, threads)
    return {
        "fn": fn, "regs": r["regs"], "smem": smem, "spill": r["spill"], "threads": threads,
        "blocks_sm": blocks, "limit": lim, "threads_sm": threads_sm,
        "insts": len(insts), "mix": Counter(classify(x[1]) for x in insts),
        "hot": len(hot), "hot_cycles": sum(max(x[3], 1) for x in hot), "hot_mix": Counter(classify(x[1]) for x in hot),
        "mma_shapes": shape_of(insts),
    }


def work(row):
    f, p = row["family"], row["params"]
    hd, heads, kvh = p.get("head_dim", 0), p.get("heads", 0), p.get("kv_heads", 1)
    kvb = 1 if p.get("kv_fp8") else 2
    k = row["kernel"]
    if f == "rmsnorm":
        h = p["hidden"]
        return p["rows"] * h * 8, p["rows"] * h * 4, "fp32"
    if f == "fused_add_rmsnorm":
        h = p["hidden"]
        return p["rows"] * h * 16, p["rows"] * h * 5, "fp32"
    if f == "silu_and_mul":
        i = p["inter"]
        return p["rows"] * i * 12, p["rows"] * i * 6, "fp32"
    if f == "rotary":
        return p["rows"] * p["qkv_row"] * 8, p["rows"] * (heads + kvh) * hd * 6, "fp32"
    if f == "decode_attention":
        seqs, ctx = p["seqs"], p["ctx"]
        if "combine" in k or "reduce" in k:
            return seqs * heads * (hd + 2) * 4 * p["splits"], seqs * heads * hd * p["splits"] * 3, "fp32"
        return seqs * ctx * kvh * hd * kvb * 2, seqs * heads * ctx * hd * 4, "fp32"
    if f == "prefill_attention":
        t = p["tokens"]
        return t * (heads + 2 * kvh) * hd * 2 + t * heads * hd * 4, 2 * heads * t * t * hd, "bf16"
    if f in ("bf16_gemm", "nvfp4_gemm", "bf16_gemv", "nvfp4_gemv"):
        m, n, kk = p["m"], p["n"], p["k"]
        if p.get("role"):
            return m * kk * 4 + m * kk * 9 // 16, m * kk * 4, "fp32"
        wb = n * kk * 9 // 16 if f.startswith("nvfp4") else n * kk * 2
        return wb + m * kk * 4 + m * n * 4, 2 * m * n * kk, "fp4" if f == "nvfp4_gemm" else ("bf16" if f == "bf16_gemm" else "fp32")
    if f == "topk_routing":
        e = p["experts"]
        if k == "moe_route_logits":
            return e * p["hidden"] * 2 + p["tokens"] * p["hidden"] * 4, 2 * p["tokens"] * e * p["hidden"], "fp32"
        return p["tokens"] * e * 4, p["tokens"] * e * 8, "fp32"
    return 0, 0, "fp32"


def roofline(gpu, b, fl, unit):
    g = GPUS[gpu]
    peak = {"fp32": g["fp32_tflops"], "bf16": g["bf16_tflops"] or g["fp32_tflops"], "fp4": g["fp4_tflops"] or g["fp32_tflops"]}[unit]
    t_mem = b / (g["bw_gbs"] * 1e3)
    t_math = fl / (peak * 1e6)
    return t_mem, t_math


def our_rows(gpu):
    m = json.loads((ROOT / "ours" / gpu / "manifest.json").read_text())
    rows, broken = [], []
    for mod in m["modules"]:
        if not mod.get("cubin"):
            broken.append((mod["spec"], (mod.get("error") or "no cubin").splitlines()[0]))
            continue
        cubin = ROOT / "ours" / gpu / mod["cubin"]
        for r in mod["rows"]:
            dyn = r["smem"]
            meas = measure(gpu, cubin, r["kernel"], r["block"], dyn)
            if meas:
                rows.append({**r, "ours": meas, "sm": mod["sm"]})
    return m, rows, broken


@lru_cache(maxsize=None)
def vllm_index(gpu):
    if not VLLM:
        return {}
    arch = "sm_120" if gpu.startswith("sm_12") else gpu
    idx = {}
    for d in (VLLM / "C", VLLM / "moe"):
        for f in sorted(d.glob(f"*.{arch}.cubin")):
            for fn in resources(str(f)):
                idx.setdefault(fn, f)
    return idx


def flashinfer_pick(row, names, gpu):
    f, p = row["family"], row["params"]
    def first(pattern):
        hits = sorted(n for n in names if re.search(pattern, n))
        return hits[0] if hits else None
    if f == "decode_attention":
        hd, group = p["head_dim"], p["heads"] // p["kv_heads"]
        kvb = 1 if p.get("kv_fp8") else 2
        vec = max(16 // kvb, hd // 32)
        bdx = hd // vec
        threads = max(128, bdx * group)
        bdz = threads // (bdx * group)
        return first(rf"BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2ELj1ELj{vec}ELj{bdx}ELj{group}ELj{bdz}E"), threads, 0
    if f == "prefill_attention":
        hd = p["head_dim"]
        tile = 128 if hd < 256 else 64
        wq, mq = 4, (2 if tile > 64 else 1)
        fixed, per = tile * hd * 2, 2 * hd * 16 * 2
        smem_sm, optin = GPUS[gpu]["smem_sm"], GPUS[gpu]["smem_sm"] - 1024
        ctas = 2 if smem_sm >= 2 * (fixed + per) else 1
        mma_kv = max(1, min((min(smem_sm // ctas, optin) - fixed) // per, 8 // mq))
        while mma_kv >= 1:
            fn = first(rf"KernelTraitsILNS_8MaskModeE1ELj{tile}ELj{mq}ELj{mma_kv}ELj{hd // 16}ELj{hd // 16}ELj{wq}ELj1E")
            if fn:
                return fn, 32 * wq, fixed + mma_kv * per
            mma_kv -= 1
        return None, 0, 0
    if f == "rmsnorm":
        return first(r"RMSNormKernelILj8E13__nv_bfloat16"), min(1024, p["hidden"] // 8), 0
    if f == "fused_add_rmsnorm":
        return first(r"FusedAddRMSNormKernelILj8E13__nv_bfloat16"), min(1024, p["hidden"] // 8), 0
    if f == "silu_and_mul":
        return first(r"act_and_mul_kernelI13__nv_bfloat16.*Lj8EEEvPT_PKS5_i$"), 256, 0
    if f == "rotary":
        return first(rf"BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernelILb0ELj{p['head_dim']}ELj8E"), 32 * 4, 0
    return None, 0, 0


FI_KEY = {"decode_attention": "batch_decode_hd{hd}{fp8}", "prefill_attention": "batch_prefill_hd{hd}", "rmsnorm": "norm", "fused_add_rmsnorm": "norm", "silu_and_mul": "silu_and_mul", "rotary": "rope"}

VLLM_PICK = {
    "rmsnorm": (r"^_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi4ELb0EE", lambda p: min(1024, p["hidden"] // 8)),
    "fused_add_rmsnorm": (r"^_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16ELi8ELb0EE", lambda p: min(1024, p["hidden"] // 8)),
    "silu_and_mul": (r"^_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E14__nv_bfloat162XadL_ZNS_11silu_kernel.*ELb1ELb0ELb1ELb0EEE", lambda p: 1024),
    "rotary": (r"^_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ES2_Lb1EE", lambda p: 512),
    "topk_routing": (r"^_ZN4vllm3moe10topkGatingILi8ELi128ELi4ELi16ELi32Ei13__nv_bfloat16LNS0_11ScoringFuncE0EE", lambda p: 128),
}


def competitors(gpu, row, fi, tr):
    f, p = row["family"], row["params"]
    out = []
    g = GPUS[gpu]
    arch = g["arch"]
    if f in FI_KEY and not ("combine" in row["kernel"] or "reduce" in row["kernel"]):
        key = FI_KEY[f].format(hd=p.get("head_dim"), fp8="_fp8kv" if p.get("kv_fp8") else "")
        files = fi.get("built", {}).get(arch, {}).get(key)
        if files:
            cubin = ROOT / "theirs" / arch / files[0]
            fn, threads, dyn = flashinfer_pick(row, list(resources(str(cubin))), gpu)
            if fn:
                meas = measure(gpu, cubin, fn, threads, dyn)
                if meas:
                    out.append(("FlashInfer " + fi.get("flashinfer", ""), meas))
    for k in tr.get("kernels", []):
        if k["arch"] != arch or k["family"] != f or "cubin" not in k:
            continue
        kp = k["params"]
        if f in ("rmsnorm", "fused_add_rmsnorm") and kp["hidden"] != p.get("hidden"):
            continue
        if f == "silu_and_mul" and kp["inter"] != p.get("inter"):
            continue
        if f == "rotary" and kp["head_dim"] != p.get("head_dim"):
            continue
        if f == "topk_routing" and (kp["experts"] != p.get("experts") or row["kernel"] != "moe_route"):
            continue
        if f == "bf16_gemm" and kp["tile"] != [128, 128, 64]:
            continue
        meas = measure(gpu, ROOT / "theirs" / arch / k["cubin"], k["kernel"], k["threads"], k.get("shared", 0))
        if meas:
            out.append(("Triton " + tr.get("triton", ""), meas))
            break
    pick = VLLM_PICK.get(f)
    if pick and not (f == "topk_routing" and row["kernel"] != "moe_route"):
        idx = vllm_index(gpu)
        hits = sorted(n for n in idx if re.search(pick[0], n))
        if hits:
            meas = measure(gpu, idx[hits[0]], hits[0], pick[1](p), 0)
            if meas:
                out.append(("vLLM prebuilt", meas))
    return out


NONE = {
    "bf16_gemv": "cuBLAS GEMV is closed source; no static artefact",
    "nvfp4_gemv": "FlashInfer 0.7.3 has no NVFP4 GEMV; vLLM runs the CUTLASS GEMM at M=1",
    "nvfp4_gemm": "FlashInfer CUTLASS fp4 GEMM: tactic chosen by its autotuner at run time, compared from the prebuilt cubin in the findings",
}


def verdict(o, ts):
    if not ts:
        return ""
    notes = []
    best = max(ts, key=lambda t: t[1]["threads_sm"])
    if o["spill"] and not any(t[1]["spill"] for t in ts):
        notes.append(f"spills {o['spill']} B")
    if o["threads_sm"] < 0.5 * best[1]["threads_sm"]:
        notes.append(f"{o['threads_sm']} threads/SM vs {best[1]['threads_sm']}")
    wide = lambda m: m.get("ldg128", 0) + m.get("ldgsts", 0)
    if any(wide(t[1]["mix"]) for t in ts) and not wide(o["mix"]) and o["mix"].get("ldg32", 0):
        notes.append("32-bit loads only")
    return "behind: " + ", ".join(notes) if notes else "matches"


def mix_str(m):
    return " ".join(f"{k} {m[k]}" for k in SHOW if m.get(k))


def short(fn):
    return re.sub(r"^_ZN\d+", "", fn)[:48]


def report_gpu(gpu, fi, tr):
    m, rows, broken = our_rows(gpu)
    g = GPUS[gpu]
    lines = [f"\n## {gpu}: {g['name']} ({g['host']})\n", f"Target `{m['target']}`, NVRTC {m['nvrtc']} ({' '.join(m['options'])}). Peak numbers: {SPEC_SOURCES[gpu]}.\n"]
    for spec, err in broken:
        lines.append(f"- `{spec}` does not compile for {gpu}: {err[:220]}")
    if broken:
        lines.append("")
    struct = {}
    for r in rows:
        key = (r["family"], r["spec"], r["kernel"])
        struct.setdefault(key, r)
    for fam in sorted({k[0] for k in struct}):
        lines.append(f"\n### {fam}\n")
        lines.append("| spec | ours | block | regs | smem | spill | blocks/SM (limit) | threads/SM | hot loop insts, cycles | hot loop mix | competitor | block | regs | smem | spill | threads/SM | hot loop insts, cycles | hot loop mix | verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key in sorted(k for k in struct if k[0] == fam):
            r = struct[key]
            o = r["ours"]
            ts = competitors(gpu, r, fi, tr)
            head = f"| {r['spec']} | `{r['kernel']}` | {o['threads']} | {o['regs']} | {o['smem']} | {o['spill']} | {o['blocks_sm']} ({o['limit']}) | {o['threads_sm']} | {o['hot']}, {o['hot_cycles']} | {mix_str(o['hot_mix'])} |"
            if not ts:
                lines.append(head + f" {NONE.get(fam, 'no competitor built for this target')} | | | | | | | | |")
                continue
            v = verdict(o, ts)
            for i, (name, t) in enumerate(ts):
                prefix = head if i == 0 else "| | | | | | | | | | |"
                lines.append(prefix + f" {name} `{short(t['fn'])}` | {t['threads']} | {t['regs']} | {t['smem']} | {t['spill']} | {t['threads_sm']} | {t['hot']}, {t['hot_cycles']} | {mix_str(t['hot_mix'])} | {v if i == 0 else ''} |")
    lines.append("\n### Roofline (algorithmic bytes and FLOPs, peak numbers above)\n")
    lines.append("| family | spec | kernel | shape | MB | GFLOP | FLOP/B | mem us | math us | bound | blocks | waves |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda r: (r["family"], r["spec"], r["kernel"], json.dumps(r["params"], sort_keys=True))):
        b, fl, unit = work(r)
        tm, tc = roofline(gpu, b, fl, unit)
        blocks = r["grid"][0] * r["grid"][1] * r["grid"][2]
        waves = blocks / max(r["ours"]["blocks_sm"] * g["sms"], 1)
        p = r["params"]
        shape = " ".join(f"{k}={p[k]}" for k in ("rows", "seqs", "ctx", "tokens", "m", "n", "k") if k in p)
        lines.append(f"| {r['family']} | {r['spec']} | `{r['kernel']}` | {shape} | {b / 1e6:.2f} | {fl / 1e9:.3f} | {fl / b if b else 0:.1f} | {tm:.2f} | {tc:.2f} | {'memory' if tm >= tc else 'compute ' + unit} | {blocks} | {waves:.2f} |")
    return "\n".join(lines), rows, broken


def main():
    fi = json.loads((ROOT / "theirs" / "competitors.json").read_text()) if (ROOT / "theirs" / "competitors.json").exists() else {}
    tr = json.loads((ROOT / "theirs" / "triton.json").read_text()) if (ROOT / "theirs" / "triton.json").exists() else {}
    parts = []
    data = {}
    for gpu in GPUS:
        if (ROOT / "ours" / gpu / "manifest.json").exists():
            text, rows, broken = report_gpu(gpu, fi, tr)
            parts.append(text)
            data[gpu] = {"rows": [{k: v for k, v in r.items() if k != "ours"} | {"ours": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in r["ours"].items()}} for r in rows], "broken": broken}
    (ROOT / "static.json").write_text(json.dumps(data, indent=1, sort_keys=True))
    text = "\n".join(parts) + "\n"
    if DOC:
        DOC.write_text(text)
    else:
        sys.stdout.write(text)


main()
