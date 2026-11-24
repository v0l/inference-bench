#!/usr/bin/env python3
import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "popular"
CACHE = Path(os.environ.get("POPULAR_CACHE", Path.home() / ".cache" / "specrt-popular"))
HUB = Path.home() / ".cache" / "huggingface" / "hub"
COMMIT_LIMIT = 100 * 1024
COMMITTED = {"config.json", "generation_config.json", "hf_quant_config.json"}
TEXT_TOP = 100
TRENDING_TOP = 20
VISION_TOP = 30
CARD_GB = 96
FILES = ["config.json", "generation_config.json", "hf_quant_config.json", "tokenizer_config.json", "chat_template.jinja"]
INDEX = "model.safetensors.index.json"
LABS = tomllib.loads((OUT / "labs.toml").read_text())
EXPAND = ["downloads", "likes", "trendingScore", "gated", "tags", "safetensors", "gguf", "pipeline_tag", "createdAt", "siblings", "config", "cardData"]
SKIP = re.compile(r"(?i)(tiny-random|internal-testing|dummy|/tiny[-_]|testing|-test$|random-)")
FORMATS = ["NVFP4", "FP8", "AWQ", "GPTQ", "GGUF"]
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1, "F8_E8M0": 1, "F4": 0.5}


_resolve = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **k: _resolve(host, port, socket.AF_INET, *a, **k)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def slug(repo):
    return repo.replace("/", "__")


last_call = [0.0]


def get(url, binary=False):
    key = hashlib.sha1(url.encode()).hexdigest()
    path = CACHE / "http" / key
    meta = CACHE / "http" / (key + ".status")
    if meta.exists():
        status = int(meta.read_text())
        return status, (path.read_bytes() if status == 200 else None)
    for attempt in range(6):
        wait = 0.35 - (time.time() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "specrt-popular-survey"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                status = 200
        except urllib.error.HTTPError as e:
            status = e.code
            body = None
            if status == 429:
                reset = re.search(r"t=(\d+)", e.headers.get("ratelimit", "") or "")
                pause = int(reset.group(1)) + 1 if reset else 30
                log(f"rate limited, waiting {pause}s")
                time.sleep(min(pause, 60))
                continue
            if status >= 500:
                time.sleep(5)
                continue
        except (urllib.error.URLError, TimeoutError) as e:
            log(f"{url}: {e}, retrying")
            time.sleep(5)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if status == 200:
            path.write_bytes(body)
        meta.write_text(str(status))
        return status, body
    raise SystemExit(f"giving up on {url}")


def api(path, params=None):
    q = []
    for k, v in (params or {}).items():
        for item in v if isinstance(v, list) else [v]:
            q.append((k, item))
    url = f"{HF}/api/{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
    status, body = get(url)
    if status != 200:
        return None
    return json.loads(body)


def listing(**params):
    params.setdefault("direction", -1)
    params["expand[]"] = EXPAND
    return api("models", params) or []


def info(repo):
    return api(f"models/{repo}", {"expand[]": EXPAND})


def parent_of(m):
    tags = m.get("tags") or []
    for kind in ["quantized", "finetune", "merge", "adapter"]:
        for t in tags:
            if t.startswith(f"base_model:{kind}:"):
                return t.split(":", 2)[2], kind
    base = (m.get("cardData") or {}).get("base_model")
    if isinstance(base, list):
        base = base[0] if len(base) == 1 else None
    if isinstance(base, str) and base != m["id"]:
        return base, "card"
    return None, None


known = {}


def remember(m):
    if m and m.get("id"):
        old = known.get(m["id"])
        if old is None or len(json.dumps(m)) > len(json.dumps(old)):
            known[m["id"]] = m


def meta(repo):
    if repo not in known:
        m = info(repo)
        if m is None:
            return None
        remember(m)
        known.setdefault(repo, m)
    return known[repo]


def root_of(repo):
    seen = [repo]
    current = repo
    for _ in range(4):
        m = meta(current)
        if m is None:
            break
        parent, _ = parent_of(m)
        if parent is None or parent in seen:
            break
        if meta(parent) is None:
            break
        seen.append(parent)
        current = parent
    return current


def quant_config(m):
    cfg = m.get("config") or {}
    return cfg.get("quantization_config") or cfg.get("quantization") or {}


def weight_format(m):
    tags = [t.lower() for t in m.get("tags") or []]
    name = m["id"].lower()
    if m.get("gguf") or "gguf" in tags:
        return "GGUF"
    if "mlx" in tags:
        return "MLX"
    q = quant_config(m)
    method = str(q.get("quant_method", "")).lower()
    algo = str(q.get("quant_algo", "")).upper()
    groups = q.get("config_groups") or {}
    bits = sorted({(g.get("weights") or {}).get("num_bits") for g in groups.values() if isinstance(g, dict)} - {None})
    types = {(g.get("weights") or {}).get("type") for g in groups.values() if isinstance(g, dict)}
    if "nvfp4" in name or "nvfp4" in tags or "NVFP4" in algo or (4 in bits and "float" in types):
        return "NVFP4"
    if method == "awq" or "awq" in name:
        return "AWQ"
    if method == "gptq" or "gptq" in name or (method == "compressed-tensors" and 4 in bits and "int" in types):
        return "GPTQ"
    if method == "fp8" or "FP8" in algo or "fp8" in name or (8 in bits and "float" in types):
        return "FP8"
    if method == "bitsandbytes" or "bnb" in name:
        return "BNB"
    if method == "mxfp4":
        return "MXFP4"
    if method:
        return method.upper()
    return "BF16"


def quant_label(m):
    fmt = weight_format(m)
    q = quant_config(m)
    method = q.get("quant_method")
    if fmt == "GGUF":
        return "gguf"
    if fmt == "MLX":
        return f"mlx {q.get('bits', '?')}bit"
    if not q:
        return "none"
    if method == "compressed-tensors":
        parts = set()
        for g in (q.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            a = g.get("input_activations") or {}
            parts.add(f"{g.get('format') or q.get('format') or '?'} w{w.get('num_bits')}{(w.get('type') or '?')[0]}-{w.get('strategy')}" + (f"-g{w.get('group_size')}" if w.get("group_size") else "") + (f" a{a.get('num_bits')}" if a else ""))
        return "compressed-tensors " + ", ".join(sorted(parts))
    if method == "modelopt":
        algos = {q.get("quant_algo")} | {v.get("quant_algo") for v in (q.get("quantized_layers") or {}).values() if isinstance(v, dict)}
        if not q.get("quant_algo") and q.get("config_groups"):
            for g in q["config_groups"].values():
                w = g.get("weights") or {}
                algos.add(f"w{w.get('num_bits')}{(w.get('type') or '?')[0]}" + (f"-g{w.get('group_size')}" if w.get("group_size") else ""))
        return "modelopt " + "+".join(sorted(str(a) for a in algos if a))
    if method == "fp8":
        return f"fp8 {q.get('fmt', 'e4m3')} block {q.get('weight_block_size')}"
    if method in ("awq", "gptq"):
        return f"{method} {q.get('bits')}bit g{q.get('group_size')}"
    if method == "bitsandbytes":
        return "bitsandbytes " + ("4bit" if q.get("load_in_4bit") else "8bit")
    return str(method)


def size_gb(m):
    st = (m.get("safetensors") or {}).get("parameters")
    if st:
        return round(sum(n * DTYPE_BYTES.get(d, 2) for d, n in st.items()) / 1e9, 1)
    g = m.get("gguf") or {}
    if g.get("totalFileSize"):
        return round(g["totalFileSize"] / 1e9, 1)
    return None


def params_b(m):
    st = m.get("safetensors") or {}
    g = m.get("gguf") or {}
    total = st.get("total") or g.get("total")
    return round(total / 1e9, 2) if total else None


GGUF_LABEL = re.compile(r"(?i)[-_.](UD-)?((?:IQ\d_[A-Z]+|Q\d_K(?:_[A-Z]+)?|Q\d_\d|F16|BF16|F32|MXFP4(?:_MOE)?|TQ\d_\d))(?:-\d+-of-\d+)?\.gguf$")


def gguf_labels(m):
    out = set()
    for s in m.get("siblings") or []:
        f = s["rfilename"]
        if "mmproj" in f.lower() or "mtp" in f.lower():
            continue
        hit = GGUF_LABEL.search(f)
        if hit:
            out.add(hit.group(2).upper())
    return sorted(out)


def has(m, name):
    return any(s["rfilename"] == name for s in m.get("siblings") or [])


def gated(m):
    return m.get("gated") not in (None, False)


def local_snapshot(repo):
    snaps = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for s in sorted(snaps.iterdir()):
        if (s / "config.json").exists():
            return s
    return None


def mirror_for(repo):
    name = repo.split("/", 1)[1]
    for org in ["unsloth", "NousResearch"]:
        for candidate in [f"{org}/{name}", f"{org}/Meta-{name}" if name.startswith("Llama") else None]:
            if candidate is None:
                continue
            m = meta(candidate)
            if m and not gated(m) and has(m, "config.json"):
                return candidate
    return None


def tensor_formats(index):
    names = list((index or {}).get("weight_map", {}).keys())
    modules = {}
    for n in names:
        base, _, leaf = n.rpartition(".")
        modules.setdefault(base, set()).add(leaf)
    counts = {}
    for base, leaves in modules.items():
        fmt = None
        if "weight_packed" in leaves and "weight_global_scale" in leaves:
            fmt = "compressed-tensors nvfp4 (weight_packed+weight_global_scale)"
        elif "weight_packed" in leaves:
            fmt = "compressed-tensors int4 (weight_packed+weight_scale)"
        elif "weight_scale_2" in leaves:
            fmt = "modelopt nvfp4 (weight_scale_2)"
        elif "qweight" in leaves and "qzeros" in leaves:
            fmt = "gptq/awq qweight+qzeros"
        elif "weight_scale_inv" in leaves:
            fmt = "fp8 block (weight_scale_inv)"
        elif "weight" in leaves and "weight_scale" in leaves:
            fmt = "fp8 or int8 per-channel (weight_scale)"
        elif any(l.endswith("absmax") or "quant_state" in l for l in leaves):
            fmt = "bitsandbytes (absmax/quant_state)"
        elif "scales" in leaves and "biases" in leaves:
            fmt = "mlx affine (scales+biases)"
        else:
            blocks = [l for l in leaves if l.endswith("_blocks")]
            if blocks:
                fmt = "mxfp4 (_blocks/_scales)"
                counts[fmt] = counts.get(fmt, 0) + len(blocks)
                continue
            if "weight" in leaves and re.search(r"(proj|fc\d?|dense|w\d|wq|wk|wv|wo|gate|up|down|qkv|out)$", base.split(".")[-1]):
                fmt = "plain weight"
        if fmt:
            counts[fmt] = counts.get(fmt, 0) + 1
    return dict(sorted(counts.items()))


def fetch_files(entry, m, config_repo, local):
    files = {}
    dest = OUT / slug(entry["repo"])
    cache = CACHE / "files" / slug(entry["repo"])
    for name in FILES + [INDEX]:
        body = None
        if local is not None and (local / name).exists() and gated(m):
            body = (local / name).read_bytes()
        elif config_repo and (config_repo != entry["repo"] or has(m, name) or not m.get("siblings")):
            src = meta(config_repo) or {}
            if src.get("siblings") and not has(src, name):
                continue
            status, body = get(f"{HF}/{config_repo}/resolve/main/{name}")
            if status != 200:
                if name == "config.json":
                    entry["config_status"] = status
                body = None
        if body is None:
            continue
        if name == INDEX:
            try:
                parsed = json.loads(body)
                entry["tensor_formats"] = tensor_formats(parsed)
                total = (parsed.get("metadata") or {}).get("total_size")
                if isinstance(total, (int, float)) and total > 0:
                    entry["size_gb"] = round(total / 1e9, 1)
            except json.JSONDecodeError:
                entry["tensor_formats"] = {"unreadable index": 1}
            files[name] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "committed": False}
            continue
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(body)
        keep = len(body) <= COMMIT_LIMIT and (name in COMMITTED or entry["role"] in ("base", "disk", "lab"))
        if keep:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / name).write_bytes(body)
        files[name] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "committed": keep}
    return files


def row(repo, role, base, lists, m):
    cfg = m.get("config") or {}
    created = (m.get("createdAt") or "")[:10]
    return {
        "repo": repo,
        "role": role,
        "base": base,
        "lists": sorted(lists),
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "trending": m.get("trendingScore", 0),
        "created": created,
        "pipeline": m.get("pipeline_tag"),
        "gated": bool(gated(m)),
        "model_type": cfg.get("model_type") or (m.get("gguf") or {}).get("architecture"),
        "architectures": cfg.get("architectures") or [],
        "format": weight_format(m),
        "quant": quant_label(m),
        "size_gb": size_gb(m),
        "params_b": params_b(m),
        "gguf_arch": (m.get("gguf") or {}).get("architecture"),
        "gguf_labels": gguf_labels(m) if (m.get("gguf") or weight_format(m) == "GGUF") else [],
        "tokenizer_json": has(m, "tokenizer.json"),
    }


def months_ago(today, months):
    d = datetime.date.fromisoformat(today)
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return datetime.date(y, m + 1, min(d.day, 28)).isoformat()


def lab_of_repo(repo, created, cutoff):
    org, _, name = repo.partition("/")
    if re.search(LABS["selection"]["exclude"], name):
        return None
    for lab in LABS["lab"]:
        if org.lower() in [o.lower() for o in lab["orgs"]] and re.search(lab["pattern"], name):
            if created >= cutoff or re.search(lab["current"], name):
                return lab["key"]
    return None


def lab_listing(cutoff):
    found = {}
    allowed = set(LABS["selection"]["pipelines"]) | {None}
    for lab in LABS["lab"]:
        for org in lab["orgs"]:
            for term in lab["search"]:
                for m in listing(author=org, search=term, sort="lastModified", limit=LABS["selection"]["limit"]):
                    remember(m)
                    if m.get("pipeline_tag") not in allowed:
                        continue
                    if lab_of_repo(m["id"], (m.get("createdAt") or "")[:10], cutoff) == lab["key"]:
                        found[m["id"]] = lab["key"]
    return found


def machine_inputs():
    cfg = LABS["machine"]
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        meminfo[k] = int(v.split()[0]) * 1024
    free = {}
    seen = set()
    for p in cfg["nvme_paths"]:
        path = os.path.expanduser(p)
        dev = os.stat(path).st_dev
        if dev in seen:
            continue
        seen.add(dev)
        free[p] = shutil.disk_usage(path).free
    return {
        "ram_total_gb": round(meminfo["MemTotal"] / 1e9, 1),
        "ram_available_gb": round(meminfo["MemAvailable"] / 1e9, 1),
        "host_share": cfg["host_share"],
        "host_bandwidth_gbs": cfg["host_bandwidth_gbs"],
        "nvme_free_gb": round(max(free.values()) / 1e9, 1),
        "nvme_path": max(free, key=free.get),
        "nvme_free_by_path_gb": {p: round(v / 1e9, 1) for p, v in sorted(free.items())},
        "cards": cfg["cards"],
        "card_gib": cfg["card_gib"],
    }


def main():
    today = datetime.date.today().isoformat()
    previous = OUT / "manifest.json"
    if "--reuse-date" in sys.argv and previous.exists():
        today = json.loads(previous.read_text())["fetched"]
    lists = {
        "text": listing(pipeline_tag="text-generation", sort="downloads", limit=1000),
        "trending": listing(pipeline_tag="text-generation", sort="trendingScore", limit=200),
        "vision": listing(pipeline_tag="image-text-to-text", sort="downloads", limit=400),
        "any": listing(pipeline_tag="any-to-any", sort="downloads", limit=100),
        "vision_trending": listing(pipeline_tag="image-text-to-text", sort="trendingScore", limit=100),
    }
    for items in lists.values():
        for m in items:
            remember(m)
    cutoff = months_ago(today, LABS["selection"]["months"])
    lab_repos = lab_listing(cutoff)
    OUT.mkdir(parents=True, exist_ok=True)
    for d in OUT.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    groups = {}
    order = {}
    for name, items in lists.items():
        for rank, m in enumerate(items):
            if SKIP.search(m["id"]):
                continue
            root = root_of(m["id"])
            g = groups.setdefault(root, {"members": set(), "lists": set(), "downloads": 0})
            if m["id"] not in g["members"]:
                g["members"].add(m["id"])
                g["downloads"] += m.get("downloads", 0)
            kind = {"vision_trending": "vision", "any": "vision"}.get(name, name)
            g["lists"].add(kind)
            order.setdefault((kind, root), rank)

    def ranked(kind, limit, exclude=()):
        keys = [r for r in groups if kind in groups[r]["lists"] and r not in exclude]
        if kind == "trending":
            keys.sort(key=lambda r: (order[(kind, r)], r))
        else:
            keys.sort(key=lambda r: (-groups[r]["downloads"], r))
        return keys[:limit]

    text = ranked("text", TEXT_TOP)
    trending = ranked("trending", TRENDING_TOP, set(text))
    vision = ranked("vision", VISION_TOP, set(text) | set(trending))
    chosen = {r: {"text"} for r in text}
    for r in trending:
        chosen.setdefault(r, set()).add("trending")
    for r in vision:
        chosen.setdefault(r, set()).add("vision")
    for repo in lab_repos:
        root = root_of(repo)
        g = groups.setdefault(root, {"members": set(), "lists": set(), "downloads": 0})
        if repo not in g["members"]:
            g["members"].add(repo)
            g["downloads"] += (meta(repo) or {}).get("downloads", 0)
        g.setdefault("lab_members", set()).add(repo)
        chosen.setdefault(root, set()).add("lab")
    on_disk = []
    if HUB.is_dir():
        for d in sorted(HUB.iterdir()):
            if not d.name.startswith("models--"):
                continue
            repo = d.name[len("models--"):].replace("--", "/", 1)
            m = meta(repo)
            local = local_snapshot(repo)
            arch = []
            if local is not None:
                try:
                    arch = json.loads((local / "config.json").read_text()).get("architectures") or []
                except (json.JSONDecodeError, OSError):
                    arch = []
            pipeline = (m or {}).get("pipeline_tag")
            causal = any(a.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")) for a in arch)
            if pipeline in ("text-generation", "image-text-to-text", "any-to-any") or (pipeline is None and causal):
                on_disk.append(repo)
    for repo in on_disk:
        root = root_of(repo)
        groups.setdefault(root, {"members": set(), "lists": set(), "downloads": 0})["members"].add(repo)
        chosen.setdefault(root, set()).add("disk")
    rows = []
    for root in sorted(chosen):
        g = groups[root]
        candidates = sorted(g["members"] | {root}, key=lambda r: (-(meta(r) or {}).get("downloads", 0), r))
        disk = [r for r in candidates if r in on_disk]
        author = root.split("/")[0]
        lab = [r for r in candidates if r.split("/")[0] == author and meta(r) and weight_format(meta(r)) in ("BF16", "MXFP4")]
        lab = lab or [r for r in candidates if r.split("/")[0] == author and meta(r) and weight_format(meta(r)) == "FP8"]
        primary = disk[0] if (disk and not lab and "disk" in chosen[root] and len(chosen[root]) == 1) else (lab[0] if lab else (root if meta(root) else candidates[0]))
        variants = {}
        seen = set(candidates)
        for parent in dict.fromkeys([primary, root]):
            for m in listing(filter=f"base_model:quantized:{parent}", sort="downloads", limit=100):
                remember(m)
                seen.add(m["id"])
        for r in sorted(seen, key=lambda r: (-(meta(r) or {}).get("downloads", 0), r)):
            m = meta(r)
            if m is None or r == primary or SKIP.search(r):
                continue
            if root_of(r) != root and r not in g["members"]:
                continue
            fmt = weight_format(m)
            if fmt in FORMATS and fmt not in variants and fmt != weight_format(meta(primary)):
                variants[fmt] = r
        picks = [(primary, "base")] + [(variants[f], f.lower()) for f in FORMATS if f in variants]
        for repo in sorted(g.get("lab_members", ())):
            if repo not in [p for p, _ in picks] and meta(repo):
                picks.append((repo, "lab"))
        for repo in disk:
            if repo not in [p for p, _ in picks]:
                picks.append((repo, "disk"))
        for repo, role in picks:
            m = meta(repo)
            entry = row(repo, role, root, chosen[root] if role == "base" else ({"disk"} if role == "disk" else set()), m)
            entry["group_downloads"] = g["downloads"] if role == "base" else 0
            lab_key = lab_repos.get(repo) or lab_repos.get(root)
            if lab_key:
                entry["lab"] = lab_key
            local = local_snapshot(repo)
            config_repo = repo
            if entry["format"] == "GGUF":
                config_repo = None
            elif gated(m) and local is None:
                config_repo = mirror_for(repo)
                entry["config_from"] = config_repo or "gated, no open mirror"
            elif gated(m):
                entry["config_from"] = "local snapshot (gated on the hub)"
            entry["files"] = fetch_files(entry, m, config_repo, local)
            if config_repo and "config.json" not in entry["files"] and entry.get("config_status") == 401:
                mirror = mirror_for(repo)
                if mirror:
                    entry["config_from"] = mirror
                    entry["files"] = fetch_files(entry, m, mirror, None)
            entry.pop("config_status", None)
            if "config.json" in entry["files"]:
                cfg = json.loads((CACHE / "files" / slug(repo) / "config.json").read_text())
                entry["model_type"] = cfg.get("model_type") or entry["model_type"]
                entry["architectures"] = cfg.get("architectures") or entry["architectures"]
                full = dict(m)
                full["config"] = dict(cfg)
                hq = CACHE / "files" / slug(repo) / "hf_quant_config.json"
                if not cfg.get("quantization_config") and hq.exists():
                    hqc = json.loads(hq.read_text()).get("quantization") or {}
                    full["config"]["quantization_config"] = {"quant_method": "modelopt", **hqc}
                entry["format"] = weight_format(full)
                entry["quant"] = quant_label(full)
            rows.append(entry)
            log(f"{repo}: {entry['format']} {entry['model_type']} {entry['size_gb']} GB")
    manifest = {
        "fetched": today,
        "source": "Hugging Face API: pipeline_tag text-generation (sort downloads, limit 1000; sort trendingScore, limit 200), image-text-to-text (downloads 400, trending 100) and any-to-any (downloads 100); downloads are the hub's last-30-day counts",
        "card_gb": CARD_GB,
        "commit_limit_bytes": COMMIT_LIMIT,
        "labs_since": cutoff,
        "machine": machine_inputs(),
        "selection": f"every model of the labs in labs.toml released since {cutoff} or still shipped by its lab, then top {TEXT_TOP} text groups by summed downloads, the top {TRENDING_TOP} trending text groups not already in, the top {VISION_TOP} vision groups not already in, and every text model in the local hub cache; a group is every repo whose base_model chain ends at the same root; repos whose names look like test fixtures are skipped",
        "models": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    log(f"{len(rows)} rows, {len(chosen)} groups")


if __name__ == "__main__":
    main()
