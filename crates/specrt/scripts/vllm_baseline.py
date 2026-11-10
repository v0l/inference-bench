import argparse, json, os, statistics, sys, threading, time, urllib.request

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "rain", "falls",
         "on", "quiet", "stone", "streets"]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def words(n, salt):
    return " ".join(WORDS[(i * 7 + i // 16 + salt) % len(WORDS)] for i in range(n))


class Bench:
    def __init__(self, base, model, ratio=None):
        self.base, self.model = base, model
        self.ratio = ratio or self.calibrate()

    def count(self, text):
        r = post(self.base + "/tokenize", {"model": self.model, "prompt": text})
        return json.load(r)["count"]

    def calibrate(self):
        sample = words(8000, 0)
        return self.count(sample) / 8000

    def prompt(self, tokens, nonce, head="Summarise the following text in one sentence.\n\n"):
        n = max(1, int((tokens - 40) / self.ratio))
        return f"[{nonce}] {head}{words(n, nonce % 5)}"

    def chat(self, prompt, max_tokens):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        first = None
        usage = {}
        with post(self.base + "/v1/chat/completions", body) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch and first is None:
                    delta = ch[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        first = time.perf_counter()
        end = time.perf_counter()
        first = first or end
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        ttft, dec = first - t0, end - first
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached, "ttft_s": ttft,
                "prefill_tok_s": pt / ttft if ttft > 0 else None,
                "decode_tok_s": (ct - 1) / dec if dec > 0 and ct > 1 else None, "total_s": end - t0}


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", required=True, help="single|concurrent|prefix|held")
    ap.add_argument("--lengths", default="128,1024,4096,16384")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--long-runs", type=int, default=3)
    ap.add_argument("--sizes", default="1,4,8,16,32")
    ap.add_argument("--conc-len", type=int, default=128)
    ap.add_argument("--ratio", type=float)
    ap.add_argument("--out-tokens", type=int, default=128)
    a = ap.parse_args()
    b = Bench(a.base, a.model, a.ratio)
    nonce = [int(time.time()) % 100000 * 100]

    def fresh():
        nonce[0] += 1
        return nonce[0]

    result = {"label": a.label, "suite": a.suite, "tokens_per_word": b.ratio, "points": []}

    def save():
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    b.chat(b.prompt(256, fresh()), 16)
    if a.suite == "single":
        for n in [int(x) for x in a.lengths.split(",")]:
            runs = a.runs if n < 100000 else a.long_runs
            rows = [b.chat(b.prompt(n, fresh()), a.out_tokens) for _ in range(runs)]
            p = {"prompt_len": n, "runs": rows, "median_ttft_s": med(rows, "ttft_s"),
                 "median_prefill_tok_s": med(rows, "prefill_tok_s"), "median_decode_tok_s": med(rows, "decode_tok_s")}
            result["points"].append(p)
            save()
            print(a.label, n, p["median_ttft_s"], p["median_prefill_tok_s"], p["median_decode_tok_s"], flush=True)
    elif a.suite == "concurrent":
        for n in [int(x) for x in a.sizes.split(",")]:
            rows = [None] * n
            prompts = [b.prompt(a.conc_len, fresh()) for _ in range(n)]

            def go(i):
                rows[i] = b.chat(prompts[i], a.out_tokens)

            ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            [t.start() for t in ths]
            [t.join() for t in ths]
            wall = time.perf_counter() - t0
            agg = sum(r["decode_tok_s"] or 0 for r in rows)
            p = {"concurrent": n, "prompt_len": a.conc_len, "wall_s": wall, "aggregate_decode_tok_s": agg,
                 "throughput_tok_s": sum(r["completion_tokens"] for r in rows) / wall,
                 "median_ttft_s": med(rows, "ttft_s"), "runs": rows}
            result["points"].append(p)
            save()
            print(a.label, "conc", n, agg, flush=True)
    elif a.suite == "prefix":
        for n in [int(x) for x in a.lengths.split(",")]:
            base_nonce = fresh()
            cold = b.chat(b.prompt(n, base_nonce), 16)
            warm = []
            for k in range(a.runs):
                body = b.prompt(n, base_nonce) + f"\n\nQuestion {k}: " + words(1000 // 1, k)[:3000]
                warm.append(b.chat(body, 16))
            p = {"prompt_len": n, "cold": cold, "warm": warm, "median_warm_ttft_s": med(warm, "ttft_s")}
            result["points"].append(p)
            save()
            print(a.label, "prefix", n, cold["ttft_s"], p["median_warm_ttft_s"], flush=True)
    elif a.suite == "held":
        for n in [int(x) for x in a.lengths.split(",")]:
            rows = [b.chat(b.prompt(n, fresh()), 512) for _ in range(a.long_runs)]
            p = {"prompt_len": n, "decode_tokens": 512, "runs": rows, "median_decode_tok_s": med(rows, "decode_tok_s")}
            result["points"].append(p)
            save()
            print(a.label, "held", n, p["median_decode_tok_s"], flush=True)
    save()


main()
