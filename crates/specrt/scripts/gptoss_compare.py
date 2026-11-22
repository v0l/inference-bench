import json
import sys
import urllib.request

PROMPTS = [
    "What is 17 times 3? Answer briefly.",
    "Name the capital of France and one landmark there.",
    "Write a haiku about autumn.",
    "Explain in two sentences what a mutex is.",
    "List three prime numbers greater than 50.",
    "Translate 'good morning' into Spanish and German.",
    "What does the Rust borrow checker prevent?",
    "Give a one line definition of entropy.",
]


def ask(url, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 120,
        "logprobs": True,
        "top_logprobs": 1,
    }
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    c = r["choices"][0]
    m = c["message"]
    lp = [(t["token"], t["logprob"]) for t in (c.get("logprobs") or {}).get("content") or []]
    return {"prompt": prompt, "reasoning": m.get("reasoning_content") or m.get("reasoning") or "", "content": m.get("content") or "", "logprobs": lp}


def collect(url, model, out):
    json.dump([ask(url, model, p) for p in PROMPTS], open(out, "w"))


def compare(a, b):
    a, b = json.load(open(a)), json.load(open(b))
    same = 0
    print("| prompt | text identical | first divergent token | mean abs logprob diff before divergence |")
    print("|---|---|---|---|")
    for x, y in zip(a, b):
        tx, ty = x["logprobs"], y["logprobs"]
        n = 0
        while n < min(len(tx), len(ty)) and tx[n][0] == ty[n][0]:
            n += 1
        diffs = [abs(tx[i][1] - ty[i][1]) for i in range(n)]
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        ident = (x["reasoning"], x["content"]) == (y["reasoning"], y["content"])
        same += ident
        div = "none" if n == len(tx) == len(ty) else str(n)
        print(f"| {x['prompt'][:40]} | {ident} | {div} of {len(tx)}/{len(ty)} | {mean:.4f} |")
    print(f"\nidentical: {same}/{len(a)}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        compare(sys.argv[2], sys.argv[3])
