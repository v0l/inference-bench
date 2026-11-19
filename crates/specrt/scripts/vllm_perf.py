import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams


def prompts(n, length, salt):
    return [{"prompt_token_ids": [1000 + (i * 7 + s * 131 + salt) % 5000 for i in range(length)]} for s in range(n)]


def timed(llm, batch, length, new, salt):
    params = SamplingParams(temperature=0, max_tokens=new, ignore_eos=True)
    t = time.perf_counter()
    llm.generate(prompts(batch, length, salt), params, use_tqdm=False)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default="1024,8192")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--util", type=float, default=0.12)
    ap.add_argument("--moe-backend", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",")]
    llm = LLM(a.model, max_model_len=max(lens) + a.new + 8, gpu_memory_utilization=a.util, enable_prefix_caching=False, max_num_seqs=max(a.batch, 8), **({"moe_backend": a.moe_backend} if a.moe_backend else {}))
    timed(llm, 1, 128, 8, 0)
    out = {"engine": "vllm", "version": __import__("vllm").__version__, "prefill": {}, "decode": {}}
    for length in lens:
        t = [timed(llm, 1, length, 1, r + 1) for r in range(a.rounds)]
        out["prefill"][str(length)] = length / statistics.median(t)
    base = [timed(llm, 1, 128, 1, r + 10) for r in range(a.rounds)]
    full = [timed(llm, 1, 128, a.new, r + 10) for r in range(a.rounds)]
    out["decode"]["1"] = (a.new - 1) / (statistics.median(full) - statistics.median(base))
    base = [timed(llm, a.batch, 128, 1, r + 20) for r in range(a.rounds)]
    full = [timed(llm, a.batch, 128, a.new, r + 20) for r in range(a.rounds)]
    out["decode"][str(a.batch)] = a.batch * (a.new - 1) / (statistics.median(full) - statistics.median(base))
    json.dump(out, open(a.out, "w"))
    print(json.dumps(out))


main()
