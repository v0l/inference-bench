import argparse
import json
import statistics
import time

import sglang as sgl


def prompts(n, length, salt):
    return [[1000 + (i * 7 + s * 131 + salt) % 5000 for i in range(length)] for s in range(n)]


def timed(engine, batch, length, new, salt):
    params = {"temperature": 0, "max_new_tokens": new, "ignore_eos": True}
    t = time.perf_counter()
    engine.generate(input_ids=prompts(batch, length, salt), sampling_params=params)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default="1024,8192")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--util", type=float, default=0.12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",")]
    engine = sgl.Engine(
        model_path=a.model,
        context_length=max(lens) + a.new + 8,
        mem_fraction_static=a.util,
        disable_radix_cache=True,
        max_running_requests=max(a.batch, 8),
    )
    timed(engine, 1, 128, 8, 0)
    out = {"engine": "sglang", "version": sgl.__version__, "prefill": {}, "decode": {}}
    for length in lens:
        t = [timed(engine, 1, length, 1, r + 1) for r in range(a.rounds)]
        out["prefill"][str(length)] = length / statistics.median(t)
    base = [timed(engine, 1, 128, 1, r + 10) for r in range(a.rounds)]
    full = [timed(engine, 1, 128, a.new, r + 10) for r in range(a.rounds)]
    out["decode"]["1"] = (a.new - 1) / (statistics.median(full) - statistics.median(base))
    base = [timed(engine, a.batch, 128, 1, r + 20) for r in range(a.rounds)]
    full = [timed(engine, a.batch, 128, a.new, r + 20) for r in range(a.rounds)]
    out["decode"][str(a.batch)] = a.batch * (a.new - 1) / (statistics.median(full) - statistics.median(base))
    json.dump(out, open(a.out, "w"))
    print(json.dumps(out))
    engine.shutdown()


if __name__ == "__main__":
    main()
