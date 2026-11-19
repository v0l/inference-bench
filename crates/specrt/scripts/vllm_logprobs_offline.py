import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from reference import REAL_PROMPTS
from vllm_logprobs import EXTRA

NEW = 32
TOP = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--util", type=float, default=0.12)
    ap.add_argument("--moe-backend", default=None)
    ap.add_argument("--linear-backend", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok(t).input_ids for t in (REAL_PROMPTS + EXTRA)[: a.prompts]]
    extra = {k: v for k, v in (("moe_backend", a.moe_backend), ("linear_backend", a.linear_backend)) if v}
    llm = LLM(a.model, max_model_len=1024, gpu_memory_utilization=a.util, enable_prefix_caching=False, max_num_seqs=8, **extra)
    params = SamplingParams(temperature=0, max_tokens=NEW, ignore_eos=True, logprobs=TOP)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    records = []
    for ids, o in zip(prompts, outs):
        c = o.outputs[0]
        rows = [{str(t): lp.logprob for t, lp in step.items()} for step in c.logprobs]
        records.append({"prompt": ids, "greedy": list(c.token_ids), "logprobs": rows})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "new_tokens": NEW, "top": TOP, "records": records}))


main()
