import json
import os
import sys
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import LLM, SamplingParams

REPO = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"


def main():
    records = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    index = int(sys.argv[3])
    new = int(sys.argv[4])
    eager = sys.argv[5] != "graphs" if len(sys.argv) > 5 else True
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        REPO,
        max_model_len=2048,
        gpu_memory_utilization=0.38,
        kv_cache_memory_bytes=2 * 1024**3,
        kv_cache_dtype="fp8",
        enforce_eager=eager,
        disable_log_stats=False,
        enable_prefix_caching=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        max_num_seqs=1,
        skip_mm_profiling=True,
        mm_processor_cache_gb=0,
        max_num_batched_tokens=1024,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    drafter = runner.drafter.model
    while hasattr(drafter, 'unwrap') or hasattr(drafter, 'runnable'):
        drafter = drafter.unwrap() if hasattr(drafter, 'unwrap') else drafter.runnable
    print('drafter', type(runner.drafter).__name__, type(runner.drafter.model).__name__, type(drafter).__name__, flush=True)
    calls = []

    orig = drafter.forward

    def wrapped(*args, **kwargs):
        c = {
            "input_ids": kwargs["input_ids"].detach().cpu().clone(),
            "positions": kwargs["positions"].detach().cpu().clone(),
            "hidden": kwargs["hidden_states"].detach().float().cpu().clone(),
        }
        out = orig(*args, **kwargs)
        c["out"] = out.detach().float().cpu().clone()
        c["argmax"] = drafter.compute_logits(out).argmax(-1).cpu().clone()
        calls.append(c)
        return out

    if eager:
        drafter.forward = wrapped
    params = SamplingParams(temperature=0, max_tokens=new, ignore_eos=True)
    lo, hi = (index, index + 1) if index >= 0 else (0, len(records))
    for i in range(lo, hi):
        calls.clear()
        ids = records[i]["prompt"]
        res = llm.generate([{"prompt_token_ids": ids}], params)
        gen = list(res[0].outputs[0].token_ids)
        keep = list(calls)
        torch.save({"prompt": ids, "gen": gen, "calls": keep}, out / f"vllm-mtp-{i}.pt")
        print("prompt", i, "calls", len(calls), flush=True)
    for m in llm.get_metrics():
        if "spec_decode" in m.name:
            print("metric", m.name, getattr(m, "values", getattr(m, "value", None)), flush=True)


if __name__ == "__main__":
    main()
