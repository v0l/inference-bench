import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MAXLEN = 48


def prompts_text():
    sys.path.insert(0, str(HERE))
    from reference import REAL_PROMPTS
    from vllm_logprobs import EXTRA

    return REAL_PROMPTS + EXTRA[:3]


def cmd_prompts(model, out):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    ids = [tok(t).input_ids[:MAXLEN] for t in prompts_text()]
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.json").write_text(json.dumps(ids))
    print("prompts", [len(i) for i in ids])


def load_prompts(out):
    return json.loads((out / "prompts.json").read_text())


def cmd_fp32(model, out):
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32, attn_implementation="eager").to(os.environ.get("FP32_DEVICE", "cuda")).eval()
    rows = []
    with torch.inference_mode():
        for p in load_prompts(out):
            rows.append(m(torch.tensor([p]).to(m.device)).logits[0].float().cpu().numpy())
    full = np.concatenate(rows)
    np.save(out / "fp32.npy", full)
    full.astype(np.float32).tofile(out / "fp32.f32")
    print("fp32", sum(len(r) for r in rows), "rows")


def cmd_vllm(model, out):
    from vllm import LLM, SamplingParams

    vocab = np.load(out / "fp32.npy", mmap_mode="r").shape[1]
    llm = LLM(model, max_model_len=256, gpu_memory_utilization=float(os.environ.get("VLLM_UTIL", "0.12")), enable_prefix_caching=False, enforce_eager=False, max_logprobs=-1)
    rows = []
    for p in load_prompts(out):
        o = llm.generate([{"prompt_token_ids": p}], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=-1, logprobs=-1), use_tqdm=False)[0]
        lp = np.full((len(p), vocab), -np.inf, dtype=np.float32)
        dists = [d for d in o.prompt_logprobs[1:]] + [o.outputs[0].logprobs[0]]
        for i, d in enumerate(dists):
            idx = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            lp[i, idx] = np.fromiter((v.logprob for v in d.values()), dtype=np.float32, count=len(d))
        rows.append(lp)
        del o
    np.save(out / "vllm.npy", np.concatenate(rows))
    print("vllm", sum(len(r) for r in rows), "rows")


def logsm(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def first_rows(prompts):
    out, at = [], 0
    for p in prompts:
        out.append(at)
        at += len(p)
    return out


def cmd_compare(name, out, files):
    prompts = load_prompts(out)
    skip = set(first_rows(prompts))
    keep = np.array([i for i in range(sum(len(p) for p in prompts)) if i not in skip])
    ref = np.load(out / "fp32.npy")
    vocab = ref.shape[1]
    p = logsm(ref[keep])
    pp = np.exp(p)
    result = {"model": name, "rows": int(len(keep))}
    for label, path in files:
        if path.suffix == ".f32":
            flat = np.fromfile(path, dtype=np.float32)
            a = flat.reshape(ref.shape[0], -1)
        else:
            a = np.load(path)
        if a.shape[1] < vocab:
            a = np.pad(a, ((0, 0), (0, vocab - a.shape[1])), constant_values=-1e30)
        a = a[:, :vocab]
        q = logsm(a[keep])
        kl = (pp * (p - q)).sum(axis=1)
        top = int((ref[keep].argmax(1) == a[keep].argmax(1)).sum())
        result[label] = {"mean_kl": float(kl.mean()), "max_kl": float(kl.max()), "top1": top, "max_abs": float(np.abs(a[keep] - ref[keep]).max()) if label != "vllm" else None}
        print(f"{name} {label:12s} mean KL {kl.mean():.3e} max KL {kl.max():.3e} top1 {top}/{len(keep)}")
    (out / "compare.json").write_text(json.dumps(result))


if __name__ == "__main__":
    cmd, model, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if cmd == "prompts":
        cmd_prompts(model, out)
    elif cmd == "fp32":
        cmd_fp32(model, out)
    elif cmd == "vllm":
        cmd_vllm(model, out)
    elif cmd == "compare":
        cmd_compare(Path(model).name, out, [(a.split("=")[0], Path(a.split("=")[1])) for a in sys.argv[4:]])
