import json
import sys
from pathlib import Path

import numpy as np
import torch

PROMPT = 300
GREEDY = 32
ROWS = 4

REPOS = {
    "olmo2-1b": "allenai/OLMo-2-0425-1B",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "granite-3.3-2b": "ibm-granite/granite-3.3-2b-base",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}


def reference(name, out, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(REPOS[name], torch_dtype=dtype, attn_implementation="eager").cuda().eval()
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(1000, 30000, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None].cuda()).logits[0, -ROWS:].float().cpu()
        greedy = model.generate(ids[None].cuda(), max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:].cpu()
    logits.numpy().astype(np.float32).tofile(out / f"{name}.ref.bin")
    meta = {"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": logits.shape[1], "rows": ROWS}
    (out / f"{name}.ref.json").write_text(json.dumps(meta))
    print(name, "->", out)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    f32 = sys.argv[2] == "f32"
    for name in (sys.argv[3:] if f32 else sys.argv[2:]) or list(REPOS):
        reference(name, out, torch.float32 if f32 else torch.bfloat16)
