import json
import sys
from pathlib import Path

import numpy as np
import torch

PROMPT = 700
GREEDY = 32
ROWS = 4


def main(model_dir, out_dir):
    from transformers import Gemma3ForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = Gemma3ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(1000, 200000, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None].cuda()).logits[0, -ROWS:].float().cpu()
        greedy = model.generate(ids[None].cuda(), max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:].cpu()
    logits.numpy().astype(np.float32).tofile(out / "gemma3-1b.ref.bin")
    meta = {"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": logits.shape[1], "rows": ROWS}
    (out / "gemma3-1b.ref.json").write_text(json.dumps(meta))
    print("gemma3-1b ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
