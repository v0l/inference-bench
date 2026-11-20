import json
import sys
from pathlib import Path

import numpy as np
import torch

PROMPT = 96
TEXT = (
    "The Irish Sea separates the island of Ireland from Great Britain. It is about 210 kilometres long and "
    "its widest point measures roughly 240 kilometres. Ferries cross it every day between Dublin, Holyhead, "
    "Liverpool and Belfast, carrying passengers, cars and freight. Fishing boats work the same waters for "
    "cod, whiting and prawns, while wind farms now stand on the shallow banks off the coast. In winter the "
    "weather can turn quickly, and the crossing from Dublin Port to Holyhead"
)
GREEDY = 16
NAME = "nemotron-nano-9b-v2"


def main(checkpoint, out_dir, dtype="float32"):
    from transformers import AutoModelForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=getattr(torch, dtype), low_cpu_mem_usage=True, device_map="cuda").eval()
    vocab = model.config.vocab_size
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ids = torch.tensor(tokenizer(TEXT)["input_ids"][:PROMPT]).cuda()
    with torch.inference_mode():
        logits = model(ids[None], use_cache=False).logits[0]
        seq = ids.tolist()
        for _ in range(GREEDY):
            seq.append(int(model(torch.tensor(seq, device="cuda")[None], use_cache=False).logits[0, -1].argmax()))
    logits.float().cpu().numpy().astype(np.float32).tofile(out / f"{NAME}.ref.bin")
    (out / f"{NAME}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": seq[PROMPT:], "vocab": vocab}))
    print(NAME, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
