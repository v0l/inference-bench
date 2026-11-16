import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

LAYERS = 4
PROMPT = 300
GREEDY = 16
ROWS = 4


def snapshot(repo):
    base = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    return next(p for p in base.iterdir() if (p / "config.json").exists())


def main(repo, name, out_dir):
    from transformers import AutoConfig, MistralForCausalLM

    snap = snapshot(repo)
    cfg = AutoConfig.from_pretrained(snap)
    cfg.quantization_config = None
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else LAYERS
    cfg.num_hidden_layers = layers
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = lambda n: n.startswith(("model.embed_tokens", "model.norm", "lm_head")) or (n.startswith("model.layers.") and int(n.split(".")[2]) < layers)
    with torch.device("meta"):
        model = MistralForCausalLM(cfg)
    model = model.to_empty(device="cuda").to(torch.bfloat16)
    params = dict(model.named_parameters())
    for file in sorted({index[n] for n in index if wanted(n)}):
        with safe_open(snap / file, "pt", device="cuda") as f:
            for n in f.keys():
                if not wanted(n) or n.endswith("weight_scale") or n.endswith("input_scale"):
                    continue
                w = f.get_tensor(n)
                if w.dtype == torch.float8_e4m3fn:
                    w = w.float() * f.get_tensor(n + "_scale").float()
                params[n].data.copy_(w.to(torch.bfloat16))
                del w
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg).cuda()
    model.eval()
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(1000, 100000, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None].cuda()).logits[0, -ROWS:].float().cpu()
        greedy = model.generate(ids[None].cuda(), max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:].cpu()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logits.numpy().astype(np.float32).tofile(out / f"{name}.ref.bin")
    (out / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": logits.shape[1], "rows": ROWS}))
    print(name, "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
