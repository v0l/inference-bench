import os
import json

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

LAYERS = 3
PROMPT = int(os.environ.get("NEXT_PROMPT", 300))
GREEDY = 16
ROWS = 8
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant(f, name):
    packed = f.get_tensor(name + ".weight")
    scale = f.get_tensor(name + ".weight_scale").float()
    global_scale = f.get_tensor(name + ".weight_scale_2").float()
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1).long()
    grid = E2M1.to(packed.device)
    values = grid[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    rows, cols = values.shape
    return (values.reshape(rows, cols // 16, 16) * (scale * global_scale)[..., None]).reshape(rows, cols)


def main(snap, out_dir, layers):
    from transformers import AutoConfig, Qwen3NextForCausalLM

    snap = Path(snap)
    cfg = AutoConfig.from_pretrained(snap)
    cfg.quantization_config = None
    cfg.num_hidden_layers = layers
    cfg._attn_implementation = "eager"
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = lambda n: n.startswith(("model.embed_tokens", "model.norm", "lm_head")) or (n.startswith("model.layers.") and int(n.split(".")[2]) < layers)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = Qwen3NextForCausalLM(cfg)
    model = model.to_empty(device="cuda")
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

    model.model.rotary_emb = Qwen3NextRotaryEmbedding(cfg).to("cuda")
    params = dict(model.named_parameters())
    inter = cfg.moe_intermediate_size
    for file in sorted({index[n] for n in index if wanted(n)}):
        with safe_open(snap / file, "pt", device="cuda") as f:
            names = [n for n in f.keys() if wanted(n)]
            for n in names:
                if n.endswith(("weight_scale", "weight_scale_2", "input_scale", "k_scale", "v_scale")):
                    continue
                if ".mlp.experts." in n:
                    parts = n.split(".")
                    layer, expert, proj = parts[2], int(parts[5]), parts[6]
                    base = n.rsplit(".weight", 1)[0]
                    w = dequant(f, base).to(torch.bfloat16)
                    prefix = f"model.layers.{layer}.mlp.experts."
                    if proj == "down_proj":
                        params[prefix + "down_proj"].data[expert].copy_(w)
                    else:
                        at = slice(0, inter) if proj == "gate_proj" else slice(inter, 2 * inter)
                        params[prefix + "gate_up_proj"].data[expert, at].copy_(w)
                    continue
                w = f.get_tensor(n)
                if w.dtype == torch.uint8:
                    w = dequant(f, n.rsplit(".weight", 1)[0])
                params[n].data.copy_(w.to(torch.bfloat16))
                del w
    model.eval()
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from reference import REAL_PROMPTS

    tok = AutoTokenizer.from_pretrained(snap)
    ids = torch.tensor(tok(" ".join(REAL_PROMPTS)).input_ids[:PROMPT])
    with torch.inference_mode():
        logits = model(ids[None].cuda(), use_cache=False).logits[0, -ROWS:].float().cpu()
        try:
            greedy = model.generate(ids[None].cuda(), max_new_tokens=GREEDY, do_sample=False)[0, len(ids):].cpu()
        except ValueError:
            greedy = torch.zeros(0, dtype=torch.long)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"qwen3-next-{layers}l"
    logits.numpy().astype(np.float32).tofile(out / f"{name}.ref.bin")
    (out / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": logits.shape[1], "rows": ROWS}))
    print(name, "->", out, "peak GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else LAYERS)
