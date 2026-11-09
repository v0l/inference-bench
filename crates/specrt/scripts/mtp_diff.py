import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from reference import TinyMtp, snapshot_dir

REPO = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"


def main():
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    torch.set_num_threads(16)
    snap = snapshot_dir(REPO)
    cfg = json.loads((snap / "config.json").read_text())["text_config"]
    cfg = {k: v for k, v in cfg.items() if k != "quantization_config"}
    config = Qwen3_5TextConfig(**cfg)
    config._attn_implementation = "eager"
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(snap / index[name], "pt") as f:
            return f.get_tensor(name)

    mtp = TinyMtp(config, {k: tensor(k) for k in index if k.startswith("mtp.")}).eval()
    embed = torch.nn.Embedding.from_pretrained(tensor("model.language_model.embed_tokens.weight").float())
    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    head.weight.data = tensor("lm_head.weight").float()

    class Shell:
        pass

    model = Shell()
    model.config = config
    model.model = Shell()
    model.model.embed_tokens = embed
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
    model.lm_head = head
    d = torch.load(sys.argv[1])
    with torch.inference_mode():
        for i, c in enumerate(d["calls"][:int(sys.argv[2])]):
            n = len(c["input_ids"])
            pos = c["positions"].reshape(-1)[:n] if c["positions"].dim() == 1 else c["positions"][0][:n]
            logits, g = mtp(model, c["input_ids"][None].long(), c["hidden"][None], pos)
            rel = float((g - c["out"]).abs().max() / c["out"].abs().max())
            mine = logits.argmax(-1)
            print(i, "rows", n, "out rel err", round(rel, 5), "argmax equal", (mine == c["argmax"]).tolist())


main()
