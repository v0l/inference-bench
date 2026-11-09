import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from reference import TinyMtp, snapshot_dir

REPO = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"


def build(variant):
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    torch.set_num_threads(4)
    snap = snapshot_dir(REPO)
    cfg = json.loads((snap / "config.json").read_text())["text_config"]
    cfg = {k: v for k, v in cfg.items() if k != "quantization_config"}
    config = Qwen3_5TextConfig(**cfg)
    config._attn_implementation = "eager"
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(snap / index[name], "pt") as f:
            return f.get_tensor(name)

    dt = torch.bfloat16 if "bf16" in variant else torch.float32
    mtp = TinyMtp(config, {k: tensor(k) for k in index if k.startswith("mtp.")}).eval().to("cuda", dt)
    embed_w = tensor("model.language_model.embed_tokens.weight").to("cuda")
    head_w = tensor("lm_head.weight").to("cuda", dt)

    class Embed:
        def __call__(self, tokens):
            return embed_w[tokens].to(dt)

    class Head:
        def __call__(self, g):
            return g @ head_w.T

    embed = Embed()
    head = Head()

    class Shell:
        pass

    model = Shell()
    model.config = config
    model.model = Shell()
    model.model.embed_tokens = embed
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config).to("cuda")
    model.lm_head = head
    if "fp8kv" in variant:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m5

        eager = m5.eager_attention_forward

        def fp8_attention(module, query, key, value, *a, **k):
            q8 = lambda t: t.to(torch.float8_e4m3fn).to(t.dtype)
            return eager(module, query, q8(key), q8(value), *a, **k)

        m5.eager_attention_forward = fp8_attention
    return mtp, model, dt


def forward(run, c):
    mtp, model, dt = run
    n = len(c["input_ids"])
    pos = c["positions"].reshape(-1)[:n] if c["positions"].dim() == 1 else c["positions"][0][:n]
    with torch.inference_mode():
        logits, g = mtp(model, c["input_ids"][None].long().cuda(), c["hidden"][None].to("cuda", dt), pos.cuda())
    return logits.float().cpu()


def main():
    d = torch.load(sys.argv[1])
    variant = sys.argv[3] if len(sys.argv) > 3 else "f32"
    run = build(variant)
    for i, c in enumerate(d["calls"][:int(sys.argv[2])]):
        logits = forward(run, c)
        n = len(c["input_ids"])
        print(variant, i, "rows", n, "argmax equal", int((logits.argmax(-1) == c["argmax"]).sum()), "of", n)


if __name__ == "__main__":
    main()
