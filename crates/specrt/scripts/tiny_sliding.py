import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
WINDOW = 16
FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-qwen3-sliding"


def config():
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=WINDOW,
        use_sliding_window=True,
        max_window_layers=0,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
        max_position_embeddings=512,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
    )


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if p.dim() == 1:
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name:
            v = torch.randn(p.shape, generator=g)
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build():
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(SEED)
    cfg = config()
    model = Qwen3ForCausalLM(cfg).float().eval()
    path = FIXTURE / "model.safetensors"
    if not path.exists():
        FIXTURE.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        cfg.to_json_file(FIXTURE / "config.json")
    model.load_state_dict({k: v.float() for k, v in load_file(path).items()}, strict=True)
    return model


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build()
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None]).logits[0]
        greedy = model.generate(ids[None], max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out / "tiny-qwen3-sliding.ref.bin")
    meta = {"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": 512, "window": WINDOW}
    (out / "tiny-qwen3-sliding.ref.json").write_text(json.dumps(meta))
    print("tiny-qwen3-sliding ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
