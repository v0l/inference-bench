import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-gemma3"


def config():
    from transformers import Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        query_pre_attn_scalar=48,
        sliding_window=16,
        layer_types=["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2 + ["full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        max_position_embeddings=512,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
    )
    for key, value in json.loads(os.environ.get("G3_OVERRIDE", "{}")).items():
        setattr(cfg, key, value)
    return cfg


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name == "lm_head.weight":
            continue
        if p.dim() == 1:
            v = 0.2 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name:
            v = torch.randn(p.shape, generator=g) * 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build():
    from safetensors.torch import load_file, save_file
    from transformers import Gemma3ForCausalLM

    torch.manual_seed(SEED)
    cfg = config()
    model = Gemma3ForCausalLM(cfg).float().eval()
    path = FIXTURE / "model.safetensors"
    if not path.exists():
        FIXTURE.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        cfg.to_json_file(FIXTURE / "config.json")
    state = {k: v.float() for k, v in load_file(path).items()}
    model.load_state_dict(state, strict=False)
    model.tie_weights()
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
    logits.float().numpy().astype(np.float32).tofile(out / f"tiny-gemma3{os.environ.get('G3_TAG', '')}.ref.bin")
    meta = {"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": 512}
    (out / f"tiny-gemma3{os.environ.get('G3_TAG', '')}.ref.json").write_text(json.dumps(meta))
    print("tiny-gemma3 ->", out)




def layer_states(out_dir):
    out = Path(out_dir)
    model = build()
    meta = json.loads((out / "tiny-gemma3.ref.json").read_text())
    ids = torch.tensor(meta["tokens"])
    seen = []
    hooks = [l.register_forward_hook(lambda _m, _i, o: seen.append(o[0] if isinstance(o, tuple) else o)) for l in model.model.layers]
    with torch.inference_mode():
        model(ids[None])
    for h in hooks:
        h.remove()
    torch.stack([s[0] for s in seen]).float().numpy().astype(np.float32).tofile(out / "tiny-gemma3.layers.bin")


if __name__ == "__main__":
    main(sys.argv[1])
    layer_states(sys.argv[1])
