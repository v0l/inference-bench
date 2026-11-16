import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURES = Path(__file__).parent.parent / "tests/fixtures"

COMMON = dict(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    eos_token_id=None,
    bos_token_id=None,
    pad_token_id=None,
    attn_implementation="eager",
)


def families():
    from transformers import GraniteConfig, MistralConfig, Olmo2Config, Phi3Config, SmolLM3Config

    return {
        "tiny-mistral": (MistralConfig, dict(COMMON, head_dim=64, sliding_window=None, tie_word_embeddings=False, rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})),
        "tiny-phi3": (Phi3Config, dict(COMMON, num_key_value_heads=4, tie_word_embeddings=False, rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5})),
        "tiny-phi3-hd96": (Phi3Config, dict(COMMON, hidden_size=192, intermediate_size=384, num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False, rope_parameters={"rope_type": "default", "rope_theta": 10000.0})),
        "tiny-smollm3": (SmolLM3Config, dict(COMMON, no_rope_layers=[1, 0, 1, 0], use_sliding_window=False, tie_word_embeddings=True, rope_parameters={"rope_type": "default", "rope_theta": 2000000.0})),
        "tiny-olmo2": (Olmo2Config, dict(COMMON, num_key_value_heads=4, tie_word_embeddings=False, rope_parameters={"rope_type": "default", "rope_theta": 500000.0})),
        "tiny-granite": (
            GraniteConfig,
            dict(COMMON, tie_word_embeddings=True, embedding_multiplier=12.0, attention_multiplier=0.125, residual_multiplier=0.22, logits_scaling=8.0, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0}),
        ),
    }


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        if p.dim() == 1:
            v = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name or name == "lm_head.weight":
            v = torch.randn(p.shape, generator=g) * 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build(name):
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    config_class, kwargs = families()[name]
    torch.manual_seed(SEED)
    cfg = config_class(**kwargs)
    model = AutoModelForCausalLM.from_config(cfg).float().eval()
    fixture = FIXTURES / name
    path = fixture / "model.safetensors"
    if not path.exists():
        fixture.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        cfg.to_json_file(fixture / "config.json")
    state = {k: v.float() for k, v in load_file(path).items()}
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    return model


def reference(name, out_dir):
    model = build(name)
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None]).logits[0]
        greedy = model.generate(ids[None], max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out_dir / f"{name}.ref.bin")
    (out_dir / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": 512}))
    print(name, "->", out_dir)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:] or list(families()):
        reference(name, out)
