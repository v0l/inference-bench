import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-gemma4"


def config():
    from transformers import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        sliding_window=16,
        layer_types=["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2 + ["full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        },
        final_logit_softcapping=30.0,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=512,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        max_position_embeddings=512,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
    )
    for key, value in json.loads(os.environ.get("G4_OVERRIDE", "{}")).items():
        setattr(cfg, key, value)
    return cfg


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name == "lm_head.weight":
            continue
        if name.endswith("layer_scalar"):
            v = 0.5 + torch.rand(p.shape, generator=g)
        elif p.dim() == 1:
            v = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name:
            v = torch.randn(p.shape, generator=g) * 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build():
    from safetensors.torch import load_file, save_file
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(SEED)
    cfg = config()
    model = Gemma4ForCausalLM(cfg).float().eval()
    path = FIXTURE / "model.safetensors"
    if not path.exists():
        FIXTURE.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        cfg.to_json_file(FIXTURE / "config.json")
        published = json.loads((FIXTURE / "config.json").read_text())
        published.pop("per_layer_config", None)
        published.update({"global_head_dim": 64, "num_global_key_value_heads": 1})
        (FIXTURE / "config.json").write_text(json.dumps(published, indent=2, sort_keys=True) + "\n")
    state = {k: v.float() for k, v in load_file(path).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(m == "lm_head.weight" for m in missing), missing
    model.tie_weights()
    reread = Gemma4TextConfig.from_json_file(FIXTURE / "config.json")
    shape = lambda c: [(c.per_layer_config[i].head_dim, c.per_layer_config[i].num_key_value_heads) for i in range(c.num_hidden_layers)]
    assert shape(reread) == shape(cfg), (shape(reread), shape(cfg))
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
    logits.float().numpy().astype(np.float32).tofile(out / "tiny-gemma4.ref.bin")
    meta = {"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": 512}
    (out / "tiny-gemma4.ref.json").write_text(json.dumps(meta))
    print("tiny-gemma4 ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
