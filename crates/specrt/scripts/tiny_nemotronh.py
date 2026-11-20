import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURES = Path(__file__).parent.parent / "tests/fixtures"
NAME = "tiny-nemotronh"
PATTERN = "M-*-M*M-"

RAW = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "attention_bias": False,
    "chunk_size": 16,
    "conv_kernel": 4,
    "head_dim": 64,
    "hidden_size": 256,
    "hybrid_override_pattern": PATTERN,
    "intermediate_size": 512,
    "layer_norm_epsilon": 1e-5,
    "mamba_head_dim": 32,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 16,
    "mamba_proj_bias": False,
    "max_position_embeddings": 512,
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "n_groups": 2,
    "num_attention_heads": 4,
    "num_hidden_layers": len(PATTERN),
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "ssm_state_size": 16,
    "tie_word_embeddings": False,
    "time_step_limit": [0.0, None],
    "torch_dtype": "bfloat16",
    "use_bias": False,
    "use_conv_bias": True,
    "vocab_size": 512,
    "bos_token_id": None,
    "eos_token_id": None,
    "pad_token_id": None,
}


def build_config():
    from transformers import NemotronHConfig

    kw = {k: v for k, v in RAW.items() if k not in ("architectures", "model_type", "torch_dtype", "time_step_limit")}
    cfg = NemotronHConfig(**kw)
    cfg._attn_implementation = "eager"
    return cfg


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name.endswith(".A_log"):
            v = torch.log(torch.rand(p.shape, generator=g) * 4.0 + 0.5)
        elif name.endswith(".dt_bias"):
            v = -1.0 + 0.5 * torch.randn(p.shape, generator=g)
        elif p.dim() == 1:
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        elif "embeddings" in name or name == "lm_head.weight":
            v = torch.randn(p.shape, generator=g)
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def to_checkpoint(name):
    return "backbone." + name[len("model."):] if name.startswith("model.") else name


def from_checkpoint(name):
    return "model." + name[len("backbone."):] if name.startswith("backbone.") else name


def make():
    from safetensors.torch import load_file, save_file
    from transformers import NemotronHForCausalLM

    cfg = build_config()
    torch.manual_seed(SEED)
    model = NemotronHForCausalLM(cfg).float().eval()
    assert [b.block_type for b in model.model.layers] == [
        {"M": "linear_attention", "*": "full_attention", "-": "mlp"}[c] for c in PATTERN
    ]
    folder = FIXTURES / NAME
    path = folder / "model.safetensors"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_file({to_checkpoint(k): v for k, v in weights(model).items()}, path)
        (folder / "config.json").write_text(json.dumps(RAW, indent=2) + "\n")
    loaded = {from_checkpoint(k): v.float() for k, v in load_file(path).items()}
    model.load_state_dict(loaded, strict=True)
    return model


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = make()
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None], use_cache=False).logits[0]
        seq = ids.tolist()
        for _ in range(GREEDY):
            step = model(torch.tensor(seq)[None], use_cache=False).logits[0, -1]
            seq.append(int(step.argmax()))
        greedy = seq[PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out / f"{NAME}.ref.bin")
    (out / f"{NAME}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy, "vocab": 512}))
    print(NAME, "->", out)


if __name__ == "__main__":
    main(sys.argv[1])
