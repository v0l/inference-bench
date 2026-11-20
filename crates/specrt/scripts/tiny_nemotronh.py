import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURES = Path(__file__).parent.parent / "tests/fixtures"
BLOCK_TYPES = {"M": "linear_attention", "*": "full_attention", "-": "mlp", "E": "moe"}
MOE_EXTRA = {
    "hidden_size": 288,
    "n_routed_experts": 8,
    "num_experts_per_tok": 3,
    "moe_intermediate_size": 128,
    "moe_shared_expert_intermediate_size": 256,
    "routed_scaling_factor": 2.5,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
}
VARIANTS = {
    "tiny-nemotronh": ("M-*-M*M-", {}),
    "tiny-nemotronh-moe": ("ME*EMEM*E", MOE_EXTRA),
    "tiny-nemotronh-moe-mma": ("ME*EMEM*E", {**MOE_EXTRA, "hidden_size": 320}),
}

RAW = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "attention_bias": False,
    "chunk_size": 16,
    "conv_kernel": 4,
    "head_dim": 64,
    "hidden_size": 256,
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


def raw_config(name):
    pattern, extra = VARIANTS[name]
    return {**RAW, "hybrid_override_pattern": pattern, "num_hidden_layers": len(pattern), **extra}


def build_config(name):
    from transformers import NemotronHConfig

    kw = {k: v for k, v in raw_config(name).items() if k not in ("architectures", "model_type", "torch_dtype", "time_step_limit")}
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


EXPERT_STACKS = (".mixer.experts.up_proj", ".mixer.experts.down_proj")


def to_checkpoint(tensors):
    out = {}
    for name, v in tensors.items():
        name = "backbone." + name[len("model."):] if name.startswith("model.") else name
        stack = next((t for t in EXPERT_STACKS if name.endswith(t)), None)
        if stack is None:
            out[name] = v
            continue
        base, leaf = name[: -len(stack)], stack.rsplit(".", 1)[1]
        for i in range(v.shape[0]):
            out[f"{base}.mixer.experts.{i}.{leaf}.weight"] = v[i].contiguous()
    return out


def from_checkpoint(tensors):
    out, stacks = {}, {}
    for name, v in tensors.items():
        name = "model." + name[len("backbone."):] if name.startswith("backbone.") else name
        if ".mixer.experts." in name and name.endswith(".weight") and name.split(".mixer.experts.")[1].split(".")[0].isdigit():
            base, rest = name.split(".mixer.experts.")
            index, leaf = rest.split(".")[0], rest.split(".")[1]
            stacks.setdefault((base, leaf), {})[int(index)] = v
        else:
            out[name] = v
    for (base, leaf), parts in stacks.items():
        out[f"{base}.mixer.experts.{leaf}"] = torch.stack([parts[i] for i in sorted(parts)])
    return out


def make(name):
    from safetensors.torch import load_file, save_file
    from transformers import NemotronHForCausalLM

    cfg = build_config(name)
    torch.manual_seed(SEED)
    model = NemotronHForCausalLM(cfg).float().eval()
    pattern = VARIANTS[name][0]
    assert [b.block_type for b in model.model.layers] == [BLOCK_TYPES[c] for c in pattern]
    folder = FIXTURES / name
    path = folder / "model.safetensors"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_file(to_checkpoint(weights(model)), path)
        (folder / "config.json").write_text(json.dumps(raw_config(name), indent=2) + "\n")
    loaded = {k: v.float() for k, v in from_checkpoint(load_file(path)).items()}
    model.load_state_dict(loaded, strict=True)
    return model


def reference(name, out):
    model = make(name)
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None], use_cache=False).logits[0]
        seq = ids.tolist()
        for _ in range(GREEDY):
            step = model(torch.tensor(seq)[None], use_cache=False).logits[0, -1]
            seq.append(int(step.argmax()))
        greedy = seq[PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out / f"{name}.ref.bin")
    (out / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy, "vocab": 512}))
    print(name, "->", out)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        reference(name, out)


if __name__ == "__main__":
    main(sys.argv[1])
