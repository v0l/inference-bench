import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURES = Path(__file__).parent.parent / "tests/fixtures"
NAME = "tiny-granitehybrid"


def build_config():
    from transformers import GraniteMoeHybridConfig

    return GraniteMoeHybridConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=256,
        shared_intermediate_size=512,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["mamba", "attention", "mamba", "attention"],
        mamba_n_heads=16,
        mamba_d_head=32,
        mamba_d_state=16,
        mamba_n_groups=2,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        position_embedding_type="nope",
        embedding_multiplier=4.0,
        attention_multiplier=0.125,
        residual_multiplier=0.22,
        logits_scaling=2.0,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
    )


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
        elif "embed_tokens" in name or name == "lm_head.weight":
            v = torch.randn(p.shape, generator=g)
        elif name.endswith("conv1d.weight"):
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


RENAMES = [
    (".block_sparse_moe.experts.gate_up_proj", ".block_sparse_moe.input_linear.weight"),
    (".block_sparse_moe.experts.down_proj", ".block_sparse_moe.output_linear.weight"),
    (".block_sparse_moe.router.weight", ".block_sparse_moe.router.layer.weight"),
]


def to_checkpoint(name):
    for a, b in RENAMES:
        if name.endswith(a):
            return name[: -len(a)] + b
    return name


def from_checkpoint(name):
    for a, b in RENAMES:
        if name.endswith(b):
            return name[: -len(b)] + a
    return name


def make():
    from safetensors.torch import load_file, save_file
    from transformers import GraniteMoeHybridForCausalLM

    cfg = build_config()
    torch.manual_seed(SEED)
    model = GraniteMoeHybridForCausalLM(cfg).float().eval()
    folder = FIXTURES / NAME
    path = folder / "model.safetensors"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_file({to_checkpoint(k): v for k, v in weights(model).items()}, path)
        raw = json.loads(cfg.to_json_string())
        raw["time_step_limit"] = [0.0, None]
        (folder / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    model.load_state_dict({from_checkpoint(k): v.float() for k, v in load_file(path).items()}, strict=True)
    return model


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = make()
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None]).logits[0]
        seq = ids.tolist()
        for _ in range(GREEDY):
            step = model(torch.tensor(seq)[None]).logits[0, -1]
            seq.append(int(step.argmax()))
        greedy = seq[PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out / f"{NAME}.ref.bin")
    (out / f"{NAME}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy, "vocab": 512}))
    print(NAME, "->", out)


if __name__ == "__main__":
    main(sys.argv[1])
