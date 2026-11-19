import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
FIXTURES = Path(__file__).parent.parent / "tests/fixtures"
NAME = "tiny-bamba"


def build_config():
    from transformers import BambaConfig

    return BambaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_layer_indices=[1, 3],
        mamba_n_heads=16,
        mamba_d_state=16,
        mamba_n_groups=2,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
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


def make():
    from safetensors.torch import load_file, save_file
    from transformers import BambaForCausalLM

    cfg = build_config()
    torch.manual_seed(SEED)
    model = BambaForCausalLM(cfg).float().eval()
    folder = FIXTURES / NAME
    path = folder / "model.safetensors"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        raw = json.loads(cfg.to_json_string())
        raw["time_step_limit"] = [0.0, None]
        (folder / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    model.load_state_dict({k: v.float() for k, v in load_file(path).items()}, strict=True)
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
