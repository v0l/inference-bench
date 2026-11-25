import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 0
PROMPT = 96
GREEDY = 32
VOCAB = 512
FIXTURE = Path(__file__).parent.parent / "tests/fixtures" / os.environ.get("G3N_FIXTURE", "tiny-gemma3n")


def config():
    from transformers import Gemma3nTextConfig

    return Gemma3nTextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=384,
        hidden_size=128,
        hidden_size_per_layer_input=16,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=2,
        laurel_rank=8,
        altup_num_inputs=4,
        activation_sparsity_pattern=[0.95, 0.0, 0.0, 0.0],
        final_logit_softcapping=30.0,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        max_position_embeddings=512,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
    )


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name == "lm_head.weight":
            continue
        if p.dim() == 1:
            v = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name:
            v = torch.randn(p.shape, generator=g) * 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build():
    from safetensors.torch import load_file, save_file
    from transformers import Gemma3nForCausalLM

    torch.manual_seed(SEED)
    cfg = config()
    model = Gemma3nForCausalLM(cfg).float().eval()
    path = FIXTURE / "model.safetensors"
    if not path.exists():
        FIXTURE.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        cfg.to_json_file(FIXTURE / "config.json")
    state = {k: v.float() for k, v in load_file(path).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(m == "lm_head.weight" for m in missing), missing
    model.tie_weights()
    return model


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build()
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, VOCAB, (PROMPT,), generator=g)
    seen = []
    hooks = [l.register_forward_hook(lambda _m, _i, o: seen.append(o[0] if isinstance(o, tuple) else o)) for l in model.model.layers]
    with torch.inference_mode():
        logits = model(ids[None]).logits[0]
    for h in hooks:
        h.remove()
    with torch.inference_mode():
        greedy = model.generate(ids[None], max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out / f"{FIXTURE.name}.ref.bin")
    streams = torch.stack([s[:, 0].permute(1, 0, 2) for s in seen[: len(model.model.layers)]])
    streams.float().numpy().astype(np.float32).tofile(out / f"{FIXTURE.name}.streams.bin")
    meta = {"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": VOCAB, "streams": list(streams.shape)}
    (out / f"{FIXTURE.name}.ref.json").write_text(json.dumps(meta))
    print(FIXTURE.name, "->", out, meta["streams"])


if __name__ == "__main__":
    main(sys.argv[1])
