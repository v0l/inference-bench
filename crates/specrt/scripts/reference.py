import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "The spec-driven runtime reads a TOML file describing the model, renders CUDA "
    "templates for every operator, and compiles them at load time. A paged key value "
    "cache keeps the attention state between decode steps, so"
)
GREEDY = 32
TINY_SEED = 0
TINY_PREFILL = 1024
TINY_DECODE = 32
TINY_PROMPT = 48
TINY_GREEDY = 64


def tiny_config(head_dim=128):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
        attn_output_gate=True,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
    )


def tiny_weights(model):
    g = torch.Generator().manual_seed(TINY_SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name.endswith("A_log"):
            v = torch.empty(p.shape).uniform_(-5.5, -1.0, generator=g)
        elif name.endswith("dt_bias"):
            v = 2.8 + 4.0 * torch.randn(p.shape, generator=g)
        elif name.endswith("linear_attn.norm.weight"):
            v = torch.empty(p.shape).uniform_(0.8, 1.0, generator=g)
        elif p.dim() == 1:
            v = 0.1 * torch.randn(p.shape, generator=g)
        elif "conv1d" in name:
            v = 0.3 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name:
            v = torch.randn(p.shape, generator=g)
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def tiny_model(fixture, head_dim=128):
    from safetensors.torch import load_file, save_file
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(TINY_SEED)
    config = tiny_config(head_dim)
    model = Qwen3_5ForCausalLM(config).float().eval()
    weights = fixture / "model.safetensors"
    if not weights.exists():
        fixture.mkdir(parents=True, exist_ok=True)
        save_file(tiny_weights(model), weights)
        config.to_json_file(fixture / "config.json")
    state = {k: v.float() for k, v in load_file(weights).items()}
    model.load_state_dict(state, strict=True)
    return model


def tiny_tokens(n, salt):
    g = torch.Generator().manual_seed(TINY_SEED + salt)
    return torch.randint(0, 1024, (n,), generator=g)


def layer_outputs(model):
    seen = []
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out: seen.append(out[0] if isinstance(out, tuple) else out))
        for layer in model.model.layers
    ]
    return seen, hooks


def linear_states(cache, model):
    conv, rec = [], []
    for i, kind in enumerate(model.config.layer_types):
        if kind == "linear_attention":
            layer = cache.layers[i]
            conv.append(layer.conv_states[0][0])
            rec.append(layer.recurrent_states[0][0])
    return torch.stack(conv), torch.stack(rec)


def save(out_dir, name, arrays, meta):
    for key, value in arrays.items():
        value.detach().float().numpy().astype(np.float32).tofile(out_dir / f"{name}.{key}.bin")
    (out_dir / f"{name}.json").write_text(json.dumps(meta))


def tiny(fixture, out_dir, head_dim=128):
    from transformers import DynamicCache

    torch.set_num_threads(4)
    model = tiny_model(fixture, head_dim)
    prefix = "tiny" if head_dim == 128 else f"tiny{head_dim}"
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    with torch.inference_mode():
        ids = tiny_tokens(TINY_PREFILL, 1)
        seen, hooks = layer_outputs(model)
        cache = DynamicCache(config=model.config)
        logits = model(ids[None], past_key_values=cache, use_cache=True).logits[0]
        for h in hooks:
            h.remove()
        conv, rec = linear_states(cache, model)
        save(
            out_dir,
            f"{prefix}-prefill",
            {"layers": torch.stack([s[0] for s in seen]), "logits": logits, "conv": conv, "state": rec},
            {"tokens": ids.tolist(), **versions},
        )
        ids = tiny_tokens(TINY_DECODE, 2)
        seen, hooks = layer_outputs(model)
        cache = DynamicCache(config=model.config)
        steps = []
        for t in ids:
            steps.append(model(t.view(1, 1), past_key_values=cache, use_cache=True).logits[0, -1])
        for h in hooks:
            h.remove()
        conv, rec = linear_states(cache, model)
        per_step = torch.stack([s[0, -1] for s in seen]).view(TINY_DECODE, -1, model.config.hidden_size)
        save(
            out_dir,
            f"{prefix}-decode",
            {"layers": per_step, "logits": torch.stack(steps), "conv": conv, "state": rec},
            {"tokens": ids.tolist(), **versions},
        )
        ids = tiny_tokens(TINY_PROMPT, 3)
        logits = model(ids[None]).logits[0]
        greedy = model.generate(ids[None], max_new_tokens=TINY_GREEDY, do_sample=False)[0, TINY_PROMPT:]
        save(out_dir, f"{prefix}-greedy", {"logits": logits}, {"tokens": ids.tolist(), "greedy": greedy.tolist(), **versions})
    print("tiny", fixture, "->", out_dir)


def main():
    if sys.argv[1] == "tiny":
        tiny(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 128)
        return
    spec_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    spec = tomllib.loads(spec_path.read_text())
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(spec["repo"])
    config = AutoConfig.from_pretrained(spec["repo"])
    config.sliding_window = None
    scaling = spec["rope"].get("scaling")
    if scaling:
        original = getattr(config, "rope_parameters", None) or {}
        theta = original.get("rope_theta", getattr(config, "rope_theta", None))
        params = {
            "rope_theta": theta,
            "rope_type": scaling["kind"],
            "factor": scaling["factor"],
            **{k: v for k, v in scaling.items() if k not in ("kind", "factor", "original_max_positions")},
            **({"original_max_position_embeddings": int(scaling["original_max_positions"])} if "original_max_positions" in scaling else {}),
        }
        config.rope_parameters = params
        config.rope_scaling = params
    if spec["mlp"]["activation"] == "geglu":
        config.hidden_act = "gelu_pytorch_tanh"
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], config=config, torch_dtype=torch.float32)
    model.eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model(ids).logits[0].float().numpy()
        greedy = model.generate(ids, max_new_tokens=GREEDY, do_sample=False)[0, ids.shape[1]:]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = spec["name"]
    logits.astype(np.float32).tofile(out_dir / f"{name}.ref.bin")
    meta = {
        "tokens": ids[0].tolist(),
        "vocab": logits.shape[1],
        "greedy": greedy.tolist(),
        "greedy_text": tok.decode(greedy),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (out_dir / f"{name}.ref.json").write_text(json.dumps(meta))
    print(name, len(meta["tokens"]), "tokens", repr(meta["greedy_text"]))


if __name__ == "__main__":
    main()
