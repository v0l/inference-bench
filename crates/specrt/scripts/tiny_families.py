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
    if name in MIXES:
        return mix_reference(name, out_dir)
    model = build(name)
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = model(ids[None]).logits[0]
        greedy = model.generate(ids[None], max_new_tokens=GREEDY, do_sample=False)[0, PROMPT:]
    logits.float().numpy().astype(np.float32).tofile(out_dir / f"{name}.ref.bin")
    (out_dir / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy.tolist(), "vocab": 512}))
    print(name, "->", out_dir)


MIXES = {
    "tiny-sandwich-scaled": dict(kv_heads=2, pre=False, sandwich=True, qk_full=True, residual=0.22, theta=500000.0, tie=False),
    "tiny-sandwich-pre-qknorm": dict(kv_heads=2, pre=True, sandwich=True, qk_full=True, theta=1000000.0, tie=True),
    "tiny-scaled-qknorm": dict(kv_heads=2, pre=True, qk_full=True, residual=0.22, embed=12.0, attention=0.125, logits_div=8.0, theta=10000000.0, tie=True),
}


def mix_config(name):
    c = dict(pre=True, sandwich=False, qk_full=False, residual=1.0, embed=1.0, attention=None, logits_div=1.0, tie=False)
    c.update(MIXES[name])
    return c


def mix_names(c):
    names = ["model.embed_tokens.weight", "model.norm.weight"] + ([] if c["tie"] else ["lm_head.weight"])
    for i in range(COMMON["num_hidden_layers"]):
        p = f"model.layers.{i}"
        names += [f"{p}.self_attn.{w}_proj.weight" for w in "qkvo"] + [f"{p}.mlp.{w}_proj.weight" for w in ("gate", "up", "down")]
        names.append(f"{p}.post_attention_layernorm.weight")
        if c["pre"]:
            names.append(f"{p}.input_layernorm.weight")
        if c["sandwich"]:
            names.append(f"{p}.post_feedforward_layernorm.weight")
        if c["sandwich"] and c["pre"]:
            names.append(f"{p}.pre_feedforward_layernorm.weight")
        if c["qk_full"]:
            names += [f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight"]
    return sorted(names)


def mix_shape(name, c):
    h, i, v = COMMON["hidden_size"], COMMON["intermediate_size"], COMMON["vocab_size"]
    hd = h // COMMON["num_attention_heads"]
    kv = c["kv_heads"] * hd
    for key, shape in {"embed_tokens": (v, h), "lm_head": (v, h), "q_proj": (h, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, h), "gate_proj": (i, h), "up_proj": (i, h), "down_proj": (h, i), "k_norm": (kv,)}.items():
        if key in name:
            return shape
    return (h,)


def mix_weights(name):
    c = mix_config(name)
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for n in mix_names(c):
        shape = mix_shape(n, c)
        if len(shape) == 1:
            w = 1.0 + 0.2 * torch.randn(shape, generator=g)
        elif "embed_tokens" in n or n == "lm_head.weight":
            w = torch.randn(shape, generator=g) * 0.5
        else:
            w = torch.randn(shape, generator=g) / shape[1] ** 0.5
        out[n] = w.to(torch.bfloat16).contiguous()
    return out


def rms(x, w, eps=COMMON["rms_norm_eps"]):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def mix_forward(c, w, ids):
    heads, kvh = COMMON["num_attention_heads"], c["kv_heads"]
    hd = COMMON["hidden_size"] // heads
    t = ids.shape[0]
    inv = 1.0 / (c["theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    freqs = torch.outer(torch.arange(t, dtype=torch.float32), inv)
    cos, sin = torch.cat((freqs, freqs), -1).cos(), torch.cat((freqs, freqs), -1).sin()
    scale = c["attention"] if c["attention"] is not None else hd ** -0.5
    mask = torch.full((t, t), float("-inf")).triu(1)
    x = w["model.embed_tokens.weight"][ids] * c["embed"]
    for i in range(COMMON["num_hidden_layers"]):
        p = lambda n: w[f"model.layers.{i}.{n}.weight"]
        h = rms(x, p("input_layernorm")) if c["pre"] else x
        q, k, v = h @ p("self_attn.q_proj").T, h @ p("self_attn.k_proj").T, h @ p("self_attn.v_proj").T
        if c["qk_full"]:
            q, k = rms(q, p("self_attn.q_norm")), rms(k, p("self_attn.k_norm"))
        q = q.view(t, heads, hd).transpose(0, 1)
        k = k.view(t, kvh, hd).transpose(0, 1).repeat_interleave(heads // kvh, 0)
        v = v.view(t, kvh, hd).transpose(0, 1).repeat_interleave(heads // kvh, 0)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        a = torch.softmax(q @ k.transpose(1, 2) * scale + mask, -1) @ v
        o = a.transpose(0, 1).reshape(t, -1) @ p("self_attn.o_proj").T
        if c["sandwich"]:
            o = rms(o, p("post_attention_layernorm"))
        x = x + c["residual"] * o
        if c["pre"]:
            h = rms(x, p("pre_feedforward_layernorm") if c["sandwich"] else p("post_attention_layernorm"))
        else:
            h = x
        m = (torch.nn.functional.silu(h @ p("mlp.gate_proj").T) * (h @ p("mlp.up_proj").T)) @ p("mlp.down_proj").T
        if c["sandwich"]:
            m = rms(m, p("post_feedforward_layernorm"))
        x = x + c["residual"] * m
    head = w["model.embed_tokens.weight"] if c["tie"] else w["lm_head.weight"]
    return rms(x, w["model.norm.weight"]) @ head.T / c["logits_div"]


def mix_build(name):
    from safetensors.torch import load_file, save_file

    path = FIXTURES / name / "model.safetensors"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(mix_weights(name), path)
    return mix_config(name), {k: v.float() for k, v in load_file(path).items()}


def mix_reference(name, out_dir):
    c, w = mix_build(name)
    g = torch.Generator().manual_seed(SEED + 3)
    ids = torch.randint(0, 512, (PROMPT,), generator=g)
    with torch.inference_mode():
        logits = mix_forward(c, w, ids)
        seq = ids.clone()
        for _ in range(GREEDY):
            seq = torch.cat([seq, mix_forward(c, w, seq)[-1].argmax().view(1)])
    logits.numpy().astype(np.float32).tofile(out_dir / f"{name}.ref.bin")
    (out_dir / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": seq[PROMPT:].tolist(), "vocab": 512}))
    print(name, "->", out_dir)


def mix_agrees_with_transformers():
    from safetensors.torch import load_file

    as_mix = {
        "tiny-olmo2": dict(kv_heads=4, pre=False, sandwich=True, qk_full=True, residual=1.0, embed=1.0, attention=None, logits_div=1.0, theta=500000.0, tie=False),
        "tiny-granite": dict(kv_heads=2, pre=True, sandwich=False, qk_full=False, residual=0.22, embed=12.0, attention=0.125, logits_div=8.0, theta=10000000.0, tie=True),
    }
    ids = torch.randint(0, 512, (PROMPT,), generator=torch.Generator().manual_seed(SEED + 3))
    for name, c in as_mix.items():
        model = build(name)
        w = {k: v.float() for k, v in load_file(FIXTURES / name / "model.safetensors").items()}
        with torch.inference_mode():
            want = model(ids[None]).logits[0]
            got = mix_forward(c, w, ids)
        diff = (want - got).abs().max().item()
        print(name, "mix reference max abs diff against transformers", diff)
        assert diff < 1e-4, name


if __name__ == "__main__":
    if sys.argv[1:] == ["check-mix"]:
        mix_agrees_with_transformers()
        sys.exit(0)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:] or list(families()) + list(MIXES):
        reference(name, out)
