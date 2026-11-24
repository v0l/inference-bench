import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/tiny-glm5-next"
REAL = ROOT / "tests/fixtures/glm5.3-flash.config.json"
NAME = "tiny-glm5-next"
PREFIX = "model.language_model."
SEED = 0
LAYERS = 4
PROMPT = 30
DECODE = 4
HIDDEN_LAYERS = [0, 1, 3]
MAX_BYTES = 5 * 1024 * 1024

SIZES = dict(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=32,
    num_hidden_layers=LAYERS,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_head_dim=32,
    v_head_dim=32,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    max_position_embeddings=512,
    swiglu_limit=1.5,
    eos_token_id=None,
    pad_token_id=None,
    layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
    mlp_layer_types=["dense"] + ["sparse"] * 3,
    indexer_types=["full"] * LAYERS,
    linear_attn_config={
        "full_attn_layers": [3],
        "gate_lower_bound": -5.0,
        "head_dim": 64,
        "kda_layers": [0, 1, 2],
        "num_heads": 2,
        "short_conv_kernel_size": 4,
    },
)

F32_SUFFIXES = ("hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale", "self_attn.A_log", "self_attn.dt_bias", "e_score_correction_bias")

RENAMES = [
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
    ("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
    ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
    ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
    ("self_attn.A_log", "self_attn.forget_gate.A_log"),
]


def text_dict():
    real = json.loads(REAL.read_text())
    text = dict(real["text_config"])
    text.update(SIZES)
    return real, text


def fixture_config():
    import transformers

    real, text = text_dict()
    return {
        "architectures": real["architectures"],
        "model_type": real["model_type"],
        "text_config": text,
        "tie_word_embeddings": False,
        "transformers_version": transformers.__version__,
    }


def text_config(layers=LAYERS):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    _, text = text_dict()
    extra = layers - LAYERS
    text["num_hidden_layers"] = layers
    text["layer_types"] = text["layer_types"] + ["deepseek_sparse_attention"] * extra
    text["mlp_layer_types"] = text["mlp_layer_types"] + ["sparse"] * extra
    text["indexer_types"] = text["indexer_types"] + ["full"] * extra
    cfg = Glm5NextTextConfig(**text)
    cfg._attn_implementation = "eager"
    return cfg


def to_real(name, value):
    for real, internal in RENAMES:
        name = name.replace(internal, real)
    if name.endswith("self_attn.conv1d.weight"):
        base = name[: -len("conv1d.weight")]
        return {f"{base}{p}_conv1d.weight": c.contiguous() for p, c in zip("qkv", value.chunk(3, dim=0))}
    if name.endswith("mlp.experts.gate_up_proj"):
        base = name[: -len("gate_up_proj")]
        half = value.shape[1] // 2
        out = {}
        for e in range(value.shape[0]):
            out[f"{base}{e}.gate_proj.weight"] = value[e, :half].contiguous()
            out[f"{base}{e}.up_proj.weight"] = value[e, half:].contiguous()
        return out
    if name.endswith("mlp.experts.down_proj"):
        base = name[: -len("down_proj")]
        return {f"{base}{e}.down_proj.weight": value[e].contiguous() for e in range(value.shape[0])}
    return {name: value}


def to_internal(tensors):
    out, convs, experts = {}, {}, {}
    for name, value in tensors.items():
        parts = name.split(".")
        if name.endswith("_conv1d.weight"):
            convs.setdefault(name[: -len("q_conv1d.weight")], {})[parts[-2][0]] = value
            continue
        if "mlp.experts." in name:
            experts.setdefault(".".join(parts[:-3]), {})[(int(parts[-3]), parts[-2])] = value
            continue
        for real, internal in RENAMES:
            name = name.replace(real, internal)
        out[name] = value
    for base, parts in convs.items():
        out[f"{base}conv1d.weight"] = torch.cat([parts[p] for p in "qkv"], dim=0)
    for base, items in experts.items():
        n = 1 + max(e for e, _ in items)
        out[f"{base}.gate_up_proj"] = torch.stack([torch.cat([items[(e, "gate_proj")], items[(e, "up_proj")]]) for e in range(n)])
        out[f"{base}.down_proj"] = torch.stack([items[(e, "down_proj")] for e in range(n)])
    return out


def random_tensor(name, shape, g):
    randn = lambda: torch.randn(shape, generator=g)
    if name.endswith(("_base", "k_norm.bias")):
        return 0.3 * randn()
    if name.endswith("_scale"):
        return 0.75 + 0.5 * torch.rand(shape, generator=g)
    if name.endswith("e_score_correction_bias"):
        return 0.2 * randn()
    if name.endswith("A_log"):
        return 0.5 * randn()
    if name.endswith("dt_bias"):
        return 0.5 * randn()
    if name.endswith("index_kpool_compress_ape"):
        return 0.5 * randn()
    if name.endswith("_conv1d.weight"):
        return 0.5 * randn()
    if len(shape) == 1:
        return 1.0 + 0.1 * randn()
    if "embed_tokens" in name or name == "lm_head.weight":
        return randn()
    return randn() / shape[-1] ** 0.5


def stored_dtype(name, mtp):
    if name == PREFIX + "norm.weight" or name.endswith(F32_SUFFIXES):
        return torch.float32
    if name.endswith("post_attention_layernorm.weight") and not mtp:
        return torch.float32
    return torch.bfloat16


def mtp_modules(cfg):
    from torch import nn
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention, Glm5NextTextMoE, Glm5NextTextRMSNorm

    h, eps = cfg.hidden_size, cfg.rms_norm_eps
    m = nn.Module()
    m.self_attn = Glm5NextTextAttention(cfg, LAYERS)
    m.mlp = Glm5NextTextMoE(cfg)
    for n in ("input_layernorm", "post_attention_layernorm", "enorm", "hnorm"):
        setattr(m, n, Glm5NextTextRMSNorm(h, eps))
    m.shared_head = nn.Module()
    m.shared_head.norm = Glm5NextTextRMSNorm(h, eps)
    m.eh_proj = nn.Linear(2 * h, h, bias=False)
    return m


def build_models():
    from torch import nn
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    cfg = text_config()
    torch.manual_seed(SEED)
    model = Glm5NextTextModel(cfg).float().eval()
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).float().eval()
    mtp = mtp_modules(text_config(LAYERS + 1)).float().eval()
    return cfg, model, head, mtp


def generate(model, head, mtp):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(SEED)
    tensors = {}
    shapes = {}
    for name, p in sorted(model.state_dict().items()):
        for real, v in to_real(PREFIX + name, p).items():
            shapes[real] = (tuple(v.shape), False)
    shapes["lm_head.weight"] = (tuple(head.weight.shape), False)
    for name, p in sorted(mtp.state_dict().items()):
        for real, v in to_real(f"{PREFIX}layers.{LAYERS}.{name}", p).items():
            shapes[real] = (tuple(v.shape), True)
    for name in sorted(shapes):
        shape, is_mtp = shapes[name]
        tensors[name] = random_tensor(name, shape, g).to(stored_dtype(name, is_mtp)).contiguous()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    save_file(tensors, FIXTURE / "model.safetensors", metadata={"format": "pt"})
    (FIXTURE / "config.json").write_text(json.dumps(fixture_config(), indent=2) + "\n")


def load(model, head, mtp):
    from safetensors.torch import load_file

    raw = {k: v.float() for k, v in load_file(FIXTURE / "model.safetensors").items()}
    mtp_prefix = f"{PREFIX}layers.{LAYERS}."
    main = {k[len(PREFIX):]: v for k, v in raw.items() if k.startswith(PREFIX) and not k.startswith(mtp_prefix)}
    model.load_state_dict(to_internal(main), strict=True)
    head.load_state_dict({"weight": raw["lm_head.weight"]}, strict=True)
    mtp.load_state_dict(to_internal({k[len(mtp_prefix):]: v for k, v in raw.items() if k.startswith(mtp_prefix)}), strict=True)
    return raw


def prompt_tokens(vocab):
    g = torch.Generator().manual_seed(SEED + 3)
    return torch.randint(0, vocab, (PROMPT,), generator=g)


def logits_of(model, head, ids, **kw):
    out = model(input_ids=ids[None], **kw)
    return head(out.last_hidden_state)[0], out


def mtp_step(model, head, mtp, tokens, hidden):
    n = tokens.shape[0]
    embeds = model.embed_tokens(tokens)
    embeds = torch.where((torch.arange(n) == 0)[:, None], torch.zeros_like(embeds), embeds)
    x = mtp.eh_proj(torch.cat([mtp.enorm(embeds), mtp.hnorm(hidden)], dim=-1))[None]
    mask = torch.ones(1, n, dtype=torch.bool)
    a, _, _ = mtp.self_attn(hidden_states=mtp.input_layernorm(x), attention_mask=mask)
    residual = a + x
    y = mtp.mlp(mtp.post_attention_layernorm(residual))
    out = mtp.shared_head.norm(y + residual)[0]
    return head(out), out


def clamp_share(model, ids):
    seen = []

    def hook(module, args):
        gate_up = None
        x = args[0]
        if hasattr(module, "gate_proj"):
            gate_up = (module.gate_proj(x), module.up_proj(x))
        if gate_up is not None:
            lim = module.swiglu_limit
            seen.append(float(((gate_up[0] > lim) | (gate_up[1].abs() > lim)).float().mean()))

    handles = [m.register_forward_pre_hook(hook) for m in model.modules() if hasattr(m, "swiglu_limit") and hasattr(m, "gate_proj")]
    model(input_ids=ids[None])
    for h in handles:
        h.remove()
    return max(seen)


def save(out, name, arrays, meta):
    for key, value in arrays.items():
        value.detach().float().numpy().astype(np.float32).tofile(out / f"{name}.{key}.bin")
    (out / f"{name}.json").write_text(json.dumps(meta) + "\n")


def references(model, head, mtp):
    import transformers

    out = FIXTURE / "reference"
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    ids = prompt_tokens(cfg.vocab_size)
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    trace = {}
    handles = [model.layers[i].register_forward_hook(lambda _m, _a, o, i=i: trace.__setitem__(i, o[0][0].detach().clone())) for i in HIDDEN_LAYERS]
    with torch.inference_mode():
        prefill, state = logits_of(model, head, ids, use_cache=True)
        for h in handles:
            h.remove()
        hidden = torch.stack([trace[i] for i in HIDDEN_LAYERS])
        past = state.past_key_values
        greedy = [int(prefill[-1].argmax())]
        steps = []
        for _ in range(DECODE):
            row, state = logits_of(model, head, torch.tensor([greedy[-1]]), past_key_values=past, use_cache=True)
            past = state.past_key_values
            steps.append(row[-1])
            greedy.append(int(row[-1].argmax()))
        decode = torch.stack(steps)
        full, _ = logits_of(model, head, torch.cat([ids, torch.tensor(greedy[:-1])]))
        recompute = float((full[PROMPT:] - decode).abs().max())
        prefill_gap = float((full[:PROMPT] - prefill).abs().max())
        final = model(input_ids=ids[None]).last_hidden_state[0]
        shifted = torch.cat([ids[1:], torch.tensor([greedy[0]])])
        mtp_logits, mtp_hidden = mtp_step(model, head, mtp, shifted, final)
        draft = int(mtp_logits[-1].argmax())
        chain, _ = mtp_step(model, head, mtp, torch.cat([shifted, torch.tensor([draft])]), torch.cat([final, mtp_hidden[-1:]]))
        clamped = clamp_share(model, ids)
    for name, t in {"prefill": prefill, "decode": decode, "hidden": hidden, "mtp": mtp_logits, "chain": chain}.items():
        assert torch.isfinite(t).all(), name
    assert recompute < 1e-3 and prefill_gap < 1e-3, (recompute, prefill_gap)
    prefill.float().numpy().astype(np.float32).tofile(out / f"{NAME}.ref.bin")
    meta = {
        "tokens": ids.tolist(),
        "greedy": greedy,
        "vocab": cfg.vocab_size,
        "decode_steps": DECODE,
        "hidden_layers": HIDDEN_LAYERS,
        "hc_mult": cfg.hc_mult,
        "hidden_size": cfg.hidden_size,
        "decode_vs_recompute": recompute,
        "swiglu_clamped_share": clamped,
        **versions,
    }
    (out / f"{NAME}.ref.json").write_text(json.dumps(meta) + "\n")
    save(out, f"{NAME}.decode", {"logits": decode}, {"rows": DECODE, "vocab": cfg.vocab_size, "greedy": greedy})
    save(out, f"{NAME}.hidden", {"streams": hidden}, {"layers": HIDDEN_LAYERS, "shape": list(hidden.shape)})
    save(out, f"{NAME}.mtp", {"logits": mtp_logits, "chain": chain[-1:]}, {
        "tokens": shifted.tolist(),
        "first": greedy[0],
        "draft": draft,
        "chain_draft": int(chain[-1].argmax()),
        "zero_embed_at_position_0": True,
        "chain_topk": "fresh",
        **versions,
    })
    print(NAME, "prefill", tuple(prefill.shape), "greedy", greedy, "decode vs recompute", recompute, "clamped", clamped)


def fixture_bytes():
    return sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())


def main():
    torch.set_num_threads(4)
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg, model, head, mtp = build_models()
    if command in ("all", "fixture"):
        generate(model, head, mtp)
    load(model, head, mtp)
    if command in ("all", "reference"):
        references(model, head, mtp)
    size = fixture_bytes()
    print(NAME, size, "bytes")
    assert size < MAX_BYTES, size


if __name__ == "__main__":
    main()
