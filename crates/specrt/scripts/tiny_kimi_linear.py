import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
REAL = Path.home() / ".cache/specrt-popular/files/moonshotai__Kimi-Linear-48B-A3B-Instruct/config.json"
SEED = 0
LAYERS = 4
PROMPT = 30
DECODE = 4
HIDDEN_LAYERS = [0, 1, 3]
MAX_BYTES = 5 * 1024 * 1024

SIZES = dict(
    vocab_size=512,
    hidden_size=128,
    head_dim=72,
    intermediate_size=256,
    moe_intermediate_size=32,
    num_hidden_layers=LAYERS,
    num_attention_heads=4,
    num_key_value_heads=4,
    kv_lora_rank=64,
    qk_nope_head_dim=48,
    qk_rope_head_dim=16,
    v_head_dim=32,
    num_experts=8,
    num_experts_per_token=2,
    first_k_dense_replace=1,
    model_max_length=512,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
    linear_attn_config={
        "full_attn_layers": [4],
        "kda_layers": [1, 2, 3],
        "head_dim": 64,
        "num_heads": 2,
        "short_conv_kernel_size": 4,
    },
)

VARIANTS = {"tiny-kimi-linear": {}}

F32_SUFFIXES = ("self_attn.A_log", "self_attn.dt_bias")

RENAMES = [
    ("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
    ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
    ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
    ("self_attn.A_log", "self_attn.forget_gate.A_log"),
    (".block_sparse_moe.", ".mlp."),
]


def fixture_dir(name):
    return ROOT / "tests/fixtures" / name


def config_dict(extra):
    real = json.loads(REAL.read_text())
    real.pop("auto_map", None)
    real.update(SIZES)
    real.update(extra)
    return real


def model_config(extra):
    from transformers.models.kimi_linear.configuration_kimi_linear import KimiLinearConfig

    cfg = KimiLinearConfig(**config_dict(extra))
    cfg._attn_implementation = "eager"
    return cfg


def to_real(name, value):
    for real, internal in RENAMES:
        name = name.replace(internal, real)
    if name.endswith("self_attn.conv1d.weight"):
        base = name[: -len("conv1d.weight")]
        return {f"{base}{p}_conv1d.weight": c.contiguous() for p, c in zip("qkv", value.chunk(3, dim=0))}
    if name.endswith("block_sparse_moe.experts.gate_up_proj"):
        base = name[: -len("gate_up_proj")]
        half = value.shape[1] // 2
        out = {}
        for e in range(value.shape[0]):
            out[f"{base}{e}.w1.weight"] = value[e, :half].contiguous()
            out[f"{base}{e}.w3.weight"] = value[e, half:].contiguous()
        return out
    if name.endswith("block_sparse_moe.experts.down_proj"):
        base = name[: -len("down_proj")]
        return {f"{base}{e}.w2.weight": value[e].contiguous() for e in range(value.shape[0])}
    return {name: value}


def to_internal(tensors):
    out, convs, experts = {}, {}, {}
    for name, value in tensors.items():
        parts = name.split(".")
        if name.endswith("_conv1d.weight"):
            convs.setdefault(name[: -len("q_conv1d.weight")], {})[parts[-2][0]] = value
            continue
        if ".experts." in name:
            experts.setdefault(".".join(parts[:-3]), {})[(int(parts[-3]), parts[-2])] = value
            continue
        for real, internal in RENAMES:
            name = name.replace(real, internal)
        out[name] = value
    for base, parts in convs.items():
        for real, internal in RENAMES:
            base = base.replace(real, internal)
        out[f"{base}conv1d.weight"] = torch.cat([parts[p] for p in "qkv"], dim=0)
    for base, items in experts.items():
        for real, internal in RENAMES:
            base = base.replace(real, internal)
        n = 1 + max(e for e, _ in items)
        out[f"{base}.gate_up_proj"] = torch.stack([torch.cat([items[(e, "w1")], items[(e, "w3")]]) for e in range(n)])
        out[f"{base}.down_proj"] = torch.stack([items[(e, "w2")] for e in range(n)])
    return out


def random_tensor(name, shape, g):
    randn = lambda: torch.randn(shape, generator=g)
    if name.endswith("e_score_correction_bias"):
        return 0.2 * randn()
    if name.endswith("A_log"):
        return 0.5 * randn()
    if name.endswith("dt_bias"):
        return 0.5 * randn()
    if name.endswith("_conv1d.weight"):
        return 0.5 * randn()
    if len(shape) == 1:
        return 1.0 + 0.1 * randn()
    if "embed_tokens" in name or name == "lm_head.weight":
        return randn()
    return randn() / shape[-1] ** 0.5


def stored_dtype(name):
    if name.endswith(F32_SUFFIXES):
        return torch.float32
    return torch.bfloat16


def build(extra):
    from transformers.models.kimi_linear.modeling_kimi_linear import KimiLinearForCausalLM

    cfg = model_config(extra)
    torch.manual_seed(SEED)
    return KimiLinearForCausalLM(cfg).float().eval()


def real_names(model):
    shapes = {}
    for name, p in sorted(model.state_dict().items()):
        name = name.replace(".mlp.experts.", ".block_sparse_moe.experts.").replace(".mlp.gate.", ".block_sparse_moe.gate.").replace(".mlp.shared_experts.", ".block_sparse_moe.shared_experts.")
        for real, v in to_real(name, p).items():
            shapes[real] = tuple(v.shape)
    return shapes


def generate(name, model, extra):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(SEED)
    shapes = real_names(model)
    tensors = {n: random_tensor(n, shapes[n], g).to(stored_dtype(n)).contiguous() for n in sorted(shapes)}
    out = fixture_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    import transformers

    cfg = config_dict(extra)
    cfg["transformers_version"] = transformers.__version__
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def load(name, model):
    from safetensors.torch import load_file

    raw = {k: v.float() for k, v in load_file(fixture_dir(name) / "model.safetensors").items()}
    model.load_state_dict(to_internal(raw), strict=True)


def prompt_tokens(vocab):
    g = torch.Generator().manual_seed(SEED + 3)
    return torch.randint(0, vocab, (PROMPT,), generator=g)


def save(out, name, arrays, meta):
    for key, value in arrays.items():
        value.detach().float().numpy().astype(np.float32).tofile(out / f"{name}.{key}.bin")
    (out / f"{name}.json").write_text(json.dumps(meta) + "\n")


def references(name, model):
    import transformers

    out = fixture_dir(name) / "reference"
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    ids = prompt_tokens(cfg.vocab_size)
    trace = {}
    layers = model.model.layers
    handles = [layers[i].register_forward_hook(lambda _m, _a, o, i=i: trace.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())) for i in HIDDEN_LAYERS]
    with torch.inference_mode():
        res = model(input_ids=ids[None], use_cache=True)
        for h in handles:
            h.remove()
        prefill = res.logits[0]
        past = res.past_key_values
        hidden = torch.stack([trace[i] for i in HIDDEN_LAYERS])
        greedy = [int(prefill[-1].argmax())]
        steps = []
        for _ in range(DECODE):
            res = model(input_ids=torch.tensor([[greedy[-1]]]), past_key_values=past, use_cache=True)
            past = res.past_key_values
            steps.append(res.logits[0, -1])
            greedy.append(int(res.logits[0, -1].argmax()))
        decode = torch.stack(steps)
        full = model(input_ids=torch.cat([ids, torch.tensor(greedy[:-1])])[None]).logits[0]
    recompute = float((full[PROMPT:] - decode).abs().max())
    prefill_gap = float((full[:PROMPT] - prefill).abs().max())
    for k, t in {"prefill": prefill, "decode": decode, "hidden": hidden}.items():
        assert torch.isfinite(t).all(), k
    assert recompute < 1e-3 and prefill_gap < 1e-3, (recompute, prefill_gap)
    prefill.float().numpy().astype(np.float32).tofile(out / f"{name}.ref.bin")
    meta = {
        "tokens": ids.tolist(),
        "greedy": greedy,
        "vocab": cfg.vocab_size,
        "decode_steps": DECODE,
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_size": cfg.hidden_size,
        "decode_vs_recompute": recompute,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (out / f"{name}.ref.json").write_text(json.dumps(meta) + "\n")
    save(out, f"{name}.decode", {"logits": decode}, {"rows": DECODE, "vocab": cfg.vocab_size, "greedy": greedy})
    save(out, f"{name}.hidden", {"streams": hidden}, {"layers": HIDDEN_LAYERS, "shape": list(hidden.shape)})
    print(name, "prefill", tuple(prefill.shape), "greedy", greedy, "decode vs recompute", recompute)


def fixture_bytes(name):
    return sum(p.stat().st_size for p in fixture_dir(name).rglob("*") if p.is_file())


def main():
    torch.set_num_threads(4)
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, extra in VARIANTS.items():
        model = build(extra)
        if command in ("all", "fixture"):
            generate(name, model, extra)
        load(name, model)
        if command in ("all", "reference"):
            references(name, model)
        size = fixture_bytes(name)
        print(name, size, "bytes")
        assert size < MAX_BYTES, size


if __name__ == "__main__":
    main()
