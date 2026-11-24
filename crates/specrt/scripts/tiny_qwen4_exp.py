import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/tiny-qwen4-exp"
NAME = "tiny-qwen4-exp"
PREFIX = "model.language_model."
SEED = 0
PROMPT = 30
DECODE = 4
HIDDEN_LAYERS = [0, 1, 3]
MAX_BYTES = 5 * 1024 * 1024
F32_SUFFIXES = ("linear_attn.A_log", "linear_attn.dt_bias", "hc_norm.weight")

TEXT = dict(
    model_type="qwen4_exp_text",
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=256,
    hidden_act="silu",
    attn_output_gate=True,
    output_gate_type="sigmoid",
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=8,
    num_experts_per_tok=2,
    hc_count=4,
    hc_lowrank=16,
    ple_layer_ids=[2],
    ple_embed_dim=128,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1000,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4,
    seed=1234,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=64,
    indexer_budget=16,
    indexer_compress_ratio=4,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    eos_token_id=1,
    pad_token_id=None,
    bos_token_id=None,
    tie_word_embeddings=False,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"},
)


def model_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    text = {k: v for k, v in TEXT.items() if k not in ("model_type", "attn_output_gate")}
    cfg = Qwen4ExpTextConfig(**text)
    cfg._attn_implementation = "eager"
    return cfg


def fixture_config():
    import transformers

    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": TEXT,
        "tie_word_embeddings": False,
        "transformers_version": transformers.__version__,
    }


def to_real(name, value):
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
    out, experts = {}, {}
    for name, value in tensors.items():
        parts = name.split(".")
        if "mlp.experts." in name:
            experts.setdefault(".".join(parts[:-3]), {})[(int(parts[-3]), parts[-2])] = value
            continue
        out[name] = value
    for base, items in experts.items():
        n = 1 + max(e for e, _ in items)
        out[f"{base}.gate_up_proj"] = torch.stack([torch.cat([items[(e, "gate_proj")], items[(e, "up_proj")]]) for e in range(n)])
        out[f"{base}.down_proj"] = torch.stack([items[(e, "down_proj")] for e in range(n)])
    return out


def random_tensor(name, shape, g):
    randn = lambda: torch.randn(shape, generator=g)
    if name.endswith(("A_log", "dt_bias")):
        return 0.5 * randn()
    if name.endswith("conv1d.weight"):
        return 0.5 * randn()
    if len(shape) == 1:
        return 1.0 + 0.1 * randn()
    if "embed_tokens" in name or name == "lm_head.weight" or name.endswith("ngram_embedding.weight"):
        return randn()
    return randn() / shape[-1] ** 0.5


def stored_dtype(name):
    if name == PREFIX + "norm.weight" or name.endswith(F32_SUFFIXES):
        return torch.float32
    return torch.bfloat16


def build_models():
    from torch import nn
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    cfg = model_config()
    torch.manual_seed(SEED)
    model = Qwen4ExpTextModel(cfg).float().eval()
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).float().eval()
    return cfg, model, head


def is_buffer(name):
    return name.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"))


def generate(model, head):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(SEED)
    shapes = {}
    for name, p in sorted(model.state_dict().items()):
        if is_buffer(name):
            continue
        for real, v in to_real(PREFIX + name, p).items():
            shapes[real] = tuple(v.shape)
    shapes["lm_head.weight"] = tuple(head.weight.shape)
    tensors = {name: random_tensor(name, shapes[name], g).to(stored_dtype(name)).contiguous() for name in sorted(shapes)}
    FIXTURE.mkdir(parents=True, exist_ok=True)
    save_file(tensors, FIXTURE / "model.safetensors", metadata={"format": "pt"})
    (FIXTURE / "config.json").write_text(json.dumps(fixture_config(), indent=2) + "\n")


def load(model, head):
    from safetensors.torch import load_file

    raw = {k: v.float() for k, v in load_file(FIXTURE / "model.safetensors").items()}
    main = {k[len(PREFIX):]: v for k, v in raw.items() if k.startswith(PREFIX)}
    missing = model.load_state_dict(to_internal(main), strict=False)
    assert all(is_buffer(k) for k in missing.missing_keys), missing.missing_keys
    assert not missing.unexpected_keys, missing.unexpected_keys
    head.load_state_dict({"weight": raw["lm_head.weight"]}, strict=True)


def prompt_tokens(vocab):
    g = torch.Generator().manual_seed(SEED + 3)
    return torch.randint(2, vocab, (PROMPT,), generator=g)


def logits_of(model, head, ids, **kw):
    out = model(input_ids=ids[None], **kw)
    return head(out.last_hidden_state)[0], out


def save(out, name, arrays, meta):
    for key, value in arrays.items():
        value.detach().float().numpy().astype(np.float32).tofile(out / f"{name}.{key}.bin")
    (out / f"{name}.json").write_text(json.dumps(meta) + "\n")


def references(model, head):
    import transformers

    out = FIXTURE / "reference"
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    ids = prompt_tokens(cfg.vocab_size)
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    trace = {}
    handles = [model.layers[i].register_forward_hook(lambda _m, _a, o, i=i: trace.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())) for i in HIDDEN_LAYERS]
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
    for name, t in {"prefill": prefill, "decode": decode, "hidden": hidden}.items():
        assert torch.isfinite(t).all(), name
    assert recompute < 1e-3 and prefill_gap < 1e-3, (recompute, prefill_gap)
    prefill.float().numpy().astype(np.float32).tofile(out / f"{NAME}.ref.bin")
    meta = {
        "tokens": ids.tolist(),
        "greedy": greedy,
        "vocab": cfg.vocab_size,
        "decode_steps": DECODE,
        "hidden_layers": HIDDEN_LAYERS,
        "hc_count": cfg.hc_count,
        "hidden_size": cfg.hidden_size,
        "decode_vs_recompute": recompute,
        **versions,
    }
    (out / f"{NAME}.ref.json").write_text(json.dumps(meta) + "\n")
    save(out, f"{NAME}.decode", {"logits": decode}, {"rows": DECODE, "vocab": cfg.vocab_size, "greedy": greedy})
    save(out, f"{NAME}.hidden", {"streams": hidden}, {"layers": HIDDEN_LAYERS, "shape": list(hidden.shape)})
    print(NAME, "prefill", tuple(prefill.shape), "greedy", greedy, "decode vs recompute", recompute)


def indexer_reference(model, head):
    out = FIXTURE / "reference"
    cfg = model.config
    layer = next(i for i, l in enumerate(model.layers) if hasattr(l, "self_attn"))
    attn = model.layers[layer].self_attn
    width = cfg.indexer_budget + cfg.indexer_compress_ratio - 1
    inputs, masks, outputs = [], [], []
    hooks = [
        attn.register_forward_pre_hook(lambda _m, args, kwargs: inputs.append((args[0] if args else kwargs["hidden_states"])[0].detach().clone()), with_kwargs=True),
        attn.indexer.register_forward_hook(lambda _m, _a, o: masks.append(o[0, 0].detach().clone())),
        attn.o_proj.register_forward_pre_hook(lambda _m, args: outputs.append(args[0][0].detach().clone())),
    ]
    ids = prompt_tokens(cfg.vocab_size)
    with torch.inference_mode():
        _, state = logits_of(model, head, ids, use_cache=True)
        past = state.past_key_values
        greedy = [int(head(state.last_hidden_state)[0, -1].argmax())]
        for _ in range(DECODE):
            row, state = logits_of(model, head, torch.tensor([greedy[-1]]), past_key_values=past, use_cache=True)
            past = state.past_key_values
            greedy.append(int(row[-1].argmax()))
    for h in hooks:
        h.remove()
    rows = []
    for mask in masks:
        for q in range(mask.shape[0]):
            chosen = torch.nonzero(mask[q]).flatten().tolist()
            assert len(chosen) <= width, (q, chosen)
            rows.append(chosen + [-1] * (width - len(chosen)))
    selected = torch.tensor(rows, dtype=torch.float32)
    hidden = torch.cat(inputs)
    gated = torch.cat(outputs)
    assert hidden.shape[0] == PROMPT + DECODE and gated.shape[0] == PROMPT + DECODE and selected.shape[0] == PROMPT + DECODE
    save(out, f"{NAME}.indexer", {"input": hidden, "selected": selected}, {"layer": layer, "rows": PROMPT + DECODE, "prompt": PROMPT, "width": width, "greedy": greedy})
    sparse = next(p for p, r in enumerate(rows) if sum(t >= 0 for t in r) < p + 1)
    kept = gated[sparse:]
    scale = float(kept.abs().max()) / 32767
    np.round(kept.numpy() / scale).astype(np.int16).tofile(out / f"{NAME}.indexer.attn.i16")
    meta = json.loads((out / f"{NAME}.indexer.json").read_text())
    (out / f"{NAME}.indexer.json").write_text(json.dumps({**meta, "attn_from": sparse, "attn_scale": scale}) + "\n")
    print(NAME, "indexer layer", layer, "rows", PROMPT + DECODE, "width", width, "attn rows from", sparse, "scale", scale)


MASK64 = (1 << 64) - 1
REAL_CONFIG = ROOT / "tests/fixtures/popular/Qwen__Qwen3.8-Flash-Next/config.json"


def np_splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def np_multipliers(vocab, ngram, ple_index, seed):
    half = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base = seed + 10007 * ple_index
    return [2 * (np_splitmix64((base + 0x9E3779B97F4A7C15 * (k + 1)) & MASK64) % half) + 1 for k in range(ngram)]


def np_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while p < 2 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            p += 1
    return p


def np_heads(hp, ple_index):
    heads = (hp["ngram_size"] - 1) * hp["heads_per_ngram"]
    sizes = [np_prime_after(hp["ngram_vocab_size_base"] - 1, ple_index * heads + h + 1) for h in range(heads)]
    offsets = [sum(sizes[:h]) for h in range(heads)]
    return sizes, offsets


def np_ngram_ids(tokens, hp, ple_index):
    n, hpn, eos = hp["ngram_size"], hp["heads_per_ngram"], hp["eos_token_id"]
    mult = np_multipliers(hp["vocab_size"], n, ple_index, hp["seed"])
    sizes, offsets = np_heads(hp, ple_index)
    hist = [eos] * (n - 1) + list(tokens)
    out = []
    for t in range(n - 1, len(hist)):
        shifted, live = [hist[t]], True
        for k in range(1, n):
            h = hist[t - k]
            shifted.append(h if live else eos)
            if h == eos:
                live = False
        row = []
        for g in range(2, n + 1):
            acc = (shifted[0] * mult[0]) & MASK64
            for k in range(1, g):
                acc ^= (shifted[k] * mult[k]) & MASK64
            signed = acc - (1 << 64) if acc >> 63 else acc
            for j in range(hpn):
                h = (g - 2) * hpn + j
                row.append(signed % sizes[h] + offsets[h])
        out.append(row)
    return np.array(out, dtype=np.int64)


class IdsOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        return ids.unsqueeze(-1)


def torch_ngram_ids(tokens, hp, ple_index):
    from types import SimpleNamespace

    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cfg = SimpleNamespace(**hp, make_ngram_vocab_size_divisible_by=128)
    with torch.device("meta"):
        emb = m.Qwen4ExpTextNGramEmbedding(cfg, (hp["ngram_size"] - 1) * hp["heads_per_ngram"], 0, ple_index)
    emb.layer_multipliers = m._build_layer_multipliers(hp["vocab_size"], hp["ngram_size"], ple_index, hp["seed"])
    emb.ngram_heads_vocab_sizes = torch.tensor(emb.head_vocab_sizes, dtype=torch.long)
    emb.ngram_heads_offsets = torch.tensor(emb.head_offsets, dtype=torch.long)
    emb.ngram_embedding = IdsOut()
    return emb(torch.tensor([tokens], dtype=torch.long), None)[0].long().numpy()


def hash_params(text):
    keys = ("vocab_size", "ngram_size", "heads_per_ngram", "ngram_vocab_size_base", "seed", "eos_token_id")
    defaults = {"seed": 1234, "heads_per_ngram": 8, "ngram_size": 3, "ngram_vocab_size_base": 20_000_000}
    return {k: text.get(k, defaults.get(k)) if text.get(k) is not None else defaults[k] for k in keys}


def hash_cases():
    real = hash_params(json.loads(REAL_CONFIG.read_text())["text_config"])
    tiny = hash_params(TEXT)
    rng = np.random.default_rng(1279)
    cases = []

    def history(hp, length, eos_rate, high):
        toks = rng.integers(0, high, length)
        toks[rng.random(length) < eos_rate] = hp["eos_token_id"]
        toks[5:7] = hp["eos_token_id"]
        return [int(t) for t in toks]

    for name, hp, ple_index, high in [
        ("tiny", tiny, 0, tiny["vocab_size"]),
        ("tiny_second_ple", tiny, 1, tiny["vocab_size"]),
        ("real", real, 0, real["vocab_size"]),
        ("real_second_ple", real, 1, real["vocab_size"]),
        ("real_wraparound", real, 0, (1 << 31) - 1),
    ]:
        tokens = history(hp, 48, 0.12, high)
        want = np_ngram_ids(tokens, hp, ple_index)
        got = torch_ngram_ids(tokens, hp, ple_index)
        assert (want == got).all(), name
        mult = np_multipliers(hp["vocab_size"], hp["ngram_size"], ple_index, hp["seed"])
        wraps = any(t * m >= 1 << 63 for t in tokens for m in mult)
        assert wraps == name.endswith("wraparound"), name
        sizes, offsets = np_heads(hp, ple_index)
        cases.append({"name": name, "params": hp, "ple_index": ple_index, "multipliers": mult, "sizes": sizes, "offsets": offsets, "tokens": tokens, "ids": want.tolist()})
    return cases


def ple_reference(model, head):
    out = FIXTURE / "reference"
    cfg = model.config
    ids = prompt_tokens(cfg.vocab_size)
    layer = cfg.ple_layer_ids[0] - 1
    calls = []
    hook = model.layers[layer].ple.register_forward_hook(lambda _m, a, o: calls.append((a[0][0].detach().clone(), a[1][0].detach().clone(), o[0].detach().clone())))
    with torch.inference_mode():
        prefill, state = logits_of(model, head, ids, use_cache=True)
        past = state.past_key_values
        greedy = [int(prefill[-1].argmax())]
        for _ in range(DECODE):
            row, state = logits_of(model, head, torch.tensor([greedy[-1]]), past_key_values=past, use_cache=True)
            past = state.past_key_values
            greedy.append(int(row[-1].argmax()))
    hook.remove()
    stored = json.loads((out / f"{NAME}.ref.json").read_text())
    assert greedy == stored["greedy"], (greedy, stored["greedy"])
    streams = torch.cat([c[0] for c in calls])
    tokens = torch.cat([c[1] for c in calls])
    ple = torch.cat([c[2] for c in calls])
    hp = hash_params(TEXT)
    ngram = np_ngram_ids(tokens.tolist(), hp, cfg.ple_layer_ids.index(layer + 1))
    save(out, f"{NAME}.ple", {"streams": streams, "out": ple}, {"layer": layer, "tokens": tokens.tolist(), "prefill": PROMPT, "decode_steps": DECODE, "ngram_ids": ngram.tolist(), "shape": list(ple.shape)})
    (out / f"{NAME}.ngram.json").write_text(json.dumps(hash_cases()) + "\n")
    print(NAME, "ple layer", layer, "rows", tuple(ple.shape), "calls", len(calls))


def fixture_bytes():
    return sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())


def main():
    torch.set_num_threads(4)
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg, model, head = build_models()
    if command in ("all", "fixture"):
        generate(model, head)
    load(model, head)
    if command in ("all", "reference"):
        references(model, head)
    if command in ("all", "reference", "indexer"):
        indexer_reference(model, head)
    if command in ("all", "reference", "ple"):
        ple_reference(model, head)
    size = fixture_bytes()
    print(NAME, size, "bytes")
    assert size < MAX_BYTES, size


if __name__ == "__main__":
    main()
