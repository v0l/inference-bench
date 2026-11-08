import json
import os
import sys
import time
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


FP8_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")
FP8_BLOCK = 128
FP8_MAX = 448.0


def quantize_fp8_block(w):
    rows, cols = w.shape
    blocks = w.float().reshape(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    s = scale.float()[:, None, :, None]
    q = (blocks / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(rows, cols), scale


def dequantize_fp8_block(q, scale):
    rows, cols = q.shape
    s = scale.float().repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return q.float() * s[:rows, :cols]


def is_fp8_projection(name):
    return name.endswith(".weight") and name.split(".")[-2] in FP8_PROJECTIONS


def write_fp8_shards(src, dst):
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    shards = {}
    for name, w in sorted(load_file(src / "model.safetensors").items()):
        parts = name.split(".")
        shard = f"layers-{parts[2]}.safetensors" if parts[1] == "layers" else "outside.safetensors"
        group = shards.setdefault(shard, {})
        if is_fp8_projection(name):
            group[name], group[name + "_scale_inv"] = quantize_fp8_block(w)
        else:
            group[name] = w
    weight_map = {}
    for shard, tensors in shards.items():
        save_file({k: v.contiguous() for k, v in tensors.items()}, dst / shard)
        weight_map.update({k: shard for k in tensors})
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))


def dequantized_state(fixture):
    from safetensors.torch import load_file

    index = json.loads((fixture / "model.safetensors.index.json").read_text())
    raw = {}
    for shard in sorted(set(index["weight_map"].values())):
        raw.update(load_file(fixture / shard))
    state = {}
    for name, w in raw.items():
        if name.endswith("_scale_inv"):
            continue
        state[name] = dequantize_fp8_block(w, raw[name + "_scale_inv"]) if w.dtype == torch.float8_e4m3fn else w.float()
    return state


def quantize_activation_fp8(x):
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shape).to(x.dtype)


def quantize_projection_inputs(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in FP8_PROJECTIONS:
            module.register_forward_pre_hook(lambda _m, args: (quantize_activation_fp8(args[0]),))


def tiny_fp8_model(fixture, act_quant=False):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    src = fixture.parent / "tiny-qwen3.5-hd256"
    if not (fixture / "model.safetensors.index.json").exists():
        write_fp8_shards(src, fixture)
    torch.manual_seed(TINY_SEED)
    model = Qwen3_5ForCausalLM(tiny_config(256)).float().eval()
    model.load_state_dict(dequantized_state(fixture), strict=True)
    if act_quant:
        quantize_projection_inputs(model)
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


def tiny(fixture, out_dir, head_dim=128, fp8=False, act_quant=False):
    from transformers import DynamicCache

    torch.set_num_threads(4)
    model = tiny_fp8_model(fixture, act_quant) if fp8 else tiny_model(fixture, head_dim)
    prefix = ("tinyfp8act" if act_quant else "tinyfp8") if fp8 else "tiny" if head_dim == 128 else f"tiny{head_dim}"
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


def snapshot_dir(repo):
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snaps = hub / f"models--{repo.replace('/', '--')}" / "snapshots"
    return next(d for d in sorted(snaps.iterdir()) if (d / "model.safetensors.index.json").exists())


def real_prompt_ids(tok):
    return tok(PROMPT + " " + PROMPT).input_ids[:64]


def slice_rows(src, vocab):
    tok = AutoTokenizer.from_pretrained(src)
    used = sorted(set(real_prompt_ids(tok)))
    rest = [i for i in range(vocab) if i not in set(used)][: vocab - len(used)]
    return sorted(used + rest)


def real_slice(repo, dst, layers, vocab):
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = snapshot_dir(repo)
    rows = torch.tensor(slice_rows(src, vocab))
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    dst.mkdir(parents=True, exist_ok=True)
    keep = {}
    for name, shard in index.items():
        parts = name.split(".")
        if name.startswith("model.language_model.layers.") and int(parts[3]) < layers:
            keep.setdefault(f"layers-{parts[3]}.safetensors", []).append((name, shard))
        elif name in ("model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight"):
            keep.setdefault("outside.safetensors", []).append((name, shard))
    weight_map = {}
    for out, names in sorted(keep.items()):
        tensors = {}
        for name, shard in names:
            with safe_open(src / shard, "pt") as f:
                t = f.get_tensor(name)
            if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
                t = t[rows]
            tensors[name] = t.contiguous()
            weight_map[name] = out
        save_file(tensors, dst / out)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = layers
    text["layer_types"] = text["layer_types"][:layers]
    text["vocab_size"] = vocab
    q = config["quantization_config"]
    config["quantization_config"] = {k: q[k] for k in ("activation_scheme", "fmt", "quant_method", "weight_block_size")}
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        (dst / f).write_bytes((src / f).read_bytes())
    (dst / "vocab_rows.json").write_text(json.dumps(rows.tolist()))


def real_slice_reference(repo, fixture, out_dir, layers, vocab):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    if not (fixture / "model.safetensors.index.json").exists():
        real_slice(repo, fixture, layers, vocab)
    torch.set_num_threads(16)
    text = json.loads((fixture / "config.json").read_text())["text_config"]
    config = Qwen3_5TextConfig(**{k: v for k, v in text.items() if k != "model_type"})
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    state = {k.replace("model.language_model.", "model."): v for k, v in dequantized_state(fixture).items()}
    model.load_state_dict(state, strict=True, assign=True)
    model = model.float().eval()
    model.model.rotary_emb.__init__(config)
    tok = AutoTokenizer.from_pretrained(fixture)
    rows = {r: i for i, r in enumerate(json.loads((fixture / "vocab_rows.json").read_text()))}
    ids = torch.tensor([rows[t] for t in real_prompt_ids(tok)])
    versions = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        seen, hooks = layer_outputs(model)
        logits = model(ids[None]).logits[0]
        for h in hooks:
            h.remove()
    save(out_dir, "real27b-slice", {"layers": torch.stack([s[0] for s in seen]), "logits": logits},
         {"tokens": ids.tolist(), "layers": layers, "vocab": vocab, **versions})
    print("real slice", fixture, len(ids), "tokens ->", out_dir)


REAL_PROMPTS = [
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil based ink and a screw press adapted from wine making. Within fifty years presses had spread to more than two hundred cities across Europe, and printers in Venice alone had issued several thousand editions. As paper mills multiplied and type founders sold standard fonts, the cost of a book fell",
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list in linear time.\"\"\"\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n",
    "Photosynthesis converts light energy into chemical energy. In the light dependent reactions, water is split and oxygen is released, while ATP and NADPH are produced. The Calvin cycle then uses that ATP and NADPH to fix carbon dioxide into three carbon sugars, regenerating the acceptor molecule ribulose bisphosphate on each turn. Measured against the sunlight that falls on a leaf, the overall efficiency of the process is",
    "Dear hiring committee, I am writing to apply for the position of senior network engineer. Over the past eight years I have designed and operated BGP edge routing for a regional provider, migrated two data centres to an EVPN fabric, and led the on call rotation for a team of six engineers across three time zones. In my current role",
    "Question: A train leaves the station at 9:00 travelling at 80 km/h. A second train leaves the same station at 10:30 on the same track travelling at 120 km/h. At what time does the second train catch the first? Answer: Let t be the time in hours after 10:30. The first train has a head start of",
]
REAL_PROMPT_LEN = 64
REAL_NEW = 32


def real_prompts(repo, out_dir):
    tok = AutoTokenizer.from_pretrained(snapshot_dir(repo))
    prompts = []
    for text in REAL_PROMPTS:
        ids = tok(text).input_ids
        assert len(ids) >= REAL_PROMPT_LEN, f"prompt has only {len(ids)} tokens"
        prompts.append(ids[:REAL_PROMPT_LEN])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real27b-prompts.json").write_text(json.dumps({"prompts": prompts, "new_tokens": REAL_NEW}))
    print("wrote", len(prompts), "prompts of", REAL_PROMPT_LEN, "tokens")


def real_full(repo, out_dir, seq_file, act_quant):
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding

    torch.set_num_threads(int(os.environ.get("REF_THREADS", "24")))
    src = snapshot_dir(repo)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    text = json.loads((src / "config.json").read_text())["text_config"]
    config = Qwen3_5TextConfig(**{k: v for k, v in text.items() if k != "model_type"})
    config._attn_implementation = "sdpa"
    seqs = json.loads(seq_file.read_text())["sequences"]
    handles = {}

    def tensor(name):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(src / shard, "pt")
        return handles[shard].get_tensor(name)

    def weights(prefix):
        names = [n for n in index if n.startswith(prefix) and not n.endswith("_scale_inv")]
        out = {}
        for n in names:
            w = tensor(n)
            out[n[len(prefix):]] = dequantize_fp8_block(w, tensor(n + "_scale_inv")) if w.dtype == torch.float8_e4m3fn else w.float()
        return out

    root = "model.language_model."
    started = time.time()
    embed = tensor(root + "embed_tokens.weight")
    hidden = [embed[torch.tensor(s)].float()[None] for s in seqs]
    del embed
    rotary = Qwen3_5TextRotaryEmbedding(config)
    trace = []
    with torch.inference_mode():
        for i in range(config.num_hidden_layers):
            layer = Qwen3_5DecoderLayer(config, i).float().eval()
            layer.load_state_dict(weights(f"{root}layers.{i}."), strict=True)
            if act_quant:
                quantize_projection_inputs(layer)
            for p, h in enumerate(hidden):
                pos = torch.arange(h.shape[1])[None]
                emb = rotary(h, pos)
                hidden[p] = layer(h, position_embeddings=emb, attention_mask=None, position_ids=pos)
            trace.append(hidden[0][0, :REAL_PROMPT_LEN].clone())
            del layer
            print(f"layer {i} {time.time() - started:.0f}s", flush=True)
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": tensor(root + "norm.weight").float()})
        head = tensor("lm_head.weight").float()
        logits = torch.stack([norm(h[0]) @ head.T for h in hidden])
    name = "real27b-ref-act" if act_quant else "real27b-ref"
    versions = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    save(out_dir, name, {"logits": logits, "layers": torch.stack(trace)},
         {"sequences": seqs, "argmax": logits.argmax(-1).tolist(), "act_quant": act_quant, "secs": time.time() - started, **versions})
    print(name, "done in", f"{time.time() - started:.0f}s")


NVFP4_REPO = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"
NVFP4_ROWS = {"gate_proj": [0, 1, 4095, 17407], "up_proj": [0, 9000], "down_proj": [0, 2561, 5119]}
NVFP4_LAYERS = [0, 30, 55]
FP8_CHANNEL_ROWS = [
    (56, "mlp.gate_proj", [0, 17407]),
    (63, "mlp.down_proj", [5119]),
    (0, "linear_attn.in_proj_qkv", [0, 10239]),
    (0, "linear_attn.in_proj_z", [6143]),
    (3, "self_attn.o_proj", [0, 4000]),
]


def nvfp4_dequant(out_dir):
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    from safetensors import safe_open

    src = snapshot_dir(NVFP4_REPO)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(src / index[name], "pt") as f:
            return f.get_tensor(name)

    rows_f32, rows_bf16, entries = [], [], []
    for layer in NVFP4_LAYERS:
        for proj, rows in NVFP4_ROWS.items():
            p = f"model.language_model.layers.{layer}.mlp.{proj}"
            sd = {k: tensor(f"{p}.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale")}
            sel = {"weight_packed": sd["weight_packed"][rows], "weight_scale": sd["weight_scale"][rows], "weight_global_scale": sd["weight_global_scale"]}
            m, n = sel["weight_packed"].shape
            q = unpack_fp4_from_uint8(sel["weight_packed"], m, n * 2, dtype=torch.float32)
            w32 = dequantize(x_q=q, scale=sel["weight_scale"].to(torch.float32), global_scale=sel["weight_global_scale"], dtype=torch.float32)
            w16 = NVFP4PackedCompressor.decompress(dict(sel), scheme=None)["weight"]
            rows_f32.append(w32)
            rows_bf16.append(w16.float())
            entries.append({"kind": "nvfp4", "layer": layer, "proj": proj, "rows": rows, "cols": n * 2})
    for layer, proj, rows in FP8_CHANNEL_ROWS:
        p = f"model.language_model.layers.{layer}.{proj}"
        w, s = tensor(f"{p}.weight")[rows], tensor(f"{p}.weight_scale")[rows]
        w32 = dequantize(x_q=w, scale=s.to(torch.float32), dtype=torch.float32)
        rows_f32.append(w32)
        rows_bf16.append(w32.to(torch.bfloat16).float())
        entries.append({"kind": "fp8_channel", "layer": layer, "proj": proj, "rows": rows, "cols": w.shape[1]})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cat([r.flatten() for r in rows_f32]).numpy().astype(np.float32).tofile(out_dir / "nvfp4-dequant.f32.bin")
    torch.cat([r.flatten() for r in rows_bf16]).numpy().astype(np.float32).tofile(out_dir / "nvfp4-dequant.bf16.bin")
    versions = {"torch": torch.__version__, "compressed_tensors": __import__("compressed_tensors").__version__}
    (out_dir / "nvfp4-dequant.json").write_text(json.dumps({"entries": entries, **versions}))
    print("nvfp4 dequant", len(entries), "tensors ->", out_dir)


def main():
    if sys.argv[1] == "nvfp4-dequant":
        nvfp4_dequant(Path(sys.argv[2]))
        return
    if sys.argv[1] == "real-prompts":
        real_prompts(sys.argv[2], Path(sys.argv[3]))
        return
    if sys.argv[1] == "real-full":
        real_full(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), len(sys.argv) > 5 and sys.argv[5] == "act")
        return
    if sys.argv[1] == "real-slice":
        real_slice_reference(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        return
    if sys.argv[1] == "tiny":
        tiny(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 128)
        return
    if sys.argv[1] == "tiny-fp8":
        tiny(Path(sys.argv[2]), Path(sys.argv[3]), 256, fp8=True)
        tiny(Path(sys.argv[2]), Path(sys.argv[3]), 256, fp8=True, act_quant=True)
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
