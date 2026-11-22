import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextModel

IMAGE_TOKEN = 248056
VIDEO_TOKEN = 248057
MERGE = 2
PREFIX = "model.language_model."
NOISE = float(os.environ.get("VISION_LAYERS_NOISE", "0"))


def dequant(f, key, device):
    w = f.get_tensor(key).to(device)
    scale_key = key + "_scale_inv"
    if scale_key in f.keys():
        s = f.get_tensor(scale_key).to(device).float()
        s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        return (w.float() * s).to(torch.bfloat16)
    return w


def load_text_model(snapshot, device):
    cfg = Qwen3_5Config.from_pretrained(snapshot)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = Qwen3_5TextModel(cfg.text_config).eval()
    torch.set_default_dtype(torch.float32)
    state = model.state_dict()
    index = json.load(open(snapshot / "model.safetensors.index.json"))["weight_map"]
    by_file = {}
    for key, file in index.items():
        if key.startswith(PREFIX):
            by_file.setdefault(file, []).append(key)
    seen = set()
    for file, keys in by_file.items():
        with safe_open(snapshot / file, "pt") as f:
            for key in keys:
                if key.endswith("_scale_inv"):
                    continue
                name = key[len(PREFIX):]
                value = dequant(f, key, device)
                with torch.no_grad():
                    state[name].copy_(value.to(state[name].dtype))
                seen.add(name)
    missing = set(state) - seen
    assert not missing, sorted(missing)[:5]
    return model, cfg


def rope_index(cfg, ids, kind, grid):
    types_row = torch.tensor([[(2 if kind == "video" else 1) if t == (VIDEO_TOKEN if kind == "video" else IMAGE_TOKEN) else 0 for t in ids]])
    config = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=MERGE))
    model = types.SimpleNamespace(config=config)
    model.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(model, *a, **k)
    g = torch.tensor([grid])
    image, video = (None, g) if kind == "video" else (g, None)
    pos, _ = Qwen3_5Model.get_rope_index(model, torch.tensor([ids]), types_row, image, video)
    return pos


def run(model, cfg, out, tag):
    meta = json.load(open(out / f"{tag}.meta.json"))
    ids = meta["ids"]
    embeds = np.fromfile(out / f"{tag}.embeds.f32", dtype="<f4").reshape(-1, meta["hidden"])
    dev = next(model.parameters()).device
    ids_t = torch.tensor([ids], device=dev)
    x = model.embed_tokens(ids_t).float()
    rows = np.array(meta["rows"])
    sel = rows >= 0
    x[0, torch.tensor(sel, device=dev)] = torch.from_numpy(embeds[rows[sel]]).to(dev)
    x = x.to(torch.bfloat16)
    x_clean = x
    pos = rope_index(cfg, ids, tag, meta["grid"]).to(dev)
    def forward(inputs):
        got = []
        hooks = [layer.register_forward_hook(lambda m, i, o: got.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu())) for layer in model.layers]
        with torch.no_grad():
            model(inputs_embeds=inputs, position_ids=pos, use_cache=False)
        for h in hooks:
            h.remove()
        return torch.stack([g[0] for g in got]).numpy()

    theirs = forward(x_clean)
    if NOISE:
        gen = torch.Generator(device=dev).manual_seed(7)
        noise = torch.randn(x_clean.shape, generator=gen, device=dev)
        ours = forward((x_clean.float() * (1 + NOISE * noise)).to(torch.bfloat16))
    else:
        ours = np.fromfile(out / f"{tag}.layers.f32", dtype="<f4").reshape(theirs.shape)
    mask = sel
    rows_out = []
    for layer in range(theirs.shape[0]):
        a, b = ours[layer], theirs[layer]
        d = a - b
        rel = float(np.sqrt((d * d).mean() / (b * b).mean()))
        rel_media = float(np.sqrt((d[mask] ** 2).mean() / (b[mask] ** 2).mean()))
        rel_text = float(np.sqrt((d[~mask] ** 2).mean() / (b[~mask] ** 2).mean()))
        rows_out.append({"layer": layer, "rel_rms": rel, "rel_rms_media": rel_media, "rel_rms_text": rel_text, "max_abs": float(np.abs(d).max())})
    (out / f"{tag}.layer_diff.json").write_text(json.dumps(rows_out))
    print(tag, "layer rel_rms rel_media rel_text max_abs")
    for r in rows_out:
        print(f"{r['layer']:3d} {r['rel_rms']:.5f} {r['rel_rms_media']:.5f} {r['rel_rms_text']:.5f} {r['max_abs']:.3f}")


def main():
    snapshot = Path(sys.argv[1])
    out = Path(sys.argv[2])
    model, cfg = load_text_model(snapshot, "cuda")
    for tag in sys.argv[3:]:
        run(model, cfg, out, tag)


main()
