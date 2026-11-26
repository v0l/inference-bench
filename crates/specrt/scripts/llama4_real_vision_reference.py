import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Llama4Config, Llama4ImageProcessor
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector, Llama4VisionModel


def vision_state(snapshot):
    out = {}
    for shard in sorted(Path(snapshot).glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.startswith(("vision_model.", "multi_modal_projector.")):
                    out[key] = f.get_tensor(key)
    return out


def build(cfg, state, dtype, device):
    torch.set_default_dtype(dtype)
    vision = Llama4VisionModel(cfg.vision_config).eval()
    projector = Llama4MultiModalProjector(cfg).eval()
    vision.load_state_dict({k[len("vision_model."):]: v.to(dtype) for k, v in state.items() if k.startswith("vision_model.")})
    projector.load_state_dict({k[len("multi_modal_projector."):]: v.to(dtype) for k, v in state.items() if k.startswith("multi_modal_projector.")})
    return vision.to(device), projector.to(device)


def embed(vision, projector, pixel, dtype, device):
    with torch.inference_mode():
        feats = vision(pixel.to(dtype).to(device)).last_hidden_state
        return projector(feats.reshape(-1, feats.shape[-1])).float().cpu().numpy().astype(np.float32)


def main(snapshot, out_dir, images):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Llama4Config.from_pretrained(snapshot)
    state = vision_state(snapshot)
    proc = Llama4ImageProcessor.from_pretrained(snapshot)
    batches = {}
    for path in images:
        name = Path(path).stem
        shutil.copy(path, out / f"{name}.png")
        batches[name] = proc(Image.open(path).convert("RGB"), return_tensors="pt")
    runs = [("", torch.float32, "cpu")]
    if torch.cuda.is_available():
        runs.append((".bf16gpu", torch.bfloat16, "cuda"))
    for suffix, dtype, device in runs:
        vision, projector = build(cfg, state, dtype, device)
        for name, batch in batches.items():
            embeds = embed(vision, projector, batch["pixel_values"], dtype, device)
            embeds.tofile(out / f"{name}.embeds{suffix}.bin")
            ratio = [int(v) for v in batch["aspect_ratios"][0]]
            (out / f"{name}.json").write_text(json.dumps({"tiles": int(batch["pixel_values"].shape[0]), "layout": ratio, "rows": int(embeds.shape[0])}))
            print(name, suffix or ".f32cpu", "tiles", batch["pixel_values"].shape[0], "layout", ratio, "rows", embeds.shape[0])
        del vision, projector


main(sys.argv[1], sys.argv[2], sys.argv[3:])
