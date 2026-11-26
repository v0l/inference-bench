import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Llama4Config, Llama4ImageProcessor
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector, Llama4VisionModel

IMAGE = Path(__file__).parent.parent / "tests/fixtures/vision/house.png"


def vision_state(snapshot):
    out = {}
    for shard in sorted(Path(snapshot).glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.startswith(("vision_model.", "multi_modal_projector.")):
                    out[key] = f.get_tensor(key)
    return out


def main(snapshot, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Llama4Config.from_pretrained(snapshot)
    dtype = torch.bfloat16 if len(sys.argv) > 3 and sys.argv[3] == "bf16" else torch.float32
    torch.set_default_dtype(dtype)
    vision = Llama4VisionModel(cfg.vision_config).eval()
    projector = Llama4MultiModalProjector(cfg).eval()
    state = vision_state(snapshot)
    vision.load_state_dict({k[len("vision_model."):]: v.to(dtype) for k, v in state.items() if k.startswith("vision_model.")})
    projector.load_state_dict({k[len("multi_modal_projector."):]: v.to(dtype) for k, v in state.items() if k.startswith("multi_modal_projector.")})
    proc = Llama4ImageProcessor.from_pretrained(snapshot)
    image = Image.open(IMAGE).convert("RGB")
    batch = proc(image, return_tensors="pt")
    pixel = batch["pixel_values"].to(dtype)
    with torch.inference_mode():
        feats = vision(pixel).last_hidden_state
        embeds = projector(feats.reshape(-1, feats.shape[-1])).float()
    embeds.numpy().astype(np.float32).tofile(out / "house.embeds.bin")
    ratio = [int(v) for v in batch["aspect_ratios"][0]]
    (out / "house.json").write_text(json.dumps({"tiles": int(pixel.shape[0]), "layout": ratio, "rows": int(embeds.shape[0])}))
    print("tiles", pixel.shape[0], "layout", ratio, "rows", embeds.shape[0])


main(sys.argv[1], sys.argv[2])
