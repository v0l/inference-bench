import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

CASES = ["image_resize", "image_big", "video_resize"]


def load_model(snapshot: Path):
    cfg = json.load(open(snapshot / "config.json"))["vision_config"]
    model = Qwen3_5VisionModel(Qwen3_5VisionConfig(**cfg)).float().eval()
    state = {}
    with safe_open(snapshot / "outside.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                state[k[len("model.visual."):]] = f.get_tensor(k).float()
    missing = model.load_state_dict(state, strict=True)
    print(missing)
    return model


def main():
    snapshot, fixtures = Path(sys.argv[1]), Path(sys.argv[2])
    model = load_model(snapshot)
    for name in CASES:
        meta = json.load(open(fixtures / f"{name}.json"))
        patches = np.fromfile(fixtures / f"{name}.f32", dtype="<f4").reshape(-1, meta["patch_dim"])
        grid = torch.tensor([meta["grid"]])
        with torch.no_grad():
            out = model(torch.from_numpy(patches), grid)
        out.pooler_output.numpy().astype("<f4").tofile(fixtures / f"{name}.vision.f32")
        out.last_hidden_state.numpy().astype("<f4").tofile(fixtures / f"{name}.vision_hidden.f32")
        print(name, tuple(out.pooler_output.shape))


main()
