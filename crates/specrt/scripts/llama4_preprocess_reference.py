import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import Llama4ImageProcessor

PHOTOS = Path(__file__).parent.parent / "tests/fixtures/tiny-llama4-vision/photos"
CONFIGS = {"t56x4": (56, 4), "t112x6": (112, 6)}


def make_photos(gallery):
    PHOTOS.mkdir(parents=True, exist_ok=True)
    g = Path(gallery)
    Image.open(g / "bc-engineering-gallery-20.jpg").convert("RGB").resize((640, 480), Image.LANCZOS).save(PHOTOS / "landscape.png")
    Image.open(g / "bc-engineering-gallery-28.jpg").convert("RGB").resize((418, 600), Image.LANCZOS).save(PHOTOS / "portrait.png")
    Image.open(g / "bc-engineering-gallery-06.jpg").convert("RGB").crop((100, 100, 300, 235)).save(PHOTOS / "small.png")


def main():
    if len(sys.argv) > 1:
        make_photos(sys.argv[1])
    cases = {}
    for tag, (tile, chunks) in CONFIGS.items():
        proc = Llama4ImageProcessor(size={"height": tile, "width": tile}, max_patches=chunks, resize_to_max_canvas=False)
        for name in ["landscape", "portrait", "small"]:
            image = Image.open(PHOTOS / f"{name}.png").convert("RGB")
            out = proc(image, return_tensors="pt")
            pixels = out["pixel_values"].float().numpy().astype(np.float32)
            ratio = [int(v) for v in out["aspect_ratios"][0]]
            key = f"{name}.{tag}"
            pixels.tofile(PHOTOS / f"{key}.bin")
            cases[key] = {"image": f"{name}.png", "tile": tile, "max_patches": chunks, "tiles": int(pixels.shape[0]), "layout": ratio}
            print(key, pixels.shape, ratio)
    (PHOTOS / "reference.json").write_text(json.dumps(cases, indent=1))


main()
