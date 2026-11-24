import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/tiny-flux2-klein"
NAME = "tiny-flux2-klein"


def load(meta, key):
    f = meta["files"][key]
    return torch.from_numpy(np.fromfile(FIXTURE / "reference" / f["file"], dtype=np.float32).reshape(f["shape"]))


def timestep_numpy(t, channels, max_period=10000.0):
    half = channels // 2
    freqs = np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = np.float32(t) * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)


class Writer:
    def __init__(self, out):
        self.out = out
        self.index = {"files": {}}

    def put(self, key, t):
        array = t.detach().float().contiguous().numpy().astype(np.float32) if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
        file = f"{NAME}.denoiser.{key}.bin"
        array.tofile(self.out / file)
        self.index["files"][key] = {"file": file, "shape": list(array.shape)}


def run(model, meta, writer, prefix, check):
    embeds = load(meta, "text.embeds")
    text_ids = Flux2KleinPipeline._prepare_text_ids(embeds)
    side = int(round(meta["image_tokens"] ** 0.5))
    image_ids = Flux2KleinPipeline._prepare_latent_ids(torch.zeros(1, 1, side, side))
    inputs = [load(meta, "noise")] + [load(meta, f"step{i}.latent") for i in range(meta["steps"] - 1)]
    captured = []

    def hook(name):
        def grab(module, args, output):
            parts = output if isinstance(output, tuple) else (output,)
            captured.append((name, torch.cat(list(parts), dim=1)))
        return grab

    handles = [b.register_forward_hook(hook(f"double{i}")) for i, b in enumerate(model.transformer_blocks)]
    handles += [b.register_forward_hook(hook(f"single{i}")) for i, b in enumerate(model.single_transformer_blocks)]
    worst = 0.0
    for step, x in enumerate(inputs):
        captured.clear()
        with torch.no_grad():
            v = model(hidden_states=x, encoder_hidden_states=embeds, timestep=torch.tensor([meta["sigmas"][step]]), img_ids=image_ids, txt_ids=text_ids, guidance=None, return_dict=False)[0]
        diff = float((v - load(meta, f"step{step}.velocity")).abs().max())
        worst = max(worst, diff)
        print(f"{prefix} step {step} velocity max diff to the pipeline {diff:.3e}")
        writer.put(f"{prefix}step{step}.velocity", v)
        for name, t in captured:
            writer.put(f"{prefix}step{step}.{name}", t)
    for h in handles:
        h.remove()
    if check and worst > 1e-5:
        sys.exit(f"the reloaded transformer does not reproduce the recorded velocity ({worst})")
    return text_ids, image_ids


def main():
    out = FIXTURE / "reference"
    meta = json.loads((out / f"{NAME}.json").read_text())
    config = json.loads((FIXTURE / "transformer/config.json").read_text())
    writer = Writer(out)
    model = Flux2Transformer2DModel.from_pretrained(FIXTURE / "transformer").eval()
    text_ids, image_ids = run(model, meta, writer, "", True)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    run(model, meta, writer, "bf16.", False)
    for step, sigma in enumerate(meta["sigmas"][: meta["steps"]]):
        t = np.float32(sigma) * np.float32(1000.0)
        writer.put(f"step{step}.timestep", timestep_numpy(t, config["timestep_guidance_channels"]))
    side = int(round(meta["image_tokens"] ** 0.5))
    noise = load(meta, "noise")
    unpacked = noise.permute(0, 2, 1).reshape(1, -1, side, side)
    writer.put("noise.grid", Flux2KleinPipeline._unpatchify_latents(unpacked))
    writer.index["text_ids"] = text_ids[0].tolist()
    writer.index["image_ids"] = image_ids[0].tolist()
    (out / f"{NAME}.denoiser.json").write_text(json.dumps(writer.index, indent=1) + "\n")


if __name__ == "__main__":
    main()
