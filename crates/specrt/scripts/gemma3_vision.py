import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

GREEDY = 24
ROWS = 4
QUESTION = "Describe this image in one sentence."


def wide_image(src, out):
    img = Image.open(src).convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w * 3, h))
    for i in range(3):
        canvas.paste(img, (i * w, 0))
    canvas.save(out)
    return out


def run(model, processor, name, path, pan, out):
    image = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=image, return_tensors="pt", do_pan_and_scan=pan, add_special_tokens=False).to("cuda")
    pixel = inputs["pixel_values"].float().cpu()
    tiles, _, h, w = pixel.shape
    patch = 14
    gh, gw = h // patch, w // patch
    patches = pixel.reshape(tiles, 3, gh, patch, gw, patch).permute(0, 2, 4, 1, 3, 5).reshape(-1)
    patches.numpy().astype(np.float32).tofile(out / f"{name}.patches.bin")
    bf16_pixels = inputs["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        feats = model.get_image_features(pixel_values=bf16_pixels).pooler_output.reshape(-1, model.config.text_config.hidden_size).float().cpu()
    feats.numpy().astype(np.float32).tofile(out / f"{name}.embeds.bin")
    model.float()
    inputs["pixel_values"] = inputs["pixel_values"].float()
    with torch.inference_mode():
        exact = model.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output.reshape(-1, feats.shape[-1]).float().cpu()
        logits = model(**inputs).logits[0, -ROWS:].float().cpu()
        gen = model.generate(**inputs, max_new_tokens=GREEDY, do_sample=False, output_scores=True, return_dict_in_generate=True)
        greedy = gen.sequences[0, inputs["input_ids"].shape[1]:].cpu()
        top2 = [torch.topk(sc[0].float(), 2).values for sc in gen.scores]
        margins = [float(t[0] - t[1]) for t in top2]
    exact.numpy().astype(np.float32).tofile(out / f"{name}.embeds32.bin")
    logits.numpy().astype(np.float32).tofile(out / f"{name}.logits.bin")
    model.to(torch.bfloat16)
    meta = {
        "ids": inputs["input_ids"][0].tolist(),
        "greedy": greedy.tolist(),
        "vocab": logits.shape[1],
        "rows": ROWS,
        "tiles": tiles,
        "grid": [tiles, gh, gw],
        "question": QUESTION,
        "image": str(path),
        "pan": pan,
        "margins": margins,
    }
    (out / f"{name}.json").write_text(json.dumps(meta))
    print(name, "tiles", tiles, "tokens", len(meta["ids"]), "embeds", tuple(feats.shape))


def main(model_dir, house, out_dir):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    run(model, processor, "gemma3-image", house, False, out)
    wide = wide_image(house, out / "wide.png")
    run(model, processor, "gemma3-pan", wide, True, out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
