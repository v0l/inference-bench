import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEED = 0
FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-gemma3-vision"
GREEDY = 16


def config():
    from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig

    text = Gemma3TextConfig(
        vocab_size=600,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        query_pre_attn_scalar=48,
        sliding_window=16,
        layer_types=["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2 + ["full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        max_position_embeddings=1024,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
    )
    vision = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=56,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        vision_use_head=False,
    )
    return Gemma3Config(
        text_config=text,
        vision_config=vision,
        mm_tokens_per_image=4,
        boi_token_index=590,
        eoi_token_index=591,
        image_token_index=592,
        attn_implementation="eager",
    )


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name == "lm_head.weight":
            continue
        if p.dim() == 1:
            v = 0.2 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name or "position_embedding" in name:
            v = torch.randn(p.shape, generator=g) * 0.5
        elif "mm_input_projection" in name:
            v = torch.randn(p.shape, generator=g) / p.shape[0] ** 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build():
    from safetensors.torch import load_file, save_file
    from transformers import Gemma3ForConditionalGeneration

    torch.manual_seed(SEED)
    cfg = config()
    model = Gemma3ForConditionalGeneration(cfg).float().eval()
    path = FIXTURE / "model.safetensors"
    if not path.exists():
        FIXTURE.mkdir(parents=True, exist_ok=True)
        save_file(weights(model), path)
        cfg.to_json_file(FIXTURE / "config.json")
        (FIXTURE / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "do_normalize": True,
                    "do_resize": True,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "size": {"height": 56, "width": 56},
                    "pan_and_scan_min_crop_size": 56,
                    "pan_and_scan_max_num_crops": 4,
                    "pan_and_scan_min_ratio_to_activate": 1.2,
                }
            )
        )
    state = {k: v.float() for k, v in load_file(path).items()}
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    return model


def picture(seed, width, height):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    base = np.stack([xs * 255 // max(width - 1, 1), ys * 255 // max(height - 1, 1), (xs + ys) * 255 // (width + height)], -1)
    noise = rng.integers(0, 40, base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build()
    cfg = model.config
    for name, (w, h, pan) in {"square": (90, 90, False), "wide": (240, 80, True)}.items():
        image = picture(1 if pan else 2, w, h)
        image.save(out / f"{name}.png")
        tiles = [image] if not pan else [image] + [image.crop((i * 80, 0, (i + 1) * 80, 80)) for i in range(3)]
        tiles = [t.resize((56, 56), Image.BILINEAR) for t in tiles]
        pixel = torch.stack([torch.from_numpy(np.asarray(t)).permute(2, 0, 1).float() / 127.5 - 1.0 for t in tiles])
        gh = 56 // 14
        patches = pixel.reshape(len(tiles), 3, gh, 14, gh, 14).permute(0, 2, 4, 1, 3, 5).reshape(-1)
        patches.numpy().astype(np.float32).tofile(out / f"{name}.patches.bin")
        seq = "\n\n" + "<boi>" + "<img>" * 4 + "<eoi>" + "\n\n"
        ids = [2, 7, 8, 9]
        for _ in tiles:
            ids += [590] + [592] * 4 + [591] + [10]
        ids += [20, 21, 22]
        input_ids = torch.tensor([ids])
        token_type = (input_ids == 592).long()
        with torch.inference_mode():
            feats = model.get_image_features(pixel_values=pixel).pooler_output.reshape(-1, 128).float()
            logits = model(input_ids=input_ids, pixel_values=pixel, token_type_ids=token_type).logits[0]
        feats.numpy().astype(np.float32).tofile(out / f"{name}.embeds.bin")
        logits.float().numpy().astype(np.float32).tofile(out / f"{name}.logits.bin")
        (out / f"{name}.json").write_text(json.dumps({"ids": ids, "tiles": len(tiles), "vocab": 600, "grid": [len(tiles), gh, gh]}))
        print(name, "tiles", len(tiles), "ids", len(ids))


if __name__ == "__main__":
    main(sys.argv[1])
