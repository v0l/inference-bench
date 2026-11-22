import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEED = 0
FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-llama4-vision"
TILE = 56
PATCH = 14


def config():
    from transformers import Llama4Config, Llama4TextConfig, Llama4VisionConfig

    text = Llama4TextConfig(
        vocab_size=600,
        hidden_size=128,
        intermediate_size=128,
        intermediate_size_mlp=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=1,
        interleave_moe_layer_step=2,
        no_rope_layers=[1, 1, 1, 0],
        attention_chunk_size=16,
        use_qk_norm=True,
        attn_temperature_tuning=True,
        attn_scale=0.5,
        floor_scale=16,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    vision = Llama4VisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=TILE,
        patch_size=PATCH,
        hidden_act="gelu",
        norm_eps=1e-5,
        pixel_shuffle_ratio=0.5,
        projector_input_dim=128,
        projector_output_dim=128,
        vision_output_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    return Llama4Config(
        text_config=text,
        vision_config=vision,
        boi_token_index=590,
        eoi_token_index=591,
        image_token_index=592,
        attn_implementation="eager",
    )


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if p.dim() == 1:
            v = 1.0 + 0.2 * torch.randn(p.shape, generator=g) if "norm" in name else 0.2 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name or "positional_embedding" in name:
            v = torch.randn(p.shape, generator=g) * 0.5
        elif "experts" in name:
            v = torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def build():
    from safetensors.torch import load_file, save_file
    from transformers import Llama4ForConditionalGeneration

    torch.manual_seed(SEED)
    cfg = config()
    model = Llama4ForConditionalGeneration(cfg).float().eval()
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
                    "size": {"height": TILE, "width": TILE},
                    "max_patches": 4,
                    "resize_to_max_canvas": False,
                }
            )
        )
    state = {k: v.float() for k, v in load_file(path).items()}
    model.load_state_dict(state, strict=False)
    return model


def picture(seed, width, height):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    base = np.stack([xs * 255 // max(width - 1, 1), ys * 255 // max(height - 1, 1), (xs + ys) * 255 // (width + height)], -1)
    return Image.fromarray(np.clip(base + rng.integers(0, 40, base.shape), 0, 255).astype(np.uint8))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build()
    for name, (w, h) in {"single": (56, 56), "wide": (112, 56)}.items():
        image = picture(1, w, h)
        image.save(out / f"{name}.png")
        ratio_w = w // TILE
        tiles = [image.crop((i * TILE, 0, (i + 1) * TILE, TILE)) for i in range(ratio_w)]
        if ratio_w > 1:
            tiles.append(image.resize((TILE, TILE), Image.BILINEAR))
        pixel = torch.stack([torch.from_numpy(np.asarray(t)).permute(2, 0, 1).float() / 127.5 - 1.0 for t in tiles])
        gh = TILE // PATCH
        patches = pixel.reshape(len(tiles), 3, gh, PATCH, gh, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(-1)
        patches.numpy().astype(np.float32).tofile(out / f"{name}.patches.bin")
        with torch.inference_mode():
            vm = model.vision_model
            feats = vm(pixel).last_hidden_state
            feats = model.multi_modal_projector(feats.reshape(-1, feats.shape[-1])).float()
        feats.numpy().astype(np.float32).tofile(out / f"{name}.embeds.bin")
        (out / f"{name}.json").write_text(json.dumps({"tiles": len(tiles), "grid": [len(tiles), gh, gh], "rows": feats.shape[0]}))
        print(name, "tiles", len(tiles), "rows", feats.shape[0])
    names = {k: list(v.shape) for k, v in model.state_dict().items() if "vision" in k or "projector" in k}
    (out / "tensors.json").write_text(json.dumps(names, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
