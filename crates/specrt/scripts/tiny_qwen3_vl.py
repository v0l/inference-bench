import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import reference as R

FIXTURE = Path(__file__).parent.parent / "tests/fixtures/tiny-qwen3-vl"
IMAGE, VIDEO, START, END = 1000, 1001, 998, 999
VISION_CASES = {"image": [(1, 8, 8)], "images2": [(1, 4, 12), (1, 8, 8)]}
PROMPT = 24


def build_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config=dict(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20], "mrope_interleaved": True},
            tie_word_embeddings=False,
            max_position_embeddings=4096,
            eos_token_id=None,
            bos_token_id=None,
            pad_token_id=None,
        ),
        vision_config=dict(depth=2, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256, num_position_embeddings=64, patch_size=16, temporal_patch_size=2, spatial_merge_size=2, deepstack_visual_indexes=[0]),
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=START,
        vision_end_token_id=END,
        tie_word_embeddings=False,
    )


def random_weights(model):
    g = torch.Generator().manual_seed(R.TINY_SEED + 31)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if name.endswith("inv_freq"):
            continue
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        elif name.endswith(".bias"):
            v = 0.02 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name:
            v = torch.randn(p.shape, generator=g)
        elif name.endswith("pos_embed.weight"):
            v = 0.5 * torch.randn(p.shape, generator=g)
        else:
            fan_in = 1
            for d in p.shape[1:]:
                fan_in *= d
            v = torch.randn(p.shape, generator=g) / fan_in ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def generate_fixture(dst):
    from safetensors.torch import save_file
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    torch.manual_seed(R.TINY_SEED)
    config = build_config()
    model = Qwen3VLForConditionalGeneration(config)
    weights = random_weights(model)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(weights, dst / "model.safetensors")
    out = config.to_dict()
    out["architectures"] = ["Qwen3VLForConditionalGeneration"]
    out["dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(out, indent=2))
    print("fixture", dst, len(weights), "tensors")


def references(fixture, out_dir):
    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    torch.set_num_threads(4)
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    config = build_config()
    model = Qwen3VLForConditionalGeneration(config).float().eval()
    model.load_state_dict({k: v.float() for k, v in load_file(fixture / "model.safetensors").items()}, strict=False)
    model.tie_weights()
    ids = R.tiny_tokens(PROMPT, 3)
    with torch.inference_mode():
        logits = model(input_ids=ids[None]).logits[0]
    R.save(out_dir, "tinyvl-text", {"logits": logits}, {"prompt": ids.tolist(), **versions})
    vc = config.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    merge = vc.spatial_merge_size
    g = torch.Generator().manual_seed(R.TINY_SEED + 41)
    for case, grids in VISION_CASES.items():
        patches = [torch.rand((t * h * w, patch_dim), generator=g) * 2 - 1 for t, h, w in grids]
        pixels = torch.cat(patches)
        grid_thw = torch.tensor(grids)
        ids = [int(x) for x in torch.randint(0, 990, (5,), generator=g)]
        types = [0] * 5
        for t, h, w in grids:
            n = t * h * w // (merge * merge)
            ids += [START] + [IMAGE] * n + [END] + [int(x) for x in torch.randint(0, 990, (3,), generator=g)]
            types += [0] + [1] * n + [0] * 4
        ids_t = torch.tensor(ids)
        with torch.inference_mode():
            out = model.model.visual(pixels, grid_thw)
            embeds = out.pooler_output
            deepstack = torch.stack(list(out.deepstack_features))
            logits = model(input_ids=ids_t[None], pixel_values=pixels, image_grid_thw=grid_thw, mm_token_type_ids=torch.tensor(types)[None]).logits[0]
        R.save(out_dir, f"tinyvl-vision-{case}", {"patches": pixels, "embeds": embeds, "deepstack": deepstack, "logits": logits}, {"grids": [list(x) for x in grids], "ids": ids, "patch_dim": patch_dim, **versions})
        print("vision", case, tuple(embeds.shape), tuple(deepstack.shape), tuple(logits.shape))


def main():
    command = sys.argv[1]
    if command == "fixture":
        generate_fixture(Path(sys.argv[2]) if len(sys.argv) > 2 else FIXTURE)
    elif command == "reference":
        references(FIXTURE, Path(sys.argv[2]) if len(sys.argv) > 2 else FIXTURE / "reference")
    elif command == "all":
        generate_fixture(FIXTURE)
        references(FIXTURE, FIXTURE / "reference")
    else:
        sys.exit("usage: tiny_qwen3_vl.py fixture|reference|all [dir]")


if __name__ == "__main__":
    main()
