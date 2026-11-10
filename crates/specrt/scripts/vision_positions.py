import json
import sys
import types
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as VllmModel

VIDEO_TOKEN = 248057
IMAGE_TOKEN = 248056
VISION_START = 248053
VISION_END = 248054
MERGE = 2
GRIDS = {"video": [3, 16, 20], "video10": [10, 16, 20]}


def prompt_ids(out, name):
    return json.loads((out / f"vllm-{name}.json").read_text())["prompt_ids"]


def vllm_positions(ids, grid):
    first = ids.index(VISION_START)
    feature = types.SimpleNamespace(
        modality="video",
        mm_position=types.SimpleNamespace(offset=first),
        data={"video_grid_thw": types.SimpleNamespace(data=torch.tensor(grid))},
    )
    config = types.SimpleNamespace(
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        vision_config=types.SimpleNamespace(spatial_merge_size=MERGE),
    )
    pos, delta = VllmModel._get_mrope_input_positions(ids, [feature], config)
    return pos.tolist(), delta


def hf_positions(ids, grid):
    types_row = torch.tensor([[2 if t == VIDEO_TOKEN else 0 for t in ids]])
    config = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=MERGE))
    model = types.SimpleNamespace(config=config)
    model.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(model, *a, **k)
    pos, delta = Qwen3_5Model.get_rope_index(model, torch.tensor([ids]), types_row, None, torch.tensor([grid]))
    return pos[:, 0, :].tolist(), int(delta[0, 0])


def main():
    out = Path(sys.argv[1])
    vllm = {name: vllm_positions(prompt_ids(out, name), grid) for name, grid in GRIDS.items()}
    (out / "vllm-pos.json").write_text(json.dumps(vllm))
    pos, delta = hf_positions(prompt_ids(out, "video"), GRIDS["video"])
    (out / "hf-pos-video.json").write_text(json.dumps({"pos": pos, "delta": delta}))
    print({k: (len(v[0][0]), v[1]) for k, v in vllm.items()}, "hf delta", delta)


main()
