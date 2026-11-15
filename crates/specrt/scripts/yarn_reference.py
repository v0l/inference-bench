import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

POSITIONS = [0, 1, 4096, 262143, 262144, 262145, 524288, 1000000]
FIXTURE = Path(__file__).resolve().parent.parent / "tests/fixtures/qwen3.8-27b-nvfp4.config.json"
OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/yarn_reference.json"


def config_with(rope_extra):
    cfg = json.loads(FIXTURE.read_text())
    text = cfg.get("text_config", cfg)
    params = dict(text["rope_parameters"])
    params.update(rope_extra)
    text["rope_parameters"] = params
    with tempfile.TemporaryDirectory() as d:
        Path(d, "config.json").write_text(json.dumps(cfg))
        full = AutoConfig.from_pretrained(d)
    return getattr(full, "text_config", full), params


def table(name, rope_extra):
    config, params = config_with(rope_extra)
    inv_freq, attention = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    pos = torch.tensor(POSITIONS, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :].float()
    return {
        "name": name,
        "rope_parameters": params,
        "head_dim": config.head_dim,
        "inv_freq": [float(x) for x in inv_freq],
        "attention_factor": float(attention),
        "positions": POSITIONS,
        "cos": [[float(x) for x in row] for row in (freqs.cos() * attention)],
        "sin": [[float(x) for x in row] for row in (freqs.sin() * attention)],
    }


def main():
    cases = [
        table("factor4", {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}),
        table("factor2_mscale", {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144, "mscale": 1.0, "mscale_all_dim": 0.5}),
        table("factor4_untruncated", {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144, "truncate": False, "beta_fast": 16, "beta_slow": 2}),
    ]
    OUT.write_text(json.dumps({"theta": cases[0]["rope_parameters"]["rope_theta"], "cases": cases}))
    print("wrote", OUT, file=sys.stderr)


main()
