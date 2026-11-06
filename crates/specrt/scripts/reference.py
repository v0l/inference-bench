import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "The spec-driven runtime reads a TOML file describing the model, renders CUDA "
    "templates for every operator, and compiles them at load time. A paged key value "
    "cache keeps the attention state between decode steps, so"
)
GREEDY = 32


def main():
    spec_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    spec = tomllib.loads(spec_path.read_text())
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(spec["repo"])
    config = AutoConfig.from_pretrained(spec["repo"])
    config.sliding_window = None
    scaling = spec["rope"].get("scaling")
    if scaling:
        config.rope_scaling = {
            "rope_type": scaling["kind"],
            "factor": scaling["factor"],
            **{k: v for k, v in scaling.items() if k not in ("kind", "factor", "original_max_positions")},
            **({"original_max_position_embeddings": int(scaling["original_max_positions"])} if "original_max_positions" in scaling else {}),
        }
    if spec["mlp"]["activation"] == "geglu":
        config.hidden_act = "gelu_pytorch_tanh"
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], config=config, torch_dtype=torch.float32)
    model.eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model(ids).logits[0].float().numpy()
        greedy = model.generate(ids, max_new_tokens=GREEDY, do_sample=False)[0, ids.shape[1]:]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = spec["name"]
    logits.astype(np.float32).tofile(out_dir / f"{name}.ref.bin")
    meta = {
        "tokens": ids[0].tolist(),
        "vocab": logits.shape[1],
        "greedy": greedy.tolist(),
        "greedy_text": tok.decode(greedy),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (out_dir / f"{name}.ref.json").write_text(json.dumps(meta))
    print(name, len(meta["tokens"]), "tokens", repr(meta["greedy_text"]))


if __name__ == "__main__":
    main()
