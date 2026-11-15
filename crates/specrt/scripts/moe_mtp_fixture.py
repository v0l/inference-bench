import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "model.safetensors", dst / "model.safetensors")
    config = json.loads((src / "config.json").read_text())
    config["mtp_num_hidden_layers"] = 1
    config["mtp_use_dedicated_embeddings"] = False
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    g = torch.Generator().manual_seed(644)
    h = config["hidden_size"]
    out = {}
    with safe_open(src / "model.safetensors", "pt") as f:
        for name in f.keys():
            if not name.startswith("model.layers.2."):
                continue
            shape = f.get_tensor(name).shape
            v = 0.1 * torch.randn(shape, generator=g) if len(shape) == 1 else torch.randn(shape, generator=g) / shape[-1] ** 0.5
            out["mtp.layers.0." + name[len("model.layers.2."):]] = v
    out["mtp.fc.weight"] = torch.randn((h, 2 * h), generator=g) / (2 * h) ** 0.5
    for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        out[f"mtp.{n}.weight"] = 0.1 * torch.randn((h,), generator=g)
    names = {}
    with safe_open(src / "model.safetensors", "pt") as f:
        names.update({k: "model.safetensors" for k in f.keys()})
    names.update({k: "mtp.safetensors" for k in out})
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": names}, indent=1))
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}, dst / "mtp.safetensors")


if __name__ == "__main__":
    main(*sys.argv[1:3])
