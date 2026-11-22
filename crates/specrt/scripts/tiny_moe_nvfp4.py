import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import reference as R

INPUT_AMAX = 4.0


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    weights = load_file(src / "model.safetensors")
    out, targets = {}, {"group_0": [], "group_1": []}
    for name, w in sorted(weights.items()):
        module = name[: -len(".weight")]
        parts = module.split(".")
        target = "model.language_model." + module[len("model."):]
        projection = name.endswith(".weight") and parts[-1] in R.FP8_PROJECTIONS and w.dim() == 2
        expert = ".mlp.experts." in name or ".mlp.shared_expert." in name
        if projection and expert:
            mlp = module.rsplit(".", 1)[0]
            peers = [mlp + ".gate_proj", mlp + ".up_proj"] if parts[-1] in ("gate_proj", "up_proj") else [module]
            wgs = R.nvfp4_global(max(float(weights[p + ".weight"].float().abs().max()) for p in peers))
            packed, scale = R.quantize_nvfp4(w, wgs)
            out[module + ".weight_packed"] = packed
            out[module + ".weight_scale"] = scale
            out[module + ".weight_global_scale"] = wgs
            out[module + ".input_global_scale"] = R.nvfp4_global(INPUT_AMAX)
            targets["group_1"].append(target)
        elif projection:
            out[name], out[module + ".weight_scale"] = R.quantize_fp8_channel(w)
            targets["group_0"].append(target)
        else:
            out[name] = w
    renamed = {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v.contiguous() for k, v in out.items()}
    save_file(renamed, dst / "model.safetensors")
    fp8 = {"num_bits": 8, "type": "float", "symmetric": True, "dynamic": False, "strategy": "channel", "group_size": None}
    fp4 = {"num_bits": 4, "type": "float", "symmetric": True, "dynamic": False, "strategy": "tensor_group", "group_size": 16, "scale_dtype": "torch.float8_e4m3fn"}
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {"format": "float-quantized", "weights": fp8, "input_activations": {**fp8, "dynamic": True, "strategy": "token"}, "targets": targets["group_0"]},
            "group_1": {"format": "nvfp4-pack-quantized", "weights": fp4, "input_activations": {**fp4, "dynamic": "local"}, "targets": targets["group_1"]},
        },
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
