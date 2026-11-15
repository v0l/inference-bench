import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import reference as R

SOURCE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b"
DEFAULT_OUT = Path.home() / ".cache/specrt-models/Qwen3.5-4B-NVFP4"
NVFP4_LAYERS = 28
CALIBRATION = Path(__file__).parent / "spec_prompts.txt"
COPIED = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json"]
DEVICE = "cuda"


def tensor_index(src):
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    return index


def read(src, index, name, device="cpu"):
    with safe_open(src / index[name], "pt", device=device) as f:
        return f.get_tensor(name)


def calibrate(src, index, config):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    text = Qwen3_5TextConfig(**config["text_config"])
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(DEVICE):
        model = Qwen3_5ForCausalLM(text)
    torch.set_default_dtype(torch.float32)
    state = {}
    for name in index:
        if name.startswith("model.language_model."):
            state["model." + name[len("model.language_model."):]] = read(src, index, name, DEVICE)
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    model.eval()
    del state
    amax = {}

    def hook(name):
        def f(_m, args):
            amax[name] = max(amax.get(name, 0.0), float(args[0].float().abs().max()))

        return f

    for name, module in model.named_modules():
        if name.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
            module.register_forward_pre_hook(hook(name))
    tok = AutoTokenizer.from_pretrained(str(src))
    documents = [d for d in CALIBRATION.read_text().split("\n---\n") if d.strip()]
    with torch.inference_mode():
        for d in documents:
            ids = tok(d, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(DEVICE)
            model(ids)
    del model
    torch.cuda.empty_cache()
    return {"model.language_model." + k[len("model."):]: v for k, v in amax.items()}


def quantize(src, out):
    index = tensor_index(src)
    config = json.loads((src / "config.json").read_text())
    amax = calibrate(src, index, config)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.safetensors"):
        f.unlink()
    shards, targets = {}, {"group_0": [], "group_1": []}

    def shard(name):
        parts = name.split(".")
        if parts[:3] == ["model", "language_model", "layers"]:
            return f"layers-{parts[3]}.safetensors"
        if parts[0] == "mtp":
            return "mtp.safetensors"
        return "outside.safetensors"

    for name in sorted(index):
        group = shards.setdefault(shard(name), {})
        w = read(src, index, name)
        language = name.startswith("model.language_model.layers.")
        if not (language and R.is_fp8_projection(name)):
            group[name] = w.cpu().contiguous()
            continue
        module = name[: -len(".weight")]
        layer = int(name.split(".")[3])
        if module.split(".")[-1] in ("gate_proj", "up_proj", "down_proj") and layer < NVFP4_LAYERS:
            mlp = module.rsplit(".", 1)[0]
            peers = [module] if module.endswith("down_proj") else [mlp + ".gate_proj", mlp + ".up_proj"]
            wgs = R.nvfp4_global(max(float(read(src, index, p + ".weight").float().abs().max()) for p in peers))
            igs = R.nvfp4_global(amax[peers[0]])
            packed, scale = R.quantize_nvfp4(w, wgs)
            group[module + ".weight_packed"] = packed.cpu().contiguous()
            group[module + ".weight_scale"] = scale.cpu().contiguous()
            group[module + ".weight_global_scale"] = wgs
            group[module + ".input_global_scale"] = igs
            targets["group_1"].append(module)
        else:
            q, s = R.quantize_fp8_channel(w)
            group[name] = q.cpu().contiguous()
            group[module + ".weight_scale"] = s.cpu().contiguous()
            targets["group_0"].append(module)
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, out / name)
        weight_map.update({k: name for k in tensors})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    fp8_args = {"num_bits": 8, "type": "float", "symmetric": True, "dynamic": False, "strategy": "channel", "group_size": None}
    fp4_args = {"num_bits": 4, "type": "float", "symmetric": True, "dynamic": False, "strategy": "tensor_group", "group_size": 16, "scale_dtype": "torch.float8_e4m3fn"}
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {"format": "float-quantized", "weights": fp8_args, "input_activations": {**fp8_args, "dynamic": True, "strategy": "token"}, "targets": sorted(targets["group_0"])},
            "group_1": {"format": "nvfp4-pack-quantized", "weights": fp4_args, "input_activations": {**fp4_args, "dynamic": "local"}, "targets": sorted(targets["group_1"])},
        },
        "ignore": ["lm_head", "re:model.visual.*", "re:mtp.*"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for f in COPIED:
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    total = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    print("wrote", out, f"{total / 1e9:.2f} GB")


if __name__ == "__main__":
    quantize(SOURCE, Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
