import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((src / "config.json").read_text())
cfg["mtp_num_hidden_layers"] = 1
(out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

h = cfg["hidden_size"]
inter = cfg["intermediate_size"]
hd = cfg["head_dim"]
heads = cfg["num_attention_heads"]
kv = cfg["num_key_value_heads"]
g = torch.Generator().manual_seed(35)


def rnd(*shape, scale=0.05):
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


p = "mtp.layers.0"
tensors = {
    "mtp.fc.weight": rnd(h, 2 * h),
    "mtp.pre_fc_norm_embedding.weight": rnd(h, scale=0.1),
    "mtp.pre_fc_norm_hidden.weight": rnd(h, scale=0.1),
    "mtp.norm.weight": rnd(h, scale=0.1),
    f"{p}.input_layernorm.weight": rnd(h, scale=0.1),
    f"{p}.post_attention_layernorm.weight": rnd(h, scale=0.1),
    f"{p}.self_attn.q_proj.weight": rnd(2 * heads * hd, h),
    f"{p}.self_attn.k_proj.weight": rnd(kv * hd, h),
    f"{p}.self_attn.v_proj.weight": rnd(kv * hd, h),
    f"{p}.self_attn.o_proj.weight": rnd(h, heads * hd),
    f"{p}.self_attn.q_norm.weight": rnd(hd, scale=0.1),
    f"{p}.self_attn.k_norm.weight": rnd(hd, scale=0.1),
    f"{p}.mlp.gate_proj.weight": rnd(inter, h),
    f"{p}.mlp.up_proj.weight": rnd(inter, h),
    f"{p}.mlp.down_proj.weight": rnd(h, inter),
}
save_file(tensors, str(out / "mtp.safetensors"))

with safe_open(str(src / "model.safetensors"), "pt") as f:
    names = list(f.keys())
weight_map = {n: "model.safetensors" for n in names}
weight_map.update({n: "mtp.safetensors" for n in tensors})
(out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=1) + "\n")
link = out / "model.safetensors"
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to((src / "model.safetensors").resolve())
