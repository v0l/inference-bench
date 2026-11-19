import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

BLOCK = 128


class Shards:
    def __init__(self, model):
        self.dir = Path(model)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(self.dir / f, framework="pt", device="cuda")
        return self.files[f].get_tensor(name)

    def dequant(self, name):
        w = self.get(name)
        if w.dtype != torch.float8_e4m3fn:
            return w.float()
        s = self.get(name.replace(".weight", ".weight_scale_inv")).float()
        w = w.float()
        s = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[: w.shape[0], : w.shape[1]]
        return w * s


def load_layer(layer, shards, i, cfg):
    p = f"model.layers.{i}"
    sd = {}
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        sd[f"self_attn.{proj}.weight"] = shards.dequant(f"{p}.self_attn.{proj}.weight")
    for norm in ["q_norm", "k_norm"]:
        sd[f"self_attn.{norm}.weight"] = shards.get(f"{p}.self_attn.{norm}.weight").float()
    sd["input_layernorm.weight"] = shards.get(f"{p}.input_layernorm.weight").float()
    sd["post_attention_layernorm.weight"] = shards.get(f"{p}.post_attention_layernorm.weight").float()
    sd["mlp.gate.weight"] = shards.get(f"{p}.mlp.gate.weight").float()
    gu, dn = [], []
    for e in range(cfg.num_experts):
        q = f"{p}.mlp.experts.{e}"
        gu.append(torch.cat([shards.dequant(f"{q}.gate_proj.weight"), shards.dequant(f"{q}.up_proj.weight")], 0))
        dn.append(shards.dequant(f"{q}.down_proj.weight"))
    sd["mlp.experts.gate_up_proj"] = torch.stack(gu)
    sd["mlp.experts.down_proj"] = torch.stack(dn)
    layer.load_state_dict(sd, assign=True)


def main(model, out):
    cfg = AutoConfig.from_pretrained(model)
    cfg.quantization_config = None
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    shards = Shards(model)
    prompts = json.loads((Path(out) / "prompts.json").read_text())
    emb = shards.get("model.embed_tokens.weight").float()
    norm_w = shards.get("model.norm.weight").float()
    head = shards.get("lm_head.weight").float()
    rot = m.model.rotary_emb.to_empty(device="cuda")
    dim = cfg.head_dim
    inv = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cuda") / dim))
    scaling = 1.0
    rot.inv_freq = inv
    rot.original_inv_freq = inv
    rot.attention_scaling = scaling
    hs = [emb[torch.tensor(p, device="cuda")][None] for p in prompts]
    pos = [torch.arange(len(p), device="cuda")[None] for p in prompts]
    pe = [rot(h, ps) for h, ps in zip(hs, pos)]
    masks = [torch.full((len(p), len(p)), float("-inf"), device="cuda").triu(1)[None, None] for p in prompts]
    with torch.inference_mode():
        for i in range(cfg.num_hidden_layers):
            layer = m.model.layers[i]
            load_layer(layer, shards, i, cfg)
            for k in range(len(prompts)):
                r = layer(hs[k], attention_mask=masks[k], position_ids=pos[k], position_embeddings=pe[k])
                hs[k] = r[0] if isinstance(r, tuple) else r
            for prm in layer.parameters():
                prm.data = torch.empty(0)
            torch.cuda.empty_cache()
            print("layer", i, flush=True)
        rows = []
        for h in hs:
            v = h.float()
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w
            rows.append((v @ head.T)[0].cpu().numpy())
    full = np.concatenate(rows)
    np.save(Path(out) / "fp32.npy", full)
    print("fp32", len(full), "rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
