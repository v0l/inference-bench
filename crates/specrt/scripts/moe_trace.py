import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def main(repo_dir, ids_file, out_dir):
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    dev = "cuda"
    src = Path(repo_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    text = json.loads((src / "config.json").read_text())["text_config"]
    config = Qwen3_5MoeTextConfig(**{k: v for k, v in text.items() if k != "model_type"})
    config._attn_implementation = "sdpa"
    ids = json.loads(Path(ids_file).read_text())
    handles = {}

    def tensor(name):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(src / shard, "pt")
        return handles[shard].get_tensor(name).to(dev)

    def dq(prefix):
        packed = tensor(prefix + ".weight_packed")
        rows, half = packed.shape
        q = unpack_fp4_from_uint8(packed, rows, half * 2, dtype=torch.float32)
        scale = tensor(prefix + ".weight_scale").float()
        glob = tensor(prefix + ".weight_global_scale").float()
        return (q.reshape(rows, -1, 16) * (scale / glob)[..., None]).reshape(rows, half * 2)

    root = "model.language_model."
    embed = tensor(root + "embed_tokens.weight")
    hidden = embed[torch.tensor(ids, device=dev)].float()[None]
    del embed
    rotary = Qwen3_5MoeTextRotaryEmbedding(config).to(dev)
    trace = []
    started = time.time()
    with torch.inference_mode():
        for i in range(config.num_hidden_layers):
            prefix = f"{root}layers.{i}."
            with torch.device(dev):
                layer = Qwen3_5MoeDecoderLayer(config, i).float().eval()
            state = {}
            for name in [n for n in index if n.startswith(prefix)]:
                short = name[len(prefix):]
                if ".mlp.experts." in name or ".mlp.shared_expert." in name:
                    continue
                if short.endswith(".weight_packed"):
                    state[short[: -len("_packed")]] = dq(name[: -len(".weight_packed")])
                    continue
                if short.endswith((".weight_scale", ".weight_global_scale", ".input_global_scale")):
                    continue
                state[short] = tensor(name).float()
            gate_up = torch.empty(config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size, device=dev)
            down = torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size, device=dev)
            for e in range(config.num_experts):
                p = f"{prefix}mlp.experts.{e}."
                gate_up[e] = torch.cat([dq(p + "gate_proj"), dq(p + "up_proj")])
                down[e] = dq(p + "down_proj")
            state["mlp.experts.gate_up_proj"] = gate_up
            state["mlp.experts.down_proj"] = down
            for proj in ("gate_proj", "up_proj", "down_proj"):
                state[f"mlp.shared_expert.{proj}.weight"] = dq(f"{prefix}mlp.shared_expert.{proj}")
            layer.load_state_dict(state, strict=True)
            pos = torch.arange(hidden.shape[1], device=dev)[None]
            emb = rotary(hidden, pos)
            hidden = layer(hidden, position_embeddings=emb, attention_mask=None, position_ids=pos)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            trace.append(hidden[0].clone().cpu())
            del layer, state, gate_up, down
            torch.cuda.empty_cache()
            print(f"layer {i} {time.time() - started:.0f}s", flush=True)
        norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dev)
        norm.load_state_dict({"weight": tensor(root + "norm.weight").float()})
        head = tensor("lm_head.weight").float()
        logits = (norm(hidden[0]) @ head.T).cpu()
    torch.stack(trace).numpy().astype(np.float32).tofile(out / "moe122-ref.layers.bin")
    logits.numpy().astype(np.float32).tofile(out / "moe122-ref.logits.bin")
    (out / "moe122-ref.json").write_text(json.dumps({"ids": ids, "argmax": logits.argmax(-1).tolist()}))
    print("done", time.time() - started)


if __name__ == "__main__":
    main(*sys.argv[1:4])
