import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED = 0
PROMPT = 96
DSA_PROMPT = 64
GREEDY = 32
FIXTURES = Path(__file__).parent.parent / "tests/fixtures"

SINKS = "tiny-sinks"
CHUNKED = "tiny-chunked"
MLA = "tiny-deepseek-v2"
MLA_MOE = "tiny-deepseek-v2-moe"
GLM = "tiny-glm4-moe"
OSS = "tiny-gpt-oss"
DSA = "tiny-glm-dsa"
HASH = "tiny-hash-glm4-moe"
WINDOW = 16
SOFTCAP = 8.0
CHUNK = 16
ATTN_SCALE = 0.5
FLOOR_SCALE = 16


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


def sinks_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = SOFTCAP * torch.tanh(scores / SOFTCAP)
    if attention_mask is not None:
        scores = scores + attention_mask
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([scores, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1, dtype=combined.dtype)
    out = torch.matmul(probs[..., :-1].to(value.dtype), value)
    return out.transpose(1, 2).contiguous(), probs[..., :-1]


def chunked_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    config = module.config
    t = query.shape[-2]
    pos = torch.arange(t)
    chunked_layer = config.layer_types[module.layer_idx] == "sliding_attention"
    visible = pos[None, :] <= pos[:, None]
    if chunked_layer:
        visible = visible & (pos[None, :] // CHUNK == pos[:, None] // CHUNK)
    if not module.use_rope:
        temperature = torch.log1p(torch.floor((pos.float() + 1.0) / FLOOR_SCALE)) * ATTN_SCALE + 1.0
        query = query * temperature.view(1, 1, t, 1)
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = scores.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=scores.dtype)
    out = torch.matmul(probs, value)
    return out.transpose(1, 2).contiguous(), probs


COMMON = dict(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=False,
    eos_token_id=None,
    bos_token_id=None,
    pad_token_id=None,
    attn_implementation="eager",
)


def build_sinks_config():
    from transformers import Qwen3Config

    return Qwen3Config(
        **COMMON,
        head_dim=64,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=WINDOW,
        use_sliding_window=True,
        max_window_layers=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


def build_chunked_config():
    from transformers import SmolLM3Config

    return SmolLM3Config(
        **COMMON,
        no_rope_layers=[1, 0, 1, 0],
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=CHUNK,
        use_sliding_window=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100000.0},
    )


def build_mla_config(dense_layers=4):
    from transformers import DeepseekV2Config

    return DeepseekV2Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=96,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        first_k_dense_replace=dense_layers,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


def build_glm_config():
    from transformers import Glm4MoeConfig

    return Glm4MoeConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        use_qk_norm=True,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    )


def build_dsa_config():
    from transformers import GlmMoeDsaConfig

    return GlmMoeDsaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=96,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=8,
        indexer_types=["full", "shared"],
        first_k_dense_replace=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


def build_oss_config():
    from transformers import GptOssConfig

    return GptOssConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        sliding_window=16,
        swiglu_alpha=1.702,
        swiglu_limit=2.5,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def mxfp4_dequant(blocks, scales):
    lut = torch.tensor(E2M1 + [-v for v in E2M1])
    lo = lut[(blocks & 0xF).long()]
    hi = lut[(blocks >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)
    out = vals * torch.pow(2.0, scales.float() - 127.0).unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


def oss_experts(model):
    g = torch.Generator().manual_seed(SEED + 9)
    tensors, dequant = {}, {}
    for i, layer in enumerate(model.model.layers):
        e = layer.mlp.experts
        base = f"model.layers.{i}.mlp.experts"
        for name, rows, cols in [("gate_up_proj", 2 * e.intermediate_size, e.hidden_size), ("down_proj", e.hidden_size, e.intermediate_size)]:
            blocks = torch.randint(0, 256, (e.num_experts, rows, cols // 32, 16), generator=g, dtype=torch.uint8)
            scales = torch.randint(121, 124, (e.num_experts, rows, cols // 32), generator=g, dtype=torch.uint8)
            tensors[f"{base}.{name}_blocks"] = blocks
            tensors[f"{base}.{name}_scales"] = scales
            dequant[f"{base}.{name}"] = mxfp4_dequant(blocks, scales).transpose(1, 2).contiguous()
        for name in ("gate_up_proj_bias", "down_proj_bias"):
            shape = getattr(e, name).shape
            v = (0.1 * torch.randn(shape, generator=g)).to(torch.bfloat16)
            tensors[f"{base}.{name}"] = v
            dequant[f"{base}.{name}"] = v
    return tensors, dequant


def weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        if ".mlp.experts." in name and "bias" not in name:
            continue
        if name.endswith("router.bias"):
            v = 0.3 * torch.randn(p.shape, generator=g)
        elif name.endswith("e_score_correction_bias"):
            v = 0.2 * torch.randn(p.shape, generator=g)
        elif name.endswith(".sinks"):
            v = 2.0 * torch.randn(p.shape, generator=g)
        elif p.dim() == 1:
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        elif "embed_tokens" in name or name == "lm_head.weight":
            v = torch.randn(p.shape, generator=g)
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[name] = v.to(torch.bfloat16).contiguous()
    return out


def split_experts(tensors):
    out = {}
    for name, v in tensors.items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            half = v.shape[1] // 2
            for e in range(v.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = v[e, :half].contiguous()
                out[f"{base}{e}.up_proj.weight"] = v[e, half:].contiguous()
        elif name.endswith(".mlp.experts.down_proj"):
            base = name[: -len("down_proj")]
            for e in range(v.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            out[name] = v
    return out


def fuse_experts(tensors):
    out = {}
    experts = {}
    for name, v in tensors.items():
        parts = name.split(".")
        if ".mlp.experts." in name and parts[-3].isdigit():
            experts.setdefault(".".join(parts[:-3]), {})[(int(parts[-3]), parts[-2])] = v
        else:
            out[name] = v
    for base, items in experts.items():
        n = 1 + max(e for e, _ in items)
        out[f"{base}.gate_up_proj"] = torch.stack([torch.cat([items[(e, "gate_proj")], items[(e, "up_proj")]]) for e in range(n)])
        out[f"{base}.down_proj"] = torch.stack([items[(e, "down_proj")] for e in range(n)])
    return out


def make(name):
    from safetensors.torch import load_file, save_file

    if name in (MLA, MLA_MOE):
        from transformers import DeepseekV2ForCausalLM as Model

        cfg = build_mla_config(4 if name == MLA else 1)
        extra = {}
    elif name == DSA:
        from transformers import GlmMoeDsaForCausalLM as Model

        cfg = build_dsa_config()
        extra = {}
    elif name == GLM:
        from transformers import Glm4MoeForCausalLM as Model

        cfg = build_glm_config()
        extra = {}
    elif name == OSS:
        from transformers import GptOssForCausalLM as Model

        cfg = build_oss_config()
        extra = {}
    elif name == SINKS:
        import transformers.models.qwen3.modeling_qwen3 as modeling
        from transformers import Qwen3ForCausalLM as Model

        cfg = build_sinks_config()
        modeling.eager_attention_forward = sinks_attention
        extra = {"model_type": "sinks_qwen3", "architectures": ["SinksQwen3ForCausalLM"], "attn_logit_softcapping": SOFTCAP}
    else:
        import transformers.models.smollm3.modeling_smollm3 as modeling
        from transformers import SmolLM3ForCausalLM as Model

        cfg = build_chunked_config()
        modeling.eager_attention_forward = chunked_attention
        extra = {
            "model_type": "chunked_smollm3",
            "architectures": ["ChunkedSmolLM3ForCausalLM"],
            "attn_scale": ATTN_SCALE,
            "floor_scale": FLOOR_SCALE,
            "attn_temperature_tuning": True,
        }
    torch.manual_seed(SEED)
    model = Model(cfg).float().eval()
    if name == SINKS:
        for layer in model.model.layers:
            layer.self_attn.sinks = nn.Parameter(torch.zeros(cfg.num_attention_heads))
    folder = FIXTURES / name
    path = folder / "model.safetensors"
    if name == OSS:
        tensors, dequant = oss_experts(model)
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            save_file({**weights(model), **tensors}, path)
            raw = json.loads(cfg.to_json_string())
            raw.update(extra)
            (folder / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
        loaded = {k: v.float() for k, v in load_file(path).items() if "_blocks" not in k and "_scales" not in k}
        loaded.update({k: v.float() for k, v in dequant.items()})
        model.load_state_dict(loaded, strict=True)
        return model
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_file(split_experts(weights(model)), path)
        raw = json.loads(cfg.to_json_string())
        raw.update(extra)
        (folder / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    model.load_state_dict({k: v.float() for k, v in fuse_experts(load_file(path)).items()}, strict=True)
    return model


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in (SINKS, CHUNKED, MLA, MLA_MOE, GLM, OSS, DSA):
        model = make(name)
        g = torch.Generator().manual_seed(SEED + 3)
        prompt = DSA_PROMPT if name == DSA else PROMPT
        ids = torch.randint(0, 512, (prompt,), generator=g)
        with torch.inference_mode():
            logits = model(ids[None]).logits[0]
            seq = ids.tolist()
            for _ in range(GREEDY):
                step = model(torch.tensor(seq)[None]).logits[0, -1]
                seq.append(int(step.argmax()))
            greedy = seq[prompt:]
        logits.float().numpy().astype(np.float32).tofile(out / f"{name}.ref.bin")
        (out / f"{name}.ref.json").write_text(json.dumps({"tokens": ids.tolist(), "greedy": greedy, "vocab": 512}))
        print(name, "->", out)


if __name__ == "__main__":
    main(sys.argv[1])
