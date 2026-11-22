import json
import sys
from types import SimpleNamespace
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
V4 = "tiny-deepseek-v4-attention"
LLAMA4 = "tiny-llama4"
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


def build_llama4_config():
    from transformers import Llama4TextConfig

    return Llama4TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=128,
        intermediate_size_mlp=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=1,
        interleave_moe_layer_step=2,
        no_rope_layers=[1, 1, 1, 0],
        attention_chunk_size=CHUNK,
        use_qk_norm=True,
        attn_temperature_tuning=True,
        attn_scale=ATTN_SCALE,
        floor_scale=FLOOR_SCALE,
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
        index_n_heads=8,
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
        if name.endswith(".feed_forward.router.weight"):
            v = 6.0 * torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        elif name.endswith("router.bias"):
            v = 0.3 * torch.randn(p.shape, generator=g)
        elif name.endswith("e_score_correction_bias"):
            v = 0.2 * torch.randn(p.shape, generator=g)
        elif name.endswith("k_norm.bias"):
            v = 0.1 * torch.randn(p.shape, generator=g)
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
    elif name == LLAMA4:
        from transformers import Llama4ForCausalLM as Model

        cfg = build_llama4_config()
        extra = {"architectures": ["Llama4ForCausalLM"]}
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
    model.load_state_dict({k: v.float() for k, v in fuse_experts(load_file(path)).items() if not k.startswith(NEXTN_PREFIX)}, strict=True)
    return model


NEXTN_PREFIX = "model.layers.4."
NEXTN_PROMPT = 96


def glm_nextn_weights(model):
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

    cfg = model.config
    g = torch.Generator().manual_seed(SEED + 11)
    h = cfg.hidden_size
    layer = Glm4MoeDecoderLayer(cfg, 4)
    out = {}
    for name, p in sorted(layer.state_dict().items()):
        if name.endswith("e_score_correction_bias"):
            v = 0.2 * torch.randn(p.shape, generator=g)
        elif p.dim() == 1:
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[f"{NEXTN_PREFIX}{name}"] = v
    out[f"{NEXTN_PREFIX}eh_proj.weight"] = torch.randn((h, 2 * h), generator=g) / (2 * h) ** 0.5
    for n in ("enorm", "hnorm", "shared_head.norm"):
        out[f"{NEXTN_PREFIX}{n}.weight"] = 1.0 + 0.1 * torch.randn((h,), generator=g)
    out[f"{NEXTN_PREFIX}shared_head.head.weight"] = torch.randn((cfg.vocab_size, h), generator=g)
    return {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}


def add_glm_nextn(model):
    from safetensors.torch import load_file, save_file

    path = FIXTURES / GLM / "model.safetensors"
    have = load_file(path)
    if not any(k.startswith(NEXTN_PREFIX) for k in have):
        save_file({**have, **split_experts(glm_nextn_weights(model))}, path)
    return {k: v.float() for k, v in fuse_experts(load_file(path)).items() if k.startswith(NEXTN_PREFIX)}


def glm_nextn_reference(model, out):
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer, Glm4MoeRMSNorm

    raw = add_glm_nextn(model)
    cfg = model.config
    h = cfg.hidden_size
    layer = Glm4MoeDecoderLayer(cfg, 4).float().eval()
    layer.load_state_dict({k[len(NEXTN_PREFIX):]: v for k, v in raw.items() if not k[len(NEXTN_PREFIX):].startswith(("eh_proj", "enorm", "hnorm", "shared_head"))}, strict=True)
    fc = raw[f"{NEXTN_PREFIX}eh_proj.weight"]
    norms = {n: Glm4MoeRMSNorm(h, eps=cfg.rms_norm_eps) for n in ("enorm", "hnorm", "shared_head.norm")}
    for n, m in norms.items():
        m.weight.data = raw[f"{NEXTN_PREFIX}{n}.weight"]
    head = raw[f"{NEXTN_PREFIX}shared_head.head.weight"]

    def step(tokens, hidden, positions):
        e = norms["enorm"](model.model.embed_tokens(tokens))
        x = torch.cat([e, norms["hnorm"](hidden)], dim=-1) @ fc.T
        pos = positions[None]
        rope = model.model.rotary_emb(x, pos)
        mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos)
        x = layer(x, position_embeddings=rope, attention_mask=mask, position_ids=pos)
        g = norms["shared_head.norm"](x)
        return (g @ head.T)[0], g[0]

    g = torch.Generator().manual_seed(SEED + 5)
    ids = torch.randint(0, 512, (NEXTN_PROMPT,), generator=g)
    with torch.inference_mode():
        hidden = model.model(ids[None]).last_hidden_state
        target = model.lm_head(hidden)[0]
        t = int(target[-1].argmax())
        nxt = torch.cat([ids[1:], torch.tensor([t])])
        logits, mtp_hidden = step(nxt[None], hidden, torch.arange(NEXTN_PROMPT))
        d1 = int(logits[-1].argmax())
        chain_tokens = torch.cat([nxt, torch.tensor([d1])])
        chain_hidden = torch.cat([hidden, mtp_hidden[None, -1:]], dim=1)
        chain, _ = step(chain_tokens[None], chain_hidden, torch.arange(NEXTN_PROMPT + 1))
    for key, value in {"logits": logits[-1:], "chain": chain[-1:]}.items():
        value.detach().float().numpy().astype(np.float32).tofile(out / f"{GLM}.mtp.{key}.bin")
    (out / f"{GLM}.mtp.json").write_text(json.dumps({"tokens": ids.tolist(), "first": t, "draft": d1, "chain_draft": int(chain[-1].argmax())}))


def build_v4_config():
    from transformers import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=512,
        hidden_size=256,
        moe_intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=64,
        q_lora_rank=64,
        partial_rotary_factor=0.25,
        layer_types=["heavily_compressed_attention", "compressed_sparse_attention", "sliding_attention"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        compress_rope_theta=160000.0,
        rope_theta=10000.0,
        sliding_window=16,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=16,
        index_head_dim=32,
        index_topk=3,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hc_mult=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=None,
        attn_implementation="eager",
    )


class V4Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention, DeepseekV4RMSNorm, DeepseekV4RotaryEmbedding

        self.cfg = cfg
        self.rotary = DeepseekV4RotaryEmbedding(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(cfg.num_hidden_layers):
            layer = nn.Module()
            layer.self_attn = DeepseekV4Attention(cfg, i)
            layer.input_layernorm = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            layer.post_attention_layernorm = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(cfg.hidden_size, 512, bias=False)
            layer.mlp.up_proj = nn.Linear(cfg.hidden_size, 512, bias=False)
            layer.mlp.down_proj = nn.Linear(512, cfg.hidden_size, bias=False)
            self.layers.append(layer)
        self.norm = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, ids, trace=None):
        cfg = self.cfg
        t = ids.shape[1]
        pos = torch.arange(t)[None]
        h = self.embed(ids)
        emb = {k: self.rotary(h, position_ids=pos, layer_type=k) for k in ("main", "compress")}
        q = torch.arange(t)[:, None]
        k = torch.arange(t)[None, :]
        visible = (k <= q) & (q - k < cfg.sliding_window)
        mask = torch.zeros(1, 1, t, t).masked_fill(~visible, float("-inf"))
        for layer in self.layers:
            a, _ = layer.self_attn(layer.input_layernorm(h), emb, pos, mask)
            h = h + a
            x = layer.post_attention_layernorm(h)
            h = h + layer.mlp.down_proj(torch.nn.functional.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x))
            if trace is not None:
                trace.append(h[0].clone())
        return SimpleNamespace(logits=self.lm_head(self.norm(h)))


def v4_weights(model):
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, p in sorted(model.state_dict().items()):
        key = name
        if name.startswith("embed."):
            key = "model.embed_tokens.weight"
        elif name.startswith("norm."):
            key = "model." + name
        elif name.startswith("layers."):
            key = "model." + name
        elif name == "lm_head.weight":
            key = name
        elif name.startswith("rotary"):
            continue
        if name.endswith(".sinks"):
            v = 2.0 * torch.randn(p.shape, generator=g)
        elif name.endswith("position_bias"):
            v = 0.5 * torch.randn(p.shape, generator=g)
        elif p.dim() == 1:
            v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
        elif "embed" in name or name == "lm_head.weight":
            v = torch.randn(p.shape, generator=g)
        else:
            v = torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5
        out[key] = v.to(torch.bfloat16).contiguous()
    return out


def make_v4():
    from safetensors.torch import load_file, save_file

    cfg = build_v4_config()
    torch.manual_seed(SEED)
    model = V4Attention(cfg).float().eval()
    folder = FIXTURES / V4
    path = folder / "model.safetensors"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_file(v4_weights(model), path)
        raw = json.loads(cfg.to_json_string())
        raw.update({"model_type": "deepseek_v4_attention", "architectures": ["DeepseekV4AttentionForCausalLM"], "intermediate_size": 512})
        (folder / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    loaded = {}
    for k, v in load_file(path).items():
        if k == "model.embed_tokens.weight":
            k = "embed.weight"
        elif k.startswith("model."):
            k = k[len("model."):]
        loaded[k] = v.float()
    model.load_state_dict(loaded, strict=False)
    return model


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in (SINKS, CHUNKED, MLA, MLA_MOE, GLM, OSS, DSA, V4, LLAMA4):
        model = make_v4() if name == V4 else make(name)
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
        if name == GLM:
            glm_nextn_reference(model, out)


if __name__ == "__main__":
    main(sys.argv[1])
